from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from rapidfuzz.fuzz import token_set_ratio

from main import run as run_queue


ACTIONS = ["notify", "digest", "mute"]
ENTITY_PATTERNS = [
    re.compile(r"@[A-Za-z0-9_]+"),
    re.compile(r"https?://[^\s,;]+|www\.[^\s,;]+", re.I),
    re.compile(r"\b[A-Za-z0-9.-]+\.(?:com|org|net|io|in|co|app|biz)\b", re.I),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)"),
    re.compile(r"(?:[$€£₹]\s?\d[\d,.]*|\b\d[\d,.]*\s?(?:USD|EUR|GBP|INR|dollars?|rupees?)\b)", re.I),
    re.compile(r"\b(?:\d{1,2}[:.]\d{2}\s?(?:am|pm)?|\d{1,2}\s?(?:am|pm)|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b", re.I),
    re.compile(r"\b(?:[A-Z]{2,}[A-Z0-9_-]*\d+[A-Z0-9_-]*|\d{4,})\b"),
    re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b"),
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"file does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty per-message evaluation")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evidence_set(value: str) -> set[str]:
    return set() if not value or value.strip().lower() == "none" else {
        item.strip() for item in value.split(";") if item.strip()
    }


def set_f1(predicted: set[str], expected: set[str]) -> float:
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    precision = len(predicted & expected) / len(predicted)
    recall = len(predicted & expected) / len(expected)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def extract_entities(text: str) -> set[str]:
    entities: set[str] = set()
    for pattern in ENTITY_PATTERNS:
        for match in pattern.findall(text or ""):
            value = " ".join(str(match).strip(".,;:!?()[]{}\"'").lower().split())
            if value:
                entities.add(value)
    return entities


def summary_score(source: str, predicted_reason: str, reference_reason: str,
                  predicted_type: str, reference_type: str) -> tuple[float, float, float]:
    predicted_entities = extract_entities(predicted_reason)
    reference_entities = extract_entities(reference_reason)
    allowed_entities = extract_entities(source) | reference_entities
    entity_precision = (
        len(predicted_entities & allowed_entities) / len(predicted_entities)
        if predicted_entities else 1.0
    )
    entity_recall = (
        len(predicted_entities & reference_entities) / len(reference_entities)
        if reference_entities else 1.0
    )
    entity_f1 = (
        2 * entity_precision * entity_recall / (entity_precision + entity_recall)
        if entity_precision + entity_recall else 0.0
    )
    intent = 0.6 * (predicted_type == reference_type) + 0.4 * (
        token_set_ratio(predicted_reason, reference_reason) / 100.0
    )
    return 0.5 * entity_f1 + 0.5 * intent, entity_f1, intent


def expected_calibration_error(confidences: list[float], correct: list[bool], bins: int = 10) -> tuple[float, list[dict[str, Any]]]:
    if not confidences:
        return 0.0, []
    details: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [i for i, value in enumerate(confidences) if low <= value <= high and (index == bins - 1 or value < high)]
        if members:
            avg_conf = sum(confidences[i] for i in members) / len(members)
            accuracy = sum(bool(correct[i]) for i in members) / len(members)
            ece += len(members) / len(confidences) * abs(avg_conf - accuracy)
        else:
            avg_conf = accuracy = 0.0
        details.append({"lower": low, "upper": high, "count": len(members), "confidence": avg_conf, "accuracy": accuracy})
    return ece, details


def _load_trace(path: Path, expected_ids: set[str]) -> tuple[str, list[dict[str, Any]]]:
    if not path.is_file():
        raise ValueError(f"trace file does not exist: {path}")
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_no}") from exc
            by_run[str(item.get("run_id", ""))].append(item)
    candidates = []
    for run_id, events in by_run.items():
        ended = {str(item.get("message_id")) for item in events if item.get("event") == "message_end"}
        if expected_ids <= ended:
            candidates.append((max(str(item.get("timestamp", "")) for item in events), run_id, events))
    if not candidates:
        raise ValueError("trace does not contain message_end events for every evaluated ID in one run")
    _, run_id, events = max(candidates)
    return run_id, events


def _tool_score(events: list[dict[str, Any]], modality: str) -> tuple[float, dict[str, float]]:
    starts = [item for item in events if item.get("event") == "tool_start"]
    names = [str(item.get("tool", "")) for item in starts]
    required = (["process_image"] if modality == "image" else ["process_audio"] if modality == "voice" else []) + ["scan_scam_heuristics", "write_final_classification"]
    positions = []
    for name in required:
        positions.append(names.index(name) if name in names else -1)
    presence_order = float(all(value >= 0 for value in positions) and positions == sorted(positions))
    expected_calls = len(required) + (1 if names.count("query_user_history") else 0)
    economy = min(1.0, expected_calls / max(1, len(names)))

    retries = [item for item in events if item.get("event") == "retry"]
    retry_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in retries:
        retry_groups[str(item.get("operation", ""))].append(item)
    retry_ok = all(
        len(items) <= 2 and [float(item.get("delay_seconds", -1)) for item in items] == [0.5, 1.0][:len(items)]
        for items in retry_groups.values()
    )
    retry_score = float(retry_ok)
    endings = [item for item in events if item.get("event") == "message_end"]
    status = str(endings[-1].get("status", "error")) if endings else "error"
    completion = 1.0 if status == "ok" else 0.75 if status == "degraded" else 0.0
    total = 0.4 * presence_order + 0.2 * economy + 0.2 * retry_score + 0.2 * completion
    return total, {"required_order": presence_order, "economy": economy, "retry_handling": retry_score, "completion": completion}


def evaluate(ground_truth_path: Path, predictions_path: Path, trace_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    truth_rows = _read_csv(ground_truth_path)
    prediction_rows = _read_csv(predictions_path)
    truth = {row["message_id"]: row for row in truth_rows}
    predictions = {row["message_id"]: row for row in prediction_rows}
    if len(truth) != len(truth_rows) or len(predictions) != len(prediction_rows):
        raise ValueError("ground truth and predictions must have unique message IDs")
    if set(truth) != set(predictions):
        missing = sorted(set(truth) - set(predictions))
        extra = sorted(set(predictions) - set(truth))
        raise ValueError(f"prediction coverage mismatch; missing={missing}, extra={extra}")

    run_id, trace_events = _load_trace(trace_path, set(truth))
    events_by_message: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in trace_events:
        message_id = str(event.get("message_id", ""))
        if message_id in truth:
            events_by_message[message_id].append(event)

    confusion = {actual: {predicted: 0 for predicted in ACTIONS} for actual in ACTIONS}
    per_message: list[dict[str, Any]] = []
    confidences: list[float] = []
    correctness: list[bool] = []
    false_notify = 0
    for message_id, reference in truth.items():
        predicted = predictions[message_id]
        actual_action, predicted_action = reference["action"], predicted["action"]
        if actual_action not in ACTIONS or predicted_action not in ACTIONS:
            raise ValueError(f"invalid action for {message_id}")
        confusion[actual_action][predicted_action] += 1
        is_correct = actual_action == predicted_action
        if predicted_action == "notify" and actual_action != "notify":
            false_notify += 1
        evidence = set_f1(evidence_set(predicted["evidence_message_ids"]), evidence_set(reference["evidence_message_ids"]))
        summary, entity_f1, intent = summary_score(
            reference.get("message_text", ""), predicted.get("reason", ""), reference.get("reason", ""),
            predicted.get("message_type", ""), reference.get("message_type", ""),
        )
        modality = reference.get("media_type", "") or "text"
        tool, tool_parts = _tool_score(events_by_message[message_id], modality)
        confidence = float(predicted["confidence"])
        if not 0 <= confidence <= 1:
            raise ValueError(f"confidence outside [0, 1] for {message_id}")
        confidences.append(confidence)
        correctness.append(is_correct)
        per_message.append({
            "message_id": message_id, "modality": modality,
            "actual_action": actual_action, "predicted_action": predicted_action,
            "correct": int(is_correct), "confidence": confidence,
            "evidence_f1": evidence, "entity_f1": entity_f1,
            "intent_score": intent, "summary_score": summary,
            "tool_score": tool, "tool_required_order": tool_parts["required_order"],
            "tool_economy": tool_parts["economy"], "tool_retry_handling": tool_parts["retry_handling"],
            "tool_completion": tool_parts["completion"],
        })

    count = len(per_message)
    accuracy = sum(item["correct"] for item in per_message) / count
    action_score = 100 * max(0.0, accuracy - 0.5 * false_notify / count)
    evidence_score = 100 * sum(item["evidence_f1"] for item in per_message) / count
    preservation_score = 100 * sum(item["summary_score"] for item in per_message) / count
    tool_score = 100 * sum(item["tool_score"] for item in per_message) / count
    composite = 0.4 * action_score + 0.3 * evidence_score + 0.2 * preservation_score + 0.1 * tool_score

    class_metrics: dict[str, dict[str, float | int]] = {}
    for action in ACTIONS:
        tp = confusion[action][action]
        fp = sum(confusion[actual][action] for actual in ACTIONS if actual != action)
        fn = sum(confusion[action][other] for other in ACTIONS if other != action)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_metrics[action] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(confusion[action].values())}

    ece, calibration = expected_calibration_error(confidences, correctness)
    modality_metrics: dict[str, dict[str, float | int]] = {}
    for modality in ["text", "image", "voice"]:
        items = [item for item in per_message if item["modality"] == modality]
        modality_metrics[modality] = {
            "count": len(items),
            "action_accuracy": sum(item["correct"] for item in items) / len(items) if items else 0.0,
        }

    tool_ends = [item for item in trace_events if item.get("event") == "tool_end" and item.get("message_id") in truth]
    retries = [item for item in trace_events if item.get("event") == "retry" and item.get("message_id") in truth]
    errors = [item for item in tool_ends if item.get("status") not in {"ok", None}]
    metrics = {
        "run_id": run_id, "message_count": count,
        "composite_health_score": composite,
        "components": {
            "action": action_score, "evidence": evidence_score,
            "summary": preservation_score, "tools": tool_score,
        },
        "action_accuracy": accuracy, "false_notify_count": false_notify,
        "per_action": class_metrics, "confusion_matrix": confusion,
        "evidence_citation_f1": evidence_score / 100,
        "entity_intent_preservation": preservation_score / 100,
        "expected_calibration_error": ece, "calibration_bins": calibration,
        "modality": modality_metrics,
        "tool_execution": {
            "call_count": len(tool_ends), "retry_count": len(retries), "error_count": len(errors),
            "success_rate": (len(tool_ends) - len(errors)) / len(tool_ends) if tool_ends else 0.0,
            "error_rate": len(errors) / len(tool_ends) if tool_ends else 0.0,
            "retry_rate": len(retries) / max(1, len(tool_ends)),
        },
    }
    return metrics, per_message


def create_dashboard(metrics: dict[str, Any], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)

    radar = fig.add_subplot(grid[0, 0], polar=True)
    labels = ["Action", "Evidence", "Summary", "Tools"]
    values = [metrics["components"][key] for key in ["action", "evidence", "summary", "tools"]]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    radar.plot(angles + angles[:1], values + values[:1], color="#1677b8", linewidth=2)
    radar.fill(angles + angles[:1], values + values[:1], color="#1677b8", alpha=0.2)
    radar.set_xticks(angles, labels)
    radar.set_ylim(0, 100)
    radar.set_title(f"Composite Health: {metrics['composite_health_score']:.1f}/100", pad=18)

    heat = fig.add_subplot(grid[0, 1])
    matrix = [[metrics["confusion_matrix"][actual][pred] for pred in ACTIONS] for actual in ACTIONS]
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=ACTIONS, yticklabels=ACTIONS, ax=heat)
    heat.set(xlabel="Predicted", ylabel="Actual", title="Action Confusion Matrix")

    calibration = fig.add_subplot(grid[1, 0])
    bins = [item for item in metrics["calibration_bins"] if item["count"]]
    calibration.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    if bins:
        calibration.plot([item["confidence"] for item in bins], [item["accuracy"] for item in bins], marker="o", color="#d95f02", label="Observed")
    calibration.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Accuracy",
                    title=f"Confidence Calibration (ECE {metrics['expected_calibration_error']:.3f})")
    calibration.legend(loc="lower right")

    breakdown = fig.add_subplot(grid[1, 1])
    modalities = ["text", "image", "voice"]
    modality_values = [100 * metrics["modality"][item]["action_accuracy"] for item in modalities]
    tool = metrics["tool_execution"]
    labels2 = modalities + ["tool success", "tool errors"]
    values2 = modality_values + [100 * tool["success_rate"], 100 * tool["error_rate"]]
    colors = ["#4c78a8"] * 3 + ["#59a14f", "#e15759"]
    bars = breakdown.bar(labels2, values2, color=colors)
    breakdown.bar_label(bars, fmt="%.1f", padding=3)
    breakdown.set_ylim(0, 110)
    breakdown.set_ylabel("Percent")
    breakdown.set_title("Modality & Tool Execution Breakdown")
    breakdown.tick_params(axis="x", rotation=20)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": "Message Router Evaluator"})
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate router predictions and create a dashboard.")
    parser.add_argument("--run-pipeline", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ground-truth", type=Path, default=REPO_ROOT / "dataset" / "sample_messages.csv")
    parser.add_argument("--predictions", type=Path, default=REPO_ROOT / "logs" / "sample_predictions.csv")
    parser.add_argument("--trace", type=Path, default=REPO_ROOT / "logs" / "evaluation_trace.jsonl")
    parser.add_argument("--metrics", type=Path, default=REPO_ROOT / "logs" / "eval_metrics.json")
    parser.add_argument("--per-message", type=Path, default=REPO_ROOT / "logs" / "eval_per_message.csv")
    parser.add_argument("--dashboard", type=Path, default=REPO_ROOT / "logs" / "eval_dashboard.png")
    return parser


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.run_pipeline:
        queue_args = [
            "--dataset-dir", str(REPO_ROOT / "dataset"), "--input", str(args.ground_truth),
            "--output", str(args.predictions), "--trace", str(args.trace), "--no-color",
        ]
        if args.force:
            queue_args.append("--force")
        queue_status = run_queue(queue_args)
        if queue_status != 0:
            return queue_status
    try:
        metrics, rows = evaluate(args.ground_truth.resolve(), args.predictions.resolve(), args.trace.resolve())
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_csv(args.per_message, rows)
        create_dashboard(metrics, args.dashboard)
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"composite_health_score={metrics['composite_health_score']:.2f}")
    print(f"metrics={args.metrics.resolve()}")
    print(f"dashboard={args.dashboard.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
