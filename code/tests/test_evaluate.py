from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from evaluate.main import create_dashboard, evaluate, expected_calibration_error, set_f1, summary_score


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class EvaluationTests(unittest.TestCase):
    def test_formula_helpers(self) -> None:
        self.assertEqual(set_f1(set(), set()), 1.0)
        self.assertEqual(set_f1({"a"}, set()), 0.0)
        self.assertAlmostEqual(set_f1({"a", "b"}, {"b", "c"}), 0.5)
        score, entity, intent = summary_score(
            "Meet Alice at 5 PM", "Meet Alice at 5 PM", "Meet Alice at 5 PM", "event", "event"
        )
        self.assertAlmostEqual(score, 1.0)
        self.assertAlmostEqual(entity, 1.0)
        self.assertAlmostEqual(intent, 1.0)
        ece, bins = expected_calibration_error([0.9, 0.2], [True, False])
        self.assertAlmostEqual(ece, 0.15)
        self.assertEqual(len(bins), 10)

    def test_metrics_and_four_panel_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            truth_rows = [
                {"message_id": "s1", "message_text": "Meet Alice at 5 PM", "media_type": "", "action": "notify", "message_type": "event", "reason": "Meet Alice at 5 PM.", "confidence": "0.9", "evidence_message_ids": "message_0001"},
                {"message_id": "s2", "message_text": "Monthly newsletter", "media_type": "image", "action": "digest", "message_type": "business_update", "reason": "A routine monthly newsletter can wait.", "confidence": "0.8", "evidence_message_ids": "none"},
                {"message_id": "s3", "message_text": "Win now", "media_type": "voice", "action": "mute", "message_type": "spam", "reason": "Unwanted prize spam should be muted.", "confidence": "0.9", "evidence_message_ids": "message_0003"},
            ]
            prediction_rows = [
                {"message_id": "s1", "action": "notify", "message_type": "event", "reason": "Meet Alice at 5 PM.", "confidence": "0.9", "evidence_message_ids": "message_0001"},
                {"message_id": "s2", "action": "notify", "message_type": "business_update", "reason": "A routine monthly newsletter can wait.", "confidence": "0.8", "evidence_message_ids": "none"},
                {"message_id": "s3", "action": "mute", "message_type": "spam", "reason": "Unwanted prize spam should be muted.", "confidence": "0.9", "evidence_message_ids": "message_0003"},
            ]
            truth_path, predictions_path, trace_path = root / "truth.csv", root / "pred.csv", root / "trace.jsonl"
            write_csv(truth_path, truth_rows)
            write_csv(predictions_path, prediction_rows)
            events = []
            for item in truth_rows:
                message_id, modality = item["message_id"], item["media_type"] or "text"
                names = (["process_image"] if modality == "image" else ["process_audio"] if modality == "voice" else []) + ["scan_scam_heuristics", "write_final_classification"]
                events.append({"timestamp": "2026-01-01T00:00:00Z", "run_id": "run", "event": "message_start", "message_id": message_id})
                for name in names:
                    events.extend([
                        {"timestamp": "2026-01-01T00:00:01Z", "run_id": "run", "event": "tool_start", "message_id": message_id, "tool": name},
                        {"timestamp": "2026-01-01T00:00:02Z", "run_id": "run", "event": "tool_end", "message_id": message_id, "tool": name, "status": "ok"},
                    ])
                events.append({"timestamp": "2026-01-01T00:00:03Z", "run_id": "run", "event": "message_end", "message_id": message_id, "status": "ok"})
            trace_path.write_text("".join(json.dumps(item) + "\n" for item in events), encoding="utf-8")

            metrics, per_message = evaluate(truth_path, predictions_path, trace_path)
            self.assertAlmostEqual(metrics["components"]["action"], 50.0)
            self.assertEqual(metrics["false_notify_count"], 1)
            self.assertEqual(len(per_message), 3)
            self.assertAlmostEqual(metrics["tool_execution"]["success_rate"], 1.0)
            dashboard = root / "dashboard.png"
            create_dashboard(metrics, dashboard)
            self.assertTrue(dashboard.is_file())
            self.assertGreater(dashboard.stat().st_size, 10_000)
            self.assertEqual(dashboard.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_prediction_coverage_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            truth = [{"message_id": "s1", "message_text": "x", "media_type": "", "action": "mute", "message_type": "spam", "reason": "x", "confidence": "1", "evidence_message_ids": "none"}]
            prediction = [{"message_id": "other", "action": "mute", "message_type": "spam", "reason": "x", "confidence": "1", "evidence_message_ids": "none"}]
            write_csv(root / "truth.csv", truth)
            write_csv(root / "pred.csv", prediction)
            (root / "trace.jsonl").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                evaluate(root / "truth.csv", root / "pred.csv", root / "trace.jsonl")


if __name__ == "__main__":
    unittest.main()
