from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from pydantic import ValidationError

from router.graph import MessageRouter, create_openai_model
from router.io import DatasetError, OutputContractError, OutputStore, load_messages
from router.tracing import TerminalReporter, TraceRecorder


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"
DEFAULT_TRACE = REPO_ROOT / "logs" / "execution_trace.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Route WhatsApp messages into notify, digest, or mute."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Input CSV (default: <dataset-dir>/messages.csv).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output CSV (default: <dataset-dir>/output.csv).",
    )
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--message-id")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    return parser


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        print("error: --limit must be at least 1", file=sys.stderr)
        return 2

    dataset_dir = args.dataset_dir.resolve()
    input_path = (args.input or dataset_dir / "messages.csv").resolve()
    output_path = (args.output or dataset_dir / "output.csv").resolve()
    reporter = TerminalReporter(color=not args.no_color)
    try:
        trace = TraceRecorder(args.trace, reporter=reporter)
        messages = load_messages(dataset_dir, input_path)
        output_store = OutputStore(dataset_dir, output_path, messages)
    except (DatasetError, OutputContractError, OSError, ValidationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    selected = messages
    if args.message_id:
        selected = [item for item in messages if item.message_id == args.message_id]
        if not selected:
            print(f"error: unknown message_id {args.message_id!r}", file=sys.stderr)
            return 2
    if args.limit is not None:
        selected = selected[: args.limit]

    pending = [
        item for item in selected
        if args.force or not output_store.is_complete(item.message_id)
    ]
    skipped = len(selected) - len(pending)
    if pending and not os.environ.get("OPENAI_API_KEY"):
        print("error: OPENAI_API_KEY is required when messages need processing", file=sys.stderr)
        return 2

    trace.emit(
        "queue_start", input=input_path.name, output=output_path.name,
        selected=len(selected), pending=len(pending), skipped=skipped, force=args.force,
    )
    written = fallback_count = system_fallback_count = failed = 0
    queue_began = time.perf_counter()
    if pending:
        try:
            router = MessageRouter(create_openai_model(), output_store, trace=trace)
        except Exception as exc:
            trace.emit("queue_end", status="configuration_error", error_category=type(exc).__name__)
            print(f"error: {exc}", file=sys.stderr)
            return 2

        for index, message in enumerate(pending, start=1):
            reporter.write(f"[{index}/{len(pending)}] routing {message.message_id}", "blue")
            trace.emit(
                "message_start", message_id=message.message_id,
                queue_index=index, modality=message.media_type or "text",
            )
            began = time.perf_counter()
            try:
                result = router.classify(message)
                written += 1
                if result.used_fallback:
                    fallback_count += 1
                    if result.diagnostics.system_failure:
                        system_fallback_count += 1
                    reporter.write(
                        f"  fallback: {result.diagnostics.error_category or 'unknown'}", "yellow"
                    )
                reporter.write(
                    f"  wrote {result.classification.action.value}/"
                    f"{result.classification.message_type.value}",
                    "green" if not result.used_fallback else "yellow",
                )
                trace.emit(
                    "message_end", message_id=message.message_id,
                    status="system_fallback" if result.diagnostics.system_failure else (
                        "degraded" if result.diagnostics.degraded else "ok"
                    ),
                    duration_ms=round((time.perf_counter() - began) * 1000, 2),
                    diagnostics=result.diagnostics,
                    classification=result.classification,
                )
            except Exception as exc:  # One bad row must not stop the queue.
                failed += 1
                reporter.write(f"  failed: {type(exc).__name__}", "red")
                trace.emit(
                    "message_end", message_id=message.message_id, status="error",
                    duration_ms=round((time.perf_counter() - began) * 1000, 2),
                    error_category=type(exc).__name__,
                )

    full_run = args.message_id is None and args.limit is None
    try:
        output_store.validate(require_complete=full_run)
    except OutputContractError as exc:
        print(f"error: output contract validation failed: {exc}", file=sys.stderr)
        failed += 1

    exit_code = 1 if failed or system_fallback_count else 0
    trace.emit(
        "queue_end", status="ok" if exit_code == 0 else "degraded",
        written=written, skipped=skipped, fallbacks=fallback_count,
        system_fallbacks=system_fallback_count, failed=failed,
        duration_ms=round((time.perf_counter() - queue_began) * 1000, 2),
        exit_code=exit_code,
    )
    reporter.write(
        "summary: "
        f"written={written} skipped={skipped} fallback={fallback_count} "
        f"system_fallback={system_fallback_count} failed={failed} output={output_path}"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
