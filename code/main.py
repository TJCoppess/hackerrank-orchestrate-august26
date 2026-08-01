from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pydantic import ValidationError

from router.graph import MessageRouter, create_openai_model
from router.io import DatasetError, OutputContractError, OutputStore, load_messages


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Route WhatsApp messages into notify, digest, or mute."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing messages.csv and context CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <dataset-dir>/output.csv).",
    )
    parser.add_argument(
        "--message-id",
        help="Process one message ID instead of the entire queue.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many selected messages.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess rows that already contain a complete prediction.",
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        print("error: --limit must be at least 1", file=sys.stderr)
        return 2

    dataset_dir = args.dataset_dir.resolve()
    output_path = (args.output or dataset_dir / "output.csv").resolve()

    try:
        messages = load_messages(dataset_dir)
        output_store = OutputStore(dataset_dir, output_path, messages)
    except (DatasetError, OutputContractError, OSError, ValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    selected = messages
    if args.message_id:
        selected = [message for message in messages if message.message_id == args.message_id]
        if not selected:
            print(f"error: unknown message_id {args.message_id!r}", file=sys.stderr)
            return 2
    if args.limit is not None:
        selected = selected[: args.limit]

    pending = [
        message
        for message in selected
        if args.force or not output_store.is_complete(message.message_id)
    ]
    skipped = len(selected) - len(pending)

    if pending and not os.environ.get("OPENAI_API_KEY"):
        print(
            "error: OPENAI_API_KEY is required when messages need processing",
            file=sys.stderr,
        )
        return 2

    written = 0
    fallback_count = 0
    system_fallback_count = 0
    failed = 0

    if pending:
        model = create_openai_model()
        router = MessageRouter(model, output_store)
        for index, message in enumerate(pending, start=1):
            print(f"[{index}/{len(pending)}] routing {message.message_id}")
            try:
                result = router.classify(message)
                written += 1
                if result.used_fallback:
                    fallback_count += 1
                    if result.diagnostics.system_failure:
                        system_fallback_count += 1
                    print(f"  fallback: {result.error or 'model did not finalize'}")
                print(
                    f"  wrote {result.classification.action.value}/"
                    f"{result.classification.message_type.value}"
                )
            except Exception as exc:  # Keep one bad row from stopping the queue.
                failed += 1
                print(f"  failed: {exc}", file=sys.stderr)

    full_run = args.message_id is None and args.limit is None
    try:
        output_store.validate(require_complete=full_run)
    except OutputContractError as exc:
        print(f"error: output contract validation failed: {exc}", file=sys.stderr)
        failed += 1

    print(
        "summary: "
        f"written={written} skipped={skipped} "
        f"fallback={fallback_count} system_fallback={system_fallback_count} "
        f"failed={failed} output={output_path}"
    )
    return 1 if failed or (full_run and system_fallback_count) else 0


if __name__ == "__main__":
    raise SystemExit(run())
