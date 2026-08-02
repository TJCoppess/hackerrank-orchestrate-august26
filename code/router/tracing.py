from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


ALLOWED_EVENTS = {
    "queue_start", "message_start", "model_attempt", "tool_start", "tool_end",
    "retry", "classification", "fallback", "message_end", "queue_end",
}
FORBIDDEN_KEYS = {
    "message_text", "ocr_text", "transcript", "history_text", "base64", "api_key",
    "raw_error", "chain_of_thought", "reasoning",
}


def _safe_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, Path):
        return value.name
    if isinstance(value, dict):
        return {
            str(key): _safe_value(item)
            for key, item in value.items()
            if str(key).lower() not in FORBIDDEN_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)[:160]


class TerminalReporter:
    COLORS = {
        "blue": "\033[36m", "green": "\033[32m", "yellow": "\033[33m",
        "red": "\033[31m", "dim": "\033[2m", "reset": "\033[0m",
    }

    def __init__(self, *, color: bool = True, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stdout
        self.color = bool(color and not os.environ.get("NO_COLOR") and self.stream.isatty())

    def write(self, text: str, color: str | None = None) -> None:
        if self.color and color:
            text = f"{self.COLORS[color]}{text}{self.COLORS['reset']}"
        print(text, file=self.stream, flush=True)


class TraceRecorder:
    """Thread-safe append-only JSONL recorder containing safe operational metadata."""

    def __init__(
        self,
        path: Path,
        *,
        run_id: str | None = None,
        reporter: TerminalReporter | None = None,
    ) -> None:
        self.path = path.resolve()
        self.run_id = run_id or uuid.uuid4().hex
        self.reporter = reporter
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, *, message_id: str | None = None, **fields: Any) -> None:
        if event not in ALLOWED_EVENTS:
            raise ValueError(f"unsupported trace event: {event}")
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "event": event,
            "run_id": self.run_id,
        }
        if message_id:
            record["message_id"] = message_id
        record.update(_safe_value(fields))
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")

    def show(self, text: str, color: str | None = None) -> None:
        if self.reporter is not None:
            self.reporter.write(text, color)
