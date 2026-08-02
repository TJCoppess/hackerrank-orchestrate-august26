from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from router.retry import retry_call
from router.tracing import TerminalReporter, TraceRecorder


class StatusError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        super().__init__("sensitive provider body")
        self.status_code = status_code


class RetryTraceTests(unittest.TestCase):
    def test_transient_success_on_third_attempt_and_backoff(self) -> None:
        attempts = 0
        sleeps: list[float] = []
        notices = []

        def call() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise StatusError(500)
            return "ok"

        self.assertEqual(retry_call("test", call, sleep=sleeps.append, on_retry=notices.append), "ok")
        self.assertEqual(attempts, 3)
        self.assertEqual(sleeps, [0.5, 1.0])
        self.assertEqual([item.error_category for item in notices], ["server_error", "server_error"])

    def test_permanent_error_is_not_retried(self) -> None:
        calls = 0

        def call() -> None:
            nonlocal calls
            calls += 1
            raise ValueError("bad schema")

        with self.assertRaises(ValueError):
            retry_call("test", call, sleep=lambda _: self.fail("must not sleep"))
        self.assertEqual(calls, 1)

    def test_trace_is_jsonl_redacted_and_color_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.jsonl"
            stream = io.StringIO()
            recorder = TraceRecorder(path, run_id="run", reporter=TerminalReporter(color=False, stream=stream))
            recorder.emit("retry", message_id="msg", raw_error="secret", message_text="private", error_category="timeout")
            recorder.show("retry timeout", "yellow")
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("raw_error", record)
            self.assertNotIn("message_text", record)
            self.assertEqual(record["error_category"], "timeout")
            self.assertNotIn("\033[", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
