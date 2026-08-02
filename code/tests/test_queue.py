from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import main
from router.models import Action, Classification, MessageType, RoutingDiagnostics, RoutingResult
from test_io import create_dataset


class DummyRouter:
    calls = 0
    system_failure = False

    def __init__(self, model: object, output_store: object, **kwargs: object) -> None:
        self.output_store = output_store

    def classify(self, message: object) -> RoutingResult:
        type(self).calls += 1
        classification = Classification(
            action=Action.DIGEST, message_type=MessageType.PERSONAL,
            reason="A routine direct message can wait for the digest.",
            confidence=0.75, evidence_message_ids=[],
        )
        self.output_store.upsert(message.message_id, classification)
        return RoutingResult(
            classification=classification,
            used_fallback=self.system_failure,
            diagnostics=RoutingDiagnostics(
                degraded=self.system_failure, system_failure=self.system_failure,
                error_category="system_exception" if self.system_failure else None,
            ),
        )


class QueueTests(unittest.TestCase):
    def test_skip_force_trace_and_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_dataset(root)
            output, trace = root / "output.csv", root / "trace.jsonl"
            args = ["--dataset-dir", str(root), "--output", str(output), "--trace", str(trace), "--no-color"]
            DummyRouter.calls = 0
            DummyRouter.system_failure = False
            with patch.dict(os.environ, {"OPENAI_API_KEY": "fake"}), patch.object(main, "create_openai_model", return_value=object()), patch.object(main, "MessageRouter", DummyRouter):
                self.assertEqual(main.run(args), 0)
                self.assertEqual(DummyRouter.calls, 1)
                self.assertEqual(main.run(args), 0)
                self.assertEqual(DummyRouter.calls, 1)
                self.assertEqual(main.run(args + ["--force"]), 0)
                self.assertEqual(DummyRouter.calls, 2)
                DummyRouter.system_failure = True
                self.assertEqual(main.run(args + ["--force"]), 1)

            records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(records)
            self.assertTrue(all("run_id" in item and "timestamp" in item for item in records))
            self.assertIn("queue_start", {item["event"] for item in records})
            self.assertIn("queue_end", {item["event"] for item in records})

    def test_input_can_include_ground_truth_columns_without_label_leak(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_dataset(root)
            source = root / "messages.csv"
            text = source.read_text(encoding="utf-8")
            lines = text.splitlines()
            labeled = root / "sample.csv"
            labeled.write_text(
                lines[0] + ",action,message_type,reason,confidence,evidence_message_ids\n" +
                lines[1] + ",notify,urgent,SECRET LABEL,0.99,message_0001\n",
                encoding="utf-8",
            )
            from router.io import load_messages
            message = load_messages(root, labeled)[0]
            dumped = message.model_dump()
            self.assertNotIn("action", dumped)
            self.assertNotIn("reason", dumped)


if __name__ == "__main__":
    unittest.main()
