from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from langchain_core.messages import AIMessage

from router.graph import MessageRouter
from router.io import OutputStore, load_messages
from router.models import Action, MessageType
from test_io import create_dataset


class FakeModel:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls = 0

    def bind_tools(self, tools: list[object], **kwargs: object) -> "FakeModel":
        self.tools = tools
        self.bind_kwargs = kwargs
        return self

    def invoke(self, messages: list[object]) -> AIMessage:
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        if not isinstance(response, AIMessage):
            raise TypeError("fake response must be an AIMessage")
        return response


def final_tool_call(call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{
            "name": "write_final_classification",
            "args": {
                "action": "notify",
                "message_type": "personal",
                "reason": "A direct personal message merits immediate attention.",
                "confidence": 0.82,
                "evidence_message_ids": [],
            },
            "id": call_id,
            "type": "tool_call",
        }],
    )


class GraphTests(unittest.TestCase):
    def make_router(self, root: Path, model: FakeModel) -> tuple[MessageRouter, object, OutputStore]:
        create_dataset(root)
        messages = load_messages(root)
        store = OutputStore(root, root / "output.csv", messages)
        return MessageRouter(model, store), messages[0], store

    def test_terminal_tool_writes_and_ends(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            router, message, store = self.make_router(Path(temp), FakeModel([final_tool_call()]))
            properties = router.write_tool.tool_call_schema.model_json_schema()["properties"]
            self.assertNotIn("message_id", properties)
            self.assertNotIn("state", properties)
            result = router.classify(message)
            self.assertFalse(result.used_fallback)
            self.assertEqual(result.classification.action, Action.NOTIFY)
            self.assertEqual(result.classification.message_type, MessageType.PERSONAL)
            self.assertTrue(store.is_complete(message.message_id))

    def test_missing_tool_call_uses_fallback_after_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            no_tool = AIMessage(content="I forgot to call the tool.")
            model = FakeModel([no_tool, no_tool])
            router, message, store = self.make_router(Path(temp), model)
            result = router.classify(message)
            self.assertTrue(result.used_fallback)
            self.assertEqual(model.calls, 2)
            self.assertTrue(store.is_complete(message.message_id))

    def test_model_exception_writes_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            router, message, store = self.make_router(Path(temp), FakeModel([RuntimeError("API unavailable")]))
            result = router.classify(message)
            self.assertTrue(result.used_fallback)
            self.assertIn("API unavailable", result.error or "")
            self.assertTrue(store.is_complete(message.message_id))

    def test_malformed_tool_arguments_fall_back(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            invalid_calls = [
                AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "write_final_classification",
                        "args": {
                            "action": "later",
                            "message_type": "newsletter",
                            "reason": "Invalid output.",
                            "confidence": 1.5,
                            "evidence_message_ids": [],
                        },
                        "id": f"bad_call_{index}",
                        "type": "tool_call",
                    }],
                )
                for index in range(2)
            ]
            router, message, store = self.make_router(Path(temp), FakeModel(invalid_calls))
            result = router.classify(message)
            self.assertTrue(result.used_fallback)
            self.assertEqual(result.classification.message_type, MessageType.UNKNOWN)
            self.assertTrue(store.is_complete(message.message_id))


if __name__ == "__main__":
    unittest.main()
