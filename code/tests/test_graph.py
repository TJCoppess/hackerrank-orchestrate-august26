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
from router.models import (
    Action,
    AudioTranscriptionResult,
    ImageExtractionResult,
    MessageType,
    ToolStatus,
)
from test_io import create_dataset, write_csv


class FakeModel:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls = 0
        self.seen_messages: list[list[object]] = []

    def bind_tools(self, tools: list[object], **kwargs: object) -> "FakeModel":
        self.tools = tools
        self.bind_kwargs = kwargs
        return self

    def invoke(self, messages: list[object]) -> AIMessage:
        self.seen_messages.append(list(messages))
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


def scam_tool_call(call_id: str = "scan_1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{
            "name": "scan_scam_heuristics",
            "args": {},
            "id": call_id,
            "type": "tool_call",
        }],
    )


class FakeMediaProcessor:
    def __init__(self, status: ToolStatus = ToolStatus.OK) -> None:
        self.status = status
        self.image_calls = 0
        self.audio_calls = 0

    def process_image(self, path: Path) -> ImageExtractionResult:
        self.image_calls += 1
        if self.status == ToolStatus.ERROR:
            return ImageExtractionResult(status=ToolStatus.ERROR, error="vision unavailable")
        return ImageExtractionResult(
            status=ToolStatus.OK, ocr_text="School closes at 2 PM",
            visual_summary="A school notice", language="en", confidence=0.9,
        )

    def process_audio(self, path: Path) -> AudioTranscriptionResult:
        self.audio_calls += 1
        return AudioTranscriptionResult(
            status=self.status,
            transcript="Please pick me up at 6 PM" if self.status == ToolStatus.OK else "",
            detected_languages=["en"] if self.status == ToolStatus.OK else [],
            error="transcription unavailable" if self.status == ToolStatus.ERROR else None,
        )


class GraphTests(unittest.TestCase):
    def make_router(self, root: Path, model: FakeModel, media_processor: object | None = None) -> tuple[MessageRouter, object, OutputStore]:
        if not (root / "messages.csv").exists():
            create_dataset(root)
        messages = load_messages(root)
        store = OutputStore(root, root / "output.csv", messages)
        return MessageRouter(model, store, media_processor=media_processor), messages[0], store

    def test_terminal_tool_writes_and_ends(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            router, message, store = self.make_router(
                Path(temp), FakeModel([scam_tool_call(), final_tool_call()])
            )
            properties = router.write_tool.tool_call_schema.model_json_schema()["properties"]
            self.assertNotIn("message_id", properties)
            self.assertNotIn("state", properties)
            self.assertNotIn("tool_call_id", properties)
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
            self.assertEqual(model.calls, 6)
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
            invalid_calls = [scam_tool_call()] + [
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
                for index in range(1)
            ]
            router, message, store = self.make_router(Path(temp), FakeModel(invalid_calls))
            result = router.classify(message)
            self.assertTrue(result.used_fallback)
            self.assertEqual(result.classification.message_type, MessageType.UNKNOWN)
            self.assertTrue(store.is_complete(message.message_id))

    def test_image_then_parallel_analysis_then_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_dataset(root)
            image_path = root / "media" / "images" / "example.jpg"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"fake")
            messages_path = root / "messages.csv"
            import csv
            with messages_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["media_type"] = "image"
            rows[0]["media_id"] = "img_1"
            write_csv(messages_path, list(rows[0]), rows)
            model = FakeModel([
                AIMessage(content="", tool_calls=[{"name": "process_image", "args": {}, "id": "media_call", "type": "tool_call"}]),
                AIMessage(content="", tool_calls=[
                    {"name": "scan_scam_heuristics", "args": {}, "id": "scan_call", "type": "tool_call"},
                    {"name": "query_user_history", "args": {"search_term": "Hello"}, "id": "history_call", "type": "tool_call"},
                ]),
                AIMessage(content="", tool_calls=[{
                    "name": "write_final_classification",
                    "args": {"action": "notify", "message_type": "urgent", "reason": "A time-sensitive school closure affects the user.", "confidence": 0.88, "evidence_message_ids": ["message_0001"]},
                    "id": "final_call", "type": "tool_call",
                }]),
            ])
            media = FakeMediaProcessor()
            router, message, store = self.make_router(root, model, media)
            result = router.classify(message)
            self.assertFalse(result.used_fallback)
            self.assertEqual(media.image_calls, 1)
            self.assertEqual(result.diagnostics.tool_trace, ["process_image", "scan_scam_heuristics", "query_user_history", "write_final_classification"])
            self.assertEqual(result.diagnostics.retrieved_evidence_ids, ["message_0001"])
            tool_message_ids = {
                getattr(item, "tool_call_id", None)
                for batch in model.seen_messages
                for item in batch
            }
            self.assertTrue({"media_call", "scan_call", "history_call"} <= tool_message_ids)
            self.assertTrue(store.is_complete(message.message_id))
            before = media.image_calls
            image_tool = router.tool_map["process_image"]
            first = image_tool.func(state={"incoming": message}, tool_call_id="direct_1")
            second = image_tool.func(
                state={
                    "incoming": message,
                    "image_extraction": first.update["image_extraction"],
                },
                tool_call_id="direct_2",
            )
            self.assertEqual(media.image_calls, before + 1)
            self.assertEqual(second.update["tool_trace"], ["process_image:cached"])

    def test_media_failure_degrades_but_can_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_dataset(root)
            image_path = root / "media" / "images" / "example.jpg"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"fake")
            import csv
            with (root / "messages.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["media_type"] = "image"
            rows[0]["media_id"] = "img_1"
            write_csv(root / "messages.csv", list(rows[0]), rows)
            model = FakeModel([
                AIMessage(content="", tool_calls=[{"name": "process_image", "args": {}, "id": "m", "type": "tool_call"}]),
                scam_tool_call(),
                AIMessage(content="", tool_calls=[{
                    "name": "write_final_classification",
                    "args": {"action": "digest", "message_type": "unknown", "reason": "Image extraction failed, so this is deferred cautiously.", "confidence": 0.55, "evidence_message_ids": []},
                    "id": "f", "type": "tool_call",
                }]),
            ])
            router, message, _ = self.make_router(root, model, FakeMediaProcessor(ToolStatus.ERROR))
            result = router.classify(message)
            self.assertFalse(result.used_fallback)
            self.assertTrue(result.diagnostics.degraded)
            self.assertFalse(result.diagnostics.system_failure)

    def test_voice_flow_and_evidence_without_retrieval_is_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_dataset(root)
            audio_path = root / "media" / "audio" / "example.mp3"
            audio_path.parent.mkdir(parents=True)
            audio_path.write_bytes(b"fake")
            import csv
            with (root / "messages.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["media_type"] = "voice"
            rows[0]["media_id"] = "vn_1"
            write_csv(root / "messages.csv", list(rows[0]), rows)
            bad_final = AIMessage(content="", tool_calls=[{
                "name": "write_final_classification",
                "args": {"action": "notify", "message_type": "personal", "reason": "Pickup request is time-sensitive.", "confidence": 0.85, "evidence_message_ids": ["message_0001"]},
                "id": "bad_evidence", "type": "tool_call",
            }])
            model = FakeModel([
                AIMessage(content="", tool_calls=[{"name": "process_audio", "args": {}, "id": "audio", "type": "tool_call"}]),
                scam_tool_call(), bad_final, final_tool_call("clean_final"),
            ])
            media = FakeMediaProcessor()
            router, message, _ = self.make_router(root, model, media)
            result = router.classify(message)
            self.assertFalse(result.used_fallback)
            self.assertEqual(media.audio_calls, 1)
            self.assertEqual(model.calls, 4)
            self.assertEqual(result.diagnostics.audio_status, ToolStatus.OK)


if __name__ == "__main__":
    unittest.main()
