from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from router.media import MAX_AUDIO_BYTES, MediaProcessor
from router.models import ImageExtractionPayload, ToolStatus


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(output_parsed=ImageExtractionPayload(
            ocr_text="SALE 20%", visual_summary="A store poster.",
            detected_urls=["shop.example"], detected_phone_numbers=[],
            detected_dates=[], language="en", confidence=0.92,
        ))


class FakeTranscriptions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(text="Meet at 5 PM.", language="en")


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()
        self.audio = SimpleNamespace(transcriptions=FakeTranscriptions())


class MediaTests(unittest.TestCase):
    def test_luna_request_and_structured_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "poster.jpg"
            image.write_bytes(b"fake-image")
            client = FakeClient()
            result = MediaProcessor(client=client).process_image(image)
            self.assertEqual(result.status, ToolStatus.OK)
            self.assertEqual(result.ocr_text, "SALE 20%")
            call = client.responses.calls[0]
            self.assertEqual(call["model"], "gpt-5.6-luna")
            self.assertEqual(call["input"][0]["content"][1]["detail"], "original")
            self.assertIs(call["text_format"], ImageExtractionPayload)

    def test_audio_request_validation_and_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "note.mp3"
            audio.write_bytes(b"fake-audio")
            client = FakeClient()
            result = MediaProcessor(client=client).process_audio(audio)
            self.assertEqual(result.status, ToolStatus.OK)
            self.assertEqual(result.transcript, "Meet at 5 PM.")
            call = client.audio.transcriptions.calls[0]
            self.assertEqual(call["model"], "gpt-transcribe")
            self.assertIn("verbatim", str(call["prompt"]))

            unsupported = root / "note.aac"
            unsupported.write_bytes(b"x")
            self.assertEqual(MediaProcessor(client=client).process_audio(unsupported).status, ToolStatus.ERROR)

    def test_audio_size_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            audio = Path(temp) / "large.mp3"
            with audio.open("wb") as handle:
                handle.truncate(MAX_AUDIO_BYTES + 1)
            result = MediaProcessor(client=FakeClient()).process_audio(audio)
            self.assertEqual(result.status, ToolStatus.ERROR)
            self.assertIn("25 MB", result.error or "")


if __name__ == "__main__":
    unittest.main()
