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
    def __init__(self, responses: list[object] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.responses = responses or [SimpleNamespace(
            text="Meet at 5 PM.", languages=[SimpleNamespace(code="en")]
        )]

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        result = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(result, Exception):
            raise result
        return result


class FakeClient:
    def __init__(self, transcriptions: FakeTranscriptions | None = None) -> None:
        self.responses = FakeResponses()
        self.audio = SimpleNamespace(transcriptions=transcriptions or FakeTranscriptions())


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
            self.assertEqual(call["response_format"], "json")
            self.assertEqual(result.detected_languages, ["en"])
            self.assertEqual(call["file"][0], "voice-note.mp3")
            self.assertEqual(call["file"][2], "audio/mpeg")

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

    def test_magic_bytes_override_misleading_extension(self) -> None:
        cases = [
            (b"ID3\x04\x00\x00" + b"x" * 20, "note.wav", "voice-note.mp3"),
            (b"RIFF\x10\x00\x00\x00WAVEfmt ", "note.mp3", "voice-note.wav"),
            (b"\x00\x00\x00\x18ftypM4A " + b"x" * 12, "note.mp3", "voice-note.m4a"),
        ]
        with tempfile.TemporaryDirectory() as temp:
            for data, filename, expected in cases:
                client = FakeClient()
                path = Path(temp) / filename
                path.write_bytes(data)
                self.assertEqual(MediaProcessor(client=client).process_audio(path).status, ToolStatus.OK)
                self.assertEqual(client.audio.transcriptions.calls[0]["file"][0], expected)

    def test_transcription_retries_and_returns_structured_error(self) -> None:
        class Transient(RuntimeError):
            status_code = 500

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "note.mp3"
            path.write_bytes(b"ID3\x04\x00\x00" + b"x" * 20)
            sleeps: list[float] = []
            transcriptions = FakeTranscriptions([Transient("one"), Transient("two"), SimpleNamespace(text="Recovered", languages=[{"code": "hi"}])])
            result = MediaProcessor(client=FakeClient(transcriptions), sleep=sleeps.append).process_audio(path)
            self.assertEqual(result.transcript, "Recovered")
            self.assertEqual(result.detected_languages, ["hi"])
            self.assertEqual(sleeps, [0.5, 1.0])

            failed = FakeTranscriptions([Transient("always")])
            degraded = MediaProcessor(client=FakeClient(failed), sleep=lambda _: None).process_audio(path)
            self.assertEqual(degraded.status, ToolStatus.ERROR)
            self.assertEqual(degraded.error_category, "server_error")

    def test_msg_086_is_detected_as_mp3(self) -> None:
        path = CODE_DIR.parent / "dataset" / "media" / "audio" / "vn_009.mp3"
        client = FakeClient()
        result = MediaProcessor(client=client).process_audio(path, "msg_086")
        self.assertEqual(result.status, ToolStatus.OK)
        self.assertEqual(client.audio.transcriptions.calls[0]["file"][0], "voice-note.mp3")


if __name__ == "__main__":
    unittest.main()
