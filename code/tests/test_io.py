from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from router.io import DatasetError, OutputContractError, OutputStore, load_messages
from router.models import Action, Classification, MessageType


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def create_dataset(root: Path, media_path: str = "media/images/example.jpg") -> None:
    write_csv(root / "images.csv", ["image_id", "file_path"], [{"image_id": "img_1", "file_path": media_path}])
    write_csv(root / "voice_notes.csv", ["voice_note_id", "file_path"], [{"voice_note_id": "vn_1", "file_path": "media/audio/example.mp3"}])
    write_csv(
        root / "message_history.csv",
        ["message_id", "user_id", "conversation_type", "group_id", "business_id", "sender_user_id", "created_at", "message_text", "media_type", "media_id", "forwarded_count"],
        [{"message_id": "message_0001", "user_id": "u_1", "conversation_type": "personal", "group_id": "", "business_id": "", "sender_user_id": "u_2", "created_at": "2026-07-30 12:00", "message_text": "Hello before", "media_type": "", "media_id": "", "forwarded_count": "0"}],
    )
    write_csv(root / "message_events.csv", ["user_id", "message_id", "message_opened", "message_replied", "reaction_time_minutes", "notification_dismissed", "muted_after_message", "message_reported"], [{"user_id": "u_1", "message_id": "message_0001", "message_opened": "1", "message_replied": "0", "reaction_time_minutes": "3", "notification_dismissed": "0", "muted_after_message": "0", "message_reported": "0"}])
    write_csv(root / "users.csv", ["user_id", "do_not_disturb_window", "messages_opened_30d", "messages_replied_30d", "notifications_dismissed_30d", "messages_reported_30d"], [{"user_id": "u_1", "do_not_disturb_window": "22:00-07:00", "messages_opened_30d": "10", "messages_replied_30d": "2", "notifications_dismissed_30d": "1", "messages_reported_30d": "0"}])
    write_csv(root / "groups.csv", ["group_id", "group_name"], [])
    write_csv(root / "group_members.csv", ["group_id", "user_id"], [])
    write_csv(root / "business_accounts.csv", ["business_id", "official_domain"], [])
    write_csv(root / "user_business_history.csv", ["user_id", "business_id"], [])
    write_csv(root / "daily_notification_summary.csv", ["user_id", "date", "notifications_sent", "notifications_dismissed"], [{"user_id": "u_1", "date": "2026-08-01", "notifications_sent": "2", "notifications_dismissed": "0"}])
    write_csv(
        root / "messages.csv",
        [
            "message_id", "user_id", "conversation_type", "group_id",
            "business_id", "sender_user_id", "created_at", "message_text",
            "media_type", "media_id", "forwarded_count",
        ],
        [{
            "message_id": "msg_1", "user_id": "u_1", "conversation_type": "personal",
            "group_id": "", "business_id": "", "sender_user_id": "u_2",
            "created_at": "2026-08-01 12:00", "message_text": "Hello",
            "media_type": "", "media_id": "", "forwarded_count": "0",
        }],
    )
    write_csv(
        root / "output.csv",
        ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"],
        [{"message_id": "msg_1", "action": "", "message_type": "", "reason": "", "confidence": "", "evidence_message_ids": ""}],
    )


class OutputStoreTests(unittest.TestCase):
    def test_upsert_is_idempotent_and_serializes_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_dataset(root)
            messages = load_messages(root)
            store = OutputStore(root, root / "output.csv", messages)
            first = Classification(action=Action.NOTIFY, message_type=MessageType.PERSONAL, reason="Direct personal message.", confidence=0.8, evidence_message_ids=[])
            second = Classification(action=Action.DIGEST, message_type=MessageType.GREETING, reason="Non-urgent greeting.", confidence=0.6, evidence_message_ids=[])
            store.upsert("msg_1", first)
            store.upsert("msg_1", second)
            with (root / "output.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["action"], "digest")
            self.assertEqual(rows[0]["evidence_message_ids"], "none")
            store.validate(require_complete=True)

    def test_unknown_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_dataset(root)
            store = OutputStore(root, root / "output.csv", load_messages(root))
            classification = Classification(action=Action.DIGEST, message_type=MessageType.UNKNOWN, reason="Deferred.", confidence=0.4, evidence_message_ids=["message_9999"])
            with self.assertRaises(OutputContractError):
                store.upsert("msg_1", classification)

    def test_media_path_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_dataset(root, media_path="../outside.jpg")
            with (root / "messages.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["media_type"] = "image"
            rows[0]["media_id"] = "img_1"
            write_csv(root / "messages.csv", list(rows[0]), rows)
            with self.assertRaises(DatasetError):
                load_messages(root)


if __name__ == "__main__":
    unittest.main()
