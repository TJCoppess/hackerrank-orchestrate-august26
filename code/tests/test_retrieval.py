from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from router.io import load_messages
from router.retrieval import HistoryRepository
from test_io import create_dataset, write_csv


class RetrievalTests(unittest.TestCase):
    def test_user_isolation_ranking_context_and_empty_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            create_dataset(root)
            columns = [
                "message_id", "user_id", "conversation_type", "group_id", "business_id",
                "sender_user_id", "created_at", "message_text", "media_type", "media_id", "forwarded_count",
            ]
            write_csv(root / "message_history.csv", columns, [
                {"message_id": "message_0001", "user_id": "u_1", "conversation_type": "personal", "group_id": "", "business_id": "", "sender_user_id": "u_2", "created_at": "2026-07-31 12:00", "message_text": "Project meeting tomorrow", "media_type": "", "media_id": "", "forwarded_count": "0"},
                {"message_id": "message_0002", "user_id": "u_1", "conversation_type": "personal", "group_id": "", "business_id": "", "sender_user_id": "u_2", "created_at": "2026-07-30 12:00", "message_text": "Project meeting update", "media_type": "", "media_id": "", "forwarded_count": "0"},
                {"message_id": "message_9999", "user_id": "u_9", "conversation_type": "personal", "group_id": "", "business_id": "", "sender_user_id": "u_2", "created_at": "2026-08-01 11:59", "message_text": "Project meeting tomorrow", "media_type": "", "media_id": "", "forwarded_count": "0"},
            ])
            event_columns = ["user_id", "message_id", "message_opened", "message_replied", "reaction_time_minutes", "notification_dismissed", "muted_after_message", "message_reported"]
            write_csv(root / "message_events.csv", event_columns, [
                {"user_id": "u_1", "message_id": "message_0001", "message_opened": "1", "message_replied": "1", "reaction_time_minutes": "1", "notification_dismissed": "0", "muted_after_message": "0", "message_reported": "0"},
                {"user_id": "u_1", "message_id": "message_0002", "message_opened": "0", "message_replied": "0", "reaction_time_minutes": "", "notification_dismissed": "1", "muted_after_message": "1", "message_reported": "0"},
                {"user_id": "u_9", "message_id": "message_9999", "message_opened": "1", "message_replied": "1", "reaction_time_minutes": "1", "notification_dismissed": "0", "muted_after_message": "0", "message_reported": "0"},
            ])
            incoming = load_messages(root)[0]
            repository = HistoryRepository(root)
            result = repository.search(incoming, "project meeting", top_k=5)
            self.assertEqual(result.user_id, "u_1")
            self.assertEqual(result.matches[0].message_id, "message_0001")
            self.assertNotIn("message_9999", [match.message_id for match in result.matches])
            self.assertIsNotNone(result.user_profile)
            self.assertIsNotNone(result.notification_summary)
            self.assertLessEqual(len(result.matches), 5)
            unrelated = incoming.model_copy(update={"sender_user_id": "u_777"})
            empty = repository.search(unrelated, "zzzz qqqq xxxx")
            self.assertEqual(empty.matches, [])


if __name__ == "__main__":
    unittest.main()
