from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from pydantic import ValidationError

from router.io import OUTPUT_COLUMNS, OutputStore, load_messages
from router.models import Classification


class ContractTests(unittest.TestCase):
    def test_real_template_matches_all_input_ids_without_writing(self) -> None:
        dataset = REPO_ROOT / "dataset"
        messages = load_messages(dataset)
        store = OutputStore(dataset, dataset / "output.csv", messages)
        store.validate(require_complete=False)
        with (dataset / "output.csv").open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        self.assertEqual(reader.fieldnames, OUTPUT_COLUMNS)
        self.assertEqual(len(rows), 110)
        self.assertEqual([row["message_id"] for row in rows], [message.message_id for message in messages])

    def test_classification_rejects_invalid_values_and_spoofed_message_id(self) -> None:
        with self.assertRaises(ValidationError):
            Classification.model_validate({
                "message_id": "msg_spoofed",
                "action": "later",
                "message_type": "newsletter",
                "reason": "Invalid.",
                "confidence": 1.5,
                "evidence_message_ids": [],
            })


if __name__ == "__main__":
    unittest.main()
