from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Iterable

from .models import Action, Classification, IncomingMessage, MessageType


MESSAGE_COLUMNS = [
    "message_id",
    "user_id",
    "conversation_type",
    "group_id",
    "business_id",
    "sender_user_id",
    "created_at",
    "message_text",
    "media_type",
    "media_id",
    "forwarded_count",
]
OUTPUT_COLUMNS = [
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
]


class DatasetError(ValueError):
    pass


class OutputContractError(ValueError):
    pass


def _read_csv(path: Path, required_columns: Iterable[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise DatasetError(f"required file does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise DatasetError(f"CSV has no header: {path}")
        missing = [column for column in required_columns if column not in reader.fieldnames]
        if missing:
            raise DatasetError(f"{path.name} is missing columns: {', '.join(missing)}")
        return [dict(row) for row in reader]


def _load_media_map(path: Path, id_column: str) -> dict[str, str]:
    rows = _read_csv(path, [id_column, "file_path"])
    result: dict[str, str] = {}
    for row in rows:
        media_id = row[id_column].strip()
        if not media_id or media_id in result:
            raise DatasetError(f"invalid or duplicate {id_column} in {path.name}: {media_id!r}")
        result[media_id] = row["file_path"].strip()
    return result


def _resolve_media_path(dataset_dir: Path, relative_path: str) -> Path:
    root = dataset_dir.resolve()
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise DatasetError(f"media path escapes dataset directory: {relative_path!r}")
    if not candidate.is_file():
        raise DatasetError(f"media file does not exist: {candidate}")
    return candidate


def load_messages(
    dataset_dir: Path, input_path: Path | None = None
) -> list[IncomingMessage]:
    dataset_dir = dataset_dir.resolve()
    image_paths = _load_media_map(dataset_dir / "images.csv", "image_id")
    audio_paths = _load_media_map(dataset_dir / "voice_notes.csv", "voice_note_id")
    source_path = (input_path or dataset_dir / "messages.csv").resolve()
    rows = _read_csv(source_path, MESSAGE_COLUMNS)

    messages: list[IncomingMessage] = []
    seen_ids: set[str] = set()
    for row in rows:
        message_id = row["message_id"].strip()
        if not message_id or message_id in seen_ids:
            raise DatasetError(f"invalid or duplicate message_id: {message_id!r}")
        seen_ids.add(message_id)

        media_type = row["media_type"].strip()
        media_id = row["media_id"].strip()
        image_path: Path | None = None
        audio_path: Path | None = None
        if media_type == "image":
            if media_id not in image_paths:
                raise DatasetError(f"unknown image media_id {media_id!r} for {message_id}")
            image_path = _resolve_media_path(dataset_dir, image_paths[media_id])
        elif media_type == "voice":
            if media_id not in audio_paths:
                raise DatasetError(f"unknown voice media_id {media_id!r} for {message_id}")
            audio_path = _resolve_media_path(dataset_dir, audio_paths[media_id])
        elif media_type:
            raise DatasetError(f"unsupported media_type {media_type!r} for {message_id}")
        elif media_id:
            raise DatasetError(f"message {message_id} has media_id without media_type")

        try:
            forwarded_count = int(row["forwarded_count"] or "0")
        except ValueError as exc:
            raise DatasetError(f"invalid forwarded_count for {message_id}") from exc

        messages.append(
            IncomingMessage(
                message_id=message_id,
                user_id=row["user_id"].strip(),
                conversation_type=row["conversation_type"].strip(),
                group_id=row["group_id"].strip(),
                business_id=row["business_id"].strip(),
                sender_user_id=row["sender_user_id"].strip(),
                created_at=row["created_at"].strip(),
                message_text=row["message_text"],
                media_type=media_type,
                media_id=media_id,
                forwarded_count=forwarded_count,
                image_path=image_path,
                audio_path=audio_path,
            )
        )
    if not messages:
        raise DatasetError(f"{source_path.name} contains no messages")
    return messages


class OutputStore:
    def __init__(
        self,
        dataset_dir: Path,
        output_path: Path,
        messages: list[IncomingMessage],
    ) -> None:
        self.dataset_dir = dataset_dir.resolve()
        self.output_path = output_path.resolve()
        self.message_ids = [message.message_id for message in messages]
        self._message_id_set = set(self.message_ids)
        self.incoming_user_by_id = {
            message.message_id: message.user_id for message in messages
        }
        history_rows = _read_csv(
            self.dataset_dir / "message_history.csv", ["message_id"]
        )
        self.history_ids = {row["message_id"].strip() for row in history_rows}
        self.history_user_by_id = {
            row["message_id"].strip(): row.get("user_id", "").strip()
            for row in history_rows
        }
        self._ensure_output_exists()
        self.validate(require_complete=False)

    def _ensure_output_exists(self) -> None:
        if self.output_path.exists():
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{column: "" for column in OUTPUT_COLUMNS} for _ in self.message_ids]
        for row, message_id in zip(rows, self.message_ids):
            row["message_id"] = message_id
        self._atomic_write(rows)

    def _load_rows(self) -> list[dict[str, str]]:
        if not self.output_path.is_file():
            raise OutputContractError(f"output file does not exist: {self.output_path}")
        with self.output_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != OUTPUT_COLUMNS:
                raise OutputContractError(
                    f"expected columns {OUTPUT_COLUMNS}, found {reader.fieldnames}"
                )
            return [dict(row) for row in reader]

    def _atomic_write(self, rows: list[dict[str, str]]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                delete=False,
                dir=self.output_path.parent,
                prefix=f".{self.output_path.name}.",
                suffix=".tmp",
            ) as handle:
                temp_path = Path(handle.name)
                writer = csv.DictWriter(
                    handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temp_path, self.output_path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    def is_complete(self, message_id: str) -> bool:
        if message_id not in self._message_id_set:
            raise OutputContractError(f"unknown output message_id: {message_id}")
        row = next(
            row for row in self._load_rows() if row["message_id"] == message_id
        )
        return all(row[column].strip() for column in OUTPUT_COLUMNS[1:])

    def upsert(self, message_id: str, classification: Classification) -> None:
        if message_id not in self._message_id_set:
            raise OutputContractError(f"cannot write unknown message_id: {message_id}")
        unknown_evidence = set(classification.evidence_message_ids) - self.history_ids
        if unknown_evidence:
            raise OutputContractError(
                "unknown historical evidence IDs: " + ", ".join(sorted(unknown_evidence))
            )
        self.validate_evidence_owner(
            self.incoming_user_by_id[message_id], classification.evidence_message_ids
        )

        rows = self._load_rows()
        matches = [row for row in rows if row["message_id"] == message_id]
        if len(matches) != 1:
            raise OutputContractError(
                f"expected one output row for {message_id}, found {len(matches)}"
            )
        row = matches[0]
        row.update(
            {
                "action": classification.action.value,
                "message_type": classification.message_type.value,
                "reason": classification.reason,
                "confidence": f"{classification.confidence:.2f}",
                "evidence_message_ids": ";".join(
                    classification.evidence_message_ids
                )
                or "none",
            }
        )
        self._atomic_write(rows)

    def validate_evidence_owner(self, user_id: str, evidence_ids: list[str]) -> None:
        wrong_user = [
            evidence_id
            for evidence_id in evidence_ids
            if self.history_user_by_id.get(evidence_id) != user_id
        ]
        if wrong_user:
            raise OutputContractError(
                "historical evidence does not belong to the active user: "
                + ", ".join(sorted(wrong_user))
            )

    def validate(self, require_complete: bool) -> None:
        rows = self._load_rows()
        row_ids = [row["message_id"] for row in rows]
        if row_ids != self.message_ids:
            raise OutputContractError(
                "output message IDs must exactly match messages.csv order"
            )
        if len(row_ids) != len(set(row_ids)):
            raise OutputContractError("output contains duplicate message IDs")

        allowed_actions = {value.value for value in Action}
        allowed_types = {value.value for value in MessageType}
        for row in rows:
            populated = [bool(row[column].strip()) for column in OUTPUT_COLUMNS[1:]]
            if not any(populated):
                if require_complete:
                    raise OutputContractError(
                        f"missing prediction for {row['message_id']}"
                    )
                continue
            if not all(populated):
                raise OutputContractError(
                    f"partially populated prediction for {row['message_id']}"
                )
            if row["action"] not in allowed_actions:
                raise OutputContractError(
                    f"invalid action for {row['message_id']}: {row['action']!r}"
                )
            if row["message_type"] not in allowed_types:
                raise OutputContractError(
                    f"invalid message_type for {row['message_id']}: "
                    f"{row['message_type']!r}"
                )
            try:
                confidence = float(row["confidence"])
            except ValueError as exc:
                raise OutputContractError(
                    f"invalid confidence for {row['message_id']}"
                ) from exc
            if not 0.0 <= confidence <= 1.0:
                raise OutputContractError(
                    f"confidence outside [0, 1] for {row['message_id']}"
                )
            evidence = row["evidence_message_ids"]
            if evidence != "none":
                evidence_ids = [item.strip() for item in evidence.split(";") if item.strip()]
                unknown = set(evidence_ids) - self.history_ids
                if unknown:
                    raise OutputContractError(
                        f"unknown evidence IDs for {row['message_id']}: "
                        + ", ".join(sorted(unknown))
                    )
                self.validate_evidence_owner(
                    self.incoming_user_by_id[row["message_id"]], evidence_ids
                )
