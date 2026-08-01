from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from .models import HistoryMatch, HistorySearchResult, IncomingMessage, JsonContext


def _clean(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def _record(row: pd.Series | None) -> JsonContext | None:
    if row is None:
        return None
    return {str(key): _clean(value) for key, value in row.to_dict().items()}


def _flag(value: Any) -> bool:
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return False


def _optional_float(value: Any) -> float | None:
    if value in (None, "") or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class HistoryRepository:
    """In-memory, user-scoped historical retrieval and personalization context."""

    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir.resolve()
        self.history = self._read("message_history.csv")
        self.events = self._read("message_events.csv")
        self.users = self._read("users.csv")
        self.groups = self._read("groups.csv")
        self.group_members = self._read("group_members.csv")
        self.businesses = self._read("business_accounts.csv")
        self.business_history = self._read("user_business_history.csv")
        self.notification_summary = self._read("daily_notification_summary.csv")

        required = {"message_id", "user_id", "created_at", "message_text"}
        missing = required - set(self.history.columns)
        if missing:
            raise ValueError(f"message_history.csv missing columns: {sorted(missing)}")
        if self.history["message_id"].duplicated().any():
            raise ValueError("message_history.csv contains duplicate message IDs")
        self.history["created_at_parsed"] = pd.to_datetime(
            self.history["created_at"], errors="coerce", utc=True
        )

    def _read(self, name: str) -> pd.DataFrame:
        path = self.dataset_dir / name
        if not path.is_file():
            raise ValueError(f"required dataset file does not exist: {path}")
        return pd.read_csv(path, dtype=str, keep_default_na=False)

    def official_domain(self, business_id: str) -> str:
        return self.business_domains(business_id)[0]

    def business_domains(self, business_id: str) -> tuple[str, str]:
        if not business_id:
            return "", ""
        rows = self.businesses[self.businesses["business_id"] == business_id]
        if rows.empty:
            return "", ""
        row = rows.iloc[0]
        return (
            str(row.get("official_domain", "")),
            str(row.get("domain_used_by_sender", "")),
        )

    def evidence_ids_for_user(self, user_id: str) -> set[str]:
        rows = self.history[self.history["user_id"] == user_id]
        return set(rows["message_id"].astype(str))

    @staticmethod
    def _entity_match(row: pd.Series, message: IncomingMessage) -> bool:
        return bool(
            (message.group_id and row.get("group_id", "") == message.group_id)
            or (message.business_id and row.get("business_id", "") == message.business_id)
            or (
                message.sender_user_id
                and row.get("sender_user_id", "") == message.sender_user_id
            )
        )

    @staticmethod
    def _recency_bonus(created_at: Any, incoming_at: datetime) -> float:
        if pd.isna(created_at):
            return 0.0
        historical = created_at.to_pydatetime()
        if historical.tzinfo is None:
            historical = historical.replace(tzinfo=timezone.utc)
        days = max(0.0, (incoming_at - historical).total_seconds() / 86400)
        return 8.0 if days <= 7 else 5.0 if days <= 30 else 2.0 if days <= 90 else 0.0

    def search(
        self,
        message: IncomingMessage,
        search_term: str,
        top_k: int = 5,
    ) -> HistorySearchResult:
        query = " ".join(search_term.split())[:500]
        scoped = self.history[self.history["user_id"] == message.user_id].copy()
        events = self.events[self.events["user_id"] == message.user_id]
        scoped = scoped.merge(events, on=["user_id", "message_id"], how="left")
        incoming_at = pd.to_datetime(message.created_at, errors="coerce", utc=True)
        if pd.isna(incoming_at):
            incoming_dt = datetime.now(timezone.utc)
        else:
            incoming_dt = incoming_at.to_pydatetime()

        ranked: list[tuple[float, str, HistoryMatch]] = []
        for _, row in scoped.iterrows():
            text = str(row.get("message_text", ""))
            similarity = float(max(fuzz.WRatio(query, text), fuzz.token_set_ratio(query, text))) if query and text else 0.0
            entity_match = self._entity_match(row, message)
            score = similarity * 0.65
            score += 20.0 if entity_match else 0.0
            score += self._recency_bonus(row.get("created_at_parsed"), incoming_dt)
            score += 5.0 if _flag(row.get("message_replied")) else 0.0
            score += 2.0 if _flag(row.get("message_opened")) else 0.0
            score -= 6.0 if _flag(row.get("notification_dismissed")) else 0.0
            score -= 14.0 if _flag(row.get("muted_after_message")) else 0.0
            score -= 25.0 if _flag(row.get("message_reported")) else 0.0
            score = max(0.0, min(100.0, score))
            if similarity < 35.0 and not entity_match:
                continue
            match = HistoryMatch(
                message_id=str(row["message_id"]),
                created_at=str(row.get("created_at", "")),
                conversation_type=str(row.get("conversation_type", "")),
                group_id=str(row.get("group_id", "")),
                business_id=str(row.get("business_id", "")),
                sender_user_id=str(row.get("sender_user_id", "")),
                message_text=text,
                media_type=str(row.get("media_type", "")),
                forwarded_count=int(row.get("forwarded_count", "0") or "0"),
                similarity_score=round(similarity, 2),
                ranking_score=round(score, 2),
                message_opened=_flag(row.get("message_opened")),
                message_replied=_flag(row.get("message_replied")),
                reaction_time_minutes=_optional_float(row.get("reaction_time_minutes")),
                notification_dismissed=_flag(row.get("notification_dismissed")),
                muted_after_message=_flag(row.get("muted_after_message")),
                message_reported=_flag(row.get("message_reported")),
            )
            ranked.append((-score, str(row["message_id"]), match))
        ranked.sort(key=lambda item: (item[0], item[1]))

        return HistorySearchResult(
            query=query,
            user_id=message.user_id,
            matches=[item[2] for item in ranked[: max(0, min(top_k, 5))]],
            user_profile=self._lookup(self.users, user_id=message.user_id),
            notification_summary=self._latest_notification(message),
            group_context=self._lookup(self.groups, group_id=message.group_id),
            business_context=self._lookup(self.businesses, business_id=message.business_id),
            relationship_context=(
                self._lookup(self.group_members, user_id=message.user_id, group_id=message.group_id)
                if message.group_id
                else self._lookup(self.business_history, user_id=message.user_id, business_id=message.business_id)
            ),
        )

    @staticmethod
    def _lookup(frame: pd.DataFrame, **keys: str) -> JsonContext | None:
        if not all(keys.values()):
            return None
        rows = frame
        for column, value in keys.items():
            if column not in rows.columns:
                return None
            rows = rows[rows[column] == value]
        return None if rows.empty else _record(rows.iloc[0])

    def _latest_notification(self, message: IncomingMessage) -> JsonContext | None:
        rows = self.notification_summary[
            self.notification_summary["user_id"] == message.user_id
        ].copy()
        if rows.empty or "date" not in rows.columns:
            return None
        rows["date_parsed"] = pd.to_datetime(rows["date"], errors="coerce", utc=True)
        cutoff = pd.to_datetime(message.created_at, errors="coerce", utc=True)
        if not pd.isna(cutoff):
            rows = rows[rows["date_parsed"] <= cutoff]
        if rows.empty:
            return None
        row = rows.sort_values(["date_parsed", "date"], ascending=[False, False]).iloc[0]
        return _record(row.drop(labels=["date_parsed"]))
