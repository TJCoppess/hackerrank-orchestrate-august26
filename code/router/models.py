from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Action(str, Enum):
    NOTIFY = "notify"
    DIGEST = "digest"
    MUTE = "mute"


class MessageType(str, Enum):
    PERSONAL = "personal"
    URGENT = "urgent"
    EVENT = "event"
    PAYMENT = "payment"
    BUSINESS_UPDATE = "business_update"
    PROMOTION = "promotion"
    GREETING = "greeting"
    FORWARD = "forward"
    SPAM = "spam"
    SCAM = "scam"
    UNKNOWN = "unknown"


class IncomingMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    message_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    conversation_type: str
    group_id: str = ""
    business_id: str = ""
    sender_user_id: str = ""
    created_at: str = Field(min_length=1)
    message_text: str = ""
    media_type: str = ""
    media_id: str = ""
    forwarded_count: int = Field(ge=0)
    image_path: Path | None = None
    audio_path: Path | None = None


class Classification(BaseModel):
    """Complete output payload for one incoming message."""

    model_config = ConfigDict(extra="forbid")

    action: Action = Field(description="Final routing action.")
    message_type: MessageType = Field(description="Best-fit message category.")
    reason: str = Field(
        min_length=1,
        max_length=240,
        description="Short, single-line, human-readable decision explanation.",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_message_ids: list[str] = Field(
        description="Historical message IDs used as evidence; empty in Phase 1."
    )

    @field_validator("reason", mode="before")
    @classmethod
    def normalize_reason(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("reason must be text")
        return " ".join(value.split())

    @field_validator("evidence_message_ids")
    @classmethod
    def validate_evidence_ids(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values:
            candidate = value.strip()
            if not candidate.startswith("message_") or not candidate[8:].isdigit():
                raise ValueError(
                    f"evidence ID {value!r} is not a historical message ID"
                )
            if candidate not in seen:
                cleaned.append(candidate)
                seen.add(candidate)
        return cleaned


class RouterState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    incoming: IncomingMessage
    attempts: int
    finalized: bool
    classification: Classification
    used_fallback: bool


class RoutingResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    classification: Classification
    used_fallback: bool = False
    error: str | None = None
