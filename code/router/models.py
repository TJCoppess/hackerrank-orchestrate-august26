from __future__ import annotations

import operator
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


class ToolStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    NOT_APPLICABLE = "not_applicable"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


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
        description="Historical message IDs returned by query_user_history."
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


class ScamSignal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    weight: int
    explanation: str


class ScamScanResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    risk_score: int = Field(ge=0, le=100)
    risk_level: RiskLevel
    urls: list[str]
    domains: list[str]
    signals: list[ScamSignal]


class HistoryMatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    message_id: str
    created_at: str
    conversation_type: str
    group_id: str
    business_id: str
    sender_user_id: str
    message_text: str
    media_type: str
    forwarded_count: int = Field(ge=0)
    similarity_score: float = Field(ge=0, le=100)
    ranking_score: float = Field(ge=0, le=100)
    message_opened: bool
    message_replied: bool
    reaction_time_minutes: float | None = None
    notification_dismissed: bool
    muted_after_message: bool
    message_reported: bool


JsonContext = dict[str, str | int | float | bool | None]


class HistorySearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    user_id: str
    matches: list[HistoryMatch]
    user_profile: JsonContext | None = None
    notification_summary: JsonContext | None = None
    group_context: JsonContext | None = None
    business_context: JsonContext | None = None
    relationship_context: JsonContext | None = None


class ImageExtractionPayload(BaseModel):
    """Schema returned directly by the Luna extraction call."""

    model_config = ConfigDict(extra="forbid")

    ocr_text: str = Field(description="All visible text, transcribed faithfully.")
    visual_summary: str = Field(description="Concise description of non-text content.")
    detected_urls: list[str]
    detected_phone_numbers: list[str]
    detected_dates: list[str]
    language: str
    confidence: float = Field(ge=0, le=1)


class ImageExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ToolStatus
    ocr_text: str = ""
    visual_summary: str = ""
    detected_urls: list[str] = Field(default_factory=list)
    detected_phone_numbers: list[str] = Field(default_factory=list)
    detected_dates: list[str] = Field(default_factory=list)
    language: str = ""
    confidence: float = Field(default=0.0, ge=0, le=1)
    error_category: str | None = None
    error: str | None = None


class AudioTranscriptionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ToolStatus
    transcript: str = ""
    detected_languages: list[str] = Field(default_factory=list)
    error_category: str | None = None
    error: str | None = None


class RoutingDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempts: int = 0
    tool_trace: list[str] = Field(default_factory=list)
    scam_risk_score: int | None = None
    retrieved_evidence_ids: list[str] = Field(default_factory=list)
    image_status: ToolStatus | None = None
    audio_status: ToolStatus | None = None
    degraded: bool = False
    system_failure: bool = False
    error_category: str | None = None


class RouterState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    incoming: IncomingMessage
    attempts: int
    finalized: bool
    classification: Classification
    used_fallback: bool
    scam_scan: ScamScanResult
    history_search: HistorySearchResult
    image_extraction: ImageExtractionResult
    audio_transcription: AudioTranscriptionResult
    tool_trace: Annotated[list[str], operator.add]
    routing_errors: Annotated[list[str], operator.add]
    fallback_error_category: str


class RoutingResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    classification: Classification
    used_fallback: bool = False
    error: str | None = None
    diagnostics: RoutingDiagnostics = Field(default_factory=RoutingDiagnostics)
