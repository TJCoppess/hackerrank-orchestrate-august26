from __future__ import annotations

import json
import inspect
import time
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing_extensions import Annotated

from .io import OutputContractError, OutputStore
from .media import MediaProcessor
from .models import (
    Action,
    AudioTranscriptionResult,
    Classification,
    ImageExtractionResult,
    MessageType,
    RouterState,
    ToolStatus,
)
from .retrieval import HistoryRepository
from .scam import scan_message
from .tracing import TraceRecorder


def _tool_message(tool_call_id: str, payload: Any) -> ToolMessage:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    return ToolMessage(
        content=json.dumps(payload, ensure_ascii=False),
        tool_call_id=tool_call_id,
    )


def effective_text(state: RouterState) -> str:
    parts = [state["incoming"].message_text]
    image = state.get("image_extraction")
    if image and image.status == ToolStatus.OK:
        parts.extend([image.ocr_text, image.visual_summary])
    audio = state.get("audio_transcription")
    if audio and audio.status == ToolStatus.OK:
        parts.append(audio.transcript)
    return "\n".join(part.strip() for part in parts if part and part.strip())


def create_phase2_tools(
    output_store: OutputStore,
    history_repository: HistoryRepository,
    media_processor: MediaProcessor,
    trace: TraceRecorder | None = None,
) -> list[BaseTool]:
    def started(name: str, state: RouterState) -> float:
        if trace is not None:
            trace.emit("tool_start", message_id=state["incoming"].message_id, tool=name)
            trace.show(f"  tool {name}", "blue")
        return time.perf_counter()

    def ended(name: str, state: RouterState, started_at: float, status: str = "ok", **extra: Any) -> None:
        if trace is not None:
            trace.emit(
                "tool_end", message_id=state["incoming"].message_id, tool=name,
                status=status, duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
                **extra,
            )

    def call_media(method: Any, path: Any, message_id: str) -> Any:
        """Keep injected test/custom processors compatible with the Phase 2 interface."""
        return method(path, message_id) if len(inspect.signature(method).parameters) >= 2 else method(path)

    @tool("process_image")
    def process_image(
        state: Annotated[RouterState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Extract structured OCR and visual evidence from this message's image."""
        began = started("process_image", state)
        cached = state.get("image_extraction")
        result = cached or (
            call_media(media_processor.process_image,
                state["incoming"].image_path, state["incoming"].message_id
            )
            if state["incoming"].image_path is not None
            else ImageExtractionResult(
                status=ToolStatus.ERROR, error="current message has no image path"
            )
        )
        ended("process_image", state, began, result.status.value, cached=bool(cached), error_category=result.error_category)
        return Command(
            update={
                "image_extraction": result,
                "tool_trace": ["process_image:cached" if cached else "process_image"],
                "messages": [_tool_message(tool_call_id, result)],
            }
        )

    @tool("process_audio")
    def process_audio(
        state: Annotated[RouterState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Transcribe this message's voice note verbatim into untrusted evidence text."""
        began = started("process_audio", state)
        cached = state.get("audio_transcription")
        result = cached or (
            call_media(media_processor.process_audio,
                state["incoming"].audio_path, state["incoming"].message_id
            )
            if state["incoming"].audio_path is not None
            else AudioTranscriptionResult(
                status=ToolStatus.ERROR, error="current message has no audio path"
            )
        )
        ended("process_audio", state, began, result.status.value, cached=bool(cached), error_category=result.error_category)
        return Command(
            update={
                "audio_transcription": result,
                "tool_trace": ["process_audio:cached" if cached else "process_audio"],
                "messages": [_tool_message(tool_call_id, result)],
            }
        )

    @tool("scan_scam_heuristics")
    def scan_scam_heuristics(
        state: Annotated[RouterState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Run deterministic scam-risk checks on current text and extracted media text."""
        began = started("scan_scam_heuristics", state)
        cached = state.get("scam_scan")
        official_domain, sender_domain = history_repository.business_domains(
            state["incoming"].business_id
        )
        result = cached or scan_message(
            state["incoming"], effective_text(state), official_domain, sender_domain
        )
        ended("scan_scam_heuristics", state, began, cached=bool(cached), risk_score=result.risk_score)
        return Command(
            update={
                "scam_scan": result,
                "tool_trace": [
                    "scan_scam_heuristics:cached" if cached else "scan_scam_heuristics"
                ],
                "messages": [_tool_message(tool_call_id, result)],
            }
        )

    @tool("query_user_history")
    def query_user_history(
        search_term: str,
        state: Annotated[RouterState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Find up to five relevant prior messages for the active user and return context."""
        began = started("query_user_history", state)
        cached = state.get("history_search")
        result = cached or history_repository.search(state["incoming"], search_term)
        ended("query_user_history", state, began, cached=bool(cached), match_count=len(result.matches))
        return Command(
            update={
                "history_search": result,
                "tool_trace": ["query_user_history:cached" if cached else "query_user_history"],
                "messages": [_tool_message(tool_call_id, result)],
            }
        )

    @tool("write_final_classification")
    def write_final_classification(
        action: Action,
        message_type: MessageType,
        reason: str,
        confidence: float,
        evidence_message_ids: list[str],
        state: Annotated[RouterState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Validate and persist the one final classification for the current message."""
        began = started("write_final_classification", state)
        classification = Classification(
            action=action,
            message_type=message_type,
            reason=reason,
            confidence=confidence,
            evidence_message_ids=evidence_message_ids,
        )
        incoming = state["incoming"]
        output_store.validate_evidence_owner(
            incoming.user_id, classification.evidence_message_ids
        )
        history = state.get("history_search")
        if history is None and classification.evidence_message_ids:
            raise OutputContractError(
                "evidence_message_ids must be empty unless query_user_history ran"
            )
        if history is not None:
            returned_ids = {match.message_id for match in history.matches}
            invalid = set(classification.evidence_message_ids) - returned_ids
            if invalid:
                raise OutputContractError(
                    "evidence IDs were not returned by query_user_history: "
                    + ", ".join(sorted(invalid))
                )
        media_results = [state.get("image_extraction"), state.get("audio_transcription")]
        if any(result and result.status == ToolStatus.ERROR for result in media_results):
            if classification.confidence > 0.60:
                raise OutputContractError(
                    "confidence must be at most 0.60 when required media extraction failed"
                )

        output_store.upsert(incoming.message_id, classification)
        ended("write_final_classification", state, began, action=classification.action.value)
        if trace is not None:
            trace.emit(
                "classification", message_id=incoming.message_id,
                classification=classification.model_dump(mode="json"), degraded=any(
                    result and result.status == ToolStatus.ERROR for result in media_results
                ), system_failure=False,
            )
        result = {
            "message_id": incoming.message_id,
            "status": "written",
            "classification": classification.model_dump(mode="json"),
        }
        return Command(
            update={
                "classification": classification,
                "finalized": True,
                "used_fallback": False,
                "tool_trace": ["write_final_classification"],
                "messages": [_tool_message(tool_call_id, result)],
            }
        )

    return [
        process_image,
        process_audio,
        scan_scam_heuristics,
        query_user_history,
        write_final_classification,
    ]


def tools_by_name(tools: list[BaseTool]) -> dict[str, BaseTool]:
    return {item.name: item for item in tools}
