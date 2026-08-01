from __future__ import annotations

import os
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .io import OutputStore
from .media import MediaProcessor
from .models import (
    Action,
    Classification,
    IncomingMessage,
    MessageType,
    RouterState,
    RoutingDiagnostics,
    RoutingResult,
    ToolStatus,
)
from .prompts import SYSTEM_PROMPT, build_message_prompt
from .retrieval import HistoryRepository
from .tools import create_phase2_tools, tools_by_name


MAX_MODEL_ATTEMPTS = 6
GRAPH_RECURSION_LIMIT = 20
TERMINAL_TOOL = "write_final_classification"


def create_openai_model() -> Any:
    from langchain_openai import ChatOpenAI

    model_name = os.environ.get("ORCHESTRATOR_MODEL", "gpt-5.6-sol")
    reasoning_effort = os.environ.get("REASONING_EFFORT", "low")
    if reasoning_effort not in {"none", "low", "medium", "high", "xhigh", "max"}:
        raise ValueError(f"unsupported REASONING_EFFORT: {reasoning_effort!r}")
    return ChatOpenAI(
        model=model_name,
        use_responses_api=True,
        reasoning={"effort": reasoning_effort},
        timeout=60,
        max_retries=2,
    )


class MessageRouter:
    def __init__(
        self,
        model: Any,
        output_store: OutputStore,
        history_repository: HistoryRepository | None = None,
        media_processor: MediaProcessor | None = None,
    ) -> None:
        self.output_store = output_store
        self.history_repository = history_repository or HistoryRepository(
            output_store.dataset_dir
        )
        self.media_processor = media_processor or MediaProcessor()
        self.tools = create_phase2_tools(
            output_store, self.history_repository, self.media_processor
        )
        self.tool_map = tools_by_name(self.tools)
        self.write_tool = self.tool_map[TERMINAL_TOOL]
        self.bound_model = model.bind_tools(
            self.tools,
            tool_choice="required",
            strict=True,
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        builder = StateGraph(RouterState)
        builder.add_node("orchestrator", self._orchestrator_node)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_node("retry", self._retry_node)
        builder.add_node("fallback", self._fallback_node)

        builder.add_edge(START, "orchestrator")
        builder.add_conditional_edges(
            "orchestrator",
            self._route_after_orchestrator,
            {"tools": "tools", "retry": "retry", "fallback": "fallback"},
        )
        builder.add_edge("retry", "orchestrator")
        builder.add_conditional_edges(
            "tools",
            self._route_after_tools,
            {"orchestrator": "orchestrator", "fallback": "fallback", "end": END},
        )
        builder.add_edge("fallback", END)
        return builder.compile()

    def _orchestrator_node(self, state: RouterState) -> dict[str, Any]:
        response = self.bound_model.invoke(state["messages"])
        return {"messages": [response], "attempts": state.get("attempts", 0) + 1}

    @staticmethod
    def _required_media_tool(state: RouterState) -> str | None:
        media_type = state["incoming"].media_type
        if media_type == "image" and "image_extraction" not in state:
            return "process_image"
        if media_type == "voice" and "audio_transcription" not in state:
            return "process_audio"
        return None

    def _validate_tool_calls(self, state: RouterState) -> str | None:
        last_message = state["messages"][-1]
        calls = list(getattr(last_message, "tool_calls", []) or [])
        if not calls:
            return "No tool was called. Follow the required workflow."
        names = [str(call.get("name", "")) for call in calls]
        if len(names) != len(set(names)):
            return "Do not call the same tool more than once in a batch."
        known = {
            "process_image", "process_audio", "scan_scam_heuristics",
            "query_user_history", TERMINAL_TOOL,
        }
        if any(name not in known for name in names):
            return "An unknown tool was requested."

        required_media = self._required_media_tool(state)
        if required_media:
            if names != [required_media]:
                return f"Call {required_media} alone before analysis or finalization."
            return None
        if "process_image" in names or "process_audio" in names:
            return "Do not call a wrong-media or already-completed media tool."

        if TERMINAL_TOOL in names:
            if names != [TERMINAL_TOOL]:
                return "Call the terminal tool alone; do not mix it with other tools."
            if "scam_scan" not in state:
                return "Run scan_scam_heuristics before final classification."
            args = calls[0].get("args", {}) or {}
            evidence = args.get("evidence_message_ids", [])
            if evidence:
                history = state.get("history_search")
                if history is None:
                    return "Do not cite evidence unless query_user_history ran."
                returned = {match.message_id for match in history.matches}
                invalid = set(evidence) - returned
                if invalid:
                    return "Cite only IDs returned by query_user_history."
                wrong_owner = [
                    item
                    for item in evidence
                    if self.output_store.history_user_by_id.get(item)
                    != state["incoming"].user_id
                ]
                if wrong_owner:
                    return "Cite historical messages only for the active user."
            confidence = args.get("confidence")
            media_results = [state.get("image_extraction"), state.get("audio_transcription")]
            if (
                isinstance(confidence, (int, float))
                and confidence > 0.60
                and any(item and item.status == ToolStatus.ERROR for item in media_results)
            ):
                return "Cap confidence at 0.60 because required media extraction failed."
            return None

        allowed_analysis = {"scan_scam_heuristics", "query_user_history"}
        if any(name not in allowed_analysis for name in names):
            return "Only scam scanning and optional history retrieval are valid now."
        if "scan_scam_heuristics" in names and "scam_scan" in state:
            return "Scam analysis is already complete; do not repeat it."
        if "query_user_history" in names and "history_search" in state:
            return "History retrieval is already complete; do not repeat it."
        if "scam_scan" not in state and "scan_scam_heuristics" not in names:
            return "Run scan_scam_heuristics before final classification."
        return None

    def _route_after_orchestrator(
        self, state: RouterState
    ) -> Literal["tools", "retry", "fallback"]:
        if self._validate_tool_calls(state) is None:
            return "tools"
        if state.get("attempts", 0) >= MAX_MODEL_ATTEMPTS:
            return "fallback"
        return "retry"

    def _retry_node(self, state: RouterState) -> dict[str, Any]:
        error = self._validate_tool_calls(state) or "The prior tool call was invalid."
        return {
            "messages": [HumanMessage(content=f"Tool workflow correction: {error}")],
            "routing_errors": [error],
        }

    @staticmethod
    def _route_after_tools(
        state: RouterState,
    ) -> Literal["orchestrator", "fallback", "end"]:
        if state.get("finalized", False):
            return "end"
        if state.get("attempts", 0) >= MAX_MODEL_ATTEMPTS:
            return "fallback"
        return "orchestrator"

    def _fallback_node(self, state: RouterState) -> dict[str, Any]:
        classification = self._fallback_classification(state["incoming"])
        self.output_store.upsert(state["incoming"].message_id, classification)
        return {
            "classification": classification,
            "finalized": True,
            "used_fallback": True,
            "fallback_error_category": "orchestrator_exhausted",
            "tool_trace": ["system_fallback"],
        }

    @staticmethod
    def _fallback_classification(message: IncomingMessage) -> Classification:
        return Classification(
            action=Action.DIGEST,
            message_type=MessageType.UNKNOWN,
            reason="The routing workflow failed, so this message is conservatively deferred.",
            confidence=0.20,
            evidence_message_ids=[],
        )

    @staticmethod
    def _diagnostics(state: RouterState) -> RoutingDiagnostics:
        image = state.get("image_extraction")
        audio = state.get("audio_transcription")
        history = state.get("history_search")
        used_fallback = bool(state.get("used_fallback", False))
        media_degraded = any(
            item is not None and item.status == ToolStatus.ERROR
            for item in (image, audio)
        )
        return RoutingDiagnostics(
            attempts=state.get("attempts", 0),
            tool_trace=list(state.get("tool_trace", [])),
            scam_risk_score=(state["scam_scan"].risk_score if state.get("scam_scan") else None),
            retrieved_evidence_ids=(
                [match.message_id for match in history.matches] if history else []
            ),
            image_status=image.status if image else None,
            audio_status=audio.status if audio else None,
            degraded=media_degraded or used_fallback,
            system_failure=used_fallback,
            error_category=state.get("fallback_error_category"),
        )

    def classify(self, message: IncomingMessage) -> RoutingResult:
        initial_state: RouterState = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=build_message_prompt(message)),
            ],
            "incoming": message,
            "attempts": 0,
            "finalized": False,
            "used_fallback": False,
            "tool_trace": [],
            "routing_errors": [],
        }
        try:
            result = self.graph.invoke(
                initial_state, {"recursion_limit": GRAPH_RECURSION_LIMIT}
            )
            classification = result.get("classification")
            if not result.get("finalized") or classification is None:
                raise RuntimeError("graph ended without a final classification")
            if not isinstance(classification, Classification):
                classification = Classification.model_validate(classification)
            return RoutingResult(
                classification=classification,
                used_fallback=bool(result.get("used_fallback", False)),
                diagnostics=self._diagnostics(result),
            )
        except Exception as exc:
            classification = self._fallback_classification(message)
            self.output_store.upsert(message.message_id, classification)
            failed_state: RouterState = {
                **initial_state,
                "classification": classification,
                "finalized": True,
                "used_fallback": True,
                "fallback_error_category": "system_exception",
                "tool_trace": ["system_fallback"],
            }
            return RoutingResult(
                classification=classification,
                used_fallback=True,
                error=f"{type(exc).__name__}: {exc}",
                diagnostics=self._diagnostics(failed_state),
            )
