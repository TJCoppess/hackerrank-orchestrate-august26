from __future__ import annotations

import os
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .io import OutputStore
from .models import (
    Action,
    Classification,
    IncomingMessage,
    MessageType,
    RouterState,
    RoutingResult,
)
from .prompts import SYSTEM_PROMPT, build_message_prompt
from .tools import create_write_final_classification_tool


MAX_MODEL_ATTEMPTS = 2


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
    def __init__(self, model: Any, output_store: OutputStore) -> None:
        self.output_store = output_store
        self.write_tool = create_write_final_classification_tool(output_store)
        self.bound_model = model.bind_tools(
            [self.write_tool],
            tool_choice="write_final_classification",
            strict=True,
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        builder = StateGraph(RouterState)
        builder.add_node("orchestrator", self._orchestrator_node)
        builder.add_node("tools", ToolNode([self.write_tool]))
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
        return {
            "messages": [response],
            "attempts": state.get("attempts", 0) + 1,
        }

    @staticmethod
    def _route_after_orchestrator(
        state: RouterState,
    ) -> Literal["tools", "retry", "fallback"]:
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", [])
        if (
            len(tool_calls) == 1
            and tool_calls[0].get("name") == "write_final_classification"
        ):
            return "tools"
        if state.get("attempts", 0) >= MAX_MODEL_ATTEMPTS:
            return "fallback"
        return "retry"

    @staticmethod
    def _retry_node(state: RouterState) -> dict[str, Any]:
        return {
            "messages": [
                HumanMessage(
                    content=(
                        "You did not call write_final_classification. "
                        "Call it now exactly once with all required fields."
                    )
                )
            ]
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
        }

    @staticmethod
    def _fallback_classification(message: IncomingMessage) -> Classification:
        if message.media_type:
            reason = (
                "Attachment contents are unavailable in Phase 1, so this message "
                "is deferred for later review."
            )
        else:
            reason = (
                "The orchestrator could not complete classification, so the message "
                "is conservatively deferred."
            )
        return Classification(
            action=Action.DIGEST,
            message_type=MessageType.UNKNOWN,
            reason=reason,
            confidence=0.20,
            evidence_message_ids=[],
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
        }
        try:
            result = self.graph.invoke(initial_state, {"recursion_limit": 8})
            classification = result.get("classification")
            if not result.get("finalized") or classification is None:
                raise RuntimeError("graph ended without a final classification")
            if not isinstance(classification, Classification):
                classification = Classification.model_validate(classification)
            return RoutingResult(
                classification=classification,
                used_fallback=bool(result.get("used_fallback", False)),
            )
        except Exception as exc:
            classification = self._fallback_classification(message)
            self.output_store.upsert(message.message_id, classification)
            return RoutingResult(
                classification=classification,
                used_fallback=True,
                error=f"{type(exc).__name__}: {exc}",
            )
