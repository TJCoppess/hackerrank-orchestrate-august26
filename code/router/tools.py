from __future__ import annotations

import json

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing_extensions import Annotated

from .io import OutputStore
from .models import Action, Classification, MessageType, RouterState


def create_write_final_classification_tool(output_store: OutputStore) -> BaseTool:
    @tool("write_final_classification")
    def write_final_classification(
        action: Action,
        message_type: MessageType,
        reason: str,
        confidence: float,
        evidence_message_ids: list[str],
        state: Annotated[RouterState, InjectedState],
    ) -> Command:
        """Validate and persist the final classification for the current message."""
        classification = Classification(
            action=action,
            message_type=message_type,
            reason=reason,
            confidence=confidence,
            evidence_message_ids=evidence_message_ids,
        )
        message_id = state["incoming"].message_id
        output_store.upsert(message_id, classification)

        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", [])
        tool_call_id = tool_calls[0]["id"] if tool_calls else "write-final"
        result = {
            "message_id": message_id,
            "status": "written",
            "classification": classification.model_dump(mode="json"),
        }
        return Command(
            update={
                "classification": classification,
                "finalized": True,
                "used_fallback": False,
                "messages": [
                    ToolMessage(
                        content=json.dumps(result, ensure_ascii=False),
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    return write_final_classification
