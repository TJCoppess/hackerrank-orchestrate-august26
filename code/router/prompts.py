from __future__ import annotations

import json

from .models import IncomingMessage


SYSTEM_PROMPT = """You are the routing orchestrator for a WhatsApp notification system.

Classify exactly one incoming message and finish by calling
write_final_classification exactly once.

Actions:
- notify: interrupt now for credible, time-sensitive, urgent, or directly relevant content.
- digest: safe and potentially useful, but not urgent.
- mute: low-value, repetitive, unwanted, suspicious, spam-like, scam-like, or unsafe.

Allowed message_type values:
personal, urgent, event, payment, business_update, promotion, greeting,
forward, spam, scam, unknown.

Phase 1 limitations:
- Attachment paths are metadata only. You cannot inspect attachment contents yet.
- Never invent text, claims, or risk signals from an image or voice attachment.
- Historical retrieval is not implemented. Always pass evidence_message_ids as [].
- If attachment contents are necessary to decide, choose a cautious low-confidence
  digest/unknown result and state that the attachment has not yet been inspected.

The reason must be a concise single-line explanation. Confidence must be between
0 and 1. Preserve the user's identifiers and message content exactly while
reasoning, but do not include a message_id argument in the terminal tool call.
"""


def build_message_prompt(message: IncomingMessage) -> str:
    payload = message.model_dump(mode="json")
    return (
        "Classify this incoming message. Call write_final_classification once "
        "with the complete output payload.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
