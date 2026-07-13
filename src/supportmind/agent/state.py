from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    thread_id: str
    messages: Annotated[list[dict[str, Any]], operator.add]

    intent: str | None
    order_id: int | None
    amount: float | None
    reason: str | None
    confidence: float
    missing_field: str | None

    order_snapshot: dict[str, Any] | None
    kb_chunks: list[dict[str, Any]]
    tool_log: list[dict[str, Any]]
    consecutive_failures: int

    escalated: bool
    escalation_reason: str | None
    final_response: str | None
