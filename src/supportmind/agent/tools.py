from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

from langgraph.types import interrupt

from supportmind.orders.client import (
    OrderNotCancellableError,
    OrderNotFoundError,
    OrdersClient,
    SimulatedOmsError,
)
from supportmind.retrieval.pipeline import RetrievalService

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    # retryable=True means a transient OMS failure (counts toward escalation);
    # False means a terminal business outcome (e.g. "order not found") to explain directly.
    retryable: bool = False


def refund_idempotency_key(thread_id: str, order_id: int, amount: float) -> str:
    payload = f"{thread_id}:{order_id}:{amount}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def order_lookup(orders_client: OrdersClient, order_id: int) -> ToolResult:
    try:
        snapshot = orders_client.lookup_order(order_id)
    except OrderNotFoundError as exc:
        return ToolResult(success=False, error=str(exc), retryable=False)
    except SimulatedOmsError as exc:
        return ToolResult(success=False, error=str(exc), retryable=True)

    return ToolResult(
        success=True,
        data={
            "order_id": snapshot.order_id,
            "customer_name": snapshot.customer_name,
            "customer_email": snapshot.customer_email,
            "product_name": snapshot.product_name,
            "amount": snapshot.amount,
            "status": snapshot.status,
        },
    )


def cancel_order(orders_client: OrdersClient, order_id: int) -> ToolResult:
    try:
        snapshot = orders_client.cancel_order(order_id)
    except OrderNotFoundError as exc:
        return ToolResult(success=False, error=str(exc), retryable=False)
    except OrderNotCancellableError as exc:
        return ToolResult(success=False, error=str(exc), retryable=False)
    except SimulatedOmsError as exc:
        return ToolResult(success=False, error=str(exc), retryable=True)

    return ToolResult(success=True, data={"order_id": snapshot.order_id, "status": snapshot.status})


def refund_initiate(
    orders_client: OrdersClient,
    thread_id: str,
    order_id: int,
    amount: float,
    reason: str,
    refund_approval_cap: float,
) -> ToolResult:
    idempotency_key = refund_idempotency_key(thread_id, order_id, amount)

    try:
        refund = orders_client.get_or_create_refund(order_id, amount, idempotency_key, reason)
    except OrderNotFoundError as exc:
        return ToolResult(success=False, error=str(exc), retryable=False)
    except SimulatedOmsError as exc:
        return ToolResult(success=False, error=str(exc), retryable=True)

    if refund.status == "completed":
        return ToolResult(success=True, data={"status": "completed", "refund_id": refund.id, "auto_approved": True})
    if refund.status == "rejected":
        return ToolResult(success=True, data={"status": "rejected", "refund_id": refund.id})

    if amount <= refund_approval_cap:
        orders_client.complete_refund(refund.id)
        return ToolResult(success=True, data={"status": "completed", "refund_id": refund.id, "auto_approved": True})

    approval = orders_client.get_or_create_approval(thread_id, refund.id, order_id, amount)
    decision = interrupt(
        {
            "kind": "refund_approval",
            "approval_id": approval.id,
            "order_id": order_id,
            "amount": amount,
            "reason": reason,
        }
    )

    if decision.get("decision") == "approved":
        orders_client.complete_refund(refund.id)
        return ToolResult(
            success=True,
            data={"status": "completed", "refund_id": refund.id, "approval_id": approval.id, "auto_approved": False},
        )

    orders_client.reject_refund(refund.id)
    return ToolResult(
        success=True,
        data={"status": "rejected", "refund_id": refund.id, "approval_id": approval.id},
    )


def escalate_to_human(orders_client: OrdersClient, thread_id: str, reason: str, order_id: int | None = None) -> ToolResult:
    escalation = orders_client.create_escalation(thread_id=thread_id, reason=reason, order_id=order_id)
    return ToolResult(success=True, data={"escalation_id": escalation.id, "reason": reason})


def kb_search(retrieval_service: RetrievalService, query: str, top_k: int = 4) -> ToolResult:
    try:
        chunks, _traces, _rerank_ms = retrieval_service.retrieve(query=query, mode="hybrid", use_reranker=True, top_k=top_k)
    except Exception as exc:  # noqa: BLE001 - retrieval failures should not crash the agent
        logger.exception("kb_search failed")
        return ToolResult(success=False, error=str(exc))

    return ToolResult(
        success=True,
        data={
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "category": chunk.metadata.get("category"),
                    "intent": chunk.metadata.get("intent"),
                }
                for chunk in chunks
            ]
        },
    )
