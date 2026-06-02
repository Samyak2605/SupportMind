from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from supportmind.agent.prompts import (
    KB_RESPOND_SYSTEM_PROMPT,
    UNDERSTAND_SYSTEM_PROMPT,
    build_kb_user_prompt,
    build_understand_user_prompt,
)
from supportmind.agent.state import AgentState
from supportmind.agent.tools import (
    ToolResult,
    cancel_order,
    escalate_to_human,
    kb_search,
    order_lookup,
    refund_initiate,
)
from supportmind.config import SupportMindConfig
from supportmind.llm.client import LLMClient
from supportmind.orders.client import OrdersClient
from supportmind.retrieval.pipeline import RetrievalService

logger = logging.getLogger(__name__)

VALID_INTENTS = {"order_status", "cancel_order", "refund", "kb_question", "escalate"}
ORDER_REQUIRED_INTENTS = {"order_status", "cancel_order", "refund"}


@dataclass
class AgentDeps:
    llm_client: LLMClient
    orders_client: OrdersClient
    retrieval_service: RetrievalService
    config: SupportMindConfig


def _last_user_message(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def understand_node(state: AgentState, deps: AgentDeps) -> dict:
    messages = state.get("messages", [])
    user_prompt = build_understand_user_prompt(messages)
    raw = deps.llm_client.complete(UNDERSTAND_SYSTEM_PROMPT, user_prompt, json_mode=True)

    try:
        parsed = json.loads(raw)
        if parsed.get("intent") not in VALID_INTENTS:
            raise ValueError(f"invalid intent: {parsed.get('intent')}")
    except (json.JSONDecodeError, ValueError, AttributeError) as exc:
        logger.warning("Failed to parse understanding output (%s); escalating", exc)
        parsed = {
            "intent": "escalate",
            "order_id": None,
            "amount": None,
            "reason": "Could not confidently understand the request.",
            "confidence": 0.0,
            "missing_field": None,
        }

    raw_order_id = parsed.get("order_id")
    try:
        order_id = int(raw_order_id) if raw_order_id is not None else None
    except (TypeError, ValueError):
        logger.warning("Understanding returned a non-numeric order_id (%r); treating as missing", raw_order_id)
        order_id = None

    return {
        "intent": parsed.get("intent"),
        "order_id": order_id,
        "amount": parsed.get("amount"),
        "reason": parsed.get("reason") or "",
        "confidence": float(parsed.get("confidence") or 0.0),
        "missing_field": parsed.get("missing_field"),
        "order_snapshot": None,
        "kb_chunks": [],
        "tool_log": [],
        "consecutive_failures": 0,
        "escalated": False,
        "escalation_reason": None,
    }


def route_after_understand(state: AgentState) -> Literal["ask", "act", "escalate"]:
    if state.get("intent") == "escalate":
        return "escalate"
    if state.get("intent") in ORDER_REQUIRED_INTENTS and state.get("order_id") is None:
        return "ask"
    return "act"


def ask_node(state: AgentState, deps: AgentDeps) -> dict:
    question = "Could you share your order number so I can look into this for you?"
    return {"messages": [{"role": "assistant", "content": question}], "final_response": question}


def act_node(state: AgentState, deps: AgentDeps) -> dict:
    intent = state.get("intent")
    thread_id = state["thread_id"]
    tool_log = list(state.get("tool_log") or [])
    consecutive_failures = state.get("consecutive_failures", 0)
    order_snapshot = state.get("order_snapshot")
    kb_chunks = list(state.get("kb_chunks") or [])
    max_iterations = deps.config.agent.max_tool_iterations
    max_failures = deps.config.agent.max_consecutive_failures

    def log(tool: str, result: ToolResult) -> None:
        tool_log.append({"tool": tool, "success": result.success, "error": result.error, "data": result.data})

    iterations = 0
    while iterations < max_iterations:
        iterations += 1

        if intent == "kb_question":
            query = state.get("reason") or _last_user_message(state.get("messages", []))
            result = kb_search(deps.retrieval_service, query)
            log("kb_search", result)
            if result.success:
                kb_chunks = result.data["chunks"]
            break

        if intent in ORDER_REQUIRED_INTENTS and order_snapshot is None:
            result = order_lookup(deps.orders_client, state["order_id"])
            log("order_lookup", result)
            if result.success:
                order_snapshot = result.data
                consecutive_failures = 0
                if intent == "order_status":
                    break
                continue
            if result.retryable:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    break
                continue
            break

        if intent == "cancel_order":
            result = cancel_order(deps.orders_client, state["order_id"])
            log("cancel_order", result)
            if result.success:
                order_snapshot = {**(order_snapshot or {}), **result.data}
                break
            if result.retryable:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    break
                continue
            break

        if intent == "refund":
            amount = state.get("amount") or (order_snapshot or {}).get("amount")
            result = refund_initiate(
                deps.orders_client,
                thread_id,
                state["order_id"],
                amount,
                state.get("reason") or "",
                deps.config.agent.refund_approval_cap,
            )
            log("refund_initiate", result)
            if result.success:
                order_snapshot = {**(order_snapshot or {}), "refund": result.data}
                break
            if result.retryable:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    break
                continue
            break

        break

    return {
        "order_snapshot": order_snapshot,
        "kb_chunks": kb_chunks,
        "tool_log": tool_log,
        "consecutive_failures": consecutive_failures,
    }


def guard_node(state: AgentState, deps: AgentDeps) -> dict:
    return {}


def route_after_guard(state: AgentState, deps: AgentDeps) -> Literal["escalate", "respond"]:
    if state.get("consecutive_failures", 0) >= deps.config.agent.max_consecutive_failures:
        return "escalate"
    return "respond"


def escalate_node(state: AgentState, deps: AgentDeps) -> dict:
    reason = state.get("escalation_reason") or state.get("reason") or "The agent could not complete this request automatically."
    result = escalate_to_human(deps.orders_client, state["thread_id"], reason, state.get("order_id"))
    message = "I've escalated this to a human specialist who will follow up with you shortly."
    tool_log = list(state.get("tool_log") or []) + [{"tool": "escalate_to_human", "success": result.success, "data": result.data}]
    return {
        "escalated": True,
        "final_response": message,
        "messages": [{"role": "assistant", "content": message}],
        "tool_log": tool_log,
    }


def _tool_error(tool_log: list[dict], tool_name: str) -> str | None:
    for entry in reversed(tool_log):
        if entry.get("tool") == tool_name:
            return entry.get("error")
    return None


def respond_node(state: AgentState, deps: AgentDeps) -> dict:
    intent = state.get("intent")
    snapshot = state.get("order_snapshot") or {}
    tool_log = state.get("tool_log") or []

    if intent == "kb_question":
        chunks = state.get("kb_chunks") or []
        if not chunks:
            text = "I do not have enough grounded information in the knowledge base to answer that confidently.\n\nSources: none"
        else:
            query = _last_user_message(state.get("messages", []))
            user_prompt = build_kb_user_prompt(query, chunks[:2])
            text = deps.llm_client.complete(KB_RESPOND_SYSTEM_PROMPT, user_prompt, json_mode=False)
    elif intent == "order_status":
        if snapshot:
            text = f"Order #{snapshot.get('order_id')} ({snapshot.get('product_name')}) is currently **{snapshot.get('status')}**."
        else:
            text = _tool_error(tool_log, "order_lookup") or f"I couldn't find order #{state.get('order_id')}."
    elif intent == "cancel_order":
        if snapshot.get("status") == "cancelled":
            text = f"Order #{snapshot.get('order_id')} has been cancelled."
        else:
            text = _tool_error(tool_log, "cancel_order") or _tool_error(tool_log, "order_lookup") or (
                f"I wasn't able to cancel order #{state.get('order_id')}."
            )
    elif intent == "refund":
        refund_info = snapshot.get("refund") or {}
        status = refund_info.get("status")
        amount = state.get("amount") or snapshot.get("amount") or 0.0
        if status == "completed":
            text = f"I've processed a refund of ${amount:.2f} for order #{state.get('order_id')}."
        elif status == "rejected":
            text = f"Your refund request of ${amount:.2f} for order #{state.get('order_id')} was reviewed and could not be approved."
        else:
            text = _tool_error(tool_log, "refund_initiate") or _tool_error(tool_log, "order_lookup") or (
                f"I wasn't able to process the refund for order #{state.get('order_id')}."
            )
    else:
        text = "I'm not sure how to help with that yet."

    return {"final_response": text, "messages": [{"role": "assistant", "content": text}]}


def build_agent_graph(deps: AgentDeps, checkpointer: BaseCheckpointSaver):
    graph = StateGraph(AgentState)
    graph.add_node("understand", lambda state: understand_node(state, deps))
    graph.add_node("ask", lambda state: ask_node(state, deps))
    graph.add_node("act", lambda state: act_node(state, deps))
    graph.add_node("guard", lambda state: guard_node(state, deps))
    graph.add_node("respond", lambda state: respond_node(state, deps))
    graph.add_node("escalate", lambda state: escalate_node(state, deps))

    graph.set_entry_point("understand")
    graph.add_conditional_edges("understand", route_after_understand, {"ask": "ask", "act": "act", "escalate": "escalate"})
    graph.add_edge("ask", END)
    graph.add_edge("act", "guard")
    graph.add_conditional_edges("guard", lambda state: route_after_guard(state, deps), {"escalate": "escalate", "respond": "respond"})
    graph.add_edge("respond", END)
    graph.add_edge("escalate", END)

    return graph.compile(checkpointer=checkpointer)
