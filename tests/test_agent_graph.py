from __future__ import annotations

import json
import uuid

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from supportmind.agent.graph import AgentDeps, build_agent_graph
from tests.conftest import FakeLLMClient, make_orders_client


def understanding(intent: str, order_id: int | None = None, amount: float | None = None, reason: str = "", missing_field: str | None = None) -> str:
    return json.dumps(
        {
            "intent": intent,
            "order_id": order_id,
            "amount": amount,
            "reason": reason,
            "confidence": 0.95,
            "missing_field": missing_field,
        }
    )


_CHECKPOINTER_KEEPALIVE: list = []


def make_graph(agent_config, orders_client, retrieval_service, llm_client, tmp_path):
    checkpoints_db = tmp_path / f"checkpoints-{uuid.uuid4().hex}.db"
    checkpointer_cm = SqliteSaver.from_conn_string(str(checkpoints_db))
    checkpointer = checkpointer_cm.__enter__()
    _CHECKPOINTER_KEEPALIVE.append(checkpointer_cm)
    deps = AgentDeps(llm_client=llm_client, orders_client=orders_client, retrieval_service=retrieval_service, config=agent_config)
    return build_agent_graph(deps, checkpointer)


def invoke(graph, thread_id: str, user_message: str):
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke({"thread_id": thread_id, "messages": [{"role": "user", "content": user_message}]}, config=config)


def test_missing_order_id_asks_for_it(agent_config, orders_client, tmp_path):
    llm = FakeLLMClient([understanding("order_status", order_id=None, missing_field="order_id")])
    graph = make_graph(agent_config, orders_client, None, llm, tmp_path)

    result = invoke(graph, "thread-ask", "What's the status of my order?")

    assert "order number" in result["final_response"].lower()


def test_order_status_happy_path(agent_config, orders_client, seeded_order, tmp_path):
    llm = FakeLLMClient([understanding("order_status", order_id=seeded_order["order_id"])])
    graph = make_graph(agent_config, orders_client, None, llm, tmp_path)

    result = invoke(graph, "thread-status", f"Where is order {seeded_order['order_id']}?")

    assert str(seeded_order["order_id"]) in result["final_response"]
    assert "placed" in result["final_response"].lower()


def test_cancel_order_happy_path(agent_config, orders_client, seeded_order, tmp_path):
    llm = FakeLLMClient([understanding("cancel_order", order_id=seeded_order["order_id"])])
    graph = make_graph(agent_config, orders_client, None, llm, tmp_path)

    result = invoke(graph, "thread-cancel", f"Cancel order {seeded_order['order_id']}")

    assert "cancelled" in result["final_response"].lower()


def test_refund_under_cap_auto_completes(agent_config, orders_client, seeded_order, tmp_path):
    cap = agent_config.agent.refund_approval_cap
    amount = min(seeded_order["amount"], cap - 1)
    llm = FakeLLMClient([understanding("refund", order_id=seeded_order["order_id"], amount=amount, reason="damaged")])
    graph = make_graph(agent_config, orders_client, None, llm, tmp_path)

    result = invoke(graph, "thread-refund-auto", f"Refund order {seeded_order['order_id']}")

    assert "processed a refund" in result["final_response"].lower()


def test_escalate_intent_routes_directly(agent_config, orders_client, tmp_path):
    llm = FakeLLMClient([understanding("escalate", reason="customer is angry")])
    graph = make_graph(agent_config, orders_client, None, llm, tmp_path)

    result = invoke(graph, "thread-escalate", "I want to speak to a human right now!")

    assert result["escalated"] is True


def test_repeated_oms_failures_trigger_escalation(agent_config, seeded_order, tmp_path):
    flaky_client = make_orders_client(agent_config, failure_rate=1.0)
    llm = FakeLLMClient([understanding("order_status", order_id=seeded_order["order_id"])])
    graph = make_graph(agent_config, flaky_client, None, llm, tmp_path)

    result = invoke(graph, "thread-flaky", f"Status of order {seeded_order['order_id']}?")

    assert result["escalated"] is True
    assert result["consecutive_failures"] >= agent_config.agent.max_consecutive_failures


@pytest.mark.requires_index
def test_kb_question_returns_grounded_answer(agent_config, orders_client, retrieval_service, tmp_path):
    llm = FakeLLMClient(
        [
            understanding("kb_question", reason="how do I reset my password"),
            "You can reset your password from the login page.\n\nSources: [row-1-chunk-0]",
        ]
    )
    graph = make_graph(agent_config, orders_client, retrieval_service, llm, tmp_path)

    result = invoke(graph, "thread-kb", "How do I reset my password?")

    assert "Sources:" in result["final_response"]
