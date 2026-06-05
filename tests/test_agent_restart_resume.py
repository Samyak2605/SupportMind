from __future__ import annotations

import json
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from supportmind.agent.graph import AgentDeps, build_agent_graph
from supportmind.orders.client import OrdersClient
from supportmind.orders.db import build_session_factory
from tests.conftest import FakeLLMClient


def understanding(order_id: int, amount: float) -> str:
    return json.dumps(
        {
            "intent": "refund",
            "order_id": order_id,
            "amount": amount,
            "reason": "item arrived broken",
            "confidence": 0.95,
            "missing_field": None,
        }
    )


def test_refund_approval_survives_restart_and_resumes_from_checkpoint(agent_config, tmp_path):
    from supportmind.orders.models import Customer, Order

    session_factory = build_session_factory(agent_config.paths.orders_db)
    with session_factory() as session:
        customer = Customer(name="Big Spender", email="big.spender@example.com")
        session.add(customer)
        session.flush()
        order = Order(customer_id=customer.id, product_name="Premium Widget", amount=500.0, status="delivered")
        session.add(order)
        session.commit()
        order_id = order.id

    cap = agent_config.agent.refund_approval_cap
    assert 500.0 > cap, "test order amount must exceed the approval cap to trigger an interrupt"

    checkpoints_path = str(agent_config.paths.checkpoints_db)
    thread_id = f"thread-restart-{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}

    orders_client_1 = OrdersClient(
        session_factory=session_factory,
        failure_rate=0.0,
        approval_timeout_hours=agent_config.agent.approval_timeout_hours,
    )
    checkpointer_1 = SqliteSaver.from_conn_string(checkpoints_path)
    checkpointer_1_ctx = checkpointer_1.__enter__()
    llm_1 = FakeLLMClient([understanding(order_id, 500.0)])
    graph_1 = build_agent_graph(
        AgentDeps(llm_client=llm_1, orders_client=orders_client_1, retrieval_service=None, config=agent_config),
        checkpointer_1_ctx,
    )

    result = graph_1.invoke(
        {"thread_id": thread_id, "messages": [{"role": "user", "content": f"Refund order {order_id}, it arrived broken"}]},
        config=config,
    )

    assert "__interrupt__" in result
    interrupt_payload = result["__interrupt__"][0].value
    assert interrupt_payload["kind"] == "refund_approval"
    approval_id = interrupt_payload["approval_id"]

    with session_factory() as session:
        from supportmind.orders.models import Approval, Refund

        approval = session.get(Approval, approval_id)
        assert approval.status == "pending"
        refund = session.get(Refund, approval.refund_id)
        assert refund.status == "pending_approval"

    checkpointer_1.__exit__(None, None, None)
    del graph_1, checkpointer_1_ctx, checkpointer_1

    orders_client_1.decide_approval(approval_id, approved=True)

    orders_client_2 = OrdersClient(
        session_factory=build_session_factory(agent_config.paths.orders_db),
        failure_rate=0.0,
        approval_timeout_hours=agent_config.agent.approval_timeout_hours,
    )
    checkpointer_2 = SqliteSaver.from_conn_string(checkpoints_path)
    checkpointer_2_ctx = checkpointer_2.__enter__()
    llm_2 = FakeLLMClient([])
    graph_2 = build_agent_graph(
        AgentDeps(llm_client=llm_2, orders_client=orders_client_2, retrieval_service=None, config=agent_config),
        checkpointer_2_ctx,
    )

    resumed = graph_2.invoke(Command(resume={"decision": "approved"}), config=config)
    checkpointer_2.__exit__(None, None, None)

    assert "processed a refund" in resumed["final_response"].lower()

    with session_factory() as session:
        from supportmind.orders.models import Approval, Refund

        approval = session.get(Approval, approval_id)
        refund = session.get(Refund, approval.refund_id)
        db_order = session.get(Order, order_id)

        assert refund.status == "completed"
        assert db_order.status == "refunded"
