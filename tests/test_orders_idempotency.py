from __future__ import annotations

from supportmind.agent.tools import refund_idempotency_key, refund_initiate


def test_get_or_create_refund_is_idempotent(orders_client, seeded_order):
    order_id = seeded_order["order_id"]
    key = "fixed-idempotency-key"

    first = orders_client.get_or_create_refund(order_id, 25.0, key, "damaged item")
    second = orders_client.get_or_create_refund(order_id, 25.0, key, "damaged item")

    assert first.id == second.id

    with orders_client.session_factory() as session:
        from supportmind.orders.models import Refund

        rows = session.query(Refund).filter_by(idempotency_key=key).all()
        assert len(rows) == 1


def test_double_refund_tool_call_does_not_double_refund(orders_client, seeded_order, agent_config):
    order_id = seeded_order["order_id"]
    thread_id = "thread-double-refund"
    cap = agent_config.agent.refund_approval_cap
    amount = min(seeded_order["amount"], cap)

    first_result = refund_initiate(orders_client, thread_id, order_id, amount, "duplicate submit", cap)
    second_result = refund_initiate(orders_client, thread_id, order_id, amount, "duplicate submit", cap)

    assert first_result.success and second_result.success
    assert first_result.data["refund_id"] == second_result.data["refund_id"]

    with orders_client.session_factory() as session:
        from supportmind.orders.models import Refund

        rows = session.query(Refund).filter_by(order_id=order_id).all()
        assert len(rows) == 1
        assert rows[0].status == "completed"


def test_refund_idempotency_key_is_deterministic():
    key_a = refund_idempotency_key("thread-1", 42, 19.99)
    key_b = refund_idempotency_key("thread-1", 42, 19.99)
    key_c = refund_idempotency_key("thread-1", 42, 20.00)

    assert key_a == key_b
    assert key_a != key_c
