from __future__ import annotations

from supportmind.agent.tools import cancel_order, escalate_to_human, order_lookup, refund_initiate
from tests.conftest import make_orders_client


def test_order_lookup_success(orders_client, seeded_order):
    result = order_lookup(orders_client, seeded_order["order_id"])
    assert result.success
    assert result.data["status"] == "placed"


def test_order_lookup_not_found_is_terminal(orders_client):
    result = order_lookup(orders_client, 999999)
    assert not result.success
    assert result.retryable is False


def test_order_lookup_transient_failure_is_retryable(agent_config, seeded_order):
    flaky_client = make_orders_client(agent_config, failure_rate=1.0)
    result = order_lookup(flaky_client, seeded_order["order_id"])
    assert not result.success
    assert result.retryable is True


def test_cancel_order_succeeds_when_eligible(orders_client, seeded_order):
    result = cancel_order(orders_client, seeded_order["order_id"])
    assert result.success
    assert result.data["status"] == "cancelled"


def test_cancel_order_rejects_ineligible_status(orders_client, seeded_order, agent_config):
    from supportmind.orders.models import Order

    with orders_client.session_factory() as session:
        order = session.get(Order, seeded_order["order_id"])
        order.status = "delivered"
        session.commit()

    result = cancel_order(orders_client, seeded_order["order_id"])
    assert not result.success
    assert result.retryable is False


def test_refund_under_cap_auto_completes(orders_client, seeded_order, agent_config):
    cap = agent_config.agent.refund_approval_cap
    amount = min(seeded_order["amount"], cap - 1)
    result = refund_initiate(orders_client, "thread-a", seeded_order["order_id"], amount, "not as described", cap)
    assert result.success
    assert result.data["status"] == "completed"
    assert result.data["auto_approved"] is True


def test_escalate_to_human_creates_record(orders_client):
    result = escalate_to_human(orders_client, "thread-b", "customer requested a human")
    assert result.success
    assert result.data["escalation_id"] is not None
