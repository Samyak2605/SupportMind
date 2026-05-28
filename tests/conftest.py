from __future__ import annotations

from typing import Any

import pytest

from supportmind.config import load_config
from supportmind.orders.client import OrdersClient
from supportmind.orders.db import build_session_factory


class FakeLLMClient:
    """Deterministic stand-in for LLMClient used by graph-level tests.

    Graph tests exist to prove agent *orchestration* (routing, tool
    execution, interrupts, checkpoints) works, not to grade LLM judgment —
    that is what the live scenario eval harness (scripts/run_agent_evals.py)
    is for. Queuing canned responses keeps these tests fast and free of
    network/API-key dependencies, matching the CI requirement that the
    default test suite runs without secrets.
    """

    def __init__(self, responses: list[str] | None = None):
        self._queue = list(responses or [])
        self.calls: list[tuple[str, str, bool]] = []

    def queue(self, response: str) -> None:
        self._queue.append(response)

    def complete(self, system_prompt: str, user_prompt: str, *, json_mode: bool = False, use_cache: bool = True) -> str:
        self.calls.append((system_prompt, user_prompt, json_mode))
        if not self._queue:
            raise AssertionError("FakeLLMClient ran out of queued responses")
        return self._queue.pop(0)


@pytest.fixture
def base_config():
    return load_config()


@pytest.fixture
def agent_config(base_config, tmp_path):
    orders_db = tmp_path / "orders.db"
    checkpoints_db = tmp_path / "checkpoints.db"
    new_paths = base_config.paths.model_copy(update={"orders_db": orders_db, "checkpoints_db": checkpoints_db})
    return base_config.model_copy(update={"paths": new_paths})


@pytest.fixture
def orders_client(agent_config):
    session_factory = build_session_factory(agent_config.paths.orders_db)
    return OrdersClient(
        session_factory=session_factory,
        failure_rate=0.0,
        approval_timeout_hours=agent_config.agent.approval_timeout_hours,
    )


def make_orders_client(agent_config, *, failure_rate: float = 0.0) -> OrdersClient:
    session_factory = build_session_factory(agent_config.paths.orders_db)
    return OrdersClient(
        session_factory=session_factory,
        failure_rate=failure_rate,
        approval_timeout_hours=agent_config.agent.approval_timeout_hours,
    )


@pytest.fixture
def seeded_order(orders_client, agent_config) -> dict[str, Any]:
    from supportmind.orders.models import Customer, Order

    session_factory = build_session_factory(agent_config.paths.orders_db)
    with session_factory() as session:
        customer = Customer(name="Test Customer", email="test.customer@example.com")
        session.add(customer)
        session.flush()
        order = Order(customer_id=customer.id, product_name="Test Widget", amount=50.0, status="placed")
        session.add(order)
        session.commit()
        session.refresh(order)
        return {"order_id": order.id, "customer_id": customer.id, "amount": order.amount}


@pytest.fixture(scope="session")
def retrieval_service():
    from supportmind.config import load_config
    from supportmind.retrieval.pipeline import RetrievalService

    return RetrievalService(load_config())
