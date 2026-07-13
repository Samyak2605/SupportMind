from __future__ import annotations

import logging
import random
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session, sessionmaker

from supportmind.orders.models import (
    CANCELLABLE_STATUSES,
    Approval,
    Customer,
    Escalation,
    Order,
    Refund,
)

logger = logging.getLogger(__name__)


class OrderNotFoundError(Exception):
    pass


class OrderNotCancellableError(Exception):
    pass


class SimulatedOmsError(Exception):
    """Raised when the injected failure rate fires, simulating a flaky order system."""


@dataclass
class OrderSnapshot:
    order_id: int
    customer_name: str
    customer_email: str
    product_name: str
    amount: float
    status: str
    created_at: datetime


class OrdersClient:
    """Thin client over the mock order system.

    Every call that would hit a real OMS (lookup, cancel, refund) goes through
    `_maybe_fail`, which injects latency and a configurable failure rate so the
    agent's retry/error-handling paths are exercised against something real
    rather than an always-succeeding mock.
    """

    def __init__(self, session_factory: sessionmaker[Session], failure_rate: float, approval_timeout_hours: int):
        self.session_factory = session_factory
        self.failure_rate = failure_rate
        self.approval_timeout_hours = approval_timeout_hours
        self._rng = random.Random()

    def _maybe_fail(self, operation: str) -> None:
        time.sleep(self._rng.uniform(0.01, 0.05))
        if self._rng.random() < self.failure_rate:
            logger.warning("Simulated OMS failure during %s", operation)
            raise SimulatedOmsError(f"Order system temporarily unavailable during {operation}")

    def lookup_order(self, order_id: int) -> OrderSnapshot:
        self._maybe_fail("order_lookup")
        with self.session_factory() as session:
            order = session.get(Order, order_id)
            if order is None:
                raise OrderNotFoundError(f"No order found with id {order_id}")
            customer = session.get(Customer, order.customer_id)
            return OrderSnapshot(
                order_id=order.id,
                customer_name=customer.name,
                customer_email=customer.email,
                product_name=order.product_name,
                amount=order.amount,
                status=order.status,
                created_at=order.created_at,
            )

    def cancel_order(self, order_id: int) -> OrderSnapshot:
        self._maybe_fail("cancel_order")
        with self.session_factory() as session:
            order = session.get(Order, order_id)
            if order is None:
                raise OrderNotFoundError(f"No order found with id {order_id}")
            if order.status not in CANCELLABLE_STATUSES:
                raise OrderNotCancellableError(
                    f"Order {order_id} has status '{order.status}' and can no longer be cancelled"
                )
            order.status = "cancelled"
            session.commit()
            customer = session.get(Customer, order.customer_id)
            return OrderSnapshot(
                order_id=order.id,
                customer_name=customer.name,
                customer_email=customer.email,
                product_name=order.product_name,
                amount=order.amount,
                status=order.status,
                created_at=order.created_at,
            )

    def get_or_create_refund(self, order_id: int, amount: float, idempotency_key: str, reason: str) -> Refund:
        """Idempotent by `idempotency_key`: a retried call with the same key
        returns the existing Refund row instead of creating a duplicate."""
        with self.session_factory() as session:
            existing = session.query(Refund).filter_by(idempotency_key=idempotency_key).one_or_none()
            if existing is not None:
                logger.info("Refund idempotency key %s already exists; returning existing refund", idempotency_key)
                return existing

            self._maybe_fail("refund_initiate")
            order = session.get(Order, order_id)
            if order is None:
                raise OrderNotFoundError(f"No order found with id {order_id}")

            refund = Refund(
                order_id=order_id,
                amount=amount,
                idempotency_key=idempotency_key,
                status="pending_approval",
                reason=reason,
            )
            session.add(refund)
            session.commit()
            session.refresh(refund)
            return refund

    def complete_refund(self, refund_id: int) -> Refund:
        with self.session_factory() as session:
            refund = session.get(Refund, refund_id)
            if refund is None:
                raise ValueError(f"No refund found with id {refund_id}")
            if refund.status == "completed":
                return refund
            refund.status = "completed"
            refund.completed_at = datetime.now(timezone.utc)
            order = session.get(Order, refund.order_id)
            order.status = "refunded"
            session.commit()
            session.refresh(refund)
            return refund

    def reject_refund(self, refund_id: int) -> Refund:
        with self.session_factory() as session:
            refund = session.get(Refund, refund_id)
            if refund is None:
                raise ValueError(f"No refund found with id {refund_id}")
            refund.status = "rejected"
            session.commit()
            session.refresh(refund)
            return refund

    def get_or_create_approval(self, thread_id: str, refund_id: int, order_id: int, amount: float) -> Approval:
        """Idempotent by `refund_id`: LangGraph replays a node's code from the
        top on every resume, so the code path leading up to `interrupt()` runs
        again each time. Without this guard a single refund would accumulate
        one duplicate Approval row per resume attempt."""
        with self.session_factory() as session:
            existing = session.query(Approval).filter_by(refund_id=refund_id).one_or_none()
            if existing is not None:
                return existing

            approval = Approval(
                id=str(uuid.uuid4()),
                thread_id=thread_id,
                refund_id=refund_id,
                order_id=order_id,
                amount=amount,
                status="pending",
                expires_at=datetime.now(timezone.utc) + timedelta(hours=self.approval_timeout_hours),
            )
            session.add(approval)
            session.commit()
            session.refresh(approval)
            return approval

    def get_approval(self, approval_id: str) -> Approval | None:
        with self.session_factory() as session:
            return session.get(Approval, approval_id)

    def list_pending_approvals(self) -> list[Approval]:
        self.expire_stale_approvals()
        with self.session_factory() as session:
            return session.query(Approval).filter_by(status="pending").all()

    def decide_approval(self, approval_id: str, approved: bool) -> Approval:
        with self.session_factory() as session:
            approval = session.get(Approval, approval_id)
            if approval is None:
                raise ValueError(f"No approval found with id {approval_id}")
            approval.status = "approved" if approved else "rejected"
            approval.decided_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(approval)
            return approval

    def expire_stale_approvals(self) -> list[Approval]:
        now = datetime.now(timezone.utc)
        expired: list[Approval] = []
        with self.session_factory() as session:
            stale = session.query(Approval).filter_by(status="pending").all()
            for approval in stale:
                expires_at = approval.expires_at
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=timezone.utc)
                if expires_at < now:
                    approval.status = "expired"
                    approval.decided_at = now
                    expired.append(approval)
            if expired:
                session.commit()
        return expired

    def create_escalation(self, thread_id: str, reason: str, order_id: int | None = None) -> Escalation:
        with self.session_factory() as session:
            escalation = Escalation(thread_id=thread_id, order_id=order_id, reason=reason)
            session.add(escalation)
            session.commit()
            session.refresh(escalation)
            return escalation
