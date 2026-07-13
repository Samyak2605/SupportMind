from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Float, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

ORDER_STATUSES = ("placed", "processing", "shipped", "delivered", "cancelled", "refunded")
CANCELLABLE_STATUSES = ("placed", "processing")
REFUND_STATUSES = ("pending_approval", "completed", "rejected")
APPROVAL_STATUSES = ("pending", "approved", "rejected", "expired")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(200), unique=True)

    orders: Mapped[list["Order"]] = relationship(back_populates="customer")


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"))
    product_name: Mapped[str] = mapped_column(String(200))
    amount: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(20), default="placed")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    customer: Mapped[Customer] = relationship(back_populates="orders")
    refunds: Mapped[list["Refund"]] = relationship(back_populates="order")


class Refund(Base):
    __tablename__ = "refunds"
    __table_args__ = (UniqueConstraint("idempotency_key", name="uq_refund_idempotency_key"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"))
    amount: Mapped[float] = mapped_column(Float)
    idempotency_key: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(20), default="pending_approval")
    reason: Mapped[str] = mapped_column(String(500), default="")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(default=None)

    order: Mapped[Order] = relationship(back_populates="refunds")


class Approval(Base):
    __tablename__ = "approvals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(64))
    refund_id: Mapped[int] = mapped_column(ForeignKey("refunds.id"))
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"))
    amount: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    expires_at: Mapped[datetime] = mapped_column()
    decided_at: Mapped[datetime | None] = mapped_column(default=None)


class Escalation(Base):
    __tablename__ = "escalations"

    id: Mapped[int] = mapped_column(primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(64))
    order_id: Mapped[int | None] = mapped_column(ForeignKey("orders.id"), default=None)
    reason: Mapped[str] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
