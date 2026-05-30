from __future__ import annotations

import logging
import random

from sqlalchemy.orm import sessionmaker

from supportmind.config import SupportMindConfig
from supportmind.orders.models import ORDER_STATUSES, Customer, Order

logger = logging.getLogger(__name__)

FIRST_NAMES = [
    "Ava", "Liam", "Noah", "Emma", "Olivia", "Mia", "Lucas", "Sofia", "Ethan", "Zoe",
    "Aarav", "Ishaan", "Meera", "Diya", "Kabir", "Ananya", "Rohan", "Priya", "Vihaan", "Nisha",
    "Mason", "Layla", "Owen", "Chloe", "Leo",
]
LAST_NAMES = [
    "Sharma", "Patel", "Nair", "Iyer", "Gupta", "Singh", "Reddy", "Khan", "Mehta", "Rao",
    "Smith", "Johnson", "Brown", "Garcia", "Miller", "Davis", "Clark", "Lewis", "Walker", "Young",
]
PRODUCTS = [
    "Wireless Earbuds", "Running Shoes", "Coffee Maker", "Yoga Mat", "Backpack",
    "Desk Lamp", "Bluetooth Speaker", "Water Bottle", "Notebook Set", "Phone Case",
    "Office Chair", "Standing Desk", "Board Game", "Skin Care Kit", "Sunglasses",
]

STATUS_WEIGHTS = {
    "placed": 0.20,
    "processing": 0.15,
    "shipped": 0.20,
    "delivered": 0.30,
    "cancelled": 0.10,
    "refunded": 0.05,
}


def seed_database(session_factory: sessionmaker, config: SupportMindConfig) -> tuple[int, int]:
    seed_cfg = config.orders_seed
    rng = random.Random(seed_cfg.random_seed)

    with session_factory() as session:
        existing_customers = session.query(Customer).count()
        if existing_customers > 0:
            logger.info("Orders DB already seeded (%s customers found); skipping", existing_customers)
            return existing_customers, session.query(Order).count()

        customers: list[Customer] = []
        used_emails: set[str] = set()
        for i in range(seed_cfg.customer_count):
            first = rng.choice(FIRST_NAMES)
            last = rng.choice(LAST_NAMES)
            email = f"{first.lower()}.{last.lower()}{i}@example.com"
            while email in used_emails:
                email = f"{first.lower()}.{last.lower()}{i}.{rng.randint(1, 999)}@example.com"
            used_emails.add(email)
            customers.append(Customer(name=f"{first} {last}", email=email))
        session.add_all(customers)
        session.flush()

        statuses = list(STATUS_WEIGHTS.keys())
        weights = list(STATUS_WEIGHTS.values())

        orders: list[Order] = []
        for _ in range(seed_cfg.order_count):
            customer = rng.choice(customers)
            status = rng.choices(statuses, weights=weights, k=1)[0]
            amount = round(rng.uniform(15.0, 450.0), 2)
            orders.append(
                Order(
                    customer_id=customer.id,
                    product_name=rng.choice(PRODUCTS),
                    amount=amount,
                    status=status,
                )
            )
        session.add_all(orders)
        session.commit()

        logger.info("Seeded %s customers and %s orders", len(customers), len(orders))
        return len(customers), len(orders)


assert set(STATUS_WEIGHTS) == set(ORDER_STATUSES)
