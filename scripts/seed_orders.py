from supportmind.config import load_config
from supportmind.logging_config import configure_logging
from supportmind.orders.db import build_session_factory
from supportmind.orders.seed import seed_database


def main() -> None:
    config = load_config()
    configure_logging(config.app.log_level)
    session_factory = build_session_factory(config.paths.orders_db)
    customers, orders = seed_database(session_factory, config)
    print(f"Orders DB ready: {customers} customers, {orders} orders at {config.paths.orders_db}")


if __name__ == "__main__":
    main()
