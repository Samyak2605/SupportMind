from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from supportmind.orders.models import Base


def build_engine(db_path: Path) -> Engine:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine


def build_session_factory(db_path: Path) -> sessionmaker[Session]:
    engine = build_engine(db_path)
    return sessionmaker(bind=engine, expire_on_commit=False)
