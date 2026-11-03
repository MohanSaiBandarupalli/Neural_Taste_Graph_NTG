"""Alembic environment: schema = ntg.api.db.Base.metadata, URL = NTG_DATABASE_URL."""
from __future__ import annotations

import os

from alembic import context
from sqlalchemy import create_engine

from ntg.api.db import Base

target_metadata = Base.metadata
URL = os.environ.get("NTG_DATABASE_URL", "sqlite:///data/v4/ntg_app.db")


def run_offline() -> None:
    context.configure(url=URL, target_metadata=target_metadata, literal_binds=True, render_as_batch=URL.startswith("sqlite"))
    with context.begin_transaction():
        context.run_migrations()


def run_online() -> None:
    engine = create_engine(URL)
    with engine.connect() as conn:
        context.configure(connection=conn, target_metadata=target_metadata, render_as_batch=URL.startswith("sqlite"))
        with context.begin_transaction():
            context.run_migrations()


run_offline() if context.is_offline_mode() else run_online()
