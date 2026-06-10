"""Database connection — read ``SENTISENSE_DATABASE_URL`` → SQLAlchemy engine.

Security contract (org policy + task spec):
    * The connection string is read ONLY from the ``SENTISENSE_DATABASE_URL``
      environment variable.
    * If it is unset the module FAILS FAST with a clear message — it never falls
      back to an embedded default that would bake the dev password into source.
    * Nothing here is run by the implementer; the operator runs it server-side.

Driver note:
    Only psycopg v3 is installed (``psycopg[binary]``). A plain ``postgresql://``
    URL makes SQLAlchemy default to psycopg2 (absent) → engine creation fails. We
    normalise the scheme to ``postgresql+psycopg://`` so SQLAlchemy uses psycopg v3.
"""

from __future__ import annotations

import os

from sqlalchemy import Engine, create_engine

_ENV_VAR = "SENTISENSE_DATABASE_URL"


def get_connection_url() -> str:
    """Return the DB URL from the environment, normalised for psycopg v3.

    Returns:
        A SQLAlchemy-compatible URL using the ``postgresql+psycopg`` dialect.

    Raises:
        RuntimeError: If ``SENTISENSE_DATABASE_URL`` is unset or blank. We refuse
            to embed a default DSN (it would carry the dev password into source).
    """
    raw = os.environ.get(_ENV_VAR, "").strip()
    if not raw:
        raise RuntimeError(
            f"{_ENV_VAR} is not set. Export the connection string before running, "
            f"e.g. export {_ENV_VAR}='postgresql+psycopg://USER:PASS@HOST:5432/sentisense'. "
            "This module never embeds a default DSN (no hardcoded credentials)."
        )
    return _normalise_driver(raw)


def _normalise_driver(url: str) -> str:
    """Force the psycopg v3 dialect so SQLAlchemy does not reach for psycopg2.

    Args:
        url: A PostgreSQL connection URL (any common scheme spelling).

    Returns:
        The same URL with a ``postgresql+psycopg://`` scheme.
    """
    if url.startswith("postgresql+psycopg://") or url.startswith("postgres+psycopg://"):
        return url.replace("postgres+psycopg://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql+psycopg2://"):
        # Explicit psycopg2 requested but only v3 is installed — redirect to v3.
        return url.replace("postgresql+psycopg2://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    # Unknown scheme — return as-is and let SQLAlchemy raise a clear error.
    return url


def get_engine(*, echo: bool = False, pool_pre_ping: bool = True) -> Engine:
    """Create a SQLAlchemy engine bound to the SentiSense database.

    Args:
        echo: If True, log every SQL statement (debugging only).
        pool_pre_ping: If True, validate pooled connections before use — guards
            against stale connections after the DB container restarts.

    Returns:
        A configured :class:`sqlalchemy.Engine`. Connections are opened lazily.

    Raises:
        RuntimeError: If the connection env var is unset (via ``get_connection_url``).
    """
    return create_engine(
        get_connection_url(),
        echo=echo,
        pool_pre_ping=pool_pre_ping,
        future=True,
    )
