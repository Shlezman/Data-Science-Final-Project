"""Settle past predictions — backfill ``model_predictions.actual`` once the outcome is known.

A prediction for decision-day T asks "will close(T+1) > close(T)?". The actual is unknowable
until T+1's close prints. This job loads realized TA-125 next-day directions (reusing
``postcutoff_directions`` — direction[T] = close(T+1) > close(T), the final unknown row dropped)
and fills ``actual`` for every still-NULL prediction whose outcome has since become known.

Idempotent: only touches rows where ``actual IS NULL`` and the outcome is now available.

Run (server-side, after TASE close, alongside the daily job):
    uv run --extra finance python scripts/settle_predictions.py
    uv run --extra finance python scripts/settle_predictions.py --dry-run
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
from loguru import logger
from sqlalchemy import text

from sentisense.db import get_engine

_PENDING = text("SELECT id, date, model_version FROM model_predictions WHERE actual IS NULL")
_SETTLE = text("UPDATE model_predictions SET actual = :actual WHERE id = :id")


def settle(engine=None, *, dry_run: bool = False) -> int:
    """Fill ``actual`` for predictions whose next-day outcome is now known. Returns count settled."""
    engine = engine or get_engine()
    from sentisense.features.dataset import postcutoff_directions

    directions = postcutoff_directions()                      # date-indexed: 1 = close(T+1) up
    by_date = {pd.Timestamp(d).date(): bool(v) for d, v in directions.items()}

    with engine.connect() as conn:
        pending = conn.execute(_PENDING).fetchall()
    settled = [{"id": r.id, "actual": by_date[r.date]} for r in pending if r.date in by_date]

    logger.info("{} predictions pending; {} now have a known outcome.", len(pending), len(settled))
    if dry_run or not settled:
        return len(settled)
    with engine.begin() as conn:
        conn.execute(_SETTLE, settled)
    logger.info("Settled {} predictions.", len(settled))
    return len(settled)


def main() -> int:
    """CLI entry: settle predictions; exit 0 always (no pending is not an error)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Report how many would settle; write nothing.")
    args = ap.parse_args()
    settle(dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
