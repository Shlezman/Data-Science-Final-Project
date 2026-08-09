"""Nightly narrative simulation — run after the daily pipeline, publish to the UI.

Generates the persona simulation for the day (each news outlet = one agent; graph from real
per-outlet stats, statements + report from the local Ollama LLM) and upserts it into the
``narrative_sim_graph`` / ``narrative_sim_report`` tables. The dashboard's Simulator tab
picks it up automatically — no service, no UI trigger.

Run (GPU box; cron after daily_live, or by hand):
    uv run python scripts/sim_daily.py                    # today (Asia/Jerusalem)
    uv run python scripts/sim_daily.py --date 2026-08-07  # a specific day
    uv run python scripts/sim_daily.py --backfill 5       # last N days that lack a sim
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from zoneinfo import ZoneInfo

from loguru import logger
from sqlalchemy import text

from sentisense.db import get_engine
from sentisense.sim.local_sim import simulate_day

_IL = ZoneInfo("Asia/Jerusalem")

_MISSING_DAYS = text(
    """
    SELECT DISTINCT rh.date FROM raw_headlines rh
    WHERE rh.date >= :since
      AND NOT EXISTS (SELECT 1 FROM narrative_sim_graph g
                      WHERE g.sim_date = rh.date AND g.mode = 'source')
    ORDER BY rh.date
    """
)


def main() -> int:
    """Simulate today (default), one --date, or --backfill N days without a sim."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=None, help="Day to simulate (YYYY-MM-DD); default today.")
    ap.add_argument("--backfill", type=int, default=0,
                    help="Also simulate the last N days that have headlines but no sim.")
    args = ap.parse_args()

    engine = get_engine()
    days = []
    if args.backfill:
        since = dt.datetime.now(_IL).date() - dt.timedelta(days=args.backfill)
        with engine.connect() as conn:
            days = [r[0] for r in conn.execute(_MISSING_DAYS, {"since": since}).all()]
    else:
        days = [args.date or str(dt.datetime.now(_IL).date())]

    failures = 0
    for day in days:
        try:
            logger.info(simulate_day(engine, str(day)))
        except Exception as exc:  # noqa: BLE001 — one bad day must not kill the batch
            failures += 1
            logger.warning("simulation for {} failed: {}", day, str(exc)[:200])
    return 1 if failures and failures == len(days) else 0


if __name__ == "__main__":
    sys.exit(main())
