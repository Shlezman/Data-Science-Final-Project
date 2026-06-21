"""Mode C — live daily MiroFish sim. Runs ONE causal sim for the most recent trading day
with headlines (seed ≤ that day), caching the feature + agent graph + report. Idempotent
(skips if already cached). Intended for a daily cron after the scrape/score stages.

Run (server-side; MiroFish service + Gemma-4 + Zep up):
    uv run python scripts/miro_daily.py                 # latest trading day
    uv run python scripts/miro_daily.py --date 2026-06-20 --seeds 2
"""

from __future__ import annotations

import argparse

import pandas as pd
from loguru import logger
from sqlalchemy import text

from sentisense.constants import TA125_CSV
from sentisense.db import get_engine
from sentisense.sim.config import SEED_LOOKBACK_DAYS
from sentisense.sim.runner import run_window


def _latest_trading_day_with_news(engine) -> pd.Timestamp | None:
    ta = pd.read_csv(TA125_CSV)
    ta["Date"] = pd.to_datetime(ta["Date"], errors="coerce")
    trading = pd.DatetimeIndex(ta.dropna(subset=["Date"])["Date"]).sort_values()
    with engine.connect() as conn:
        mx = conn.execute(text("SELECT MAX(date) FROM raw_headlines")).scalar()
    if mx is None:
        return None
    elig = trading[trading <= pd.Timestamp(mx)]
    return elig.max() if len(elig) else None


def main() -> None:
    p = argparse.ArgumentParser(description="Run today's (or --date's) MiroFish sim (mode C).")
    p.add_argument("--date", default="", help="Trading day YYYY-MM-DD (default: latest with news).")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--lookback", type=int, default=SEED_LOOKBACK_DAYS)
    p.add_argument("--miro-url", default="")
    args = p.parse_args()

    engine = get_engine()
    day = pd.Timestamp(args.date) if args.date else _latest_trading_day_with_news(engine)
    if day is None:
        logger.error("no trading day with news found — scrape first.")
        return
    logger.info("Mode C daily sim for {} ({} seed(s), lookback {}d)", day.date(), args.seeds, args.lookback)
    run_window([day], seeds=args.seeds, lookback=args.lookback, engine=engine,
               base_url=args.miro_url or None)


if __name__ == "__main__":
    main()
