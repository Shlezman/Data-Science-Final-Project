"""Run MiroFish causal day-sims over a window of TA-125 trading days (mode A / C).

Each day's sim seeds ONLY on headlines ≤ that day (leak-safe), caches the numeric
feature + agent graph + report. Idempotent + resumable. Needs: the MiroFish service up
(Gemma-4 + self-hosted Zep), the DB, and the 'miro' extra.

Run (server-side, from repo root):
    # last 90 trading days, 1 seed each:
    uv run python scripts/run_miro_window.py --last-n 90
    # explicit range, 3 seeds (variance), 7-day seed lookback:
    uv run python scripts/run_miro_window.py --from 2024-01-01 --to 2024-03-31 --seeds 3 --lookback 7
"""

from __future__ import annotations

import argparse

import pandas as pd
from loguru import logger

from sentisense.constants import TA125_CSV
from sentisense.sim.config import SEED_LOOKBACK_DAYS
from sentisense.sim.runner import run_window


def _trading_days() -> pd.DatetimeIndex:
    ta = pd.read_csv(TA125_CSV)
    ta["Date"] = pd.to_datetime(ta["Date"], errors="coerce")
    return pd.DatetimeIndex(ta.dropna(subset=["Date"])["Date"]).sort_values()


def main() -> None:
    p = argparse.ArgumentParser(description="Run MiroFish causal day-sims over a TA-125 trading-day window.")
    p.add_argument("--from", dest="dfrom", default="", help="Start date YYYY-MM-DD.")
    p.add_argument("--to", dest="dto", default="", help="End date YYYY-MM-DD.")
    p.add_argument("--last-n", type=int, default=0, help="Use the last N trading days (overrides --from).")
    p.add_argument("--seeds", type=int, default=1, help="Multi-seed runs per day (variance → mean±std).")
    p.add_argument("--lookback", type=int, default=SEED_LOOKBACK_DAYS, help="Seed lookback in days.")
    p.add_argument("--miro-url", default="", help="Override MiroFish base URL.")
    args = p.parse_args()

    days = _trading_days()
    if args.dto:
        days = days[days <= pd.Timestamp(args.dto)]
    if args.last_n > 0:
        days = days[-args.last_n:]
    elif args.dfrom:
        days = days[days >= pd.Timestamp(args.dfrom)]
    if len(days) == 0:
        logger.error("no trading days selected — check --from/--to/--last-n.")
        return

    logger.info("MiroFish window: {} trading days [{} … {}], {} seed(s), lookback {}d",
                len(days), days.min().date(), days.max().date(), args.seeds, args.lookback)
    run_window(list(days), seeds=args.seeds, lookback=args.lookback,
               base_url=args.miro_url or None)


if __name__ == "__main__":
    main()
