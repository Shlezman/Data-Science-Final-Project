"""
scripts.backfill_history
=========================
Backfill the ``raw_headlines`` table *backwards* from the current oldest
stored date until the scraper reports no more history available.

Complements :mod:`scripts.daily_scrape_to_db` (which moves forward in
time, today + yesterday) by pushing the dataset further into the past.

Workflow
--------
1. Query ``raw_headlines`` for the oldest stored date.
2. Scrape a window of ``--window`` days ending at ``(oldest - 1)``.
3. Insert with ``ON CONFLICT DO NOTHING``.
4. Advance the cursor backwards by ``--window`` days and repeat.
5. Stop after ``--empty-streak`` consecutive windows produce zero new
   inserts (site history exhausted), or after ``--max-days`` total days.

Stop conditions
---------------
An "empty" window is one where either

* the scraper returned zero rows (most common boundary signal), or
* every row returned was already in the database (all duplicates).

Both increment the streak counter; hitting either stop condition is
what we want — no progress means no reason to keep scraping.

Usage
-----
::

    # Default: 7-day windows, stop after 2 consecutive empty windows
    python scripts/backfill_history.py

    # Bigger windows, stop on first empty window
    python scripts/backfill_history.py --window 30 --empty-streak 1

    # Safety cap: scan at most 365 days of history
    python scripts/backfill_history.py --max-days 365

    # Skip the DB query and start from a specific cutoff
    python scripts/backfill_history.py --start-before 2023-01-01

    # Dry run — scrape but don't insert
    python scripts/backfill_history.py --dry-run

Prerequisites
-------------
Same as ``daily_scrape_to_db.py`` — PostgreSQL, Playwright, psycopg.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from loguru import logger

# ─────────────────────────────────────────────────────────────────────
# Reuse helpers from the daily cronjob so fixes (timezone, ON CONFLICT,
# temp-file cleanup) propagate automatically.
# ─────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from daily_scrape_to_db import (  # noqa: E402
    DEFAULT_DB_URL,
    get_connection,
    insert_rows,
    scrape_dates,
)

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

_IL_TZ = ZoneInfo("Asia/Jerusalem")
DATE_FMT = "%Y-%m-%d"
PROJECT_ROOT = _HERE.parent
LOG_DIR = PROJECT_ROOT / "logs"


# ─────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<level>{level:<8}</level> | {message}",
    )
    logger.add(
        LOG_DIR / "backfill_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
    )


# ─────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────


def get_oldest_date(db_url: str) -> datetime | None:
    """
    Return the oldest ``date`` in ``raw_headlines`` as a naive ``datetime``
    at midnight, or ``None`` if the table is empty.
    """
    conn = get_connection(db_url)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT MIN(date) FROM raw_headlines")
        row = cursor.fetchone()
        if not row or row[0] is None:
            return None
        # psycopg returns a datetime.date for DATE columns
        oldest_date = row[0]
        return datetime(oldest_date.year, oldest_date.month, oldest_date.day)
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────
# Backfill loop
# ─────────────────────────────────────────────────────────────────────


def backfill(
    db_url: str,
    window: int,
    empty_streak_limit: int,
    pages: int,
    max_days: int | None,
    start_before: datetime | None,
    *,
    dry_run: bool = False,
) -> None:
    """
    Scrape history backwards from the oldest stored date until the scraper
    stops returning new data.

    Parameters
    ----------
    window
        Days per iteration (each iteration spawns one scraper subprocess).
    empty_streak_limit
        Stop after this many *consecutive* windows with zero inserts.
    pages
        ``--pages`` arg forwarded to the scraper (max pages per date).
    max_days
        Hard cap on total days scanned (``None`` = unlimited).
    start_before
        Override the DB lookup — scraping starts at ``start_before - 1``.
        Useful for resuming a crashed backfill or scanning beyond what
        the DB currently knows about.
    dry_run
        If true, scrape and count but do not insert.
    """
    t_start = time.perf_counter()

    # ── Resolve the initial cursor (end date of the first batch) ────────
    if start_before is not None:
        cursor_date = start_before - timedelta(days=1)
        logger.info(
            "Using --start-before override. First batch ends at {}",
            cursor_date.strftime(DATE_FMT),
        )
    else:
        oldest = get_oldest_date(db_url)
        if oldest is None:
            today = datetime.now(_IL_TZ).replace(
                hour=0, minute=0, second=0, microsecond=0, tzinfo=None
            )
            cursor_date = today - timedelta(days=1)
            logger.info(
                "raw_headlines is empty — bootstrapping from yesterday: {}",
                cursor_date.strftime(DATE_FMT),
            )
        else:
            cursor_date = oldest - timedelta(days=1)
            logger.info(
                "Oldest row in raw_headlines: {}. First batch ends at {}",
                oldest.strftime(DATE_FMT),
                cursor_date.strftime(DATE_FMT),
            )

    # ── Iterate batches backward ─────────────────────────────────────────
    total_scraped = 0
    total_inserted = 0
    total_skipped = 0
    total_days = 0
    empty_streak = 0
    iteration = 0

    while True:
        iteration += 1

        # Cap the batch window by the remaining --max-days budget
        if max_days is not None:
            remaining = max_days - total_days
            if remaining <= 0:
                logger.info("Reached --max-days cap of {}. Stopping.", max_days)
                break
            batch_window = min(window, remaining)
        else:
            batch_window = window

        batch_start = cursor_date - timedelta(days=batch_window - 1)
        logger.info("─" * 60)
        logger.info(
            "Batch {}: scraping {} day(s) [{} → {}]",
            iteration,
            batch_window,
            batch_start.strftime(DATE_FMT),
            cursor_date.strftime(DATE_FMT),
        )

        # ── Scrape via subprocess (reused helper) ───────────────────────
        rows = scrape_dates(end_date=cursor_date, days=batch_window, pages=pages)
        total_scraped += len(rows)
        total_days += batch_window

        if not rows:
            empty_streak += 1
            logger.warning(
                "Batch {} returned 0 headlines (empty-streak {}/{})",
                iteration,
                empty_streak,
                empty_streak_limit,
            )
        else:
            if dry_run:
                logger.info(
                    "[DRY RUN] Would insert up to {:,} rows from batch {}",
                    len(rows),
                    iteration,
                )
                # In dry-run mode we cannot know the true insert count, so
                # treat every row as progress to avoid false empty-streak.
                inserted = len(rows)
                skipped = 0
            else:
                inserted, skipped = insert_rows(db_url, rows)

            total_inserted += inserted
            total_skipped += skipped
            logger.info(
                "Batch {}: scraped {:,} | inserted {:,} | skipped {:,}",
                iteration,
                len(rows),
                inserted,
                skipped,
            )

            if inserted > 0:
                empty_streak = 0
            else:
                empty_streak += 1
                logger.warning(
                    "Batch {} yielded only duplicates (empty-streak {}/{})",
                    iteration,
                    empty_streak,
                    empty_streak_limit,
                )

        # ── Stop condition ───────────────────────────────────────────────
        if empty_streak >= empty_streak_limit:
            logger.info(
                "Empty-streak limit ({}) reached — site history appears exhausted.",
                empty_streak_limit,
            )
            break

        # ── Step the cursor backward by the full batch window ────────────
        # Current batch covered [cursor - window + 1, cursor];
        # next batch must end at cursor - window.
        cursor_date = cursor_date - timedelta(days=batch_window)

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    logger.info("=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info("  Iterations:     {:,}", iteration)
    logger.info("  Days scanned:   {:,}", total_days)
    logger.info("  Rows scraped:   {:,}", total_scraped)
    logger.info("  Rows inserted:  {:,}", total_inserted)
    logger.info("  Rows skipped:   {:,} (duplicate)", total_skipped)
    logger.info("  Final cursor:   {}", cursor_date.strftime(DATE_FMT))
    logger.info("  Elapsed:        {:.1f}s", elapsed)
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────


def _parse_date(value: str) -> datetime:
    try:
        return datetime.strptime(value, DATE_FMT)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date {value!r}. Expected YYYY-MM-DD."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill raw_headlines backwards from the current oldest stored "
            "date until the scraper returns no further data "
            "(--empty-streak consecutive empty windows) or --max-days is hit."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Default: 7-day windows, stop after 2 empty windows\n"
            "  python scripts/backfill_history.py\n\n"
            "  # Larger windows, stop on first empty\n"
            "  python scripts/backfill_history.py --window 30 --empty-streak 1\n\n"
            "  # Safety cap: scan at most 365 days\n"
            "  python scripts/backfill_history.py --max-days 365\n\n"
            "  # Resume from a known cutoff\n"
            "  python scripts/backfill_history.py --start-before 2023-01-01\n\n"
            "  # Dry run\n"
            "  python scripts/backfill_history.py --dry-run\n"
        ),
    )
    parser.add_argument(
        "--window",
        type=int,
        default=7,
        help="Days to scrape per iteration (default: 7).",
    )
    parser.add_argument(
        "--empty-streak",
        type=int,
        default=2,
        dest="empty_streak",
        help=(
            "Stop after this many consecutive iterations that produce zero "
            "new inserts (default: 2)."
        ),
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=100,
        help="Max pages to scrape per date (default: 100).",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="Safety cap on total days scanned (default: unlimited).",
    )
    parser.add_argument(
        "--start-before",
        type=_parse_date,
        default=None,
        help=(
            "Override DB lookup — scrape backwards starting from "
            "(this date - 1). Format: YYYY-MM-DD."
        ),
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=DEFAULT_DB_URL,
        help="PostgreSQL connection URL.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scrape but do not insert into DB.",
    )
    args = parser.parse_args()

    # ── Validate bounds ──────────────────────────────────────────────────
    if args.window < 1:
        parser.error("--window must be >= 1")
    if args.empty_streak < 1:
        parser.error("--empty-streak must be >= 1")
    if args.max_days is not None and args.max_days < 1:
        parser.error("--max-days must be >= 1 when provided")
    if args.pages < 1:
        parser.error("--pages must be >= 1")

    today_il = datetime.now(_IL_TZ).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=None
    )
    if args.start_before is not None and args.start_before > today_il:
        parser.error(
            f"--start-before ({args.start_before.strftime(DATE_FMT)}) "
            f"is in the future (today IL: {today_il.strftime(DATE_FMT)})"
        )

    setup_logging()
    backfill(
        db_url=args.db_url,
        window=args.window,
        empty_streak_limit=args.empty_streak,
        pages=args.pages,
        max_days=args.max_days,
        start_before=args.start_before,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
