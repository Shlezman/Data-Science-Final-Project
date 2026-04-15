"""
scripts.daily_scrape_to_db
===========================
Daily cronjob script: scrape today's headlines and insert directly into
the PostgreSQL ``raw_headlines`` table.

Designed to run once per day (e.g. via cron or K8s CronJob).  Does NOT
touch data.csv — it writes straight to the DB.

Workflow
--------
1. Scrape headlines for today (and optionally yesterday to catch late-night posts).
2. Insert into ``raw_headlines`` with ``ON CONFLICT DO NOTHING``.
3. Log summary: new rows inserted, skipped duplicates, elapsed time.

Usage
-----
::

    # Scrape today + yesterday, insert to DB
    python scripts/daily_scrape_to_db.py

    # Scrape only today
    python scripts/daily_scrape_to_db.py --days 1

    # Scrape last 7 days (backfill)
    python scripts/daily_scrape_to_db.py --days 7

    # Dry run — scrape but don't insert
    python scripts/daily_scrape_to_db.py --dry-run

Cron Example
------------
::

    # Every day at 06:00 IST (03:00 UTC)
    0 3 * * * cd /opt/sentisense && /usr/bin/python3 scripts/daily_scrape_to_db.py >> logs/cron.log 2>&1

Prerequisites
-------------
- PostgreSQL running (``docker compose up -d``)
- Playwright installed (``uv run playwright install firefox`` in mivzakim_scraper/)
- ``pip install psycopg[binary]``
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

DEFAULT_DB_URL = os.environ.get(
    "SENTISENSE_DATABASE_URL",
    "postgresql://sentisense:sentisense_dev@localhost:5432/sentisense",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
SCRAPER_DIR = PROJECT_ROOT / "mivzakim_scraper"

# ─────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{level:<8}</level> | {message}")
    logger.add(
        LOG_DIR / "daily_scrape_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
    )


# ─────────────────────────────────────────────────────────────────────
# Scraper
# ─────────────────────────────────────────────────────────────────────


def scrape_dates(
    end_date: datetime,
    days: int,
    pages: int = 100,
    batch_size: int = 5,
) -> list[dict[str, str]]:
    """
    Run the mivzakim_scraper as a subprocess under its own venv.

    The scraper writes to ``../headlines.csv`` relative to its own directory.
    We read that file back, then delete it (we don't want to accumulate files).

    Raises
    ------
    RuntimeError
        If the scraper crashes — allows the cron scheduler to detect failure.
    """
    import csv
    import subprocess

    logger.info(
        "Running scraper for {} day(s) ending {}",
        days,
        end_date.strftime("%Y-%m-%d"),
    )

    # Run the scraper in its own venv via `uv run` to avoid cross-venv
    # import issues (playwright, pandas, etc. live in mivzakim_scraper's venv).
    scraper_script = (
        "from scrape import get_data; "
        "from datetime import datetime; "
        f"get_data(start_date=datetime({end_date.year},{end_date.month},{end_date.day}), "
        f"days={days}, pages={pages}, batch_size={batch_size})"
    )

    # Clear VIRTUAL_ENV so uv doesn't get confused by the parent's venv
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}

    try:
        # Stream stdout/stderr to terminal in real-time (no buffering)
        result = subprocess.run(
            ["uv", "run", "python", "-u", "-c", scraper_script],
            cwd=str(SCRAPER_DIR),
            env=env,
            timeout=3600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Scraper exited with code {result.returncode}")
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Scraper timed out after 1 hour") from exc
    except FileNotFoundError as exc:
        raise RuntimeError(
            "uv not found — install from https://docs.astral.sh/uv/"
        ) from exc

    # Read the scraper output
    output_file = SCRAPER_DIR.parent / "headlines.csv"
    if not output_file.exists():
        logger.warning("Scraper produced no output file")
        return []

    with open(output_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Apply column renames
    for row in rows:
        if "importance_level" in row and "popularity" not in row:
            row["popularity"] = row.pop("importance_level")

    # Clean up
    output_file.unlink()
    logger.info("Scraped {:,} rows", len(rows))
    return rows


# ─────────────────────────────────────────────────────────────────────
# DB insertion
# ─────────────────────────────────────────────────────────────────────


def get_connection(db_url: str) -> Any:
    """Create a psycopg connection."""
    try:
        import psycopg

        return psycopg.connect(db_url, autocommit=False)
    except ImportError:
        pass

    try:
        import psycopg2

        return psycopg2.connect(db_url)
    except ImportError:
        logger.error(
            "Neither psycopg nor psycopg2 is installed.\n"
            "  Install with: pip install 'psycopg[binary]'"
        )
        sys.exit(1)


INSERT_SQL = """
    INSERT INTO raw_headlines (date, source, hour, popularity, headline)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (date, source, hour, headline) DO NOTHING
"""


def insert_rows(db_url: str, rows: list[dict[str, str]]) -> tuple[int, int]:
    """
    Insert rows into raw_headlines.

    Returns (inserted_count, skipped_count).
    """
    conn = get_connection(db_url)
    try:
        cursor = conn.cursor()

        params = [
            (
                row.get("date"),
                row.get("source"),
                row.get("hour"),
                row.get("popularity", ""),
                row.get("headline"),
            )
            for row in rows
            if row.get("headline")
        ]

        inserted = 0
        for param in params:
            cursor.execute(INSERT_SQL, param)
            inserted += cursor.rowcount
        conn.commit()

        skipped = len(params) - inserted
        return inserted, skipped

    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def daily_update(
    days: int,
    db_url: str,
    pages: int,
    *,
    dry_run: bool = False,
) -> None:
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    t_start = time.perf_counter()

    logger.info("SentiSense Daily Scrape — {} day(s) ending {}", days, today.strftime("%Y-%m-%d"))

    # 1. Scrape
    rows = scrape_dates(end_date=today, days=days, pages=pages)
    if not rows:
        logger.info("No headlines scraped. Done.")
        return

    # 2. Insert into DB
    if dry_run:
        dates = {r.get("date") for r in rows if r.get("date")}
        logger.info(
            "[DRY RUN] Would insert {:,} rows across {:,} date(s). No DB writes.",
            len(rows),
            len(dates),
        )
        return

    inserted, skipped = insert_rows(db_url, rows)
    elapsed = time.perf_counter() - t_start

    logger.info("─" * 50)
    logger.info("DAILY SCRAPE COMPLETE")
    logger.info("  Scraped rows:  {:,}", len(rows))
    logger.info("  DB inserted:   {:,}", inserted)
    logger.info("  DB skipped:    {:,} (duplicate)", skipped)
    logger.info("  Elapsed:       {:.1f}s", elapsed)
    logger.info("─" * 50)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Daily cronjob: scrape today's headlines from mivzakim.net "
            "and insert into the PostgreSQL raw_headlines table."
        )
    )
    parser.add_argument(
        "--days",
        type=int,
        default=2,
        help="Number of days to scrape backwards from today (default: 2 — today + yesterday).",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=100,
        help="Max pages to scrape per date (default: 100).",
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

    setup_logging()
    daily_update(args.days, args.db_url, args.pages, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
