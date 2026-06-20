"""Gap-fill: bring the dataset forward from the latest stored date to today.

One orchestrated command that runs, in order (each stage guarded + ``--dry-run`` aware):

  1. SCRAPE forward — from ``MAX(raw_headlines.date) + 1`` to ``--to`` (default: today),
     window by window, inserting with ``ON CONFLICT DO NOTHING``.
  2. SCORE          — ``process_headlines.py --fast`` over the new range (no cutoff;
     post-cutoff dates ARE scored here — the hard cutoff applies to *modeling*, not
     ingestion).
  3. EMBED          — ``embed_missing(scope='postcutoff')`` for the new headlines.
  4. FINANCE        — S&P/VIX/Brent (yfinance) + USD-ILS (Frankfurter) auto-fetch at
     feature-build time, so nothing is stored here. TA-125 / VTA-35 come from MANUAL
     CSV exports (investing.com) — this stage only REPORTS how stale those CSVs are.

KNOWN LIMITATION (2026-06): the mivzakim scraper's headline XPath currently returns
empty for some pages, so the forward SCRAPE may insert 0 rows ("silent zero") until the
parser is fixed. This orchestrator is wired and ready — it fills the gap as soon as the
scraper returns rows. A loud warning fires if a window scrapes pages but inserts nothing.

Run (server-side, from repo root):
    uv run python scripts/gap_fill.py --dry-run
    uv run python scripts/gap_fill.py --window 7 --headlines-per-call 4 --concurrency 50
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from loguru import logger

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from daily_scrape_to_db import (  # noqa: E402
    DEFAULT_DB_URL,
    get_connection,
    insert_rows,
    scrape_dates,
)

_IL_TZ = ZoneInfo("Asia/Jerusalem")
DATE_FMT = "%Y-%m-%d"
REPO_ROOT = _HERE.parent
PROCESSING_ENGINE = REPO_ROOT / "processing_engine"
PROCESS_SCRIPT = REPO_ROOT / "scripts" / "process_headlines.py"


def get_latest_date(db_url: str) -> datetime | None:
    """Return the newest ``date`` in raw_headlines (naive midnight), or None if empty."""
    conn = get_connection(db_url)
    try:
        cur = conn.cursor()
        cur.execute("SELECT MAX(date) FROM raw_headlines")
        row = cur.fetchone()
        if not row or row[0] is None:
            return None
        d = row[0]
        return datetime(d.year, d.month, d.day)
    finally:
        conn.close()


def scrape_forward(db_url: str, start: datetime, end: datetime, *, window: int,
                   pages: int, batch_size: int, dry_run: bool) -> tuple[int, int]:
    """Scrape ``[start, end]`` forward in ``window``-day batches; insert each batch.

    Returns ``(total_scraped, total_inserted)``. Emits a loud warning for any window
    that scrapes pages but inserts 0 rows (the scraper-parser silent-zero symptom).
    """
    total_scraped = total_inserted = 0
    cursor = start
    while cursor <= end:
        w_end = min(cursor + timedelta(days=window - 1), end)
        days = (w_end - cursor).days + 1
        logger.info("─" * 60)
        logger.info("Scrape window [{} → {}] ({} day(s))",
                    cursor.strftime(DATE_FMT), w_end.strftime(DATE_FMT), days)
        rows = scrape_dates(end_date=w_end, days=days, pages=pages, batch_size=batch_size)
        total_scraped += len(rows)
        if not rows:
            logger.warning("  0 headlines scraped for this window.")
        elif dry_run:
            logger.info("  [DRY RUN] would insert up to {:,} rows", len(rows))
        else:
            inserted, skipped = insert_rows(db_url, rows)
            total_inserted += inserted
            logger.info("  scraped {:,} | inserted {:,} | skipped {:,}", len(rows), inserted, skipped)
            if inserted == 0:
                logger.warning("  ⚠ scraped {:,} rows but inserted 0 — likely the scraper "
                               "headline-parser bug (silent zero) or all duplicates.", len(rows))
        cursor = w_end + timedelta(days=1)
    return total_scraped, total_inserted


def score_new(date_from: datetime, *, headlines_per_call: int, concurrency: int,
              dry_run: bool) -> int:
    """Score the new range via process_headlines.py --fast (no cutoff). Returns rc."""
    cmd = [
        "uv", "run", "--project", str(PROCESSING_ENGINE),
        "python", str(PROCESS_SCRIPT), "--fast",
        "--date-from", date_from.strftime(DATE_FMT),
        "--unscored-any-model",
        "--concurrency", str(concurrency),
        "--headlines-per-call", str(headlines_per_call),
    ]
    if dry_run:
        cmd += ["--dry-run"]
    logger.info("Scoring → {}", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode


def embed_new(*, dry_run: bool) -> int:
    """Embed post-cutoff headlines (scope='postcutoff'). Returns count written."""
    from sentisense.embed import embed_missing
    return embed_missing(dry_run=dry_run, scope="postcutoff")


def finance_status(db_url: str) -> None:
    """Report finance freshness — market/FX are live at build; TA-125/VTA-35 are manual."""
    logger.info("Finance: S&P/VIX/Brent (yfinance) + USD-ILS (Frankfurter) are fetched "
                "LIVE at feature-build time — nothing to store here.")
    logger.warning("TA-125 / VTA-35 come from MANUAL investing.com CSV exports. Refresh "
                   "'TA 125 Historical Data.csv' + the VTA35 CSV to cover the new dates, "
                   "else those days drop out of the modeling frames at build time.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill the gap from the latest stored date to today: scrape → score "
        "→ embed → finance status.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db-url", default=DEFAULT_DB_URL)
    parser.add_argument("--to", type=str, default="", help="End date YYYY-MM-DD (default: today, IL).")
    parser.add_argument("--window", type=int, default=7, help="Days per scrape window (default 7).")
    parser.add_argument("--pages", type=int, default=100, help="Max pages per date (default 100).")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Dates scraped concurrently per window (default 5).")
    parser.add_argument("--headlines-per-call", type=int, default=4,
                        help="Headlines per LLM call when scoring (default 4 — context-safe).")
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true", help="No writes; scrape+count only.")
    parser.add_argument("--skip-scrape", action="store_true")
    parser.add_argument("--skip-score", action="store_true")
    parser.add_argument("--skip-embed", action="store_true")
    args = parser.parse_args()
    if args.window < 1 or args.batch_size < 1:
        parser.error("--window and --batch-size must be >= 1")

    today = datetime.now(_IL_TZ).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    end = datetime.strptime(args.to, DATE_FMT) if args.to else today

    latest = get_latest_date(args.db_url)
    if latest is None:
        logger.error("raw_headlines is empty — use backfill_history.py to bootstrap first.")
        sys.exit(1)
    start = latest + timedelta(days=1)
    logger.info("Latest stored date: {}. Gap to fill: [{} → {}]",
                latest.strftime(DATE_FMT), start.strftime(DATE_FMT), end.strftime(DATE_FMT))
    if start > end:
        logger.info("No gap — latest stored date is already at/after the target. Done.")
        return

    if not args.skip_scrape:
        scraped, inserted = scrape_forward(args.db_url, start, end, window=args.window,
                                           pages=args.pages, batch_size=args.batch_size,
                                           dry_run=args.dry_run)
        logger.info("Scrape done: scraped {:,} | inserted {:,}", scraped, inserted)

    if not args.skip_score:
        rc = score_new(start, headlines_per_call=args.headlines_per_call,
                       concurrency=args.concurrency, dry_run=args.dry_run)
        if rc != 0:
            logger.warning("Scoring exited {} — continuing to embed/finance.", rc)

    if not args.skip_embed:
        written = embed_new(dry_run=args.dry_run)
        logger.info("Embed done: {:,} new post-cutoff vectors.", written)

    finance_status(args.db_url)
    logger.info("Gap-fill complete: [{} → {}].", start.strftime(DATE_FMT), end.strftime(DATE_FMT))


if __name__ == "__main__":
    main()
