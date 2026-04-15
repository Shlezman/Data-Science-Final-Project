"""
scripts.update_data_csv
========================
Update the master data.csv file by running the mivzakim_scraper for
new dates not yet present in the dataset.

This script:
1. Reads the existing data.csv and discovers the most recent date.
2. Runs the scraper for all dates from (most_recent + 1) through today.
3. Merges the newly scraped headlines into data.csv, deduplicating.
4. Renames legacy column names if needed (``importance_level`` → ``popularity``).
5. Logs a summary: how many new dates and new rows were added.

Usage
-----
::

    python scripts/update_data_csv.py
    python scripts/update_data_csv.py --data-file /path/to/data.csv
    python scripts/update_data_csv.py --dry-run   # show what would happen
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

CANONICAL_COLUMNS = ["date", "source", "hour", "popularity", "headline"]

# Legacy column name → canonical name
COLUMN_RENAMES = {
    "importance_level": "popularity",
}

DATE_FMT = "%Y-%m-%d"

# ─────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def setup_logging() -> None:
    """Configure loguru with stderr + rotating file sink."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{level:<8}</level> | {message}")
    logger.add(
        LOG_DIR / "update_data_csv_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
    )


# ─────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """
    Read a CSV file and return (columns, rows).

    Applies any column renames defined in ``COLUMN_RENAMES``.
    """
    if not path.exists():
        logger.warning("File not found: {} — starting with empty dataset", path)
        return CANONICAL_COLUMNS, []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_columns = list(reader.fieldnames or [])
        rows = list(reader)

    # Apply renames
    renamed = False
    columns = []
    for col in raw_columns:
        canonical = COLUMN_RENAMES.get(col, col)
        columns.append(canonical)
        if canonical != col:
            renamed = True
            logger.info("Column renamed: '{}' → '{}'", col, canonical)
            for row in rows:
                row[canonical] = row.pop(col)

    if renamed:
        logger.info("Legacy column names updated to canonical schema")

    return columns, rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write rows to CSV with canonical column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CANONICAL_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def deduplicate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Remove duplicate rows based on (date, source, hour, headline)."""
    seen: set[tuple[str, ...]] = set()
    unique: list[dict[str, str]] = []
    for row in rows:
        key = (
            row.get("date", ""),
            row.get("source", ""),
            row.get("hour", ""),
            row.get("headline", ""),
        )
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def get_latest_date(rows: list[dict[str, str]]) -> datetime | None:
    """Return the most recent date in the dataset, or None if empty."""
    dates: list[datetime] = []
    for row in rows:
        try:
            dates.append(datetime.strptime(row["date"], DATE_FMT))
        except (KeyError, ValueError):
            continue
    return max(dates) if dates else None


def get_unique_dates(rows: list[dict[str, str]]) -> set[str]:
    """Return the set of unique date strings in the dataset."""
    return {row.get("date", "") for row in rows if row.get("date")}


# ─────────────────────────────────────────────────────────────────────
# Scraper integration
# ─────────────────────────────────────────────────────────────────────


def scrape_new_dates(
    start_date: datetime,
    end_date: datetime,
    output_file: Path,
    pages: int = 100,
    batch_size: int = 5,
) -> list[dict[str, str]]:
    """
    Run the mivzakim_scraper for dates in [start_date, end_date].

    Invokes the scraper as a subprocess under the mivzakim_scraper venv
    (via ``uv run``) to avoid cross-venv import issues.

    Returns the newly scraped rows (with canonical column names).
    """
    import subprocess

    project_root = Path(__file__).resolve().parent.parent
    scraper_dir = project_root / "mivzakim_scraper"

    days = (end_date - start_date).days + 1
    if days <= 0:
        logger.info("No new dates to scrape")
        return []

    logger.info(
        "Scraping {} day(s): {} → {}",
        days,
        start_date.strftime(DATE_FMT),
        end_date.strftime(DATE_FMT),
    )

    # Build a small Python snippet that calls get_data with our arguments.
    # This runs inside the mivzakim_scraper venv where playwright/pandas exist.
    scraper_script = (
        "from scrape import get_data; "
        "from datetime import datetime; "
        f"get_data(start_date=datetime({end_date.year},{end_date.month},{end_date.day}), "
        f"days={days}, pages={pages}, batch_size={batch_size})"
    )

    # Clear VIRTUAL_ENV so uv doesn't get confused by the parent's venv
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}

    try:
        result = subprocess.run(
            ["uv", "run", "python", "-c", scraper_script],
            cwd=str(scraper_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max for large scrapes
        )
        if result.stdout:
            logger.info("Scraper stdout:\n{}", result.stdout.rstrip())
        if result.returncode != 0:
            logger.error("Scraper stderr:\n{}", result.stderr.rstrip())
            raise RuntimeError(f"Scraper exited with code {result.returncode}")
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Scraper timed out after 1 hour") from exc
    except FileNotFoundError as exc:
        raise RuntimeError(
            "uv not found — install from https://docs.astral.sh/uv/"
        ) from exc

    # Scraper writes to ../headlines.csv relative to mivzakim_scraper/
    scraper_output = project_root / "headlines.csv"
    if scraper_output.exists():
        _, new_rows = read_csv(scraper_output)
        scraper_output.unlink()  # clean up temp file
        return new_rows
    else:
        logger.warning("Scraper did not produce output file")
        return []


# ─────────────────────────────────────────────────────────────────────
# Main logic
# ─────────────────────────────────────────────────────────────────────


def update_dataset(data_file: Path, *, dry_run: bool = False) -> None:
    """
    Main update logic:
    1. Read existing data.csv (apply column renames).
    2. Determine date range to scrape.
    3. Scrape new headlines.
    4. Merge, deduplicate, and write back.
    """
    logger.info("Loading existing dataset: {}", data_file)
    columns, existing_rows = read_csv(data_file)

    existing_count = len(existing_rows)
    existing_dates = get_unique_dates(existing_rows)
    latest = get_latest_date(existing_rows)

    logger.info("Existing dataset: {:,} rows, {:,} unique dates", existing_count, len(existing_dates))
    if latest:
        logger.info("Latest date in dataset: {}", latest.strftime(DATE_FMT))

    # If columns were renamed, write back even without new data
    if columns != CANONICAL_COLUMNS and not dry_run:
        logger.info("Writing dataset with canonical column names")
        write_csv(data_file, existing_rows)

    # Determine scrape range
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if latest:
        start_scrape = latest + timedelta(days=1)
    else:
        # Empty dataset — start from 30 days ago
        start_scrape = today - timedelta(days=30)

    if start_scrape > today:
        logger.info("Dataset is up-to-date through {}. Nothing to scrape.", latest.strftime(DATE_FMT))
        return

    days_to_scrape = (today - start_scrape).days + 1
    logger.info(
        "Will scrape {} new day(s): {} → {}",
        days_to_scrape,
        start_scrape.strftime(DATE_FMT),
        today.strftime(DATE_FMT),
    )

    if dry_run:
        logger.info("[DRY RUN] Would scrape {} day(s). Exiting.", days_to_scrape)
        return

    # Scrape
    new_rows = scrape_new_dates(start_scrape, today, data_file)

    if not new_rows:
        logger.info("No new rows scraped")
        return

    # Merge
    merged = existing_rows + new_rows
    deduped = deduplicate(merged)
    new_row_count = len(deduped) - existing_count
    new_date_count = len(get_unique_dates(deduped)) - len(existing_dates)

    logger.info("─" * 50)
    logger.info("MERGE SUMMARY")
    logger.info("  New rows added:  {:,}", max(0, new_row_count))
    logger.info("  New dates added: {:,}", max(0, new_date_count))
    logger.info("  Total rows:      {:,}", len(deduped))
    logger.info("  Total dates:     {:,}", len(get_unique_dates(deduped)))
    logger.info("─" * 50)

    write_csv(data_file, deduped)
    logger.info("Dataset written to {}", data_file)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update data.csv with newly scraped headlines from mivzakim.net"
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data.csv",
        help="Path to the master data.csv (default: <project_root>/data.csv)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be scraped without actually running the scraper.",
    )
    args = parser.parse_args()

    setup_logging()
    logger.info("SentiSense — Data Update Script")
    update_dataset(args.data_file, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
