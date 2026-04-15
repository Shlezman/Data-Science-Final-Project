"""
scripts.migrate_csv_to_db
==========================
Migrate the master data.csv into the PostgreSQL ``raw_headlines`` table.

This script:
1. Connects to the SentiSense PostgreSQL instance.
2. Reads data.csv (applying column renames if needed).
3. Inserts rows in batches using ``ON CONFLICT DO NOTHING`` to skip
   duplicates — safe to run repeatedly (idempotent).
4. Logs progress: total imported, skipped (duplicate), elapsed time.

Usage
-----
::

    # Default: read data.csv from project root, connect to local Postgres
    python scripts/migrate_csv_to_db.py

    # Custom paths / connection
    python scripts/migrate_csv_to_db.py --data-file /path/to/data.csv
    python scripts/migrate_csv_to_db.py --db-url postgresql://user:pass@host:5432/dbname

    # Dry run — count rows without inserting
    python scripts/migrate_csv_to_db.py --dry-run

Prerequisites
-------------
- PostgreSQL running (``docker compose up -d``)
- Schema initialized (auto-runs via ``init_db.sql`` on first start)
- ``pip install psycopg[binary]`` (or ``psycopg2-binary``)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

BATCH_SIZE = 5000

DEFAULT_DB_URL = os.environ.get(
    "SENTISENSE_DATABASE_URL",
    "postgresql://sentisense:sentisense_dev@localhost:5432/sentisense",
)

# ─────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{level:<8}</level> | {message}")
    logger.add(
        LOG_DIR / "migrate_csv_to_db_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
    )


# ─────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────


def get_connection(db_url: str) -> Any:
    """Create a psycopg connection (tries psycopg 3, falls back to psycopg2)."""
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
            "  Install with: pip install 'psycopg[binary]'  (recommended)\n"
            "            or: pip install psycopg2-binary"
        )
        sys.exit(1)


INSERT_SQL = """
    INSERT INTO raw_headlines (date, source, hour, popularity, headline)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (date, source, hour, headline) DO NOTHING
"""


def insert_batch(
    cursor: Any,
    rows: list[dict[str, str]],
) -> int:
    """
    Insert a batch of rows. Returns the number actually inserted
    (i.e. not skipped by ON CONFLICT).

    Uses ``cursor.rowcount`` after each execute rather than issuing
    separate ``COUNT(*)`` queries — avoids full-table scans.
    """
    params = [
        (
            row.get("date"),
            row.get("source"),
            row.get("hour"),
            row.get("popularity", row.get("importance_level", "")),
            row.get("headline"),
        )
        for row in rows
        if row.get("headline")  # skip rows with no headline
    ]

    inserted = 0
    for param in params:
        cursor.execute(INSERT_SQL, param)
        inserted += cursor.rowcount
    return inserted


# ─────────────────────────────────────────────────────────────────────
# CSV reader (reuse logic from update_data_csv)
# ─────────────────────────────────────────────────────────────────────


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read all rows from data.csv, applying column renames."""
    import csv

    if not path.exists():
        logger.error("Data file not found: {}", path)
        sys.exit(1)

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Apply legacy column renames in-place
    renames = {"importance_level": "popularity"}
    for row in rows:
        for old_name, new_name in renames.items():
            if old_name in row and new_name not in row:
                row[new_name] = row.pop(old_name)

    return rows


# ─────────────────────────────────────────────────────────────────────
# Main migration
# ─────────────────────────────────────────────────────────────────────


def migrate(data_file: Path, db_url: str, *, dry_run: bool = False) -> None:
    logger.info("Reading CSV: {}", data_file)
    rows = read_csv_rows(data_file)
    logger.info("Loaded {:,} rows from CSV", len(rows))

    if dry_run:
        dates = {r.get("date") for r in rows if r.get("date")}
        logger.info("[DRY RUN] {:,} rows, {:,} unique dates. No DB writes.", len(rows), len(dates))
        return

    logger.info("Connecting to database: {}", db_url.split("@")[-1])  # hide password
    conn = get_connection(db_url)

    try:
        cursor = conn.cursor()
        total_inserted = 0
        t_start = time.perf_counter()
        n_batches = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_idx in range(n_batches):
            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(rows))
            batch = rows[start:end]

            inserted = insert_batch(cursor, batch)
            total_inserted += inserted
            conn.commit()

            if (batch_idx + 1) % 20 == 0 or batch_idx == n_batches - 1:
                elapsed = time.perf_counter() - t_start
                logger.info(
                    "  Batch {}/{} — {:,} inserted so far ({:.0f}s)",
                    batch_idx + 1,
                    n_batches,
                    total_inserted,
                    elapsed,
                )

        elapsed = time.perf_counter() - t_start
        skipped = len(rows) - total_inserted

        logger.info("─" * 50)
        logger.info("MIGRATION COMPLETE")
        logger.info("  Total CSV rows:   {:,}", len(rows))
        logger.info("  Rows inserted:    {:,}", total_inserted)
        logger.info("  Rows skipped:     {:,} (duplicate)", skipped)
        logger.info("  Elapsed:          {:.1f}s", elapsed)
        logger.info("─" * 50)

    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate data.csv into the PostgreSQL raw_headlines table"
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data.csv",
        help="Path to data.csv (default: <project_root>/data.csv)",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=DEFAULT_DB_URL,
        help="PostgreSQL connection URL (default: from SENTISENSE_DATABASE_URL env var)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read CSV and report stats without inserting into DB.",
    )
    args = parser.parse_args()

    setup_logging()
    logger.info("SentiSense — CSV to PostgreSQL Migration")
    migrate(args.data_file, args.db_url, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
