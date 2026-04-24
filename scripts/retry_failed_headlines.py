"""
scripts.retry_failed_headlines
===============================
Re-process headlines that **failed** on a previous ``process_headlines.py``
run.

A "failed" headline is one with an entry in ``nlp_vectors`` where
``validation_passed = FALSE`` for the active model.  Note that
``process_headlines.py`` inserts a row for *every* headline, including
failures — it marks them ``validation_passed=FALSE`` and records the
error in ``errors``.  As a result, the existing "unprocessed" query
(``nv.id IS NULL``) will *not* return them, and the INSERT
``ON CONFLICT DO NOTHING`` in the happy path is a no-op on a retry.

This script fixes both: it queries for the failed rows, deletes them,
then re-runs the affected raw_headlines through the exact same pipeline
as ``process_headlines.py`` (by importing its runners, so any future
tuning propagates automatically).

Workflow
--------
1. Query ``nlp_vectors`` for rows with ``validation_passed = FALSE``
   (for ``--model-name`` or the auto-detected active model).
2. ``DELETE`` those rows.
3. Re-run the associated raw_headlines through ``run_batch_fast_batched``
   (or ``run_batch_fast`` / ``run_batch_standard``).
4. New rows are inserted with correct scores.

Usage
-----
::

    # Retry all failed headlines using fast-batch mode (recommended)
    cd processing_engine && uv run python ../scripts/retry_failed_headlines.py \\
        --fast --headlines-per-call 50 --concurrency 32

    # Also retry headlines that were never processed at all
    cd processing_engine && uv run python ../scripts/retry_failed_headlines.py \\
        --fast --headlines-per-call 50 --concurrency 32 --include-missing

    # Only retry failures in a date window
    cd processing_engine && uv run python ../scripts/retry_failed_headlines.py \\
        --fast --headlines-per-call 20 --date-from 2024-01-01

    # Dry run — show the count without deleting/re-processing
    cd processing_engine && uv run python ../scripts/retry_failed_headlines.py --dry-run

Safety
------
- ``DELETE`` and ``INSERT`` happen in separate transactions.  If the retry
  aborts mid-way, the deleted rows are lost but the new inserts so far are
  preserved — running again re-scans for remaining failures.
- ``--dry-run`` never writes to the DB.
- ``--limit`` caps how many headlines are retried per invocation, useful
  for testing on a small slice before running over millions.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from loguru import logger

# ─────────────────────────────────────────────────────────────────────
# Reuse everything from process_headlines so runner behavior is identical
# and any future fixes propagate automatically.
# ─────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from process_headlines import (  # noqa: E402
    DEFAULT_DB_URL,
    get_active_model_name,
    get_connection,
    run_batch_fast,
    run_batch_fast_batched,
    run_batch_standard,
)

PROJECT_ROOT = _HERE.parent
LOG_DIR = PROJECT_ROOT / "logs"


# ─────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────


def setup_logging() -> None:
    """Separate log file so retries don't pollute the main batch log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<level>{level:<8}</level> | {message}",
    )
    logger.add(
        LOG_DIR / "retry_failed_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="50 MB",
        retention="30 days",
    )


# ─────────────────────────────────────────────────────────────────────
# Failed-headline query
# ─────────────────────────────────────────────────────────────────────


def find_failed_headlines(
    conn: Any,
    model_name: str,
    *,
    include_missing: bool,
    date_from: str,
    date_to: str,
    limit: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    """
    Return ``(headlines_to_retry, ids_with_existing_failed_row)``.

    ``headlines_to_retry``
        List of raw_headline rows (dicts) ready to feed back into a runner.
    ``ids_with_existing_failed_row``
        Subset whose ``nlp_vectors`` row exists with
        ``validation_passed=FALSE`` — these need to be DELETEd before
        re-inserting.  Never-processed headlines (``include_missing``
        path) have no row, so they don't need a delete.
    """
    if include_missing:
        query = """
            SELECT rh.id, rh.date, rh.source, rh.hour, rh.popularity, rh.headline,
                   nv.id AS _nv_id
            FROM raw_headlines rh
            LEFT JOIN nlp_vectors nv
                ON nv.headline_id = rh.id
                AND nv.model_name = %s
            WHERE nv.id IS NULL OR nv.validation_passed = FALSE
        """
    else:
        query = """
            SELECT rh.id, rh.date, rh.source, rh.hour, rh.popularity, rh.headline,
                   nv.id AS _nv_id
            FROM raw_headlines rh
            JOIN nlp_vectors nv
                ON nv.headline_id = rh.id
                AND nv.model_name = %s
            WHERE nv.validation_passed = FALSE
        """
    params: list[Any] = [model_name]

    if date_from:
        query += " AND rh.date >= %s"
        params.append(date_from)
    if date_to:
        query += " AND rh.date <= %s"
        params.append(date_to)

    query += " ORDER BY rh.id"

    if limit > 0:
        query += " LIMIT %s"
        params.append(limit)

    cursor = conn.cursor()
    cursor.execute(query, params)
    columns = [desc[0] for desc in cursor.description]
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    cursor.close()

    # Separate "has existing failed row" from "never processed"
    ids_with_row = [row["id"] for row in rows if row["_nv_id"] is not None]

    # Strip the join-only column before handing rows to the runner
    for row in rows:
        row.pop("_nv_id", None)

    return rows, ids_with_row


# ─────────────────────────────────────────────────────────────────────
# Delete stale failed rows so the retry INSERT is not a no-op
# ─────────────────────────────────────────────────────────────────────


_DELETE_CHUNK_SIZE = 10_000

_DELETE_SQL = """
    DELETE FROM nlp_vectors
    WHERE model_name = %s
      AND validation_passed = FALSE
      AND headline_id = ANY(%s)
"""


def delete_failed_rows(
    conn: Any, model_name: str, failed_ids: list[int]
) -> int:
    """
    Delete failed rows in chunks of ``_DELETE_CHUNK_SIZE``.
    Returns the total number of rows deleted.
    """
    if not failed_ids:
        return 0

    total_deleted = 0
    cursor = conn.cursor()
    try:
        for start in range(0, len(failed_ids), _DELETE_CHUNK_SIZE):
            chunk = failed_ids[start : start + _DELETE_CHUNK_SIZE]
            cursor.execute(_DELETE_SQL, (model_name, chunk))
            total_deleted += cursor.rowcount
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
    return total_deleted


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


async def retry(
    db_url: str,
    model_name: str,
    *,
    include_missing: bool,
    date_from: str,
    date_to: str,
    limit: int,
    batch_size: int,
    fast: bool,
    concurrency: int,
    headlines_per_call: int,
    dry_run: bool,
) -> None:
    conn = get_connection(db_url)
    try:
        logger.info("Scanning for failed headlines (model={!r})…", model_name)
        headlines, ids_with_row = find_failed_headlines(
            conn,
            model_name,
            include_missing=include_missing,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
        )

        total = len(headlines)
        with_row = len(ids_with_row)
        missing = total - with_row

        logger.info("Found {:,} headline(s) to retry", total)
        logger.info("  ‣ with validation_passed=FALSE: {:,}", with_row)
        if include_missing:
            logger.info("  ‣ missing from nlp_vectors:     {:,}", missing)

        if total == 0:
            logger.info("Nothing to retry. Done.")
            return

        if dry_run:
            dates = {str(h.get("date", "")) for h in headlines}
            logger.info(
                "[DRY RUN] Would delete {:,} failed row(s) and retry {:,} "
                "headline(s) across {:,} date(s). No DB writes.",
                with_row, total, len(dates),
            )
            return

        # Step 1: DELETE existing failed rows so the runner's INSERT
        # ON CONFLICT DO NOTHING can actually insert new data.
        if ids_with_row:
            logger.info(
                "Deleting {:,} existing failed row(s) from nlp_vectors…",
                with_row,
            )
            deleted = delete_failed_rows(conn, model_name, ids_with_row)
            logger.info("Deleted {:,} row(s).", deleted)

        # Step 2: Re-run through the identical process_headlines pipeline.
        # Dispatch matches process_headlines.run_batch exactly.
        if fast and headlines_per_call > 1:
            await run_batch_fast_batched(
                conn, headlines, model_name, batch_size,
                headlines_per_call, concurrency,
            )
        elif fast:
            await run_batch_fast(conn, headlines, model_name, batch_size, concurrency)
        else:
            await run_batch_standard(conn, headlines, model_name, batch_size)
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retry headlines that failed on a previous process_headlines.py "
            "run. Queries nlp_vectors for rows with validation_passed=FALSE, "
            "deletes them, and re-runs the associated raw_headlines through "
            "the same fast/standard pipeline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Retry all failed headlines (recommended)\n"
            "  cd processing_engine && uv run python ../scripts/retry_failed_headlines.py \\\n"
            "      --fast --headlines-per-call 50 --concurrency 32\n\n"
            "  # Also retry headlines that were never processed\n"
            "  cd processing_engine && uv run python ../scripts/retry_failed_headlines.py \\\n"
            "      --fast --headlines-per-call 50 --include-missing\n\n"
            "  # Dry run\n"
            "  cd processing_engine && uv run python ../scripts/retry_failed_headlines.py --dry-run\n"
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Rows to process per commit (default: 50).",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max headlines to retry (default: 0 = all).",
    )
    parser.add_argument(
        "--db-url", type=str, default=DEFAULT_DB_URL,
        help="PostgreSQL connection URL.",
    )
    parser.add_argument(
        "--model-name", type=str, default="",
        help="Override model name (default: auto-detect from config).",
    )
    parser.add_argument(
        "--date-from", type=str, default="",
        help="Only retry failures on/after this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--date-to", type=str, default="",
        help="Only retry failures on/before this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help=(
            "Also retry headlines with NO nlp_vectors row at all "
            "(never-processed). Default: only rows with validation_passed=FALSE."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would happen without deleting or re-processing.",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Use fast single-prompt mode (strongly recommended).",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4,
        help="Concurrent headlines in fast mode (default: 4, max: 32).",
    )
    parser.add_argument(
        "--headlines-per-call", type=int, default=0,
        help=(
            "Headlines packed per LLM call (default: 0 = disabled). "
            "Requires --fast. Max: 150."
        ),
    )
    args = parser.parse_args()

    # Safety bounds — match process_headlines.py to keep behavior consistent.
    _MAX_CONCURRENCY = 32
    _MAX_HEADLINES_PER_CALL = 150
    if args.concurrency < 1 or args.concurrency > _MAX_CONCURRENCY:
        parser.error(
            f"--concurrency must be between 1 and {_MAX_CONCURRENCY} "
            f"(got {args.concurrency})."
        )
    if args.headlines_per_call < 0 or args.headlines_per_call > _MAX_HEADLINES_PER_CALL:
        parser.error(
            f"--headlines-per-call must be between 0 and "
            f"{_MAX_HEADLINES_PER_CALL} (got {args.headlines_per_call})."
        )
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    setup_logging()

    model_name = args.model_name or get_active_model_name()
    hpc = args.headlines_per_call
    if args.fast and hpc > 1:
        mode = f"fast-batch ({hpc}/call)"
    elif args.fast:
        mode = "fast"
    else:
        mode = "standard"

    logger.info("SentiSense — Retry Failed Headlines")
    logger.info("  Model:       {}", model_name)
    logger.info("  Mode:        {}", mode)
    if args.fast:
        logger.info("  Concurrency: {}", args.concurrency)
    if hpc > 1:
        logger.info("  Headlines/call: {}", hpc)
    if args.include_missing:
        logger.info("  Scope:       failed + never-processed")
    else:
        logger.info("  Scope:       failed only (validation_passed=FALSE)")
    if args.date_from:
        logger.info("  Date from:   {}", args.date_from)
    if args.date_to:
        logger.info("  Date to:     {}", args.date_to)
    if args.limit > 0:
        logger.info("  Limit:       {:,}", args.limit)

    asyncio.run(retry(
        db_url=args.db_url,
        model_name=model_name,
        include_missing=args.include_missing,
        date_from=args.date_from,
        date_to=args.date_to,
        limit=args.limit,
        batch_size=args.batch_size,
        fast=args.fast,
        concurrency=args.concurrency,
        headlines_per_call=hpc,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
