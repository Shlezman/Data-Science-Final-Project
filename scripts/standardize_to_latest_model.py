"""
scripts.standardize_to_latest_model
====================================
Standardise the ``nlp_vectors`` table on a single "latest" model.

A headline qualifies for re-processing if it has at least one row in
``nlp_vectors`` (under any model) **but does not yet have a successful
row under the latest model**.  That covers two distinct populations:

* **Cross-model legacy rows** — headlines scored only by an older model
  (e.g. ``mistral-large-2``, ``mistral-small3.2``).  Useful when the
  active production model has changed and you want a uniform dataset.
* **Failed latest-model rows** — headlines whose ``validation_passed``
  is ``FALSE`` under the latest model.  Same population as
  ``retry_failed_headlines.py``, intentionally re-covered here so this
  script is a complete one-shot standardiser.

Workflow
--------
1. Discover the latest model (``--latest-model`` or ``get_active_model_name()``).
2. Query for qualifying ``raw_headlines`` rows.
3. Delete stale rows in chunks:

   * **Failed latest-model rows** are always deleted so the runner's
     ``INSERT … ON CONFLICT DO NOTHING`` is not a no-op.
   * **Non-latest rows** are deleted unless ``--keep-old-rows`` is set
     (use that flag if you want to preserve the multi-model history).

4. Re-run the headlines through the same fast / fast-batched / standard
   pipeline as ``process_headlines.py``.

Safety
------
* ``--dry-run`` prints counts and exits without writing.
* If the re-run fails mid-way, the deleted rows are lost but every
  headline that *was* re-processed has a fresh latest-model row
  (success or failure — same semantics as ``process_headlines.py``).
* Concurrency / headlines-per-call caps mirror ``process_headlines.py``.

Usage
-----
::

    # Standardise everything onto the auto-detected active model
    # (deletes legacy non-latest rows after re-scoring).
    cd processing_engine && uv run python ../scripts/standardize_to_latest_model.py \\
        --fast --headlines-per-call 50 --concurrency 50

    # Dry run — see what would happen.
    cd processing_engine && uv run python ../scripts/standardize_to_latest_model.py --dry-run

    # Keep the legacy non-latest rows (still re-scores under latest).
    cd processing_engine && uv run python ../scripts/standardize_to_latest_model.py \\
        --fast --headlines-per-call 50 --keep-old-rows

    # Pin the latest model name explicitly.
    cd processing_engine && uv run python ../scripts/standardize_to_latest_model.py \\
        --latest-model mistral-small-4 --fast --headlines-per-call 50 --concurrency 50

    # Restrict to a date window.
    cd processing_engine && uv run python ../scripts/standardize_to_latest_model.py \\
        --fast --headlines-per-call 50 --date-from 2024-01-01 --date-to 2024-12-31
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from loguru import logger

# ─────────────────────────────────────────────────────────────────────
# Reuse runners from process_headlines so behaviour stays in lock-step
# with the main batch CLI.  Adding scripts/ to sys.path because the
# scripts/ directory is not a package.
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
    """Separate log file so standardisation runs don't pollute the main batch log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<level>{level:<8}</level> | {message}",
    )
    logger.add(
        LOG_DIR / "standardize_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="50 MB",
        retention="30 days",
    )


# ─────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────


def find_headlines_to_standardize(
    conn: Any,
    latest_model: str,
    *,
    rescore_legacy: bool,
    date_from: str,
    date_to: str,
    limit: int,
) -> tuple[list[dict[str, Any]], int, int, int]:
    """
    Return ``(headlines, n_non_latest, n_failed_latest, n_success_latest)``.

    Discovery semantics depend on ``rescore_legacy``:

    * ``False`` (default) — include only headlines that have *some*
      ``nlp_vectors`` history but no **successful** row under
      ``latest_model``.  Skips headlines already standardised.
    * ``True`` — include *every* headline that has at least one row
      under a non-latest ``model_name``, even if a successful latest
      row already exists.  All latest-model rows for these headlines
      will be deleted before re-scoring.

    The four count return values are the per-status row totals that
    will be deleted during cleanup, surfaced upfront so ``--dry-run``
    shows the exact impact:

    * ``n_non_latest_rows`` — rows whose ``model_name <> latest``.
    * ``n_failed_latest_rows`` — failed rows under ``latest_model``.
    * ``n_success_latest_rows`` — successful rows under ``latest_model``
      (only deleted when ``rescore_legacy`` is True).
    """
    if rescore_legacy:
        query = """
            SELECT DISTINCT
                   rh.id, rh.date, rh.source, rh.hour, rh.popularity, rh.headline
            FROM raw_headlines rh
            WHERE EXISTS (
                SELECT 1 FROM nlp_vectors nv
                WHERE nv.headline_id = rh.id
                  AND nv.model_name <> %s
            )
        """
    else:
        query = """
            SELECT DISTINCT
                   rh.id, rh.date, rh.source, rh.hour, rh.popularity, rh.headline
            FROM raw_headlines rh
            WHERE EXISTS (
                SELECT 1 FROM nlp_vectors nv
                WHERE nv.headline_id = rh.id
            )
            AND NOT EXISTS (
                SELECT 1 FROM nlp_vectors nv
                WHERE nv.headline_id = rh.id
                  AND nv.model_name = %s
                  AND nv.validation_passed = TRUE
            )
        """
    params: list[Any] = [latest_model]

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

    cur = conn.cursor()
    cur.execute(query, params)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    cur.close()

    if not rows:
        return [], 0, 0, 0

    # Pre-count what we'll delete so --dry-run is accurate.
    headline_ids = [r["id"] for r in rows]
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            COALESCE(SUM(
                CASE WHEN model_name <> %s THEN 1 ELSE 0 END
            ), 0) AS non_latest,
            COALESCE(SUM(
                CASE WHEN model_name = %s AND validation_passed = FALSE THEN 1 ELSE 0 END
            ), 0) AS failed_latest,
            COALESCE(SUM(
                CASE WHEN model_name = %s AND validation_passed = TRUE THEN 1 ELSE 0 END
            ), 0) AS success_latest
        FROM nlp_vectors
        WHERE headline_id = ANY(%s)
        """,
        (latest_model, latest_model, latest_model, headline_ids),
    )
    n_non_latest, n_failed_latest, n_success_latest = cur.fetchone()
    cur.close()

    return (
        rows,
        int(n_non_latest),
        int(n_failed_latest),
        int(n_success_latest),
    )


# ─────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────


_DELETE_CHUNK_SIZE = 10_000


def delete_stale_rows(
    conn: Any,
    latest_model: str,
    headline_ids: list[int],
    *,
    keep_non_latest: bool,
    rescore_legacy: bool,
) -> tuple[int, int, int]:
    """
    Delete the rows that block re-insertion.

    * Failed latest-model rows — always deleted (otherwise the runner's
      ``ON CONFLICT DO NOTHING`` would skip the retry insert).
    * Successful latest-model rows — deleted only when
      ``rescore_legacy`` is True; the user explicitly opted to re-score
      headlines that already have a latest score.
    * Non-latest rows — deleted unless ``keep_non_latest`` is True.

    Returns ``(n_failed_deleted, n_success_deleted, n_non_latest_deleted)``.
    """
    if not headline_ids:
        return 0, 0, 0

    n_failed_deleted = 0
    n_success_deleted = 0
    n_non_latest_deleted = 0
    cur = conn.cursor()
    try:
        for start in range(0, len(headline_ids), _DELETE_CHUNK_SIZE):
            chunk = headline_ids[start : start + _DELETE_CHUNK_SIZE]

            cur.execute(
                """
                DELETE FROM nlp_vectors
                WHERE model_name = %s
                  AND validation_passed = FALSE
                  AND headline_id = ANY(%s)
                """,
                (latest_model, chunk),
            )
            n_failed_deleted += cur.rowcount

            if rescore_legacy:
                cur.execute(
                    """
                    DELETE FROM nlp_vectors
                    WHERE model_name = %s
                      AND validation_passed = TRUE
                      AND headline_id = ANY(%s)
                    """,
                    (latest_model, chunk),
                )
                n_success_deleted += cur.rowcount

            if not keep_non_latest:
                cur.execute(
                    """
                    DELETE FROM nlp_vectors
                    WHERE model_name <> %s
                      AND headline_id = ANY(%s)
                    """,
                    (latest_model, chunk),
                )
                n_non_latest_deleted += cur.rowcount
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()

    return n_failed_deleted, n_success_deleted, n_non_latest_deleted


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


async def standardize(
    db_url: str,
    latest_model: str,
    *,
    keep_old_rows: bool,
    rescore_legacy: bool,
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
        scope = (
            "every headline that has any non-latest row"
            if rescore_legacy
            else f"headlines lacking a successful row under {latest_model!r}"
        )
        logger.info("Scanning for {}…", scope)
        (
            headlines,
            n_non_latest,
            n_failed_latest,
            n_success_latest,
        ) = find_headlines_to_standardize(
            conn,
            latest_model,
            rescore_legacy=rescore_legacy,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
        )

        total = len(headlines)
        logger.info("Headlines to (re)process:                       {:,}", total)
        logger.info("  ‣ failed latest rows that will be DELETEd:    {:,}", n_failed_latest)
        if rescore_legacy:
            logger.info(
                "  ‣ successful latest rows that will be DELETEd: {:,}",
                n_success_latest,
            )
        logger.info(
            "  ‣ non-latest rows that will be {}: {:,}",
            "KEPT (--keep-old-rows)" if keep_old_rows else "DELETEd",
            n_non_latest,
        )

        if total == 0:
            logger.info("Dataset already standardised on {!r}. Nothing to do.", latest_model)
            return

        if dry_run:
            dates = {str(h.get("date", "")) for h in headlines}
            logger.info(
                "[DRY RUN] Would re-score {:,} headline(s) across {:,} date(s). "
                "No DB writes.",
                total, len(dates),
            )
            return

        # Step 1: clean up stale rows so re-INSERT can proceed.
        ids = [h["id"] for h in headlines]
        logger.info("Deleting stale rows…")
        n_f, n_s, n_nl = delete_stale_rows(
            conn,
            latest_model,
            ids,
            keep_non_latest=keep_old_rows,
            rescore_legacy=rescore_legacy,
        )
        if rescore_legacy:
            logger.info(
                "Deleted {:,} failed-latest, {:,} successful-latest, "
                "and {:,} non-latest row(s).",
                n_f, n_s, n_nl,
            )
        else:
            logger.info(
                "Deleted {:,} failed-latest row(s) and {:,} non-latest row(s).",
                n_f, n_nl,
            )

        # Step 2: re-run through the identical process_headlines pipeline.
        if fast and headlines_per_call > 1:
            await run_batch_fast_batched(
                conn, headlines, latest_model, batch_size,
                headlines_per_call, concurrency,
            )
        elif fast:
            await run_batch_fast(
                conn, headlines, latest_model, batch_size, concurrency,
            )
        else:
            await run_batch_standard(
                conn, headlines, latest_model, batch_size,
            )
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Standardise nlp_vectors on a single 'latest' model.  Finds "
            "every headline whose only / best score is under a non-latest "
            "model (or whose latest-model row failed validation), deletes "
            "those stale rows, and re-scores under the latest model so "
            "the dataset is uniform."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Standardise on the auto-detected active model (delete legacy)\n"
            "  cd processing_engine && uv run python ../scripts/standardize_to_latest_model.py \\\n"
            "      --fast --headlines-per-call 50 --concurrency 50\n\n"
            "  # Force-rescore EVERY headline that any non-latest model touched,\n"
            "  # even if a successful latest row already exists.  Existing latest\n"
            "  # rows for those headlines are deleted before re-scoring.\n"
            "  cd processing_engine && uv run python ../scripts/standardize_to_latest_model.py \\\n"
            "      --fast --headlines-per-call 50 --concurrency 50 --rescore-legacy\n\n"
            "  # Dry run — show counts without touching the DB\n"
            "  cd processing_engine && uv run python ../scripts/standardize_to_latest_model.py --dry-run\n\n"
            "  # Keep history (don't delete non-latest rows)\n"
            "  cd processing_engine && uv run python ../scripts/standardize_to_latest_model.py \\\n"
            "      --fast --headlines-per-call 50 --keep-old-rows\n"
        ),
    )
    parser.add_argument(
        "--latest-model", type=str, default="",
        help="Override which model_name is treated as 'latest' (default: auto-detect).",
    )
    parser.add_argument(
        "--keep-old-rows", action="store_true",
        help="Don't delete non-latest rows after re-scoring (preserves multi-model history).",
    )
    parser.add_argument(
        "--rescore-legacy", action="store_true",
        help=(
            "Re-score EVERY headline that has any row under a non-latest "
            "model_name, even if a successful latest row already exists.  "
            "Existing latest-model rows for those headlines are DELETEd "
            "before re-scoring.  Use this to force a uniform re-scoring "
            "after rolling out a new model."
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Rows to process per DB commit (default: 50).",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Cap headlines per run (default: 0 = all).",
    )
    parser.add_argument(
        "--db-url", type=str, default=DEFAULT_DB_URL,
        help="PostgreSQL connection URL.",
    )
    parser.add_argument(
        "--date-from", type=str, default="",
        help="Only re-score headlines on/after this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--date-to", type=str, default="",
        help="Only re-score headlines on/before this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show counts without deleting or re-processing.",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Use fast single-prompt mode (recommended).",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4,
        help="Concurrent headlines/batches in fast mode (default: 4, max: 128).",
    )
    parser.add_argument(
        "--headlines-per-call", type=int, default=0,
        help=(
            "Headlines packed per LLM call (default: 0 = disabled). "
            "Requires --fast.  Max: 150."
        ),
    )
    args = parser.parse_args()

    # Safety bounds — match process_headlines.py exactly.
    _MAX_CONCURRENCY = 128
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

    latest_model = args.latest_model or get_active_model_name()
    hpc = args.headlines_per_call

    if args.fast and hpc > 1:
        mode = f"fast-batch ({hpc}/call)"
    elif args.fast:
        mode = "fast"
    else:
        mode = "standard"

    logger.info("SentiSense — Standardise to Latest Model")
    logger.info("  Latest model:   {}", latest_model)
    logger.info("  Mode:           {}", mode)
    if args.fast:
        logger.info("  Concurrency:    {}", args.concurrency)
    if hpc > 1:
        logger.info("  Headlines/call: {}", hpc)
    logger.info(
        "  Scope:          {}",
        "every legacy headline (--rescore-legacy)"
        if args.rescore_legacy
        else "headlines without a successful latest row",
    )
    logger.info(
        "  Old rows:       {}",
        "kept" if args.keep_old_rows else "deleted after re-scoring",
    )
    if args.date_from:
        logger.info("  Date from:      {}", args.date_from)
    if args.date_to:
        logger.info("  Date to:        {}", args.date_to)
    if args.limit:
        logger.info("  Limit:          {:,}", args.limit)

    asyncio.run(standardize(
        db_url=args.db_url,
        latest_model=latest_model,
        keep_old_rows=args.keep_old_rows,
        rescore_legacy=args.rescore_legacy,
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
