"""
scripts.process_headlines
==========================
Batch-process headlines from the PostgreSQL ``raw_headlines`` table
through the SentiSense multi-agent pipeline and write results to
``nlp_vectors``.

Workflow
--------
1. Query ``raw_headlines`` rows that have no matching ``nlp_vectors``
   entry for the active model.
2. Run each headline through ``process_single_observation()``.
3. Insert the 7 scores + metadata into ``nlp_vectors``.
4. Commit after each batch, log progress.

Usage
-----
::

    # Dry run — show how many unprocessed headlines exist
    cd processing_engine && uv run python ../scripts/process_headlines.py --dry-run

    # Process 100 headlines in batches of 10
    cd processing_engine && uv run python ../scripts/process_headlines.py --limit 100 --batch-size 10

    # Process all headlines (default)
    cd processing_engine && uv run python ../scripts/process_headlines.py

    # Process with external Mistral API
    SENTISENSE_LLM_BACKEND=openai \\
    SENTISENSE_OPENAI_BASE_URL=https://10.10.248.21/v1 \\
    SENTISENSE_OPENAI_MODEL=mistral-large-2 \\
    SENTISENSE_OPENAI_HOST_HEADER=mistral-large-instruct-2407-gptq-runai-model-120b.cs.colman.ac.il \\
    SENTISENSE_OPENAI_VERIFY_SSL=false \\
    cd processing_engine && uv run python ../scripts/process_headlines.py --limit 10
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
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


# ─────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{level:<8}</level> | {message}")
    logger.add(
        LOG_DIR / "process_headlines_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="50 MB",
        retention="30 days",
    )


# ─────────────────────────────────────────────────────────────────────
# DB helpers
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


def get_active_model_name() -> str:
    """Determine the model name from the active LLM config."""
    from processing_engine.config import LLM_BACKEND, OllamaConfig, OpenAIConfig

    backend = LLM_BACKEND.lower()
    if backend == "openai":
        return OpenAIConfig().model
    return OllamaConfig().model


def get_unprocessed_headlines(
    conn: Any,
    model_name: str,
    limit: int = 0,
    date_from: str = "",
    date_to: str = "",
) -> list[dict[str, Any]]:
    """
    Query raw_headlines that have no nlp_vectors entry for the given model.
    """
    query = """
        SELECT rh.id, rh.date, rh.source, rh.hour, rh.popularity, rh.headline
        FROM raw_headlines rh
        LEFT JOIN nlp_vectors nv
            ON nv.headline_id = rh.id
            AND nv.model_name = %s
        WHERE nv.id IS NULL
    """
    params: list[Any] = [model_name]

    if date_from:
        query += " AND rh.date >= %s"
        params.append(date_from)
    if date_to:
        query += " AND rh.date <= %s"
        params.append(date_to)

    query += " ORDER BY rh.date DESC, rh.id"

    if limit > 0:
        query += " LIMIT %s"
        params.append(limit)

    cursor = conn.cursor()
    cursor.execute(query, params)
    columns = [desc[0] for desc in cursor.description]
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    cursor.close()
    return rows


INSERT_NLP_SQL = """
    INSERT INTO nlp_vectors (
        headline_id, model_name,
        relevance_politics, relevance_economy, relevance_security,
        relevance_health, relevance_science, relevance_technology,
        global_sentiment, validation_passed, processing_time_seconds, errors
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (headline_id, model_name) DO NOTHING
"""


def insert_nlp_vector(cursor: Any, headline_id: int, model_name: str, result: dict[str, Any]) -> None:
    """Insert a single processing result into nlp_vectors."""
    errors_list = result.get("errors", [])
    # Convert Python list to PostgreSQL text array literal
    errors_pg = errors_list if errors_list else None

    cursor.execute(INSERT_NLP_SQL, (
        headline_id,
        model_name,
        result.get("relevance_category_1", 0),
        result.get("relevance_category_2", 0),
        result.get("relevance_category_3", 0),
        result.get("relevance_category_4", 0),
        result.get("relevance_category_5", 0),
        result.get("relevance_category_6", 0),
        result.get("global_sentiment", 0),
        result.get("validation_passed", False),
        result.get("processing_time_seconds"),
        errors_pg,
    ))


# ─────────────────────────────────────────────────────────────────────
# Processing
# ─────────────────────────────────────────────────────────────────────


async def process_one(observation: dict[str, Any]) -> dict[str, Any]:
    """Process a single headline through the pipeline."""
    from processing_engine import process_single_observation

    return await process_single_observation(observation)


async def run_batch(
    db_url: str,
    model_name: str,
    batch_size: int,
    limit: int,
    date_from: str,
    date_to: str,
    *,
    dry_run: bool = False,
) -> None:
    """Main batch processing loop (async — single event loop for all headlines)."""
    t_start = time.perf_counter()

    conn = get_connection(db_url)
    headlines = get_unprocessed_headlines(conn, model_name, limit=limit, date_from=date_from, date_to=date_to)

    total = len(headlines)
    logger.info("Found {:,} unprocessed headlines for model '{}'", total, model_name)

    if total == 0:
        logger.info("Nothing to process. Done.")
        conn.close()
        return

    if dry_run:
        dates = {str(h.get("date", "")) for h in headlines}
        logger.info(
            "[DRY RUN] Would process {:,} headlines across {:,} date(s). No LLM calls.",
            total,
            len(dates),
        )
        conn.close()
        return

    # Reset graph to pick up current LLM config
    from processing_engine import reset_graph
    reset_graph()

    processed = 0
    succeeded = 0
    failed = 0
    failed_ids: list[int] = []

    for batch_start in range(0, total, batch_size):
        batch = headlines[batch_start : batch_start + batch_size]
        cursor = conn.cursor()

        for headline_row in batch:
            headline_id = headline_row["id"]
            obs = {
                "date": str(headline_row.get("date", "")),
                "source": str(headline_row.get("source", "")),
                "hour": str(headline_row.get("hour", "")),
                "popularity": str(headline_row.get("popularity", "")),
                "headline": str(headline_row.get("headline", "")),
            }

            try:
                result = await process_one(obs)
                insert_nlp_vector(cursor, headline_id, model_name, result)
                succeeded += 1
            except Exception as exc:
                logger.error(
                    "Failed headline id={}: {}", headline_id, exc
                )
                failed += 1
                failed_ids.append(headline_id)

            processed += 1

        # Commit after each batch
        conn.commit()
        cursor.close()

        elapsed = time.perf_counter() - t_start
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (total - processed) / rate if rate > 0 else 0

        logger.info(
            "Progress: {:,}/{:,} ({:.0f}%) | OK: {:,} | Failed: {:,} | "
            "{:.1f}s elapsed | ~{:.0f}s remaining",
            processed,
            total,
            100 * processed / total,
            succeeded,
            failed,
            elapsed,
            eta,
        )

    conn.close()
    elapsed = time.perf_counter() - t_start

    logger.info("─" * 60)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("  Model:       {}", model_name)
    logger.info("  Total:       {:,}", total)
    logger.info("  Succeeded:   {:,}", succeeded)
    logger.info("  Failed:      {:,}", failed)
    logger.info("  Elapsed:     {:.1f}s ({:.1f} min)", elapsed, elapsed / 60)
    if failed_ids:
        logger.info("  Failed IDs:  {}", failed_ids[:20])
        if len(failed_ids) > 20:
            logger.info("  ... and {:,} more", len(failed_ids) - 20)
    logger.info("─" * 60)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-process headlines from raw_headlines through the NLP pipeline into nlp_vectors."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of headlines to process before committing (default: 50).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max total headlines to process (default: 0 = all).",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=DEFAULT_DB_URL,
        help="PostgreSQL connection URL.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="",
        help="Override model name for nlp_vectors (default: auto-detect from config).",
    )
    parser.add_argument(
        "--date-from",
        type=str,
        default="",
        help="Only process headlines from this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--date-to",
        type=str,
        default="",
        help="Only process headlines up to this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show count of unprocessed headlines without processing.",
    )
    args = parser.parse_args()

    setup_logging()

    model_name = args.model_name or get_active_model_name()
    logger.info("SentiSense — Batch Headline Processing")
    logger.info("  LLM backend: {}", os.environ.get("SENTISENSE_LLM_BACKEND", "ollama"))
    logger.info("  Model:       {}", model_name)

    asyncio.run(run_batch(
        db_url=args.db_url,
        model_name=model_name,
        batch_size=args.batch_size,
        limit=args.limit,
        date_from=args.date_from,
        date_to=args.date_to,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
