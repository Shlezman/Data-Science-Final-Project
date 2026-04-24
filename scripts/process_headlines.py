"""
scripts.process_headlines
==========================
Batch-process headlines from the PostgreSQL ``raw_headlines`` table
through the SentiSense pipeline and write results to ``nlp_vectors``.

Three modes (slowest → fastest):
  - **Standard** (default) — 7 ReAct agents per headline (~50-60s each)
  - **Fast** (``--fast``) — single-prompt scoring with concurrent
    headlines (~3-5s each, ~10-15x faster)
  - **Fast-batch** (``--fast --headlines-per-call N``) — N headlines packed
    into each LLM call (~0.3-0.5s per headline effective, ~100x faster)

Usage
-----
::

    # Fast-batch mode (fastest) — 15 headlines per LLM call, 4 calls at a time
    cd processing_engine && uv run python ../scripts/process_headlines.py \\
        --fast --headlines-per-call 15 --limit 100

    # Fast-batch with higher throughput
    cd processing_engine && uv run python ../scripts/process_headlines.py \\
        --fast --headlines-per-call 20 --concurrency 8

    # Fast mode — 4 headlines concurrently, 1 LLM call each
    cd processing_engine && uv run python ../scripts/process_headlines.py --fast --limit 100

    # Standard multi-agent mode
    cd processing_engine && uv run python ../scripts/process_headlines.py --limit 100

    # Dry run
    cd processing_engine && uv run python ../scripts/process_headlines.py --dry-run
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
    """Create a psycopg connection.

    Tries psycopg v3 first, then psycopg2 as a fallback.  Only a missing
    driver (ImportError) triggers the fallback — any other connection
    failure (bad URL, auth, network) propagates immediately so the user
    sees the real error instead of a misleading "driver not installed"
    message.
    """
    try:
        import psycopg  # noqa: F401
    except ImportError:
        pass
    else:
        import psycopg as _psycopg

        return _psycopg.connect(db_url, autocommit=False)

    try:
        import psycopg2
    except ImportError:
        logger.error(
            "Neither psycopg nor psycopg2 is installed.\n"
            "  Install with: pip install 'psycopg[binary]'"
        )
        sys.exit(1)
    return psycopg2.connect(db_url)


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
    """Process a single headline through the multi-agent pipeline."""
    from processing_engine import process_single_observation

    return await process_single_observation(observation)


def _make_obs(headline_row: dict[str, Any]) -> dict[str, Any]:
    """Convert a DB row dict into an observation dict."""
    return {
        "date": str(headline_row.get("date", "")),
        "source": str(headline_row.get("source", "")),
        "hour": str(headline_row.get("hour", "")),
        "popularity": str(headline_row.get("popularity", "")),
        "headline": str(headline_row.get("headline", "")),
    }


def _log_progress(
    processed: int, total: int, succeeded: int, failed: int, t_start: float,
) -> None:
    """Log a progress line with ETA."""
    elapsed = time.perf_counter() - t_start
    rate = processed / elapsed if elapsed > 0 else 0
    eta = (total - processed) / rate if rate > 0 else 0
    logger.info(
        "Progress: {:,}/{:,} ({:.0f}%) | OK: {:,} | Failed: {:,} | "
        "{:.1f}s elapsed | {:.1f} headlines/min | ~{:.0f}s remaining",
        processed, total, 100 * processed / total,
        succeeded, failed, elapsed, rate * 60, eta,
    )


def _log_summary(
    model_name: str, mode: str, total: int, succeeded: int,
    failed: int, failed_ids: list[int], t_start: float,
) -> None:
    """Log the final batch summary."""
    elapsed = time.perf_counter() - t_start
    rate = total / elapsed * 60 if elapsed > 0 else 0
    logger.info("─" * 60)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("  Mode:        {}", mode)
    logger.info("  Model:       {}", model_name)
    logger.info("  Total:       {:,}", total)
    logger.info("  Succeeded:   {:,}", succeeded)
    logger.info("  Failed:      {:,}", failed)
    logger.info("  Elapsed:     {:.1f}s ({:.1f} min)", elapsed, elapsed / 60)
    logger.info("  Throughput:  {:.1f} headlines/min", rate)
    if failed_ids:
        logger.info("  Failed IDs:  {}", failed_ids[:20])
        if len(failed_ids) > 20:
            logger.info("  ... and {:,} more", len(failed_ids) - 20)
    logger.info("─" * 60)


# ─────────────────────────────────────────────────────────────────────
# Standard multi-agent batch runner
# ─────────────────────────────────────────────────────────────────────


async def run_batch_standard(
    conn: Any,
    headlines: list[dict[str, Any]],
    model_name: str,
    batch_size: int,
) -> None:
    """Process headlines one-at-a-time through the 7-agent pipeline.

    .. deprecated::
        The standard multi-agent path depends on native function-/tool-
        calling via ``/v1/chat/completions``.  It is incompatible with
        ``FORCE_COMPLETIONS_API=true`` (vLLM Mistral-Small 4) and only
        works against Ollama or an OpenAI-compatible chat endpoint that
        supports ``bind_tools``.  Prefer ``--fast`` for all new runs.
    """
    if os.environ.get("SENTISENSE_FORCE_COMPLETIONS_API", "").lower() in (
        "true", "1", "yes",
    ):
        logger.warning(
            "Standard multi-agent mode is incompatible with "
            "FORCE_COMPLETIONS_API=true (no tool-calling on /v1/completions). "
            "Use --fast instead; aborting."
        )
        sys.exit(2)

    from processing_engine import reset_graph
    reset_graph()

    total = len(headlines)
    t_start = time.perf_counter()
    processed = 0
    succeeded = 0
    failed = 0
    failed_ids: list[int] = []

    for batch_start in range(0, total, batch_size):
        batch = headlines[batch_start : batch_start + batch_size]
        cursor = conn.cursor()

        for headline_row in batch:
            headline_id = headline_row["id"]
            try:
                result = await process_one(_make_obs(headline_row))
                insert_nlp_vector(cursor, headline_id, model_name, result)
                succeeded += 1
            except Exception as exc:
                logger.error("Failed headline id={}: {}", headline_id, exc)
                failed += 1
                failed_ids.append(headline_id)
            processed += 1

        conn.commit()
        cursor.close()
        _log_progress(processed, total, succeeded, failed, t_start)

    _log_summary(model_name, "standard (7 agents)", total, succeeded, failed, failed_ids, t_start)


# ─────────────────────────────────────────────────────────────────────
# Fast single-prompt batch runner with concurrent headlines
# ─────────────────────────────────────────────────────────────────────


async def run_batch_fast(
    conn: Any,
    headlines: list[dict[str, Any]],
    model_name: str,
    batch_size: int,
    concurrency: int,
) -> None:
    """
    Process headlines using the fast single-prompt pipeline.

    Each headline = 1 LLM call (instead of ~21).  Multiple headlines
    are processed concurrently (default: 4 at a time).
    """
    from processing_engine.fast_pipeline import score_headlines_concurrent
    from processing_engine.prompts import build_llm

    llm = build_llm()
    total = len(headlines)
    t_start = time.perf_counter()
    processed = 0
    succeeded = 0
    failed = 0
    failed_ids: list[int] = []

    logger.info(
        "Fast mode: {} concurrent headlines, 1 LLM call each",
        concurrency,
    )

    for batch_start in range(0, total, batch_size):
        batch = headlines[batch_start : batch_start + batch_size]
        obs_list = [_make_obs(row) for row in batch]
        ids = [row["id"] for row in batch]

        # Process the whole batch concurrently
        results = await score_headlines_concurrent(
            obs_list, llm=llm, concurrency=concurrency,
        )

        cursor = conn.cursor()
        for headline_id, result in zip(ids, results):
            try:
                insert_nlp_vector(cursor, headline_id, model_name, result)
                if result.get("validation_passed", False):
                    succeeded += 1
                else:
                    failed += 1
                    failed_ids.append(headline_id)
            except Exception as exc:
                logger.error("DB insert failed id={}: {}", headline_id, exc)
                failed += 1
                failed_ids.append(headline_id)
            processed += 1

        conn.commit()
        cursor.close()
        _log_progress(processed, total, succeeded, failed, t_start)

    _log_summary(model_name, f"fast (concurrency={concurrency})", total, succeeded, failed, failed_ids, t_start)


# ─────────────────────────────────────────────────────────────────────
# Fast batch mode — multiple headlines per single LLM call
# ─────────────────────────────────────────────────────────────────────


async def run_batch_fast_batched(
    conn: Any,
    headlines: list[dict[str, Any]],
    model_name: str,
    commit_size: int,
    headlines_per_call: int,
    concurrency: int,
) -> None:
    """
    Process headlines using batched single-prompt pipeline.

    Multiple headlines are packed into each LLM call (``headlines_per_call``),
    and multiple batches run concurrently.  This is the fastest mode:
    N headlines ÷ headlines_per_call = total LLM calls needed.
    """
    from processing_engine.fast_pipeline import score_headlines_batch
    from processing_engine.prompts import build_llm

    llm = build_llm()
    total = len(headlines)
    t_start = time.perf_counter()
    processed = 0
    succeeded = 0
    failed = 0
    failed_ids: list[int] = []

    total_calls = (total + headlines_per_call - 1) // headlines_per_call
    logger.info(
        "Fast-batch mode: {} headlines/call, {} concurrent batches → ~{} LLM calls total",
        headlines_per_call, concurrency, total_calls,
    )

    for commit_start in range(0, total, commit_size):
        commit_chunk = headlines[commit_start : commit_start + commit_size]
        obs_list = [_make_obs(row) for row in commit_chunk]
        ids = [row["id"] for row in commit_chunk]

        results = await score_headlines_batch(
            obs_list,
            llm=llm,
            batch_size=headlines_per_call,
            concurrency=concurrency,
        )

        cursor = conn.cursor()
        for headline_id, result in zip(ids, results):
            try:
                insert_nlp_vector(cursor, headline_id, model_name, result)
                if result.get("validation_passed", False):
                    succeeded += 1
                else:
                    failed += 1
                    failed_ids.append(headline_id)
            except Exception as exc:
                logger.error("DB insert failed id={}: {}", headline_id, exc)
                failed += 1
                failed_ids.append(headline_id)
            processed += 1

        conn.commit()
        cursor.close()
        _log_progress(processed, total, succeeded, failed, t_start)

    _log_summary(
        model_name,
        f"fast-batch ({headlines_per_call}/call, concurrency={concurrency})",
        total, succeeded, failed, failed_ids, t_start,
    )


# ─────────────────────────────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────────────────────────────


async def run_batch(
    db_url: str,
    model_name: str,
    batch_size: int,
    limit: int,
    date_from: str,
    date_to: str,
    *,
    dry_run: bool = False,
    fast: bool = False,
    concurrency: int = 4,
    headlines_per_call: int = 0,
) -> None:
    """Main entry point — dispatches to standard, fast, or fast-batch runner."""
    conn = get_connection(db_url)
    headlines = get_unprocessed_headlines(
        conn, model_name, limit=limit, date_from=date_from, date_to=date_to,
    )

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
            total, len(dates),
        )
        conn.close()
        return

    if fast and headlines_per_call > 1:
        await run_batch_fast_batched(
            conn, headlines, model_name, batch_size,
            headlines_per_call, concurrency,
        )
    elif fast:
        await run_batch_fast(conn, headlines, model_name, batch_size, concurrency)
    else:
        await run_batch_standard(conn, headlines, model_name, batch_size)

    conn.close()


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
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast single-prompt mode (1 LLM call per headline instead of ~21). ~10-15x faster.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of headlines to process simultaneously in fast mode (default: 4).",
    )
    parser.add_argument(
        "--headlines-per-call",
        type=int,
        default=0,
        help=(
            "Pack N headlines into each LLM call (batch mode). "
            "Requires --fast. 0=disabled (1 headline per call). "
            "Recommended: 15-20 on 32K-context models, 50-80 on 128K. "
            "Upper bound auto-derived from SENTISENSE_CONTEXT_WINDOW "
            "(fast_pipeline.MAX_BATCH_SIZE). Exceeding the bound is clamped."
        ),
    )
    args = parser.parse_args()

    # Safety bounds — prevent accidental resource exhaustion.
    # These caps match what the inference server can comfortably handle;
    # raise them only after load-testing the vLLM backend.
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

    setup_logging()

    model_name = args.model_name or get_active_model_name()
    hpc = args.headlines_per_call
    if hpc > 1 and args.fast:
        mode = f"fast-batch ({hpc}/call)"
    elif args.fast:
        mode = "fast"
    else:
        mode = "standard"

    completions_mode = os.environ.get("SENTISENSE_FORCE_COMPLETIONS_API", "").lower() in ("true", "1", "yes")
    logger.info("SentiSense — Batch Headline Processing")
    logger.info("  LLM backend: {}", os.environ.get("SENTISENSE_LLM_BACKEND", "ollama"))
    logger.info("  Model:       {}", model_name)
    logger.info("  Mode:        {}", mode)
    if completions_mode:
        logger.info("  API:         /v1/completions (raw text)")
    if args.fast:
        logger.info("  Concurrency: {}", args.concurrency)
    if hpc > 1:
        logger.info("  Headlines/call: {}", hpc)

    asyncio.run(run_batch(
        db_url=args.db_url,
        model_name=model_name,
        batch_size=args.batch_size,
        limit=args.limit,
        date_from=args.date_from,
        date_to=args.date_to,
        dry_run=args.dry_run,
        fast=args.fast,
        concurrency=args.concurrency,
        headlines_per_call=hpc,
    ))


if __name__ == "__main__":
    main()
