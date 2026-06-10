"""Live whole-pipeline ETA — up-front estimate + a running per-stage clock.

The estimate is deliberately rough (rate priors in :mod:`sentisense.config`, all
env-overridable). It is then refined *live*: each stage's actual wall-time replaces
its prior as it completes, and the two dominant stages stream their own progress —
scoring via the ``process_headlines.py`` subprocess ("~Ns remaining"), HPO via an
Optuna callback (see :mod:`sentisense.hpo.optuna_lstm`).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

from loguru import logger
from sqlalchemy import text

from sentisense import config
from sentisense.constants import CUTOFF_DATE, resolve_active_model
from sentisense.config import EMBED_MODEL


def fmt_duration(secs: float | None) -> str:
    """Human-readable duration, e.g. '2h 14m 03s'. None → 'n/a'."""
    if secs is None:
        return "n/a"
    secs = int(round(secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def eta_clock(secs_from_now: float | None) -> str:
    """Absolute wall-clock ETA string from a remaining-seconds estimate."""
    if secs_from_now is None:
        return "n/a"
    finish = datetime.now() + timedelta(seconds=secs_from_now)
    return finish.strftime("%Y-%m-%d %H:%M:%S")


def _scalar(engine, sql: str, params: dict) -> int:
    with engine.connect() as conn:
        return int(conn.execute(text(sql), params).scalar_one())


def count_unscored(engine, model: str | None = None) -> int:
    """Headlines ≤ cutoff with no nlp_vectors row for the dataset (read) model."""
    model = model or resolve_active_model(engine)
    return _scalar(engine, """
        SELECT COUNT(*) FROM raw_headlines rh
        LEFT JOIN nlp_vectors nv ON nv.headline_id = rh.id AND nv.model_name = :model
        WHERE nv.id IS NULL AND rh.date <= :cutoff
    """, {"model": model, "cutoff": CUTOFF_DATE})


def count_unembedded(engine) -> int:
    """Headlines ≤ cutoff lacking an embedding. Falls back to all-headlines if the
    embedding-cache table doesn't exist yet (created by the embed stage)."""
    try:
        return _scalar(engine, """
            SELECT COUNT(*) FROM raw_headlines rh
            LEFT JOIN headline_embeddings he
                ON he.headline_id = rh.id AND he.embed_model = :emodel
            WHERE he.headline_id IS NULL AND rh.date <= :cutoff
        """, {"emodel": EMBED_MODEL, "cutoff": CUTOFF_DATE})
    except Exception:  # table missing → nothing embedded yet
        return _scalar(engine, "SELECT COUNT(*) FROM raw_headlines WHERE date <= :cutoff",
                       {"cutoff": CUTOFF_DATE})


def estimate(stages: list[str], engine) -> dict[str, float | None]:
    """Estimate per-stage seconds for the selected stages and log a table + total ETA.

    Returns a dict ``{stage: seconds}`` (``None`` for the unpredictable backfill).
    DB-count stages (score/embed) query live counts; others use fixed priors.
    """
    fixed = config.ETA_SECS_FIXED_STAGE
    dataset_model = resolve_active_model(engine)
    logger.info("Dataset (read) model: '{}'", dataset_model)

    counts: dict[str, int] = {}
    if "score" in stages:
        counts["unscored"] = count_unscored(engine, dataset_model)
        from sentisense.constants import scoring_model_name
        scoring = scoring_model_name()
        if scoring != dataset_model:
            logger.warning(
                "score stage would write '{}' but the corpus is modelled on '{}'. "
                "Your data is already scored — skip scoring with `--from embed` (or "
                "`--from features`), or re-score everything under '{}' on purpose.",
                scoring, dataset_model, scoring,
            )
    if "embed" in stages:
        counts["unembedded"] = count_unembedded(engine)

    est: dict[str, float | None] = {}
    for s in stages:
        if s == "backfill":
            est[s] = None  # network-bound scrape; unpredictable
        elif s == "score":
            est[s] = counts["unscored"] * config.ETA_SECS_PER_SCORED_HEADLINE
        elif s == "embed":
            est[s] = counts["unembedded"] * config.ETA_SECS_PER_EMBEDDED_HEADLINE
        elif s == "tune":
            est[s] = config.OPTUNA_TRIALS * config.ETA_SECS_PER_HPO_TRIAL
        elif s in ("cluster", "baselines", "final"):
            est[s] = fixed * 2
        else:  # coverage, features
            est[s] = fixed

    known = [v for v in est.values() if v is not None]
    total = sum(known)
    logger.info("─── pipeline ETA estimate ───")
    for s in stages:
        extra = ""
        if s == "score":
            extra = f"  ({counts['unscored']:,} unscored × {config.ETA_SECS_PER_SCORED_HEADLINE}s)"
        elif s == "embed":
            extra = f"  ({counts['unembedded']:,} unembedded × {config.ETA_SECS_PER_EMBEDDED_HEADLINE}s)"
        elif s == "tune":
            extra = f"  ({config.OPTUNA_TRIALS} trials × {config.ETA_SECS_PER_HPO_TRIAL}s)"
        logger.info("  {:10s} {:>12s}{}", s, fmt_duration(est[s]), extra)
    logger.info("  {:10s} {:>12s}  → ETA {}", "TOTAL", fmt_duration(total), eta_clock(total))
    if any(v is None for v in est.values()):
        logger.info("  (backfill is network-bound and excluded from the total)")
    return est


class StageClock:
    """Tracks real per-stage wall-time and prints a running ETA as stages finish."""

    def __init__(self, estimates: dict[str, float | None]):
        self.est = estimates
        self.actual: dict[str, float] = {}
        self._stage_start: float | None = None
        self._run_start = time.perf_counter()

    def start_stage(self, stage: str) -> None:
        self._stage_start = time.perf_counter()

    def end_stage(self, stage: str, remaining: list[str]) -> None:
        elapsed = time.perf_counter() - (self._stage_start or time.perf_counter())
        self.actual[stage] = elapsed
        remaining_secs = sum((self.est.get(s) or 0) for s in remaining)
        total_elapsed = time.perf_counter() - self._run_start
        logger.info(
            "✓ {} done in {} | elapsed {} | ~remaining {} | ETA {}",
            stage, fmt_duration(elapsed), fmt_duration(total_elapsed),
            fmt_duration(remaining_secs) if remaining else "—",
            eta_clock(remaining_secs) if remaining else "now",
        )
