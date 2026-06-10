"""Single source of truth for project-wide constants.

Centralises the hard cutoff, the active model name, the score-column contract, and
the cross-layer column-name mapping so no magic strings leak into feature/model code.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# Hard data cutoff — see module docstring in sentisense/__init__.py.
# Applied as a WHERE clause on raw_headlines.date / daily_features.date in
# EVERY ingest, query, and feature step. NEVER use created_at (ingestion time).
# ─────────────────────────────────────────────────────────────────────
CUTOFF_DATE: _dt.date = _dt.date(2023, 10, 7)
CUTOFF_DATE_ISO: str = CUTOFF_DATE.isoformat()  # "2023-10-07"

# ─────────────────────────────────────────────────────────────────────
# Active LLM model whose scores we train on. The DB may hold legacy rows
# (mistral-large-2, mistral-small3.2); every analytical query MUST filter
# model_name = ACTIVE_MODEL_NAME or it double-counts headlines.
# Overridable via env for forward-compat, but defaults to the standardised model.
# ─────────────────────────────────────────────────────────────────────
import os as _os

# ─────────────────────────────────────────────────────────────────────
# TWO distinct model identities — do NOT conflate them:
#
#   * DATASET / READ model — the model_name the analytical queries (features,
#     embeddings join, coverage, ETA) train on. Should be whatever model actually
#     scored the corpus. Resolved at runtime by resolve_active_model() (auto-detect
#     the most-populated model), env-overridable via SENTISENSE_ACTIVE_MODEL.
#
#   * SCORING model — what a NEW score-run writes, derived from the LLM backend
#     (local ollama → qwen2.5:14b; openai → mistral-small-4). See scoring_model_name().
#
# Conflating them is exactly what made an already-scored (mistral-small-4) corpus look
# "unscored" while running a local qwen2.5:14b backend.
# ─────────────────────────────────────────────────────────────────────

# Static fallback for the DATASET (read) model — used only when there is no override
# and the DB is empty/unavailable. resolve_active_model() is the real entry point.
ACTIVE_MODEL_NAME: str = _os.environ.get("SENTISENSE_ACTIVE_MODEL", "mistral-small-4")


def scoring_model_name() -> str:
    """The model a NEW score-run writes, mirroring process_headlines.get_active_model_name()."""
    backend = _os.environ.get("SENTISENSE_LLM_BACKEND", "ollama").lower()
    if backend == "openai":
        return _os.environ.get("SENTISENSE_OPENAI_MODEL", "mistral-large-2")
    return _os.environ.get("SENTISENSE_OLLAMA_MODEL", "qwen2.5:14b")


def resolve_active_model(engine, *, fallback: str = ACTIVE_MODEL_NAME) -> str:
    """Resolve the DATASET (read) model for the analytical queries.

    Priority:
      1. ``SENTISENSE_ACTIVE_MODEL`` env override (explicit operator choice).
      2. The ``model_name`` with the most ``validation_passed=TRUE`` rows in
         ``nlp_vectors`` — so an already-scored corpus is used as-is, regardless of
         which LLM backend is configured locally.
      3. ``fallback`` (mistral-small-4) when the DB is empty/unavailable.
    """
    override = _os.environ.get("SENTISENSE_ACTIVE_MODEL")
    if override:
        return override
    try:
        from sqlalchemy import text

        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT model_name, COUNT(*) AS n FROM nlp_vectors "
                    "WHERE validation_passed = TRUE "
                    "GROUP BY model_name ORDER BY n DESC LIMIT 1"
                )
            ).first()
        return row[0] if row and row[0] else fallback
    except Exception:
        return fallback

# ─────────────────────────────────────────────────────────────────────
# Score-column contract. DB column order is canonical (matches init_db.sql and
# the positional INSERT in scripts/process_headlines.py).
# ─────────────────────────────────────────────────────────────────────
DB_RELEVANCE_COLUMNS: tuple[str, ...] = (
    "relevance_politics",
    "relevance_economy",
    "relevance_security",
    "relevance_health",
    "relevance_science",
    "relevance_technology",
)
SENTIMENT_COLUMN: str = "global_sentiment"
SCORE_COLUMNS: tuple[str, ...] = (*DB_RELEVANCE_COLUMNS, SENTIMENT_COLUMN)

# Cross-layer name map (DB → engine result dict → golden CSV). Documented in
# docs/sentisense-understanding.md §1. Use when bridging to the scoring pipeline
# or the golden dataset, never to rename DB columns silently.
DB_TO_ENGINE_KEY: dict[str, str] = {
    "relevance_politics": "relevance_category_1",
    "relevance_economy": "relevance_category_2",
    "relevance_security": "relevance_category_3",
    "relevance_health": "relevance_category_4",
    "relevance_science": "relevance_category_5",
    "relevance_technology": "relevance_category_6",
    "global_sentiment": "global_sentiment",
}
DB_TO_GOLDEN_KEY: dict[str, str] = {
    "relevance_politics": "politics_government",
    "relevance_economy": "economy_finance",
    "relevance_security": "security_military",
    "relevance_health": "health_medicine",
    "relevance_science": "science_climate",
    "relevance_technology": "technology",
}

# Score bounds (mirror processing_engine.config).
RELEVANCE_MIN, RELEVANCE_MAX = 0, 10
SENTIMENT_MIN, SENTIMENT_MAX = -10, 10

# ─────────────────────────────────────────────────────────────────────
# Repo paths. This file is sentisense/constants.py → repo root is two levels up.
# ─────────────────────────────────────────────────────────────────────
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
TA125_CSV: Path = REPO_ROOT / "TA 125 Historical Data.csv"
VTA35_CSV: Path = REPO_ROOT / "Tel Aviv Volatility Index VTA35 Historical Data.csv"
REPORTS_DIR: Path = REPO_ROOT / "sentisense_reports"

# VTA-35 (Israeli volatility index) inception — earlier values must be NaN-masked.
VTA35_INCEPTION: _dt.date = _dt.date(2019, 7, 17)


def parse_iso_date(value: str) -> _dt.date:
    """Parse a ``YYYY-MM-DD`` string into a date, raising a clear error otherwise.

    Used to validate operator-supplied date CLI args at the boundary.

    Args:
        value: A date string expected in ISO ``YYYY-MM-DD`` form.

    Returns:
        The parsed :class:`datetime.date`.

    Raises:
        ValueError: If ``value`` is not a valid ISO date.
    """
    return _dt.date.fromisoformat(value)
