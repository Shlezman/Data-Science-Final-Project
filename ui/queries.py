"""Read-only DB queries + confusion-matrix math for the UI (reuses the live Postgres).

Only stable column names are referenced (``global_sentiment``, ``validation_passed``,
``model_name``) so the headline views don't break on per-category relevance naming. The
confusion matrix is computed directly from ``model_predictions`` (``prediction`` vs ``actual``)
— the financial direction outcome, not the golden-dataset LLM-scoring eval.
"""

from __future__ import annotations

import math
import os

from sqlalchemy import text

from sentisense.db import get_engine

ACTIVE_MODEL = os.environ.get("SENTISENSE_ACTIVE_MODEL", "mistral-small-4")

_LATEST_DATE = text("SELECT MAX(date) AS d FROM raw_headlines")

_HEADLINES_FOR_DATE = text(
    """
    SELECT rh.id, rh.date, rh.source, rh.hour, rh.headline,
           nv.global_sentiment, nv.validation_passed,
           (nv.headline_id IS NOT NULL) AS scored
    FROM raw_headlines rh
    LEFT JOIN nlp_vectors nv
           ON nv.headline_id = rh.id AND nv.model_name = :model
    WHERE rh.date = :d
    ORDER BY rh.hour DESC NULLS LAST, rh.id DESC
    OFFSET :offset LIMIT :limit
    """
)
_COUNT_FOR_DATE = text("SELECT COUNT(*) AS n FROM raw_headlines WHERE date = :d")
_DISTINCT_DATES = text(
    "SELECT DISTINCT date FROM raw_headlines ORDER BY date DESC OFFSET :offset LIMIT :limit"
)
_PREDICTIONS = text(
    """
    SELECT date, model_version, prediction, confidence, actual
    FROM model_predictions
    WHERE (:version IS NULL OR model_version = :version)
    ORDER BY date DESC
    LIMIT :limit
    """
)


def latest_date(engine=None):
    """Most recent date present in ``raw_headlines`` (None if empty)."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        return conn.execute(_LATEST_DATE).scalar()


def headlines_for_date(engine=None, *, day, page: int = 0, page_size: int = 50) -> dict:
    """Paginated headlines for one date (+ total count), with the active model's sentiment."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        total = conn.execute(_COUNT_FOR_DATE, {"d": day}).scalar() or 0
        rows = conn.execute(_HEADLINES_FOR_DATE, {
            "model": ACTIVE_MODEL, "d": day, "offset": page * page_size, "limit": page_size,
        }).mappings().all()
    return {"date": str(day), "page": page, "page_size": page_size, "total": int(total),
            "headlines": [dict(r) for r in rows]}


def available_dates(engine=None, *, page: int = 0, page_size: int = 60) -> list[str]:
    """Distinct headline dates, newest first (for the archive date picker)."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        rows = conn.execute(_DISTINCT_DATES, {"offset": page * page_size, "limit": page_size}).all()
    return [str(r[0]) for r in rows]


def prediction_rows(engine=None, *, version=None, limit: int = 365) -> list[dict]:
    """Recent rows from ``model_predictions`` (optionally one model_version)."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        rows = conn.execute(_PREDICTIONS, {"version": version, "limit": limit}).mappings().all()
    return [dict(r) for r in rows]


def confusion_matrix(rows: list[dict]) -> dict:
    """Confusion matrix + accuracy/precision/recall/F1/MCC over SETTLED predictions.

    ``positive`` class = predicted/actual UP. Rows with ``actual IS NULL`` count as pending and
    are excluded from the matrix. Returns counts + derived metrics (all rounded).
    """
    settled = [r for r in rows if r.get("actual") is not None]
    tp = sum(1 for r in settled if r["prediction"] and r["actual"])
    tn = sum(1 for r in settled if not r["prediction"] and not r["actual"])
    fp = sum(1 for r in settled if r["prediction"] and not r["actual"])
    fn = sum(1 for r in settled if not r["prediction"] and r["actual"])
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": n, "pending": len(rows) - len(settled),
            "accuracy": round(acc, 4), "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "mcc": round(mcc, 4)}
