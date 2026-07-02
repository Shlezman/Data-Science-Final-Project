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
    WHERE (CAST(:version AS text) IS NULL OR model_version = :version)
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


_FULL_EVAL = text(
    """
    SELECT date, model_version, prediction, actual
    FROM champion_full_eval
    WHERE (CAST(:version AS text) IS NULL OR model_version = :version)
    ORDER BY date
    """
)
_TODAY_PRED = text(
    """
    SELECT date, model_version, prediction, confidence
    FROM model_predictions
    ORDER BY date DESC, created_at DESC
    LIMIT 1
    """
)
_EDA_VOLUME = text("SELECT date, COUNT(*) AS n FROM raw_headlines GROUP BY date ORDER BY date")
_EDA_SENT_TS = text(
    """
    SELECT rh.date AS date, AVG(nv.global_sentiment)::float AS mean_sentiment
    FROM raw_headlines rh
    JOIN nlp_vectors nv ON nv.headline_id = rh.id AND nv.model_name = :model
    WHERE nv.validation_passed
    GROUP BY rh.date ORDER BY rh.date
    """
)
_EDA_SENT_HIST = text(
    """
    SELECT global_sentiment AS bin, COUNT(*) AS n
    FROM nlp_vectors WHERE model_name = :model AND validation_passed
    GROUP BY global_sentiment ORDER BY global_sentiment
    """
)
_EDA_REL_HIST = text(
    """
    SELECT GREATEST(relevance_politics, relevance_economy, relevance_security,
                    relevance_health, relevance_science, relevance_technology) AS bin,
           COUNT(*) AS n
    FROM nlp_vectors WHERE model_name = :model AND validation_passed
    GROUP BY bin ORDER BY bin
    """
)
_EDA_VALIDATION = text(
    """
    SELECT COUNT(*) FILTER (WHERE validation_passed) AS passed,
           COUNT(*) FILTER (WHERE NOT validation_passed) AS failed
    FROM nlp_vectors WHERE model_name = :model
    """
)
# 15 pairwise Pearson correlations among the 6 relevance categories (one table pass).
_CORR_COLS = ["relevance_politics", "relevance_economy", "relevance_security",
              "relevance_health", "relevance_science", "relevance_technology"]
_CORR_LABELS = ["politics", "economy", "security", "health", "science", "technology"]
_EDA_CORR = text(
    "SELECT " + ", ".join(
        f"corr({_CORR_COLS[i]}, {_CORR_COLS[j]}) AS c{i}{j}"
        for i in range(6) for j in range(i + 1, 6)
    ) + " FROM nlp_vectors WHERE model_name = :model AND validation_passed"
)
_CENTROIDS = text(
    """
    SELECT d.date AS date,
           (d.features->>'embpca_000')::float AS x,
           (d.features->>'embpca_001')::float AS y,
           (d.features->>'embpca_002')::float AS z,
           e.actual AS actual,
           COALESCE(v.n, 0) AS n_headlines
    FROM daily_embedding_derived d
    LEFT JOIN champion_full_eval e ON e.date = d.date
    LEFT JOIN (SELECT date, COUNT(*) AS n FROM raw_headlines GROUP BY date) v ON v.date = d.date
    WHERE d.embed_model = (
        SELECT embed_model FROM daily_embedding_derived
        GROUP BY embed_model ORDER BY COUNT(*) DESC LIMIT 1)
      AND d.features ? 'embpca_000' AND d.features ? 'embpca_001' AND d.features ? 'embpca_002'
    ORDER BY d.date
    """
)


def full_eval_rows(engine=None, *, version=None) -> list[dict]:
    """All ``champion_full_eval`` rows (in-sample, all labeled days), optionally one version."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        rows = conn.execute(_FULL_EVAL, {"version": version}).mappings().all()
    return [dict(r) for r in rows]


def today_prediction(engine=None) -> dict | None:
    """The most recent ``model_predictions`` row → ``{date, up, confidence, model_version}``."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        row = conn.execute(_TODAY_PRED).mappings().first()
    if not row:
        return None
    return {"date": str(row["date"]), "up": bool(row["prediction"]),
            "confidence": round(float(row["confidence"]), 4), "model_version": row["model_version"]}


def eda_aggregates(engine=None) -> dict:
    """Server-side EDA aggregates for the dashboard panels (efficient SQL, not full-table pandas)."""
    engine = engine or get_engine()
    m = {"model": ACTIVE_MODEL}
    with engine.connect() as conn:
        volume = [{"date": str(r["date"]), "count": int(r["n"])}
                  for r in conn.execute(_EDA_VOLUME).mappings()]
        sent_ts = [{"date": str(r["date"]), "mean_sentiment": round(float(r["mean_sentiment"]), 3)}
                   for r in conn.execute(_EDA_SENT_TS, m).mappings()]
        sent_hist = [{"bin": int(r["bin"]), "count": int(r["n"])}
                     for r in conn.execute(_EDA_SENT_HIST, m).mappings()]
        rel_hist = [{"bin": int(r["bin"]), "count": int(r["n"])}
                    for r in conn.execute(_EDA_REL_HIST, m).mappings()]
        val = conn.execute(_EDA_VALIDATION, m).mappings().first() or {"passed": 0, "failed": 0}
        corr = conn.execute(_EDA_CORR, m).mappings().first()
    matrix = [[1.0 if i == j else None for j in range(6)] for i in range(6)]
    for i in range(6):
        for j in range(i + 1, 6):
            v = corr[f"c{i}{j}"] if corr else None
            v = round(float(v), 3) if v is not None else None
            matrix[i][j] = matrix[j][i] = v
    passed, failed = int(val["passed"] or 0), int(val["failed"] or 0)
    total = passed + failed
    return {"volume": volume, "sentiment_ts": sent_ts, "sentiment_hist": sent_hist,
            "relevance_hist": rel_hist, "category_corr": {"labels": _CORR_LABELS, "matrix": matrix},
            "validation": {"passed": passed, "failed": failed,
                           "rate": round(passed / total, 4) if total else 0.0}}


def centroid_points(engine=None) -> dict:
    """Per-day 3D news centroids (leak-safe embpca_000..002) + actual up/down + headline count."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        rows = conn.execute(_CENTROIDS).mappings().all()
    points = [{"date": str(r["date"]), "x": float(r["x"]), "y": float(r["y"]), "z": float(r["z"]),
               "actual": (None if r["actual"] is None else bool(r["actual"])),
               "n_headlines": int(r["n_headlines"])} for r in rows]
    return {"points": points}


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
