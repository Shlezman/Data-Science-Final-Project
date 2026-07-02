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
    WHERE nv.validation_passed AND nv.global_sentiment IS NOT NULL
    GROUP BY rh.date ORDER BY rh.date
    """
)
_EDA_SENT_HIST = text(
    """
    SELECT global_sentiment AS bin, COUNT(*) AS n
    FROM nlp_vectors
    WHERE model_name = :model AND validation_passed AND global_sentiment IS NOT NULL
    GROUP BY global_sentiment ORDER BY global_sentiment
    """
)
_EDA_REL_HIST = text(
    """
    SELECT GREATEST(relevance_politics, relevance_economy, relevance_security,
                    relevance_health, relevance_science, relevance_technology) AS bin,
           COUNT(*) AS n
    FROM nlp_vectors WHERE model_name = :model AND validation_passed
      AND COALESCE(relevance_politics, relevance_economy, relevance_security,
                   relevance_health, relevance_science, relevance_technology) IS NOT NULL
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


_BASIS = text("SELECT * FROM embedding_pca_basis ORDER BY created_at DESC LIMIT 1")
_DAY_EMBED = text(
    """
    SELECT he.headline_id, he.dim, he.embedding, rh.source, rh.headline,
           nv.global_sentiment AS sentiment
    FROM headline_embeddings he
    JOIN raw_headlines rh ON rh.id = he.headline_id
    LEFT JOIN nlp_vectors nv ON nv.headline_id = rh.id AND nv.model_name = :model
                             AND nv.validation_passed
    WHERE rh.date = :d AND he.embed_model = :em
    ORDER BY he.headline_id
    LIMIT :cap
    """
)
_DAY_POINT_CAP = 2000


def day_centroid_points(engine=None, *, day) -> dict:
    """Project one day's headline embeddings into the dataset's 16-d ``embpca`` space.

    Uses the persisted leak-safe basis (``embedding_pca_basis``: scaler→PCA fit on the train
    window by ``build_embedding_derived``). Each headline's 768-d vector and the day centroid
    (mean of the raw vectors) go through the SAME transform, so the centroid matches the
    ``embpca_*`` features the models actually consume.

    Returns:
        ``{date, n_pca, points: [{id, source, headline, sentiment, v: [n_pca floats]}],
           centroid: [n_pca floats]}`` — or ``{error}`` when the basis/embeddings are absent.
    """
    import numpy as np

    engine = engine or get_engine()
    with engine.connect() as conn:
        b = conn.execute(_BASIS).mappings().first()
        if b is None:
            return {"date": str(day), "points": [], "centroid": None,
                    "error": "no PCA basis — rerun scripts/build_embedding_derived.py"}
        rows = conn.execute(_DAY_EMBED, {"d": day, "em": b["embed_model"],
                                         "model": ACTIVE_MODEL, "cap": _DAY_POINT_CAP}).mappings().all()
    if not rows:
        return {"date": str(day), "points": [], "centroid": None,
                "error": "no embeddings stored for that date"}

    nf, np_pca = int(b["n_features"]), int(b["n_pca"])
    mean = np.frombuffer(b["scaler_mean"], dtype=np.float32)
    scale = np.frombuffer(b["scaler_scale"], dtype=np.float32).copy()
    scale[scale == 0] = 1.0
    pmean = np.frombuffer(b["pca_mean"], dtype=np.float32)
    comps = np.frombuffer(b["pca_components"], dtype=np.float32).reshape(np_pca, nf)

    vecs = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])

    def _project(x: np.ndarray) -> np.ndarray:
        return ((x - mean) / scale - pmean) @ comps.T

    proj = _project(vecs)
    centroid = _project(vecs.mean(axis=0, keepdims=True))[0]
    points = [{"id": int(r["headline_id"]), "source": r["source"], "headline": r["headline"],
               "sentiment": (None if r["sentiment"] is None else int(r["sentiment"])),
               "v": [round(float(x), 4) for x in proj[i]]}
              for i, r in enumerate(rows)]
    return {"date": str(day), "n_pca": np_pca, "points": points,
            "centroid": [round(float(x), 4) for x in centroid]}


_PERSONA_SOURCES = text(
    """
    SELECT rh.source AS source, COUNT(*) AS n,
           AVG(nv.global_sentiment)::float AS mean_sentiment
    FROM raw_headlines rh
    JOIN nlp_vectors nv ON nv.headline_id = rh.id AND nv.model_name = :model
                        AND nv.validation_passed AND nv.global_sentiment IS NOT NULL
    WHERE rh.date = :d
    GROUP BY rh.source
    HAVING COUNT(*) >= :min_n
    ORDER BY COUNT(*) DESC
    LIMIT :top
    """
)
_PERSONA_PRED = text(
    """
    SELECT model_version, prediction, confidence, actual
    FROM model_predictions WHERE date = :d
    ORDER BY created_at DESC LIMIT 1
    """
)
# Mean global sentiment (−10..+10) beyond ±0.5 reads as a directional stance; inside is noise.
_PERSONA_THRESHOLD = 0.5


def _vote(mean_sentiment: float) -> str:
    """Maps a persona's mean sentiment to an up/down/neutral stance."""
    if mean_sentiment >= _PERSONA_THRESHOLD:
        return "up"
    if mean_sentiment <= -_PERSONA_THRESHOLD:
        return "down"
    return "neutral"


def persona_votes(engine=None, *, day, top: int = 12, min_n: int = 3) -> dict:
    """Per-source "personas" for a day: each provider's sentiment stance (up/down/neutral),
    a General persona over all sources, the served model's prediction, and the settled actual.

    Returns:
        ``{date, personas: [{source, n, mean_sentiment, vote}], general: {...},
           model: {model_version, prediction, confidence} | None, actual: bool | None}``
    """
    engine = engine or get_engine()
    with engine.connect() as conn:
        rows = conn.execute(_PERSONA_SOURCES,
                            {"d": day, "model": ACTIVE_MODEL, "min_n": min_n, "top": top}
                            ).mappings().all()
        pred = conn.execute(_PERSONA_PRED, {"d": day}).mappings().first()
    personas = [{"source": r["source"], "n": int(r["n"]),
                 "mean_sentiment": round(float(r["mean_sentiment"]), 3),
                 "vote": _vote(float(r["mean_sentiment"]))} for r in rows]
    general = None
    if personas:
        tot = sum(p["n"] for p in personas)
        gmean = sum(p["mean_sentiment"] * p["n"] for p in personas) / tot
        general = {"source": "General (all sources)", "n": tot,
                   "mean_sentiment": round(gmean, 3), "vote": _vote(gmean)}
    model = None
    actual = None
    if pred:
        model = {"model_version": pred["model_version"], "prediction": bool(pred["prediction"]),
                 "confidence": round(float(pred["confidence"]), 4)}
        actual = None if pred["actual"] is None else bool(pred["actual"])
    return {"date": str(day), "personas": personas, "general": general,
            "model": model, "actual": actual}


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
