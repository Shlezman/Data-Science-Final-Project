"""Leak-safe PCA + cluster-distance features derived from the daily e5 centroid.

These are *extra* per-day features (on top of the raw ``embc_*`` centroid): a compact PCA of
the centroid (``embpca_*``) plus the distance from the day's centroid to each KMeans cluster
centroid (``embclus_dist_*``). The transform basis (StandardScaler → PCA → KMeans) is fit
ONCE on the daily centroids of a TRAIN window (dates ≤ ``fit_cutoff``) and then applied to
EVERY date, so the features for any later out-of-sample window never see their own basis —
the same leakage contract the rest of the pipeline honours. Persisted to
``daily_embedding_derived`` (one JSONB row per date) so the model builders join them cheaply.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import inspect, text

from sentisense.config import EMBED_MODEL, SEED
from sentisense.constants import REPO_ROOT
from sentisense.db import get_engine

DERIVED_TABLE = "daily_embedding_derived"
_MIGRATION = REPO_ROOT / "sentisense" / "db" / "migrations" / "004_embedding_derived.sql"
_FAR_FUTURE = dt.date(2100, 1, 1)


def fit_transform_derived(centroid_by_date: pd.DataFrame, *, fit_cutoff,
                          n_pca: int, n_clusters: int, seed: int = SEED) -> pd.DataFrame:
    """Fit scaler→PCA→KMeans on dates ≤ ``fit_cutoff``; transform ALL dates (leak-safe).

    Args:
        centroid_by_date: date-indexed DataFrame of the raw ``embc_*`` daily centroid.
        fit_cutoff: only rows with index ≤ this train the basis (the leakage boundary).
        n_pca: PCA components (capped to ≤ n_features and ≤ n_train_samples).
        n_clusters: KMeans clusters → that many centroid-distance features.
        seed: RNG seed for PCA/KMeans determinism.

    Returns:
        Date-indexed DataFrame (aligned to the input index) with columns
        ``embpca_000..`` and ``embclus_dist_0..``.

    Raises:
        ValueError: if the train window has too few days for the requested dims.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    idx = centroid_by_date.index
    X = centroid_by_date.to_numpy(dtype=np.float64)
    fit_mask = np.asarray(idx <= pd.Timestamp(fit_cutoff))
    n_fit = int(fit_mask.sum())
    if n_fit < max(n_clusters, n_pca) + 1:
        raise ValueError(
            f"too few train days ({n_fit}) for n_pca={n_pca}, n_clusters={n_clusters} — "
            "lower them or widen fit_cutoff")

    scaler = StandardScaler().fit(X[fit_mask])
    Xs = scaler.transform(X)
    p = min(n_pca, X.shape[1], n_fit)
    pca = PCA(n_components=p, random_state=seed).fit(Xs[fit_mask])
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(Xs[fit_mask])

    pca_df = pd.DataFrame(pca.transform(Xs), index=idx,
                          columns=[f"embpca_{i:03d}" for i in range(p)])
    dist_df = pd.DataFrame(km.transform(Xs), index=idx,
                           columns=[f"embclus_dist_{i}" for i in range(n_clusters)])
    logger.info("Derived basis fit on {} train days (≤ {}); {} PCA + {} cluster-dist over {} days",
                n_fit, pd.Timestamp(fit_cutoff).date(), p, n_clusters, len(idx))
    return pca_df.join(dist_df)


def ensure_derived_table(engine=None) -> None:
    """Apply the derived-features migration (idempotent CREATE TABLE IF NOT EXISTS)."""
    engine = engine or get_engine()
    ddl = _MIGRATION.read_text(encoding="utf-8")
    with engine.begin() as conn:
        for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
            conn.execute(text(stmt))


_UPSERT_SQL = text(
    """
    INSERT INTO daily_embedding_derived (date, embed_model, features, n_pca, n_clusters, fit_cutoff)
    VALUES (:date, :model, :features, :n_pca, :n_clusters, :fit_cutoff)
    ON CONFLICT (date, embed_model) DO UPDATE
        SET features = EXCLUDED.features, n_pca = EXCLUDED.n_pca,
            n_clusters = EXCLUDED.n_clusters, fit_cutoff = EXCLUDED.fit_cutoff,
            created_at = NOW()
    """
)


def persist_derived(derived: pd.DataFrame, *, fit_cutoff, n_pca: int, n_clusters: int,
                    engine=None, model: str = EMBED_MODEL) -> int:
    """Upsert the derived per-date features as JSONB rows. Returns the row count."""
    engine = engine or get_engine()
    ensure_derived_table(engine)
    rows = [{
        "date": (d.date() if hasattr(d, "date") else d),
        "model": model,
        "features": json.dumps({c: float(v) for c, v in derived.loc[d].items()}),
        "n_pca": int(n_pca),
        "n_clusters": int(n_clusters),
        "fit_cutoff": pd.Timestamp(fit_cutoff).date(),
    } for d in derived.index]
    with engine.begin() as conn:
        conn.execute(_UPSERT_SQL, rows)
    logger.info("Persisted {} derived rows (model={}, fit_cutoff={})",
                len(rows), model, pd.Timestamp(fit_cutoff).date())
    return len(rows)


_LOAD_SQL = text(
    "SELECT date, features FROM daily_embedding_derived "
    "WHERE embed_model = :model AND date <= :cutoff ORDER BY date"
)


def load_embedding_derived(engine=None, cutoff=None, model: str = EMBED_MODEL) -> pd.DataFrame:
    """Load derived features as a date-indexed wide frame. Empty frame if table absent/empty.

    Used by the dataset builders to join ``embpca_*``/``embclus_dist_*`` as extra columns; a
    missing table (derived step not yet run) degrades gracefully to no extra features.
    """
    engine = engine or get_engine()
    if not inspect(engine).has_table(DERIVED_TABLE):
        return pd.DataFrame()
    cutoff = cutoff if cutoff is not None else _FAR_FUTURE
    with engine.connect() as conn:
        df = pd.read_sql(_LOAD_SQL, conn, params={"model": model, "cutoff": cutoff})
    if df.empty:
        return pd.DataFrame()
    parsed = df["features"].apply(lambda s: s if isinstance(s, dict) else json.loads(s))
    feats = pd.json_normalize(parsed)
    feats.index = pd.to_datetime(df["date"])
    return feats
