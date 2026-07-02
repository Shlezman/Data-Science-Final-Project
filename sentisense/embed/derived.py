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
_BASIS_MIGRATION = REPO_ROOT / "sentisense" / "db" / "migrations" / "007_embedding_basis.sql"
_FAR_FUTURE = dt.date(2100, 1, 1)


def fit_transform_derived(centroid_by_date: pd.DataFrame, *, fit_cutoff,
                          n_pca: int, n_clusters: int, seed: int = SEED,
                          return_basis: bool = False):
    """Fit scaler→PCA→KMeans on dates ≤ ``fit_cutoff``; transform ALL dates (leak-safe).

    Args:
        centroid_by_date: date-indexed DataFrame of the raw ``embc_*`` daily centroid.
        fit_cutoff: only rows with index ≤ this train the basis (the leakage boundary).
        n_pca: PCA components (capped to ≤ n_features and ≤ n_train_samples).
        n_clusters: KMeans clusters → that many centroid-distance features.
        seed: RNG seed for PCA/KMeans determinism.
        return_basis: also return the fitted scaler/PCA arrays (for ``persist_basis`` so
            the UI can project per-headline embeddings into the same embpca space).

    Returns:
        Date-indexed DataFrame (aligned to the input index) with columns
        ``embpca_000..`` and ``embclus_dist_0..`` — or ``(frame, basis_dict)`` when
        ``return_basis=True``.

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
    out = pca_df.join(dist_df)
    if not return_basis:
        return out
    basis = {
        "n_features": int(X.shape[1]), "n_pca": int(p),
        "fit_cutoff": pd.Timestamp(fit_cutoff).date(),
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "pca_mean": pca.mean_.astype(np.float32),
        "pca_components": pca.components_.astype(np.float32),   # (n_pca, n_features)
        "n_clusters": int(n_clusters),
        "kmeans_centers": km.cluster_centers_.astype(np.float32),   # (k, n_features), SCALED space
    }
    return out, basis


def _split_sql(ddl: str) -> list[str]:
    """Split a migration into statements, ignoring ';' that appear inside ``--`` comments.

    A naive ``ddl.split(';')`` breaks on a semicolon inside a line comment (e.g.
    ``-- Idempotent (IF NOT EXISTS); ...``), leaking the comment tail out as bare SQL. Strip
    each line's ``--`` comment first, then split. (Our migrations have no ``--`` inside string
    literals, so this is safe.)
    """
    stripped = "\n".join(line.split("--", 1)[0] for line in ddl.splitlines())
    return [s.strip() for s in stripped.split(";") if s.strip()]


def ensure_derived_table(engine=None) -> None:
    """Apply the derived-features migration (idempotent CREATE TABLE IF NOT EXISTS)."""
    engine = engine or get_engine()
    with engine.begin() as conn:
        for stmt in _split_sql(_MIGRATION.read_text(encoding="utf-8")):
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


_BASIS_UPSERT = text(
    """
    INSERT INTO embedding_pca_basis
        (embed_model, n_features, n_pca, fit_cutoff, scaler_mean, scaler_scale, pca_mean,
         pca_components, n_clusters, kmeans_centers)
    VALUES (:model, :n_features, :n_pca, :fit_cutoff, :scaler_mean, :scaler_scale, :pca_mean,
            :pca_components, :n_clusters, :kmeans_centers)
    ON CONFLICT (embed_model) DO UPDATE
        SET n_features = EXCLUDED.n_features, n_pca = EXCLUDED.n_pca,
            fit_cutoff = EXCLUDED.fit_cutoff, scaler_mean = EXCLUDED.scaler_mean,
            scaler_scale = EXCLUDED.scaler_scale, pca_mean = EXCLUDED.pca_mean,
            pca_components = EXCLUDED.pca_components, n_clusters = EXCLUDED.n_clusters,
            kmeans_centers = EXCLUDED.kmeans_centers, created_at = NOW()
    """
)


def persist_basis(basis: dict, *, engine=None, model: str = EMBED_MODEL) -> None:
    """Upsert the fitted scaler→PCA basis (float32 BYTEA) so the UI can project headlines.

    Args:
        basis: the dict returned by ``fit_transform_derived(..., return_basis=True)``.
        engine: SQLAlchemy engine; created from env if None.
        model: embedding model name the basis belongs to.
    """
    engine = engine or get_engine()
    with engine.begin() as conn:
        for stmt in _split_sql(_BASIS_MIGRATION.read_text(encoding="utf-8")):
            conn.execute(text(stmt))
        conn.execute(_BASIS_UPSERT, {
            "model": model, "n_features": basis["n_features"], "n_pca": basis["n_pca"],
            "fit_cutoff": basis["fit_cutoff"],
            "scaler_mean": basis["scaler_mean"].tobytes(),
            "scaler_scale": basis["scaler_scale"].tobytes(),
            "pca_mean": basis["pca_mean"].tobytes(),
            "pca_components": basis["pca_components"].tobytes(),
            "n_clusters": basis.get("n_clusters"),
            "kmeans_centers": (basis["kmeans_centers"].tobytes()
                               if basis.get("kmeans_centers") is not None else None),
        })
    logger.info("Persisted PCA basis (model={}, {}→{} dims, fit_cutoff={})",
                model, basis["n_features"], basis["n_pca"], basis["fit_cutoff"])


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
