"""Per-trading-day narrative features from headline embeddings — leakage-safe.

For each trading day T, the cluster model is fit ONLY on embeddings strictly before
T (expanding window with a refit cadence), then day T's headlines are *assigned* with
that past-fit model. So no day-T or future information enters the clustering that
labels day T — the prompt's hard "fit only on data up to T" rule.

Output (indexed by date):
  * ``dominant_cluster_ratio`` — share of the day's headlines in its most common cluster
    (high = one narrative dominates the day; low = fragmented news).
  * ``cluster_entropy`` — Shannon entropy of the day's cluster distribution (normalised).
  * ``narrative_n_headlines`` — embedded headlines that day (context for the ratio).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from sentisense.config import CLUSTER_K, CLUSTER_REFIT_EVERY, EMBED_MODEL, SEED
from sentisense.constants import REPO_ROOT
from sentisense.embed import load_embeddings

# Cap the per-refit fit sample. MiniBatchKMeans converges fine on a sample, so we never
# materialise/scan the full (multi-GB, growing) past-embedding matrix on each of the
# ~thousands of refits — that recopying was the hours-long bottleneck.
_FIT_SAMPLE_CAP = 50_000
_PROGRESS_EVERY = 500
_CACHE_DIR = REPO_ROOT / "sentisense_cache"


def _cache_path(n_embeddings: int, k: int, refit_every: int):
    """Disk-cache key for the computed features (so tune/final never recompute).

    CSV (not pickle): a small date-indexed numeric frame — no arbitrary-object
    deserialization, safe to read back even though we're also the only writer.
    """
    safe = EMBED_MODEL.replace("/", "_")
    return _CACHE_DIR / f"narrative_{safe}_n{n_embeddings}_k{k}_r{refit_every}.csv"


def _entropy(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    p = p[p > 0]
    h = -(p * np.log(p)).sum()
    return float(h / np.log(len(p))) if len(p) > 1 else 0.0


def build_narrative_features(engine=None, *, k: int = CLUSTER_K,
                             refit_every: int = CLUSTER_REFIT_EVERY) -> pd.DataFrame:
    """Compute causal per-day narrative features from cached embeddings.

    Returns:
        DataFrame indexed by date with dominant_cluster_ratio / cluster_entropy /
        narrative_n_headlines. Empty if no embeddings are cached.
    """
    from sklearn.cluster import MiniBatchKMeans

    meta, vectors = load_embeddings(engine)
    if len(meta) == 0:
        logger.warning("No embeddings cached — run sentisense.embed.embeddings first. "
                       "Returning empty narrative features.")
        return pd.DataFrame()

    # Reuse a prior computation if the embedding set + params are unchanged — tune and
    # final both call this, and it's the slow stage.
    cache = _cache_path(len(meta), k, refit_every)
    if cache.exists():
        logger.info("Narrative features: loading cached {} ({:,} embeddings).",
                    cache.name, len(meta))
        return pd.read_csv(cache, index_col="date", parse_dates=["date"])

    meta = meta.reset_index(drop=True)
    order = np.argsort(meta["date"].values, kind="stable")
    meta = meta.iloc[order].reset_index(drop=True)
    vectors = vectors[order]
    date_vals = meta["date"].values            # sorted ascending after the argsort
    days = pd.DatetimeIndex(sorted(meta["date"].unique()))
    logger.info("Narrative clustering over {:,} days / {:,} embeddings (k={}, refit_every={}) …",
                len(days), len(meta), k, refit_every)

    # Row index ranges per day for fast slicing.
    day_to_rows = {d: np.where(date_vals == d)[0] for d in days}

    rng = np.random.default_rng(SEED)
    model: MiniBatchKMeans | None = None
    days_since_fit = 10**9
    rows_out: list[dict] = []

    for i, d in enumerate(days):
        if i % _PROGRESS_EVERY == 0:
            logger.info("  narrative clustering: day {:,}/{:,}", i, len(days))
        # Count of STRICTLY-earlier embeddings (sorted → O(log n), no full-array scan).
        n_past = int(np.searchsorted(date_vals, np.datetime64(d), side="left"))
        if n_past >= k and (model is None or days_since_fit >= refit_every):
            # Fit on a bounded random sample of the strictly-past rows (causal): caps
            # cost regardless of how large the prefix has grown.
            if n_past > _FIT_SAMPLE_CAP:
                fit_idx = rng.choice(n_past, _FIT_SAMPLE_CAP, replace=False)
            else:
                fit_idx = np.arange(n_past)
            model = MiniBatchKMeans(n_clusters=k, random_state=SEED, n_init=3, batch_size=256)
            model.fit(vectors[fit_idx])
            days_since_fit = 0
        else:
            days_since_fit += 1

        idx = day_to_rows[d]
        if model is None or len(idx) == 0:
            rows_out.append({"date": d, "dominant_cluster_ratio": np.nan,
                             "cluster_entropy": np.nan, "narrative_n_headlines": len(idx)})
            continue

        labels = model.predict(vectors[idx])
        counts = np.bincount(labels, minlength=k).astype(float)
        counts = counts[counts > 0]
        ratio = counts.max() / counts.sum()
        rows_out.append({"date": d, "dominant_cluster_ratio": float(ratio),
                         "cluster_entropy": _entropy(counts),
                         "narrative_n_headlines": int(len(idx))})

    out = pd.DataFrame(rows_out).set_index("date").sort_index()
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache)
    logger.info("Narrative features built for {:,} days (k={}, refit_every={}) — cached → {}.",
                len(out), k, refit_every, cache.name)
    return out
