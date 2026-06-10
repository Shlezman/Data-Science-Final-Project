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

from sentisense.config import CLUSTER_K, CLUSTER_REFIT_EVERY, SEED
from sentisense.embed import load_embeddings


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

    meta = meta.reset_index(drop=True)
    order = np.argsort(meta["date"].values, kind="stable")
    meta = meta.iloc[order].reset_index(drop=True)
    vectors = vectors[order]
    days = pd.DatetimeIndex(sorted(meta["date"].unique()))

    # Precompute row index ranges per day for fast slicing.
    day_to_rows = {d: np.where(meta["date"].values == d)[0] for d in days}

    model: MiniBatchKMeans | None = None
    days_since_fit = 10**9
    rows_out: list[dict] = []

    for i, d in enumerate(days):
        # (Re)fit on STRICTLY-earlier embeddings only — causal, no day-d leak.
        past_mask = meta["date"].values < d
        n_past = int(past_mask.sum())
        if n_past >= k and (model is None or days_since_fit >= refit_every):
            model = MiniBatchKMeans(n_clusters=k, random_state=SEED, n_init=3, batch_size=256)
            model.fit(vectors[past_mask])
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
    logger.info("Narrative features built for {:,} days (k={}, refit_every={}).",
                len(out), k, refit_every)
    return out
