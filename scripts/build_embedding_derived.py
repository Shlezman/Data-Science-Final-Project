"""Build the leak-safe derived embedding features (PCA + cluster distances) into Postgres.

Pulls the daily e5 centroid (streamed, OOM-safe), fits a StandardScaler→PCA→KMeans basis on a
TRAIN window only, applies it to every date, and upserts the result into
``daily_embedding_derived``. The dataset builders then join ``embpca_*``/``embclus_dist_*`` as
extra features automatically.

Leakage boundary: the basis is fit on dates ≤ ``--fit-cutoff``. The default is the
``EMBED_DERIVED_TRAIN_FRAC`` (0.85) quantile date of the ≤ CUTOFF modeling corpus, which
precedes both the CUT and FULL last-15% out-of-sample windows, so no OOS row ever influences
its own transform. Coverage (which dates get rows) is independent — by default all dates.

Run (server-side, from repo root):
    uv run --extra ml python scripts/build_embedding_derived.py
    uv run --extra ml python scripts/build_embedding_derived.py --dry-run
    uv run --extra ml python scripts/build_embedding_derived.py --n-pca 16 --n-clusters 8
"""

from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd
from loguru import logger

from sentisense.config import (
    EMBED_DERIVED_CLUSTERS,
    EMBED_DERIVED_PCA,
    EMBED_DERIVED_TRAIN_FRAC,
)
from sentisense.constants import CUTOFF_DATE
from sentisense.db import get_engine
from sentisense.embed import daily_embedding_centroid
from sentisense.embed.derived import fit_transform_derived, persist_basis, persist_derived

_FAR_FUTURE = dt.date(2100, 1, 1)


def _resolve_fit_cutoff(dates: pd.DatetimeIndex, train_frac: float):
    """The quantile date of the ≤ CUTOFF corpus that ends the leak-safe train window."""
    modeling = dates[dates <= pd.Timestamp(CUTOFF_DATE)].sort_values()
    if len(modeling) == 0:
        raise RuntimeError("no embedding dates ≤ CUTOFF — cannot place the train window.")
    k = max(int(len(modeling) * train_frac) - 1, 0)
    return modeling[k]


def main() -> None:
    """Fit the derived basis train-only, transform all dates, upsert to Postgres."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-pca", type=int, default=EMBED_DERIVED_PCA)
    ap.add_argument("--n-clusters", type=int, default=EMBED_DERIVED_CLUSTERS)
    ap.add_argument("--train-frac", type=float, default=EMBED_DERIVED_TRAIN_FRAC,
                    help="Quantile of the ≤ CUTOFF corpus used as the fit window (leak-safe).")
    ap.add_argument("--fit-cutoff", default=None,
                    help="Explicit YYYY-MM-DD fit boundary (overrides --train-frac).")
    ap.add_argument("--coverage-cutoff", default=str(_FAR_FUTURE),
                    help="Populate derived rows for dates ≤ this (default: all).")
    ap.add_argument("--dry-run", action="store_true", help="Compute + report; do not write.")
    args = ap.parse_args()

    engine = get_engine()
    cen = daily_embedding_centroid(engine, cutoff=args.coverage_cutoff)
    if cen.empty:
        raise SystemExit("No embeddings cached — run the embed stage first.")
    centroid = cen[[c for c in cen.columns if c.startswith("embc_")]]

    fit_cutoff = (pd.Timestamp(args.fit_cutoff) if args.fit_cutoff
                  else _resolve_fit_cutoff(centroid.index, args.train_frac))
    logger.info("Centroid days={} | fit window ≤ {} | n_pca={} n_clusters={}",
                len(centroid), pd.Timestamp(fit_cutoff).date(), args.n_pca, args.n_clusters)

    derived, basis = fit_transform_derived(centroid, fit_cutoff=fit_cutoff,
                                           n_pca=args.n_pca, n_clusters=args.n_clusters,
                                           return_basis=True)

    if args.dry_run:
        logger.info("[dry-run] would upsert {} rows × {} cols: {}",
                    len(derived), derived.shape[1], list(derived.columns)[:6] + ["…"])
        return

    n = persist_derived(derived, fit_cutoff=fit_cutoff,
                        n_pca=args.n_pca, n_clusters=args.n_clusters, engine=engine)
    persist_basis(basis, engine=engine)   # UI day-view projects headlines through this basis
    logger.info("Done — {} derived rows in daily_embedding_derived.", n)


if __name__ == "__main__":
    main()
