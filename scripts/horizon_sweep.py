"""Prediction-horizon sweep — find the window where the all-feature signal is strongest.

For each horizon H, builds the FUSED dataset (every extracted feature: LLM scores + per-source
+ sentiment×relevance interactions + e5 centroid `embc_*` + derived PCA/cluster `embpca_*`/
`embclus_dist_*` + finance + cross-asset + overnight global block) with the target generalised
to ``close(T+H) > close(T)``, and scores a GPU XGBoost on the same chronological last-15% OOS
window. XGBoost is the workhorse here because it consumes *all* feature families at once
(trees handle the 768-d centroid fine) and is fast enough to sweep many windows; escalate the
winning horizon to the full model zoo via ``pipeline_compare.py`` afterwards.

Honesty notes:
  • H>1 targets OVERLAP (consecutive rows share H-1 days) → the iid bootstrap is optimistic.
    The ROC-AUC CI here uses a MOVING-BLOCK bootstrap with block=H to respect that
    autocorrelation, so the lower bound is an honest "distinguishable from chance" gauge.
  • The "best" window = the one whose ROC-AUC lower bound (`auc_lo`) is highest.

Run (server-side, from repo root):
    uv run --extra finance --extra ml python scripts/horizon_sweep.py
    uv run --extra finance --extra ml python scripts/horizon_sweep.py --horizons 1,2,3,5,10 --regimes FULL,CUT
    uv run --extra finance --extra ml python scripts/horizon_sweep.py --xgb-trials 80 --no-overnight
"""

from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd
from loguru import logger

from sentisense.constants import CUTOFF_DATE
from sentisense.features import build_fused_dataset

_FAR_FUTURE = dt.date(2100, 1, 1)
_REGIMES = {"CUT": CUTOFF_DATE, "FULL": _FAR_FUTURE}
_COLS = ["roc_auc", "auc_lo", "auc_hi", "f1", "mcc", "accuracy", "up_rate", "n"]
SEED = 42


def _auc_ci_block(scores, labels, *, block: int = 1, n_boot: int = 500, seed: int = SEED):
    """Moving-block bootstrap 95% ROC-AUC CI. ``block``>1 respects horizon-overlap autocorrelation.

    block=1 reduces to the standard iid bootstrap. Returns (lo, hi); (nan, nan) if single-class.
    """
    from sklearn.metrics import roc_auc_score

    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    n = len(y)
    if n == 0 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    aucs = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(st, st + block) for st in starts])[:n]
        yb, sb = y[idx], s[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, sb))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _run_cell(df: pd.DataFrame, *, horizon: int, xgb_trials: int) -> dict:
    """XGBoost (all features) on the OOS tail → metrics + block-bootstrap ROC-AUC CI."""
    from sentisense.models.backtest import direction_metrics
    from sentisense.models.xgb_hpo import xgb_hpo

    _, scores, labels = xgb_hpo(df, n_trials=xgb_trials)
    m = direction_metrics(scores.to_numpy(), labels.to_numpy(), 0.5)
    lo, hi = _auc_ci_block(scores.to_numpy(), labels.to_numpy(), block=horizon)
    return {
        "roc_auc": m["roc_auc"], "auc_lo": lo, "auc_hi": hi,
        "f1": m.get("f1", float("nan")), "mcc": m.get("mcc", float("nan")),
        "accuracy": m.get("accuracy", float("nan")),
        "up_rate": float(labels.mean()), "n": int(len(labels)),
    }


def sweep(horizons, regimes, *, overnight: bool, xgb_trials: int) -> pd.DataFrame:
    """Build fused+all-feature data per (regime, horizon) and score XGBoost. Returns a table."""
    rows = {}
    for regime in regimes:
        cutoff = _REGIMES[regime]
        for h in horizons:
            label = f"H={h} [{regime}{'+ovn' if overnight else ''}]"
            logger.info("══ {} — building fused dataset (all features) ══", label)
            df = build_fused_dataset(cutoff=cutoff, overnight=overnight, horizon=h)
            if df.empty or len(df) < 200:
                logger.warning("{}: too few rows ({}) — skipped.", label, 0 if df.empty else len(df))
                continue
            rows[label] = _run_cell(df, horizon=h, xgb_trials=xgb_trials)
            r = rows[label]
            logger.info("{}: ROC-AUC {:.4f} CI[{:.4f},{:.4f}] acc {:.4f} n={}",
                        label, r["roc_auc"], r["auc_lo"], r["auc_hi"], r["accuracy"], r["n"])
    return pd.DataFrame.from_dict(rows, orient="index")[_COLS] if rows else pd.DataFrame()


def _write_md(board: pd.DataFrame, path: str, overnight: bool) -> None:
    """Write the sweep table + the best-window verdict (highest ROC-AUC lower bound)."""
    lines = ["# SentiSense prediction-horizon sweep (fused / all features, out-of-sample)", ""]
    lines.append("| window [regime] | " + " | ".join(_COLS) + " |")
    lines.append("|---|" + "|".join(["---"] * len(_COLS)) + "|")
    for label, r in board.iterrows():
        cells = [f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in _COLS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    clears = board[board["auc_lo"] > 0.5].sort_values("auc_lo", ascending=False)
    if len(clears):
        b = clears.iloc[0]
        lines.append(f"**Best window:** `{clears.index[0]}` — ROC-AUC {b['roc_auc']:.4f} "
                     f"CI[{b['auc_lo']:.4f}, {b['auc_hi']:.4f}] (lower bound clears 0.5). "
                     f"{len(clears)} of {len(board)} cells clear chance.")
    else:
        best = board.sort_values("auc_lo", ascending=False)
        b = best.iloc[0]
        lines.append(f"**No window clears chance** — every ROC-AUC CI straddles 0.5. "
                     f"Highest lower bound: `{best.index[0]}` auc_lo={b['auc_lo']:.4f} "
                     f"(CI[{b['auc_lo']:.4f}, {b['auc_hi']:.4f}]).")
    lines.append("")
    lines.append("_CI = moving-block bootstrap (block=H) — honest under H>1 target overlap. "
                 "XGBoost on every feature family; escalate the winning window to the full zoo._")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Wrote {}", path)


def main() -> None:
    """Sweep horizons × regimes on the all-feature fused dataset; write the verdict."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizons", default="1,2,3,5,10", help="Comma list of day-ahead windows.")
    ap.add_argument("--regimes", default="FULL", help="Comma list of CUT,FULL.")
    ap.add_argument("--no-overnight", action="store_true", help="Drop the overnight global block.")
    ap.add_argument("--xgb-trials", type=int, default=60, help="Optuna trials per cell.")
    ap.add_argument("--out", default="horizon_sweep.md")
    args = ap.parse_args()

    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip() in _REGIMES]
    board = sweep(horizons, regimes, overnight=not args.no_overnight, xgb_trials=args.xgb_trials)
    if board.empty:
        raise SystemExit("No cells produced — check the DB / embeddings / derived table.")
    _write_md(board, args.out, overnight=not args.no_overnight)


if __name__ == "__main__":
    main()
