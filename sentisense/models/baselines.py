"""Phase 5 baselines — naive (majority, persistence) + XGBoost on TimeSeriesSplit.

A model only earns attention if it beats these. All evaluation is chronological /
TimeSeriesSplit — never a random split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from sentisense.config import SEED


def _metrics(y_true, y_pred, y_proba=None) -> dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    else:
        out["roc_auc"] = 0.5
    return out


def naive_baselines(df: pd.DataFrame, *, test_frac: float = 0.15) -> dict[str, dict]:
    """Majority-class + persistence on the chronological test tail."""
    y = df["Target"].values.astype(int)
    n_test = int(len(y) * test_frac)
    y_tr, y_te = y[:-n_test], y[-n_test:]

    majority = int(round(y_tr.mean()))
    maj_pred = np.full_like(y_te, majority)

    # Persistence: predict tomorrow's direction = the previously realised direction.
    persist_pred = np.empty_like(y_te)
    prev = y_tr[-1]
    for i, actual in enumerate(y_te):
        persist_pred[i] = prev
        prev = actual
    return {
        "MajorityClass": _metrics(y_te, maj_pred, maj_pred.astype(float)),
        "Persistence": _metrics(y_te, persist_pred, persist_pred.astype(float)),
    }


def xgboost_timeseries_cv(df: pd.DataFrame, *, n_splits: int = 5) -> dict[str, float]:
    """XGBoost with scale_pos_weight, evaluated as mean over TimeSeriesSplit folds."""
    import xgboost as xgb

    y = df["Target"].values.astype(int)
    X = df.drop(columns=["Target"]).values.astype(np.float32)
    tss = TimeSeriesSplit(n_splits=n_splits)

    fold_metrics: list[dict[str, float]] = []
    for tr_idx, te_idx in tss.split(X):
        ytr = y[tr_idx]
        n_pos = max(int(ytr.sum()), 1)
        n_neg = max(len(ytr) - int(ytr.sum()), 1)
        clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=n_neg / n_pos, eval_metric="logloss",
            random_state=SEED, verbosity=0,
        )
        clf.fit(X[tr_idx], ytr)
        proba = clf.predict_proba(X[te_idx])[:, 1]
        pred = (proba > 0.5).astype(int)
        fold_metrics.append(_metrics(y[te_idx], pred, proba))

    keys = fold_metrics[0].keys()
    return {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}


def run_baselines(df: pd.DataFrame) -> dict[str, dict]:
    """Compute all Phase 5 baselines on the daily-mean frame and log a table."""
    results = naive_baselines(df)
    try:
        results["XGBoost_TSCV"] = xgboost_timeseries_cv(df)
    except ImportError:
        logger.warning("xgboost not installed — skipping XGBoost baseline (install --extra ml).")

    logger.info("Phase 5 baselines:")
    for name, m in results.items():
        logger.info("  {:16s} acc={:.4f} balacc={:.4f} f1={:.4f} auc={:.4f} mcc={:.4f}",
                    name, m["accuracy"], m["balanced_accuracy"], m["f1"], m["roc_auc"], m["mcc"])
    return results
