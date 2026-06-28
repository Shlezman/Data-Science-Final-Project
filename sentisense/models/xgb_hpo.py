"""Wide Optuna HPO for the XGBoost direction model (torch-free).

Tunes on a chronological VALIDATION slice (never the held-out test), refits the winner on
train+val, predicts the test tail — mirroring the leaderboard's leak-safe contract. Returns
the best params + held-out ``(scores, labels)`` for the shared metrics/backtest to score.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from loguru import logger

SEED = 42
_DEVICE: str | None = None


def _xgb_device() -> str:
    """'cuda' if XGBoost can actually train on GPU here, else 'cpu' (detected once, cached).

    Honors ``SENTISENSE_XGB_DEVICE`` (default 'cuda'). A GPU-less box (or a CPU-only XGBoost
    build) falls back to 'cpu' instead of crashing every cell.
    """
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    want = os.environ.get("SENTISENSE_XGB_DEVICE", "cuda").lower()
    if want == "cuda":
        try:
            import xgboost as xgb
            xgb.XGBClassifier(n_estimators=1, tree_method="hist", device="cuda").fit(
                np.zeros((4, 2)), np.array([0, 1, 0, 1]))
        except Exception as exc:  # noqa: BLE001 — any GPU/build failure → CPU
            logger.warning("XGBoost GPU unavailable ({}…) — using CPU.", str(exc)[:60])
            want = "cpu"
    _DEVICE = want
    logger.info("XGBoost device: {}", _DEVICE)
    return _DEVICE


def _fit_predict(params: dict, Xtr, ytr, Xpred):
    import xgboost as xgb
    pos = max(int(ytr.sum()), 1)
    neg = max(len(ytr) - int(ytr.sum()), 1)
    clf = xgb.XGBClassifier(eval_metric="logloss", random_state=SEED, verbosity=0,
                            scale_pos_weight=neg / pos, tree_method="hist",
                            device=_xgb_device(), **params)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xpred)[:, 1]


def xgb_hpo(df: pd.DataFrame, *, n_trials: int = 40):
    """Tune XGBoost on a chronological 70/15/15 split; eval the winner on the test tail.

    Optuna maximises validation ROC-AUC (threshold-free) over a WIDE space, then refits on
    train+val and predicts the test tail. Returns ``(best_params, scores, labels)`` aligned
    on the test index.
    """
    import optuna

    y = df["Target"].to_numpy().astype(int)
    X = df.drop(columns=["Target"])
    n = len(df); ntr = int(n * 0.70); nva = int(n * 0.15)
    Xtr, ytr = X.iloc[:ntr], y[:ntr]
    Xva, yva = X.iloc[ntr:ntr + nva], y[ntr:ntr + nva]
    Xte, yte = X.iloc[ntr + nva:], y[ntr + nva:]

    def objective(trial):
        from sentisense.models.backtest import direction_metrics
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1200, step=100),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
        p = _fit_predict(params, Xtr, ytr, Xva)
        return direction_metrics(p, yva, 0.5)["roc_auc"]   # roc_auc is threshold-free

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("XGBoost HPO: best val ROC-AUC {:.4f} | {}", study.best_value, study.best_params)

    Xtv, ytv = X.iloc[:ntr + nva], y[:ntr + nva]   # refit train+val with the winner
    p_te = _fit_predict(study.best_params, Xtv, ytv, Xte)
    return (study.best_params,
            pd.Series(p_te, index=Xte.index), pd.Series(yte, index=Xte.index))
