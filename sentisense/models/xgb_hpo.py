"""Compact Optuna HPO for the XGBoost direction model (torch-free).

Mirrors the LSTM/TimesFM tuning discipline so the leaderboard's XGBoost rows are tuned
on a VALIDATION slice (never the held-out test) — and can be re-tuned on the
sim-augmented features for the with-sim ablation. Returns the best params + held-out
(scores, labels) for the existing metrics/backtest to score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from sentisense.models.metrics import metrics_at

SEED = 42


def _fit_predict(params: dict, Xtr, ytr, Xpred):
    import xgboost as xgb
    pos = max(int(ytr.sum()), 1); neg = max(len(ytr) - int(ytr.sum()), 1)
    clf = xgb.XGBClassifier(eval_metric="logloss", random_state=SEED, verbosity=0,
                            scale_pos_weight=neg / pos, **params)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xpred)[:, 1]


def xgb_hpo(df: pd.DataFrame, *, n_trials: int = 30):
    """Tune XGBoost on a chronological train/val split; eval the winner on the test tail.

    Splits df 70/15/15 (train/val/test). Optuna maximises val ROC-AUC (threshold-free),
    then refits on train+val and predicts the test tail. Returns
    ``(best_params, scores: Series, labels: Series)`` aligned on the test index.
    """
    import optuna

    y = df["Target"].to_numpy().astype(int)
    X = df.drop(columns=["Target"])
    n = len(df); ntr = int(n * 0.70); nva = int(n * 0.15)
    Xtr, ytr = X.iloc[:ntr], y[:ntr]
    Xva, yva = X.iloc[ntr:ntr + nva], y[ntr:ntr + nva]
    Xte = X.iloc[ntr + nva:]
    yte = y[ntr + nva:]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=100),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        p = _fit_predict(params, Xtr, ytr, Xva)
        return metrics_at(p, yva)["roc_auc"]

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("XGBoost HPO: best val ROC-AUC {:.4f} | {}", study.best_value, study.best_params)

    # Refit on train+val with the winner, predict the held-out test tail.
    Xtv = X.iloc[:ntr + nva]; ytv = y[:ntr + nva]
    p_te = _fit_predict(study.best_params, Xtv, ytv, Xte)
    return (study.best_params,
            pd.Series(p_te, index=Xte.index), pd.Series(yte, index=Xte.index))
