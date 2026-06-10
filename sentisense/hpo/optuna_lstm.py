"""Stateful, crash-resumable Optuna LSTM HPO (Phase 6) + sacred-holdout eval (Phase 7).

Design (per task spec):
  * Storage = SENTISENSE_DATABASE_URL RDBStorage; ``create_study(load_if_exists=True)``
    so a killed 7-day run resumes where it left off.
  * Validation = ``TimeSeriesSplit`` on the dev region (train+val); the last
    ``TEST_FRAC`` is held SACRED and never touched until :func:`final_holdout_eval`.
  * Objective = maximise mean Validation ROC-AUC across folds × seeds (≥3 seeds);
    each fold's score is reported to a MedianPruner for early termination.
  * Small-data guard (Gate B): if dev trading-days < LSTM_VIABILITY_MIN_DAYS, the
    search space is capped (depth ≤ 2, units ≤ 64) to fight overfit.
  * Per-trial seed list recorded in ``trial.user_attrs``.

Launch (server-side, long-running) — see docs/RUNBOOK.md for the tmux/nohup form:
    uv run python -m sentisense.hpo.optuna_lstm --trials 100
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from sentisense.config import (
    HPO_SEEDS,
    LSTM_VIABILITY_MIN_DAYS,
    OPTUNA_TRIALS,
    TEST_FRAC,
)
from sentisense.db import get_connection_url

_HPO_EPOCHS = 40  # reduced per-trial budget; final retrain uses full MAX_EPOCHS
_STUDY_NAME = "sentisense_lstm_hpo"


def _set_seeds(seed: int) -> None:
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dev_test_split(df: pd.DataFrame):
    """Split off the sacred test tail; return (dev_df, test_df)."""
    n_test = int(len(df) * TEST_FRAC)
    return df.iloc[:-n_test].copy(), df.iloc[-n_test:].copy()


def _fold_auc(dev: pd.DataFrame, params: dict, seed: int, n_splits: int = 3) -> float:
    """Mean fold ROC-AUC for one param set + seed, TimeSeriesSplit on dev (no leak)."""
    import torch  # noqa: F401
    from sentisense.models.lstm import LSTMClassifier
    from sentisense.models.sequence import windowed_loader
    from sentisense.models.train import evaluate_on_test, train_model

    window = params["window"]
    y = dev["Target"].values.astype(np.float32)
    X = dev.drop(columns=["Target"]).values.astype(np.float32)
    tss = TimeSeriesSplit(n_splits=n_splits)

    aucs: list[float] = []
    for tr_idx, va_idx in tss.split(X):
        if len(va_idx) <= window or len(tr_idx) <= window:
            continue  # fold too short to window
        scaler = StandardScaler().fit(X[tr_idx])
        Xtr, Xva = scaler.transform(X[tr_idx]), scaler.transform(X[va_idx])
        dl_tr = windowed_loader(Xtr, y[tr_idx], window, shuffle=False, drop_last=True)
        dl_va = windowed_loader(Xva, y[va_idx], window, shuffle=False)
        model = LSTMClassifier(
            X.shape[1], hidden=params["units"], n_layers=params["n_layers"],
            dropout=params["dropout"], recurrent_dropout=params["recurrent_dropout"],
            dense_act=params["dense_act"], pooling=params["pooling"],
        )
        train_model(model, dl_tr, dl_va, lr=params["lr"], weight_decay=params["weight_decay"],
                    max_epochs=_HPO_EPOCHS, patience=6, model_name="hpo_trial")
        metrics = evaluate_on_test(model, dl_va)  # ROC-AUC on the fold's val slice
        aucs.append(metrics["roc_auc"])
    return float(np.mean(aucs)) if aucs else 0.5


def run_hpo(df_lstm: pd.DataFrame, *, n_trials: int = OPTUNA_TRIALS,
            study_name: str = _STUDY_NAME):
    """Run/resume the Optuna study against RDBStorage. Returns the study."""
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    dev, _ = _dev_test_split(df_lstm)
    small = len(dev) < LSTM_VIABILITY_MIN_DAYS
    max_units = 64 if small else 128
    max_layers = 2 if small else 3
    if small:
        logger.warning("Dev region {} days < {} viability bar — capping units<={}, depth<={}.",
                       len(dev), LSTM_VIABILITY_MIN_DAYS, max_units, max_layers)

    def objective(trial) -> float:
        params = {
            "window": trial.suggest_int("window", 3, 14),
            "units": trial.suggest_int("units", 16, max_units, log=True),
            "n_layers": trial.suggest_int("n_layers", 1, max_layers),
            "dropout": trial.suggest_float("dropout", 0.2, 0.6),
            "recurrent_dropout": trial.suggest_float("recurrent_dropout", 0.1, 0.5),
            "dense_act": trial.suggest_categorical("dense_act", ["relu", "elu", "tanh", "gelu"]),
            "pooling": trial.suggest_categorical("pooling", ["last", "avg", "max"]),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        }
        trial.set_user_attr("seeds", list(HPO_SEEDS))
        seed_scores: list[float] = []
        for step, seed in enumerate(HPO_SEEDS):
            _set_seeds(seed)
            seed_scores.append(_fold_auc(dev, params, seed))
            trial.report(float(np.mean(seed_scores)), step)  # running mean → pruner
            if trial.should_prune():
                import optuna as _o
                raise _o.TrialPruned()
        return float(np.mean(seed_scores))

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=get_connection_url(),       # RDBStorage on the project DB → resumable
        load_if_exists=True,
        sampler=TPESampler(seed=HPO_SEEDS[0]),
        pruner=MedianPruner(n_warmup_steps=1),
    )
    logger.info("Optuna study '{}' — {} trials (resumes if interrupted). Storage=project DB.",
                study_name, n_trials)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    # best_value/best_params raise ValueError on a study with zero COMPLETE trials
    # (e.g. n_trials=0 resume on a fresh study, or an all-pruned run). Guard it so the
    # 'load existing study' path (n_trials=0) returns cleanly instead of crashing.
    if has_completed_trials(study):
        logger.info("Best val ROC-AUC {:.4f} | params {}", study.best_value, study.best_params)
    else:
        logger.warning("Study '{}' has no completed trials yet.", study_name)
    return study


def has_completed_trials(study) -> bool:
    """True iff the study has at least one COMPLETE trial (so best_* is safe to read)."""
    import optuna
    return len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))) > 0


def final_holdout_eval(df_lstm: pd.DataFrame, best_params: dict, *,
                       n_seeds: int = 3) -> dict:
    """Phase 7 — retrain on train+val with best params, evaluate ONCE on the sacred test.

    Multi-seed (mean±std) so the headline number isn't a single-seed lottery.
    """
    from sentisense.models.lstm import LSTMClassifier
    from sentisense.models.sequence import chronological_split, windowed_loader
    from sentisense.models.train import evaluate_on_test, train_model

    window = best_params["window"]
    runs: list[dict] = []
    for seed in list(HPO_SEEDS)[:n_seeds]:
        _set_seeds(seed)
        # Train on train+val (everything before the sacred test tail), eval on test.
        X_tr, y_tr, X_va, y_va, X_te, y_te, nf = chronological_split(df_lstm)
        X_dev = np.vstack([X_tr, X_va])
        y_dev = np.concatenate([y_tr, y_va])
        # Carve a disjoint chronological tail of dev for the early-stop monitor so it
        # is NOT part of the fit (the prior code monitored on data it trained on —
        # optimistic). Test tail stays sacred (only evaluate_on_test sees dl_te).
        cut = int(len(X_dev) * 0.9)
        dl_fit = windowed_loader(X_dev[:cut], y_dev[:cut], window, shuffle=False, drop_last=True)
        dl_mon = windowed_loader(X_dev[cut:], y_dev[cut:], window, shuffle=False)
        dl_te = windowed_loader(X_te, y_te, window, shuffle=False)
        model = LSTMClassifier(
            nf, hidden=best_params["units"], n_layers=best_params["n_layers"],
            dropout=best_params["dropout"], recurrent_dropout=best_params["recurrent_dropout"],
            dense_act=best_params["dense_act"], pooling=best_params["pooling"],
        )
        train_model(model, dl_fit, dl_mon, lr=best_params["lr"],
                    weight_decay=best_params["weight_decay"], model_name=f"lstm_final_s{seed}")
        runs.append(evaluate_on_test(model, dl_te))

    keys = runs[0].keys()
    summary = {k: {"mean": float(np.mean([r[k] for r in runs])),
                   "std": float(np.std([r[k] for r in runs]))} for k in keys}
    logger.info("Phase 7 holdout (mean±std over {} seeds):", n_seeds)
    for k, v in summary.items():
        logger.info("  {:18s} {:.4f} ± {:.4f}", k, v["mean"], v["std"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 LSTM HPO (Optuna, resumable).")
    parser.add_argument("--trials", type=int, default=OPTUNA_TRIALS)
    parser.add_argument("--study-name", type=str, default=_STUDY_NAME)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be >= 1")

    from sentisense.features import build_datasets
    _, ml = build_datasets()
    run_hpo(ml, n_trials=args.trials, study_name=args.study_name)


if __name__ == "__main__":
    main()
