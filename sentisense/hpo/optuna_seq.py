"""Generic, arch-parametric Optuna HPO for the sequence-classifier zoo (GRU/TCN/PatchTST).

Mirrors :mod:`sentisense.hpo.optuna_lstm`'s discipline — RDBStorage (resumable),
``TimeSeriesSplit`` CV on the dev region, sacred test tail, multi-seed variance-penalised
objective — but dispatches the model via :data:`sentisense.models.seq_zoo.ARCHITECTURES`
with an arch-specific search space. The well-tuned LSTM path stays in ``optuna_lstm``; this
serves the additional architectures for the leaderboard.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from sentisense.config import HPO_SEEDS, LSTM_VIABILITY_MIN_DAYS, OPTUNA_TRIALS, TEST_FRAC
from sentisense.db import get_connection_url
from sentisense.hpo.optuna_lstm import (
    _HPO_EPOCHS,
    _dev_test_split,
    _set_seeds,
    _youden_threshold,
    has_completed_trials,
)

_RECURRENT = {"LSTM", "GRU"}


def study_name_for(arch: str) -> str:
    """Per-architecture study name (e.g. 'sentisense_tcn_scores')."""
    return f"sentisense_{arch.lower()}_scores"


def _suggest(trial, arch: str, *, max_units: int, max_layers: int, window_choices: list[int]) -> dict:
    """Arch-specific Optuna search space (shared optimisation/regularisation knobs + body)."""
    p = {
        "window": trial.suggest_categorical("window", window_choices),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.6),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
        "grad_clip": trial.suggest_categorical("grad_clip", [0.5, 1.0, 5.0]),
        "lr": trial.suggest_float("lr", 5e-5, 1e-2, log=True),
        "dense_act": trial.suggest_categorical("dense_act", ["relu", "elu", "tanh", "gelu"]),
        "d_dense": trial.suggest_categorical("d_dense", [16, 32, 64]),
    }
    if arch in _RECURRENT:
        p |= {
            "units": trial.suggest_int("units", 16, max_units, log=True),
            "n_layers": trial.suggest_int("n_layers", 1, max_layers),
            "recurrent_dropout": trial.suggest_float("recurrent_dropout", 0.0, 0.5),
            "bidirectional": trial.suggest_categorical("bidirectional", [False, True]),
            "pooling": trial.suggest_categorical("pooling", ["last", "avg", "max", "attn"]),
        }
    elif arch == "TCN":
        p |= {
            "channels": trial.suggest_int("channels", 16, max_units, log=True),
            "levels": trial.suggest_int("levels", 2, max_layers + 2),
            "kernel_size": trial.suggest_categorical("kernel_size", [2, 3, 5]),
            "pooling": trial.suggest_categorical("pooling", ["last", "avg", "max"]),
        }
    elif arch == "PatchTST":
        d_model = trial.suggest_categorical("d_model", [32, 64, 128])
        p |= {
            "d_model": d_model,
            "n_heads": trial.suggest_categorical("n_heads", [2, 4]),
            "depth": trial.suggest_int("depth", 1, max_layers),
            "patch_len": trial.suggest_categorical("patch_len", [4, 8, 16]),
            "stride": trial.suggest_categorical("stride", [2, 4, 8]),
        }
    return p


def _model_kwargs(arch: str, params: dict) -> dict:
    """Map a param dict to the constructor kwargs for ``arch``."""
    common = {"dropout": params["dropout"], "dense_act": params["dense_act"],
              "d_dense": params.get("d_dense", 32)}
    if arch in _RECURRENT:
        return {**common, "hidden": params["units"], "n_layers": params["n_layers"],
                "recurrent_dropout": params.get("recurrent_dropout", 0.0),
                "pooling": params.get("pooling", "last"),
                "bidirectional": params.get("bidirectional", False)}
    if arch == "TCN":
        return {**common, "channels": params["channels"], "levels": params["levels"],
                "kernel_size": params["kernel_size"], "pooling": params.get("pooling", "last")}
    if arch == "PatchTST":
        return {**common, "d_model": params["d_model"], "n_heads": params["n_heads"],
                "depth": params["depth"], "patch_len": params["patch_len"], "stride": params["stride"]}
    raise ValueError(f"unknown arch {arch!r}")


def _build(arch: str, n_features: int, params: dict):
    from sentisense.models.seq_zoo import ARCHITECTURES
    return ARCHITECTURES[arch](n_features, **_model_kwargs(arch, params))


def _fold_auc(dev: pd.DataFrame, arch: str, params: dict, n_splits: int = 3) -> float:
    """Mean fold ROC-AUC for one (arch, params) set, TimeSeriesSplit on dev (train-only scaling)."""
    from sentisense.models.sequence import windowed_loader
    from sentisense.models.train import evaluate_on_test, train_model

    window, bs = params["window"], params["batch_size"]
    y = dev["Target"].values.astype(np.float32)
    X = dev.drop(columns=["Target"]).values.astype(np.float32)
    aucs: list[float] = []
    for tr_idx, va_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
        if (len(tr_idx) - window) < bs or (len(va_idx) - window) < 1:
            continue
        scaler = StandardScaler().fit(X[tr_idx])
        Xtr, Xva = scaler.transform(X[tr_idx]), scaler.transform(X[va_idx])
        dl_tr = windowed_loader(Xtr, y[tr_idx], window, batch_size=bs, shuffle=False, drop_last=True)
        dl_va = windowed_loader(Xva, y[va_idx], window, batch_size=bs, shuffle=False)
        model = _build(arch, Xtr.shape[1], params)
        train_model(model, dl_tr, dl_va, lr=params["lr"], weight_decay=params["weight_decay"],
                    max_grad_norm=params.get("grad_clip", 1.0),
                    max_epochs=_HPO_EPOCHS, patience=6, model_name=f"{arch}_hpo")
        aucs.append(evaluate_on_test(model, dl_va)["roc_auc"])
    return float(np.mean(aucs)) if aucs else 0.5


def run_seq_hpo(df: pd.DataFrame, arch: str, *, n_trials: int = OPTUNA_TRIALS,
                study_name: str | None = None):
    """Run/resume an arch-specific Optuna study (RDBStorage). Returns the study."""
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    name = study_name or study_name_for(arch)
    dev, _ = _dev_test_split(df)
    small = len(dev) < LSTM_VIABILITY_MIN_DAYS
    max_units = 96 if small else 192
    max_layers = 2 if small else 3
    window_choices = [5, 10, 15, 20] if small else [5, 10, 15, 20, 30]

    def objective(trial) -> float:
        params = _suggest(trial, arch, max_units=max_units, max_layers=max_layers,
                          window_choices=window_choices)
        scores: list[float] = []
        for step, seed in enumerate(HPO_SEEDS):
            _set_seeds(seed)
            scores.append(_fold_auc(dev, arch, params))
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores)) - 0.25 * float(np.std(scores))   # variance-penalised

    study = optuna.create_study(
        direction="maximize", study_name=name, storage=get_connection_url(),
        load_if_exists=True, sampler=TPESampler(seed=HPO_SEEDS[0]),
        pruner=MedianPruner(n_warmup_steps=1))
    logger.info("[{}] Optuna study '{}' — {} trials (resumes if interrupted).", arch, name, n_trials)
    t0 = time.perf_counter()

    def _eta(study, trial):
        done = trial.number + 1
        if done and n_trials:
            per = (time.perf_counter() - t0) / done
            logger.info("  [{}] HPO {}/{} | {:.0f}s/trial | ~{:.0f}s left", arch, done, n_trials,
                        per, per * max(n_trials - done, 0))

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True,
                   callbacks=[_eta] if n_trials > 0 else None)
    if has_completed_trials(study):
        logger.info("[{}] best val ROC-AUC {:.4f} | {}", arch, study.best_value, study.best_params)
    return study


def seq_holdout_eval(df: pd.DataFrame, arch: str, best_params: dict, *, n_seeds: int = 2):
    """Retrain on train+val per seed, evaluate ONCE on the sacred test tail.

    Returns ``(proba_series, label_series)`` — mean test probability over seeds + aligned
    labels, date-indexed to the held-out window (same contract as LSTM ``final_holdout_eval``
    so the leaderboard scores it identically).
    """
    from sentisense.models.sequence import chronological_split, windowed_loader
    from sentisense.models.train import proba_and_labels, train_model

    window, bs = best_params["window"], best_params["batch_size"]
    n = len(df)
    n_val, n_test = int(n * 0.15), int(n * 0.15)
    test_index = df.index[n - n_test:]

    proba_accum, labels_ref = None, None
    for seed in list(HPO_SEEDS)[:n_seeds]:
        _set_seeds(seed)
        X_tr, y_tr, X_va, y_va, X_te, y_te, nf = chronological_split(df)
        X_dev = np.vstack([X_tr, X_va]); y_dev = np.concatenate([y_tr, y_va])
        cut = int(len(X_dev) * 0.9)   # disjoint dev tail for early-stop monitor (never test)
        dl_fit = windowed_loader(X_dev[:cut], y_dev[:cut], window, batch_size=bs, shuffle=False, drop_last=True)
        dl_mon = windowed_loader(X_dev[cut:], y_dev[cut:], window, batch_size=bs, shuffle=False)
        dl_te = windowed_loader(X_te, y_te, window, batch_size=bs, shuffle=False)
        model = _build(arch, nf, best_params)
        train_model(model, dl_fit, dl_mon, lr=best_params["lr"], weight_decay=best_params["weight_decay"],
                    max_grad_norm=best_params.get("grad_clip", 1.0), model_name=f"{arch}_final_s{seed}")
        te_p, te_y = proba_and_labels(model, dl_te)
        proba_accum = te_p if proba_accum is None else proba_accum + te_p
        labels_ref = te_y

    mean_proba = proba_accum / n_seeds
    dates = test_index[window - 1: window - 1 + len(mean_proba)]
    return (pd.Series(mean_proba, index=dates, name="proba"),
            pd.Series(labels_ref, index=dates, name="label"))
