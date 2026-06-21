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
import time

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
STUDY_SCORES = "sentisense_lstm_scores"   # score-feature LSTM study
STUDY_EMB = "sentisense_lstm_emb"         # embedding-centroid LSTM study
STUDY_FUSED = "sentisense_lstm_fused"     # scored ⊕ embedded fused LSTM study
_STUDY_NAME = STUDY_SCORES                # default for the standalone CLI


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


def _fold_auc(dev: pd.DataFrame, params: dict, seed: int, n_splits: int = 3,
              pca_components: int | None = None, pca_prefix: str | None = None) -> float:
    """Mean fold ROC-AUC for one param set + seed, TimeSeriesSplit on dev (no leak)."""
    import torch  # noqa: F401
    from sentisense.models.lstm import LSTMClassifier
    from sentisense.models.sequence import windowed_loader
    from sentisense.models.train import evaluate_on_test, train_model

    window = params["window"]
    batch_size = params.get("batch_size", 64)
    feat_cols = dev.drop(columns=["Target"]).columns
    y = dev["Target"].values.astype(np.float32)
    X = dev.drop(columns=["Target"]).values.astype(np.float32)
    # PCA scope mask (centroid block only when pca_prefix given).
    pca_mask = (np.array([c.startswith(pca_prefix) for c in feat_cols])
                if (pca_components and pca_prefix) else
                (np.ones(len(feat_cols), dtype=bool) if pca_components else None))
    tss = TimeSeriesSplit(n_splits=n_splits)

    aucs: list[float] = []
    for tr_idx, va_idx in tss.split(X):
        # Need at least one full batch of windows in train (drop_last=True), else the
        # fold trains on nothing and scores an untrained net.
        if (len(tr_idx) - window) < batch_size or (len(va_idx) - window) < 1:
            continue
        scaler = StandardScaler().fit(X[tr_idx])
        Xtr, Xva = scaler.transform(X[tr_idx]), scaler.transform(X[va_idx])
        if pca_mask is not None and pca_components < int(pca_mask.sum()):
            from sklearn.decomposition import PCA
            pca = PCA(n_components=pca_components, random_state=0).fit(Xtr[:, pca_mask])  # TRAIN-only
            Xtr = np.hstack([pca.transform(Xtr[:, pca_mask]), Xtr[:, ~pca_mask]])
            Xva = np.hstack([pca.transform(Xva[:, pca_mask]), Xva[:, ~pca_mask]])
        dl_tr = windowed_loader(Xtr, y[tr_idx], window, batch_size=batch_size,
                                shuffle=False, drop_last=True)
        dl_va = windowed_loader(Xva, y[va_idx], window, batch_size=batch_size, shuffle=False)
        model = LSTMClassifier(
            Xtr.shape[1], hidden=params["units"], n_layers=params["n_layers"],
            dropout=params["dropout"], recurrent_dropout=params["recurrent_dropout"],
            dense_act=params["dense_act"], pooling=params["pooling"],
            d_dense=params.get("d_dense", 32), bidirectional=params.get("bidirectional", False),
        )
        train_model(model, dl_tr, dl_va, lr=params["lr"], weight_decay=params["weight_decay"],
                    max_grad_norm=params.get("grad_clip", 1.0),
                    max_epochs=_HPO_EPOCHS, patience=6, model_name="hpo_trial")
        metrics = evaluate_on_test(model, dl_va)  # ROC-AUC on the fold's val slice
        aucs.append(metrics["roc_auc"])
    return float(np.mean(aucs)) if aucs else 0.5


def run_hpo(df_lstm: pd.DataFrame, *, n_trials: int = OPTUNA_TRIALS,
            study_name: str = _STUDY_NAME, pca_components: int | None = None,
            pca_prefix: str | None = None):
    """Run/resume the Optuna study against RDBStorage. Returns the study.

    ``pca_components`` (e.g. 50 for the embedding centroid dataset) reduces dims
    inside each CV fold, TRAIN-fit only. ``pca_prefix`` scopes PCA to the centroid
    block (e.g. 'embc_') so finance/TA-125 features pass through un-reduced.
    """
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    dev, _ = _dev_test_split(df_lstm)
    small = len(dev) < LSTM_VIABILITY_MIN_DAYS
    # Capacity caps scale with data size to fight overfit on a short dev region.
    max_units = 96 if small else 192
    max_layers = 2 if small else 3
    window_choices = [5, 10, 15, 20] if small else [5, 10, 15, 20, 30]
    if small:
        logger.warning("Dev region {} days < {} viability bar — capping units<={}, depth<={}, "
                       "window<=20.", len(dev), LSTM_VIABILITY_MIN_DAYS, max_units, max_layers)

    def objective(trial) -> float:
        params = {
            # memory / capacity
            "window": trial.suggest_categorical("window", window_choices),
            "units": trial.suggest_int("units", 16, max_units, log=True),
            "n_layers": trial.suggest_int("n_layers", 1, max_layers),
            "bidirectional": trial.suggest_categorical("bidirectional", [False, True]),
            "d_dense": trial.suggest_categorical("d_dense", [16, 32, 64]),
            # regularisation
            "dropout": trial.suggest_float("dropout", 0.1, 0.6),
            "recurrent_dropout": trial.suggest_float("recurrent_dropout", 0.0, 0.5),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
            "grad_clip": trial.suggest_categorical("grad_clip", [0.5, 1.0, 5.0]),
            # head + optimisation
            "dense_act": trial.suggest_categorical("dense_act", ["relu", "elu", "tanh", "gelu"]),
            "pooling": trial.suggest_categorical("pooling", ["last", "avg", "max", "attn"]),
            "lr": trial.suggest_float("lr", 5e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        }
        trial.set_user_attr("seeds", list(HPO_SEEDS))
        seed_scores: list[float] = []
        for step, seed in enumerate(HPO_SEEDS):
            _set_seeds(seed)
            seed_scores.append(_fold_auc(dev, params, seed, pca_components=pca_components,
                                         pca_prefix=pca_prefix))
            trial.report(float(np.mean(seed_scores)), step)  # running mean → pruner
            if trial.should_prune():
                import optuna as _o
                raise _o.TrialPruned()
        # Variance-penalised: prefer configs that are good AND stable across seeds —
        # a high-mean/high-variance config is a lucky-seed trap in finance.
        mean, std = float(np.mean(seed_scores)), float(np.std(seed_scores))
        return mean - 0.25 * std

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

    # Live HPO ETA: after each trial, log trials-done / mean-per-trial / wall-clock ETA.
    from sentisense.eta import eta_clock, fmt_duration
    _hpo_t0 = time.perf_counter()

    def _eta_callback(study, trial) -> None:
        done = trial.number + 1
        if done <= 0 or n_trials <= 0:
            return
        mean_per = (time.perf_counter() - _hpo_t0) / done
        remaining = mean_per * max(n_trials - done, 0)
        logger.info("  HPO trial {}/{} | {:.0f}s/trial | ~remaining {} | ETA {}",
                    done, n_trials, mean_per, fmt_duration(remaining), eta_clock(remaining))

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True,
                   callbacks=[_eta_callback] if n_trials > 0 else None)
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


def _youden_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """Decision threshold maximising Youden's J (TPR − FPR). 0.5 if single-class."""
    from sklearn.metrics import roc_curve
    if len(np.unique(labels)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(labels, probs)
    return float(thr[int(np.argmax(tpr - fpr))])


def final_holdout_eval(df_lstm: pd.DataFrame, best_params: dict, *, n_seeds: int = 3,
                       pca_components: int | None = None, pca_prefix: str | None = None):
    """Phase 7 — retrain on train+val, evaluate ONCE on the sacred test.

    Per seed: isotonic-calibrate test probs (fit on a disjoint dev tail) and pick a
    Youden-J threshold on that same dev tail (never the test). Reports metrics at the
    default 0.5 threshold AND at the tuned threshold, plus Brier (raw vs calibrated).

    Returns ``(summary, proba_series, label_series)`` — the date-indexed MEAN calibrated
    test probability (averaged over seeds) + aligned labels, for soft-vote ensembling.
    """
    import pandas as pd
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss

    from sentisense.models.lstm import LSTMClassifier
    from sentisense.models.sequence import chronological_split, windowed_loader
    from sentisense.models.train import metrics_at, proba_and_labels, train_model

    window = best_params["window"]
    bs = best_params.get("batch_size", 64)

    n = len(df_lstm)
    n_val, n_test = int(n * 0.15), int(n * 0.15)
    n_train = n - n_val - n_test
    test_index = df_lstm.index[n_train + n_val:]  # rows in the sacred test slice

    raw_runs, tuned_runs, brier_raw, brier_cal = [], [], [], []
    cal_kept, collapse_frac = [], []
    proba_accum, labels_ref = None, None

    for seed in list(HPO_SEEDS)[:n_seeds]:
        _set_seeds(seed)
        X_tr, y_tr, X_va, y_va, X_te, y_te, nf = chronological_split(
            df_lstm, pca_components=pca_components, pca_prefix=pca_prefix)
        X_dev = np.vstack([X_tr, X_va])
        y_dev = np.concatenate([y_tr, y_va])
        # Disjoint dev tail for early-stop monitor + calibration + threshold (never test).
        cut = int(len(X_dev) * 0.9)
        dl_fit = windowed_loader(X_dev[:cut], y_dev[:cut], window, batch_size=bs,
                                 shuffle=False, drop_last=True)
        dl_mon = windowed_loader(X_dev[cut:], y_dev[cut:], window, batch_size=bs, shuffle=False)
        dl_te = windowed_loader(X_te, y_te, window, batch_size=bs, shuffle=False)
        model = LSTMClassifier(
            nf, hidden=best_params["units"], n_layers=best_params["n_layers"],
            dropout=best_params["dropout"], recurrent_dropout=best_params["recurrent_dropout"],
            dense_act=best_params["dense_act"], pooling=best_params["pooling"],
            d_dense=best_params.get("d_dense", 32),
            bidirectional=best_params.get("bidirectional", False),
        )
        train_model(model, dl_fit, dl_mon, lr=best_params["lr"],
                    weight_decay=best_params["weight_decay"],
                    max_grad_norm=best_params.get("grad_clip", 1.0),
                    model_name=f"lstm_final_s{seed}")

        mon_p, mon_y = proba_and_labels(model, dl_mon)
        te_p, te_y = proba_and_labels(model, dl_te)
        # Isotonic calibration fit on the dev tail — ADOPTED ONLY IF it lowers the
        # dev-tail Brier (else identity). Unconditional isotonic over-fit the short dev
        # tail and HURT test Brier in the embedding run (brier_cal 0.269 > raw 0.251).
        # The keep/drop decision is made on dev only, so it never peeks at sacred test.
        use_cal = False
        if len(np.unique(mon_y)) > 1:
            cal = IsotonicRegression(out_of_bounds="clip").fit(mon_p, mon_y)
            if brier_score_loss(mon_y, cal.transform(mon_p)) < brier_score_loss(mon_y, mon_p):
                use_cal = True
        if use_cal:
            mon_p_cal, te_p_cal = cal.transform(mon_p), cal.transform(te_p)
        else:
            mon_p_cal, te_p_cal = mon_p, te_p
        cal_kept.append(int(use_cal))
        thr = _youden_threshold(mon_y, mon_p_cal)

        # Class-collapse guard: a model predicting ~one side at the chosen threshold has
        # no usable skill regardless of AUC. Track the dominant-class fraction of the
        # test decision so the scorecard exposes it instead of hiding behind acc≈0.5.
        te_dec = (te_p_cal > thr).astype(int)
        collapse_frac.append(float(max(te_dec.mean(), 1.0 - te_dec.mean())))

        raw_runs.append(metrics_at(te_p, te_y, 0.5))
        tuned_runs.append(metrics_at(te_p_cal, te_y, thr))
        brier_raw.append(float(brier_score_loss(te_y, te_p)) if len(np.unique(te_y)) > 1 else float("nan"))
        brier_cal.append(float(brier_score_loss(te_y, te_p_cal)) if len(np.unique(te_y)) > 1 else float("nan"))
        proba_accum = te_p_cal if proba_accum is None else proba_accum + te_p_cal
        labels_ref = te_y

    mean_proba = proba_accum / n_seeds
    dates = test_index[window - 1: window - 1 + len(mean_proba)]
    proba_series = pd.Series(mean_proba, index=dates, name="proba")
    label_series = pd.Series(labels_ref, index=dates, name="label")

    def _agg(runs, suffix):
        return {f"{k}{suffix}": {"mean": float(np.mean([r[k] for r in runs])),
                                 "std": float(np.std([r[k] for r in runs]))} for k in runs[0]}
    summary = {**_agg(raw_runs, "@0.5"), **_agg(tuned_runs, "@tuned")}
    summary["brier_raw"] = {"mean": float(np.nanmean(brier_raw)), "std": float(np.nanstd(brier_raw))}
    summary["brier_cal"] = {"mean": float(np.nanmean(brier_cal)), "std": float(np.nanstd(brier_cal))}
    # Diagnostics: how often calibration was kept, and how degenerate the decisions are.
    summary["calibration_kept"] = {"mean": float(np.mean(cal_kept)), "std": float(np.std(cal_kept))}
    summary["collapse_frac"] = {"mean": float(np.mean(collapse_frac)), "std": float(np.std(collapse_frac))}

    logger.info("Phase 7 holdout (mean±std over {} seeds):", n_seeds)
    for k, v in summary.items():
        logger.info("  {:24s} {:.4f} ± {:.4f}", k, v["mean"], v["std"])
    if summary["collapse_frac"]["mean"] > 0.9:
        logger.warning("⚠ Near class-collapse: model predicts one side {:.0%} of the time — "
                       "tuned-threshold metrics are not trustworthy skill.",
                       summary["collapse_frac"]["mean"])
    if summary["roc_auc@tuned"]["mean"] < 0.53 and abs(summary["mcc@tuned"]["mean"]) < 0.05:
        logger.warning("⚠ No detectable edge (ROC-AUC≈0.5, |MCC|≈0): treat as chance-level.")
    return summary, proba_series, label_series


def postcutoff_buy_overlay(test_proba: pd.Series, test_labels: pd.Series, *,
                           threshold: float = 0.5) -> dict:
    """Combined scorecard: model decisions on the pre-cutoff sacred test, FORCED BUY on
    every post-cutoff trading day.

    Post-cutoff days are NEVER used in training/tuning (per operator spec); here the
    trained model's decisions on the held-out pre-cutoff test are concatenated with a
    'always buy' (decision=1) stance over the real post-cutoff direction series. Reports
    accuracy + a confusion matrix over the union, plus the post-cutoff buy-only accuracy.

    Args:
        test_proba: date-indexed calibrated probabilities on the pre-cutoff sacred test.
        test_labels: aligned real labels for that test slice.
        threshold: decision threshold for the pre-cutoff model segment.

    Returns:
        dict with combined accuracy, confusion (tn/fp/fn/tp), and per-segment accuracy.
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix

    from sentisense.features.dataset import postcutoff_directions

    pre_dec = (test_proba.values > threshold).astype(int)
    pre_truth = test_labels.values.astype(int)

    post = postcutoff_directions()
    post_truth = post.values.astype(int)
    post_dec = np.ones(len(post_truth), dtype=int)  # forced BUY on every post-cutoff day

    dec = np.concatenate([pre_dec, post_dec])
    truth = np.concatenate([pre_truth, post_truth])
    tn, fp, fn, tp = confusion_matrix(truth, dec, labels=[0, 1]).ravel()
    out = {
        "combined_accuracy": float(accuracy_score(truth, dec)),
        "pre_cutoff_accuracy": float(accuracy_score(pre_truth, pre_dec)) if len(pre_truth) else float("nan"),
        "postcutoff_buy_accuracy": float(post_truth.mean()) if len(post_truth) else float("nan"),
        "n_pre": int(len(pre_truth)), "n_post": int(len(post_truth)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    logger.info("Buy-overlay scorecard (pre-cutoff model + post-cutoff forced BUY):")
    logger.info("  pre-cutoff days={}  post-cutoff days={}", out["n_pre"], out["n_post"])
    logger.info("  combined accuracy   {:.4f}", out["combined_accuracy"])
    logger.info("  pre-cutoff accuracy {:.4f}  | post-cutoff buy-only accuracy {:.4f}",
                out["pre_cutoff_accuracy"], out["postcutoff_buy_accuracy"])
    logger.info("  confusion [tn={} fp={} fn={} tp={}]", tn, fp, fn, tp)
    return out


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
