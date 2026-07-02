"""Multi-model HPO → model registry. Evaluates a VAST zoo out-of-sample (same leak-safe
chronological 70/15/15, tune-on-val, score-the-sacred-test-tail contract as the leaderboard),
registers each model's OOS metrics into ``model_registry``, and auto-selects the most accurate.

Zoo:
  * Trees  — XGBoost / LightGBM / CatBoost           → refit-on-all → joblib artifact (servable).
  * Seq NN — LSTM / GRU / TCN / PatchTST              → refit-on-all → torch state_dict (servable).
  * Foundation / zero-shot forecasters — Chronos, TimesFM, TFT, NHiTS, NBEATS → reuse the
    leaderboard adapters (pipeline_compare). No stored artifact: registered as
    ``artifact_format='reforecast'`` with a val-tuned threshold; serving a reforecast winner is a
    live re-forecast (a phase-2 champion addition — until then the champion falls back to pinned).

Selection: ``registry.auto_select_best(metric=--select-metric)`` (oos_roc_auc | oos_accuracy),
sticky vs a manual UI override. Top-K rank-normalized soft-vote ensemble over the TREE models
only (forecaster/seq tails aren't index-aligned with the tabular tail).

Run (server-side, GPU box; extras: finance+ml always, tft+chronos for forecasters, TimesFM manual):
    # smoke — validate every adapter cheaply (register nothing)
    uv run --extra finance --extra ml --extra tft --extra chronos python scripts/train_registry.py \
        --trials 5 --seq-models lstm,gru,tcn,patchtst --seq-trials 2 --seq-seeds 1 \
        --forecasters chronos,tft --ctx-grid 64,128 --pf-trials 2 --pf-epochs 5 \
        --select-metric oos_accuracy --dry-run
    # thorough — the real 'find the most accurate model across the vast zoo' run
    uv run --extra finance --extra ml --extra tft --extra chronos python scripts/train_registry.py \
        --trials 100 --seq-models lstm,gru,tcn,patchtst --seq-trials 40 --seq-seeds 3 \
        --forecasters chronos,timesfm,tft,nhits,nbeats --pf-trials 8 --pf-epochs 30 \
        --select-metric oos_accuracy
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from loguru import logger

from sentisense.db import get_engine

_FAR_FUTURE = dt.date(2100, 1, 1)
_IL = ZoneInfo("Asia/Jerusalem")
SEED = 42


def _auc_ci(scores, labels, *, n_boot: int = 500) -> tuple[float, float]:
    """Bootstrap 95% ROC-AUC CI (fixed seed). (nan, nan) if single-class."""
    from sklearn.metrics import roc_auc_score

    s, y = np.asarray(scores, float), np.asarray(labels, int)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(SEED)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], s[idx]))
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))) if aucs else (float("nan"), float("nan"))


def _space(model_type: str, trial) -> dict:
    if model_type == "lgbm":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 60)}
    if model_type == "catboost":
        return {"iterations": trial.suggest_int("iterations", 200, 1000, step=100),
                "depth": trial.suggest_int("depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 12.0)}
    raise ValueError(model_type)


def _estimator(model_type: str, params: dict):
    """Build an unfitted estimator with a sane class-imbalance handling."""
    if model_type == "xgboost":
        import xgboost as xgb

        from sentisense.models.xgb_hpo import _xgb_device
        return xgb.XGBClassifier(eval_metric="logloss", random_state=SEED, verbosity=0,
                                 tree_method="hist", device=_xgb_device(), **params)
    if model_type == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=SEED, class_weight="balanced", n_jobs=-1,
                              verbosity=-1, **params)
    if model_type == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(random_seed=SEED, verbose=0, auto_class_weights="Balanced", **params)
    raise ValueError(model_type)


def _tune(model_type: str, df: pd.DataFrame, n_trials: int):
    """HPO on val, return (best_params, test_proba, test_labels) on the last-15% tail."""
    from sentisense.models.backtest import direction_metrics

    if model_type == "xgboost":                       # reuse the existing wide XGBoost HPO
        from sentisense.models.xgb_hpo import xgb_hpo
        best, te_scores, te_labels = xgb_hpo(df, n_trials=n_trials)
        return best, te_scores.to_numpy(), te_labels.to_numpy()

    import optuna
    y = df["Target"].to_numpy(int)
    X = df.drop(columns=["Target"])
    n = len(df); ntr, nva = int(n * 0.70), int(n * 0.15)
    Xtr, Xva, Xte = X.iloc[:ntr], X.iloc[ntr:ntr + nva], X.iloc[ntr + nva:]
    ytr, yva, yte = y[:ntr], y[ntr:ntr + nva], y[ntr + nva:]

    def objective(trial):
        est = _estimator(model_type, _space(model_type, trial)).fit(Xtr, ytr)
        return direction_metrics(est.predict_proba(Xva)[:, 1], yva, 0.5)["roc_auc"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    est = _estimator(model_type, study.best_params).fit(X.iloc[:ntr + nva], y[:ntr + nva])
    return study.best_params, est.predict_proba(Xte)[:, 1], yte


def _metrics(proba, labels, *, threshold: float = 0.5) -> dict:
    """OOS metrics at ``threshold``. Trees/seq select threshold-free on roc_auc → 0.5 is fine;
    forecaster/pf families carry a val-tuned threshold that MUST be passed so ``accuracy`` (and
    thus ``--select-metric oos_accuracy``) is honest. roc_auc is threshold-independent."""
    from sentisense.models.backtest import direction_metrics
    m = direction_metrics(np.asarray(proba), np.asarray(labels), threshold)
    lo, hi = _auc_ci(proba, labels)
    return {"roc_auc": m["roc_auc"], "auc_lo": lo, "auc_hi": hi,
            "mcc": m.get("mcc"), "accuracy": m.get("accuracy"), "n": int(len(labels))}


_ARCH_MAP = {"lstm": "LSTM", "gru": "GRU", "tcn": "TCN", "patchtst": "PatchTST"}


def _seq_hpo_eval(arch: str, df: pd.DataFrame, n_trials: int, n_seeds: int, stamp: str):
    """Tune a torch arch in a REGISTRY-namespaced Optuna study, score its OOS test tail.

    Uses ``registry_<arch>`` (not the leaderboard's ``sentisense_<arch>_scores``) so the registry
    never collides with the tuning studies' search space. If a pre-existing registry study has an
    incompatible categorical space (e.g. ``window`` choices changed), fall back to a fresh stamped
    study rather than crashing. Returns ``(best_params, test_proba, test_labels)``.
    """
    from sentisense.hpo.optuna_seq import run_seq_hpo, seq_holdout_eval

    name = f"registry_{arch.lower()}"
    try:
        study = run_seq_hpo(df, arch, n_trials=n_trials, study_name=name)
    except ValueError as exc:  # incompatible existing study → don't crash the whole zoo
        if "dynamic value space" not in str(exc):
            raise
        logger.warning("Seq study {} has an incompatible space ({}); using a fresh study.",
                       name, str(exc)[:80])
        study = run_seq_hpo(df, arch, n_trials=n_trials, study_name=f"{name}_{stamp}")
    best = dict(study.best_params)
    proba_s, label_s = seq_holdout_eval(df, arch, best, n_seeds=n_seeds)
    return best, proba_s.to_numpy(), label_s.to_numpy()


def _train_seq_all(arch: str, df: pd.DataFrame, params: dict, feat_cols: list) -> bytes:
    """Refit a seq model on ALL labeled rows, serialize a weights_only-safe torch bundle.

    A chronological 90/10 tail is held out only to drive early-stopping (never trained on).
    The bundle carries arch/params/window, the fit-region scaler stats (as plain lists so
    ``torch.load(weights_only=True)`` accepts it), the feature order, and the state_dict.
    """
    import io

    import torch
    from sklearn.preprocessing import StandardScaler

    from sentisense.hpo.optuna_seq import _build
    from sentisense.models.sequence import windowed_loader
    from sentisense.models.train import train_model

    window, bs = int(params["window"]), int(params["batch_size"])
    y = df["Target"].to_numpy(np.float32)
    X = df[feat_cols].to_numpy(np.float32)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)
    cut = min(int(len(Xs) * 0.9), len(Xs) - window - 1)          # tail keeps ≥1 monitor window
    dl_fit = windowed_loader(Xs[:cut], y[:cut], window, batch_size=bs, shuffle=False, drop_last=True)
    dl_mon = windowed_loader(Xs[cut:], y[cut:], window, batch_size=bs, shuffle=False)
    model = _build(arch, len(feat_cols), params)
    train_model(model, dl_fit, dl_mon, lr=params["lr"], weight_decay=params["weight_decay"],
                max_grad_norm=params.get("grad_clip", 1.0), model_name=f"{arch}_registry")
    bundle = {"arch": arch, "params": dict(params), "window": window, "feature_cols": list(feat_cols),
              "scaler_mean": scaler.mean_.astype(np.float32).tolist(),
              "scaler_scale": scaler.scale_.astype(np.float32).tolist(),
              "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}}
    buf = io.BytesIO(); torch.save(bundle, buf)
    return buf.getvalue()


_PF_ARCHS = {"tft": "TFT", "nhits": "NHiTS", "nbeats": "NBEATS"}


def _register_forecasters(engine, families: list, stamp: str, scored: list, *,
                          ctx_grid, pf_trials: int, pf_epochs: int, dry_run: bool) -> None:
    """Evaluate + register the zero-shot / foundation forecasters over the SAME sacred test tail.

    Reuses pipeline_compare's leak-safe adapters (Chronos / TimesFM / TFT / NHiTS / NBEATS). Each
    family is guarded so a missing extra (chronos/tft/timesfm) skips just that family. These have
    no persistable artifact — they register with ``artifact=None`` + ``artifact_format='reforecast'``
    and a val-tuned ``threshold`` (threaded into ``_metrics`` so their ``oos_accuracy`` is honest).
    Serving a reforecast winner is a live re-forecast — a phase-2 serve-path addition; until then
    the champion safely falls back to the pinned model. Buy&Hold is intentionally NOT registered
    (chance-level, would top accuracy in a rising regime without skill).
    """
    from sentisense.serve import registry
    try:
        from pipeline_compare import (_FAR_FUTURE as _FF, _chronos_forms, _cov_cols, _load_price,
                                       _pf, _timesfm_forms)
    except Exception as exc:  # noqa: BLE001 — leaderboard adapters unavailable
        logger.warning("Cannot import leaderboard adapters ({}) — skipping forecasters.", str(exc)[:150])
        return

    price = _load_price()

    def _emit(family: str, name: str, s, lab, thr: float, params: dict) -> None:
        m = _metrics(s.to_numpy(), lab.to_numpy(), threshold=thr)
        version = f"{name}-{stamp}"
        logger.info("{}: OOS roc_auc={:.4f} acc={:.4f} thr={:.3f} n={}",
                    version, m["roc_auc"], m["accuracy"], thr, m["n"])
        scored.append((version, family, m["roc_auc"]))
        if not dry_run:
            registry.register_model(engine, version=version, name=name, model_type=family,
                                    params={"family": family, "threshold": float(thr), **params},
                                    metrics=m, artifact=None, artifact_format="reforecast",
                                    feature_cols=params.get("cov_cols"), trained_rows=int(m["n"]))

    if "chronos" in families:
        try:
            for name, (s, lab, thr) in _chronos_forms(price, _FF, ctx_grid=ctx_grid).items():
                _emit("chronos", name, s, lab, thr, {"form": name})
        except Exception as exc:  # noqa: BLE001 — extra missing or forecast failure
            logger.warning("Chronos skipped: {}", str(exc)[:200])

    if "timesfm" in families:
        try:
            from sentisense.features import build_datasets
            mt, ml = build_datasets(cutoff=_FF, overnight=True)
            for name, (s, lab, thr) in _timesfm_forms(mt, ml, price, _FF, ctx_grid=ctx_grid).items():
                _emit("timesfm", name, s, lab, thr, {"form": name})
        except Exception as exc:  # noqa: BLE001
            logger.warning("TimesFM skipped: {}", str(exc)[:200])

    pf_want = [f for f in families if f in _PF_ARCHS]
    if pf_want:
        try:
            from sentisense.features import build_datasets
            mt, _ml = build_datasets(cutoff=_FF, overnight=True)
            cov = _cov_cols(mt, "scored")
            cov_cols = list(cov.columns) if cov is not None else None
            for fam in pf_want:
                arch = _PF_ARCHS[fam]
                use_cov = None if arch == "NBEATS" else cov
                res = _pf(arch, price, _FF, cov=use_cov, n_trials=pf_trials, max_epochs=pf_epochs)
                if res is None:
                    logger.warning("{} produced no result (pytorch-forecasting absent / no test rows).", arch)
                    continue
                s, lab, thr = res
                _emit("pf", arch, s, lab, thr,
                      {"arch": arch, "cov_cols": (cov_cols if use_cov is not None else None),
                       "n_trials": pf_trials, "max_epochs": pf_epochs})
        except Exception as exc:  # noqa: BLE001
            logger.warning("pytorch-forecasting family skipped: {}", str(exc)[:200])


def main() -> int:
    """Train the tree zoo, register each + a top-K ensemble, auto-activate the best."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="xgboost,lgbm,catboost", help="Comma list of tree models.")
    ap.add_argument("--trials", type=int, default=40, help="Optuna trials per tree model.")
    ap.add_argument("--seq-models", default="", help="Comma list of torch seq models "
                    "(lstm,gru,tcn,patchtst). Empty → skip the torch zoo.")
    ap.add_argument("--seq-trials", type=int, default=15, help="Optuna trials per torch seq model.")
    ap.add_argument("--seq-seeds", type=int, default=2, help="Seeds averaged in the seq OOS eval.")
    ap.add_argument("--forecasters", default="", help="Comma list of foundation forecasters: "
                    "chronos,timesfm,tft,nhits,nbeats. Empty → skip. (needs tft/chronos extras; "
                    "timesfm is a manual install.)")
    ap.add_argument("--ctx-grid", default="", help="Comma ints overriding the Chronos/TimesFM "
                    "context-length sweep (default 64,128,256,384,512,768,1024).")
    ap.add_argument("--pf-trials", type=int, default=8, help="Optuna trials per pytorch-forecasting model.")
    ap.add_argument("--pf-epochs", type=int, default=30, help="Max epochs per pytorch-forecasting fit.")
    ap.add_argument("--top-k", type=int, default=3, help="Members in the tree soft-vote ensemble.")
    ap.add_argument("--no-activate", action="store_true", help="Register only; don't change active.")
    ap.add_argument("--select-metric", choices=["oos_roc_auc", "oos_accuracy"],
                    default="oos_roc_auc", help="Metric auto_select_best ranks the active model by.")
    ap.add_argument("--dry-run", action="store_true", help="Train + score; register nothing.")
    args = ap.parse_args()

    import joblib

    from sentisense.features import build_fused_dataset
    from sentisense.serve import registry

    engine = get_engine()
    df = build_fused_dataset(engine, cutoff=_FAR_FUTURE, overnight=True)   # labeled rows only
    if df.empty or len(df) < 300:
        raise SystemExit(f"too few labeled rows ({0 if df.empty else len(df)}).")
    feat_cols = [c for c in df.columns if c != "Target"]
    X_all, y_all = df[feat_cols], df["Target"].to_numpy(int)
    stamp = dt.datetime.now(_IL).strftime("%Y%m%d-%H%M")
    logger.info("Training registry on {} labeled days × {} features.", len(df), len(feat_cols))

    te_probas: dict[str, np.ndarray] = {}
    te_labels = None
    scored: list[tuple[str, str, float]] = []          # (version, model_type, roc_auc)
    for mtype in [m.strip() for m in args.models.split(",") if m.strip()]:
        logger.info("── tuning {} ({} trials) ──", mtype, args.trials)
        best, te, yte = _tune(mtype, df, args.trials)
        m = _metrics(te, yte)
        te_probas[f"{mtype}-{stamp}"] = np.asarray(te, float)
        te_labels = np.asarray(yte, int)
        version = f"{mtype}-{stamp}"
        logger.info("{}: OOS roc_auc={:.4f} CI[{:.4f},{:.4f}] mcc={:.4f} acc={:.4f}",
                    version, m["roc_auc"], m["auc_lo"], m["auc_hi"], m["mcc"], m["accuracy"])
        scored.append((version, mtype, m["roc_auc"]))
        if args.dry_run:
            continue
        est_all = _estimator(mtype, best).fit(X_all, y_all)               # refit on ALL labeled → served artifact
        buf = io.BytesIO(); joblib.dump(est_all, buf)
        registry.register_model(engine, version=version, name=mtype.upper(), model_type=mtype,
                                params=best, metrics=m, artifact=buf.getvalue(),
                                artifact_format="joblib", feature_cols=feat_cols, trained_rows=len(df))

    # Torch sequence zoo — HPO (resumable), OOS eval on the windowed test tail, refit on all
    # labeled, serialize a weights_only-safe state_dict bundle. These do NOT join the tree
    # ensemble (their windowed test tails aren't index-aligned with the trees'); they compete
    # individually via auto_select_best on OOS ROC-AUC.
    for raw in [m.strip().lower() for m in args.seq_models.split(",") if m.strip()]:
        arch = _ARCH_MAP.get(raw)
        if arch is None:
            logger.warning("Unknown seq model {!r} — skipping (valid: {}).", raw, list(_ARCH_MAP))
            continue
        logger.info("── tuning {} ({} trials, {} seeds) ──", arch, args.seq_trials, args.seq_seeds)
        best, te, yte = _seq_hpo_eval(arch, df, args.seq_trials, args.seq_seeds, stamp)
        m = _metrics(te, yte)
        version = f"{raw}-{stamp}"
        logger.info("{}: OOS roc_auc={:.4f} CI[{:.4f},{:.4f}] mcc={:.4f} acc={:.4f}",
                    version, m["roc_auc"], m["auc_lo"], m["auc_hi"], m["mcc"], m["accuracy"])
        scored.append((version, arch, m["roc_auc"]))
        if args.dry_run:
            continue
        artifact = _train_seq_all(arch, df, best, feat_cols)                # refit on ALL labeled
        registry.register_model(engine, version=version, name=arch, model_type=arch.lower(),
                                params=best, metrics=m, artifact=artifact,
                                artifact_format="torch", feature_cols=feat_cols, trained_rows=len(df))

    # Foundation / zero-shot forecasters (Chronos / TimesFM / TFT / NHiTS / NBEATS) — reuse the
    # leaderboard adapters, register as 'reforecast' (no artifact; served by live re-forecast, phase 2).
    families = [f.strip().lower() for f in args.forecasters.split(",") if f.strip()]
    if families:
        ctx_grid = [int(x) for x in args.ctx_grid.split(",") if x.strip()] or None
        logger.info("── forecasters: {} ──", families)
        _register_forecasters(engine, families, stamp, scored, ctx_grid=ctx_grid,
                              pf_trials=args.pf_trials, pf_epochs=args.pf_epochs, dry_run=args.dry_run)

    # Top-K soft-vote ensemble (rank-normalized) over the TREE models — its own activatable entry.
    tree_scored = [s for s in scored if s[0] in te_probas]                  # torch tails aren't aligned
    top = sorted(tree_scored, key=lambda t: (t[2] if t[2] == t[2] else -1), reverse=True)[:args.top_k]
    if len(top) >= 2 and te_labels is not None and not args.dry_run:
        ens_te = np.mean([pd.Series(te_probas[v]).rank(pct=True).to_numpy() for v, _, _ in top], axis=0)
        ens_metrics = _metrics(ens_te, te_labels)
        ens_version = f"ensemble-top{len(top)}-{stamp}"
        registry.register_model(engine, version=ens_version, name=f"Ensemble(top{len(top)})",
                                model_type="ensemble", params={"members": [v for v, _, _ in top]},
                                metrics=ens_metrics, artifact=None, artifact_format="ensemble",
                                members=[v for v, _, _ in top], feature_cols=feat_cols, trained_rows=len(df))
        logger.info("Ensemble {} OOS roc_auc={:.4f} (members: {})",
                    ens_version, ens_metrics["roc_auc"], [v for v, _, _ in top])

    if args.dry_run:
        logger.info("[dry-run] scored {} models; registered/activated nothing.", len(scored))
        return 0
    if not args.no_activate:
        active = registry.auto_select_best(engine, metric=args.select_metric)
        logger.info("Active model → {} (by {})", active, args.select_metric)
    return 0


if __name__ == "__main__":
    sys.exit(main())
