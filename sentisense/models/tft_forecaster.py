"""pytorch-forecasting forecasters → next-day direction (TFT, N-HiTS, N-BEATS).

Each forecasts the next-day log-return of TA-125; the sign of the forecast → direction,
with the decision threshold tuned on a validation slice (Youden's J) — never the test tail.
HPO is a compact per-architecture Optuna search, selected on validation direction ROC-AUC.

- TFT / N-HiTS use sentiment covariates as time-varying reals.
- N-BEATS is univariate (the library requires a target-only dataset) — covariates dropped.

Heavyweight + version-sensitive (pytorch-forecasting + lightning); the leaderboard guards
each so any failure skips just that row. Install (server-side):
    uv pip install pytorch-forecasting lightning   (MIT / Apache-2.0)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

MAX_ENCODER_LEN = 30
SEED = 42
PF_ARCHS = ("TFT", "NHiTS", "NBEATS")
_UNIVARIATE = {"NBEATS"}   # library forbids covariates for these


def _install_hint() -> str:
    return ("pytorch-forecasting/lightning not installed. Run:\n"
            "    uv pip install pytorch-forecasting lightning")


def _make_frame(price: pd.Series, cutoff, covariate_frame: pd.DataFrame | None):
    """Long frame for pytorch-forecasting: time_idx, group, target r, optional cov reals."""
    p = price[price.index <= pd.Timestamp(cutoff)].sort_index()
    r = np.log(p / p.shift(1)).dropna().rename("r")
    dates = r.index                              # time_idx t ↔ dates[t]
    df = r.to_frame()
    cov_cols: list[str] = []
    if covariate_frame is not None and not covariate_frame.empty:
        cov = covariate_frame.reindex(r.index).fillna(0.0)
        cov_cols = [str(c) for c in cov.columns]
        df = df.join(cov)
    df = df.reset_index(drop=True)
    df["time_idx"] = np.arange(len(df), dtype=int)
    df["group"] = "ta125"
    return df, cov_cols, dates


def _predicted_returns(model, training, frame: pd.DataFrame) -> pd.Series:
    """Predict one-step returns over ``frame``; return a Series indexed by predicted time_idx."""
    from pytorch_forecasting import TimeSeriesDataSet
    ds = TimeSeriesDataSet.from_dataset(training, frame, predict=False, stop_randomization=True)
    dl = ds.to_dataloader(train=False, batch_size=128)
    out = model.predict(dl, mode="prediction", return_index=True)
    preds = getattr(out, "output", out[0] if isinstance(out, tuple) else out)
    index = getattr(out, "index", out[1] if isinstance(out, tuple) else None)
    arr = np.asarray(preds).reshape(len(preds), -1)[:, 0]
    tidx = index["time_idx"].to_numpy() if index is not None else np.arange(len(arr))
    return pd.Series(arr, index=tidx).sort_index()


def _param_space(trial, arch: str) -> dict:
    """Per-architecture Optuna search space (shared optimisation knobs + arch body)."""
    p = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }
    if arch == "TFT":
        p |= {
            "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64]),
            "attention_head_size": trial.suggest_categorical("attention_head_size", [1, 2, 4]),
            "hidden_continuous_size": trial.suggest_categorical("hidden_continuous_size", [8, 16, 32]),
        }
    elif arch == "NHiTS":
        p |= {"hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256])}
    elif arch == "NBEATS":
        p |= {
            "widths": trial.suggest_categorical("widths", ["32x512", "16x256", "64x1024"]),
            "backcast_loss_ratio": trial.suggest_categorical("backcast_loss_ratio", [0.0, 0.1, 1.0]),
        }
    return p


def _build_dataset(frame, cov_cols, arch, train_max_idx, enc):
    from pytorch_forecasting import TimeSeriesDataSet
    rows = frame[frame.time_idx < train_max_idx]
    common = dict(time_idx="time_idx", target="r", group_ids=["group"], max_encoder_length=enc,
                  min_encoder_length=max(enc // 2, 5), max_prediction_length=1, allow_missing_timesteps=True)
    if arch in _UNIVARIATE:   # N-BEATS: target-only dataset (library requirement)
        return TimeSeriesDataSet(rows, time_varying_unknown_reals=["r"],
                                 add_relative_time_idx=False, add_target_scales=False, **common)
    return TimeSeriesDataSet(rows, time_varying_unknown_reals=["r", *cov_cols],
                             time_varying_known_reals=["time_idx"],
                             add_relative_time_idx=True, add_target_scales=True, **common)


def _from_dataset(training, arch, params):
    from pytorch_forecasting import NBeats, NHiTS, TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss
    lr, dr = params["learning_rate"], params["dropout"]
    if arch == "TFT":
        return TemporalFusionTransformer.from_dataset(
            training, learning_rate=lr, hidden_size=params["hidden_size"],
            attention_head_size=params["attention_head_size"], dropout=dr,
            hidden_continuous_size=min(params["hidden_continuous_size"], params["hidden_size"]),
            loss=QuantileLoss(), optimizer="adam")
    if arch == "NHiTS":
        return NHiTS.from_dataset(training, learning_rate=lr, hidden_size=params["hidden_size"],
                                  dropout=dr, loss=QuantileLoss(), optimizer="adam")
    if arch == "NBEATS":
        widths = {"32x512": [32, 512], "16x256": [16, 256], "64x1024": [64, 1024]}[params["widths"]]
        return NBeats.from_dataset(training, learning_rate=lr, widths=widths, dropout=dr,
                                   backcast_loss_ratio=params["backcast_loss_ratio"], optimizer="adam")
    raise ValueError(f"unknown pytorch-forecasting arch {arch!r}")


def _train_one(frame, cov_cols, params, train_max_idx, *, arch, max_epochs, enc):
    """Train one model on rows with time_idx < ``train_max_idx``; return (model, training_ds)."""
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping

    pl.seed_everything(SEED, workers=True)
    training = _build_dataset(frame, cov_cols, arch, train_max_idx, enc)
    train_dl = training.to_dataloader(train=True, batch_size=params["batch_size"])
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="auto", devices=1, enable_progress_bar=False,
                         enable_checkpointing=False, logger=False, gradient_clip_val=0.1,
                         callbacks=[EarlyStopping(monitor="train_loss", patience=5, mode="min")])
    model = _from_dataset(training, arch, params)
    trainer.fit(model, train_dataloaders=train_dl)
    return model, training


def pf_directions(arch: str, price: pd.Series, cutoff, *, covariate_frame: pd.DataFrame | None = None,
                  n_trials: int = 8, enc: int = MAX_ENCODER_LEN, max_epochs: int = 30):
    """HPO ``arch`` (TFT/NHiTS/NBEATS) → ``(scores, labels, threshold)`` on the held-out test tail.

    Splits the return series 70/15/15. Optuna maximises validation direction ROC-AUC; the
    winner is retrained on train+val and evaluated on the last 15% (same OOS window as the
    other models). Returns ``None`` if pytorch-forecasting is absent or no test rows.
    """
    try:
        import optuna
        from sentisense.models.backtest import forecast_to_proba
        from sklearn.metrics import roc_auc_score, roc_curve
    except ImportError as exc:
        raise ImportError(_install_hint()) from exc
    if arch in _UNIVARIATE:
        covariate_frame = None   # library forbids covariates for these

    frame, cov_cols, dates = _make_frame(price, cutoff, covariate_frame)
    n = len(frame)
    v0, v1 = int(n * 0.70), int(n * 0.85)
    r = frame["r"].to_numpy()

    def _eval(model, training, lo_idx, hi_idx):
        """(scores, labels) indexed by DECISION day for predicted days t ∈ [lo_idx, hi_idx)."""
        preds = _predicted_returns(model, training, frame[frame.time_idx < hi_idx])
        rows = [(t, preds.loc[t]) for t in preds.index if lo_idx <= t < hi_idx and 1 <= t < n]
        if not rows:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        t_arr = np.array([t for t, _ in rows])
        f_arr = np.array([f for _, f in rows])
        dec_dates = dates[t_arr - 1]                    # decision day d; label is r at d+1 (= t)
        labels = (r[t_arr] > 0).astype(int)
        return (pd.Series(forecast_to_proba(f_arr), index=dec_dates, name="proba"),
                pd.Series(labels, index=dec_dates, name="label"))

    def objective(trial) -> float:
        params = _param_space(trial, arch)
        try:
            model, training = _train_one(frame, cov_cols, params, v0, arch=arch,
                                         max_epochs=max_epochs, enc=enc)
            s, lab = _eval(model, training, v0, v1)
            if len(s) and len(np.unique(lab)) > 1:
                return float(roc_auc_score(lab, s))
        except Exception as exc:  # noqa: BLE001 — bad config shouldn't kill the study
            logger.warning("{} trial failed: {}", arch, str(exc)[:120])
        return 0.5

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("{} HPO: best val ROC-AUC {:.4f} | {}", arch, study.best_value, study.best_params)

    model, training = _train_one(frame, cov_cols, dict(study.best_params), v1, arch=arch,
                                 max_epochs=max_epochs, enc=enc)
    sv, lv = _eval(model, training, v0, v1)   # tune threshold on validation
    thr = 0.5
    if len(sv) and lv.nunique() > 1:
        fpr, tpr, t = roc_curve(lv.to_numpy(), sv.to_numpy())
        thr = float(t[int(np.argmax(tpr - fpr))])
    st, lt = _eval(model, training, v1, n)
    if not len(st):
        return None
    return st, lt, thr


def tft_directions(price: pd.Series, cutoff, **kw):
    """Back-compat shim — TFT via :func:`pf_directions`."""
    return pf_directions("TFT", price, cutoff, **kw)
