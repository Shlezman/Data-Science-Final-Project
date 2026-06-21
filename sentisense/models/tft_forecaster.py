"""Temporal Fusion Transformer (TFT) → next-day direction, via pytorch-forecasting.

TFT forecasts the next-day log-return of TA-125 (optionally with sentiment covariates as
time-varying reals); the sign of the forecast → direction, with the decision threshold
tuned on a validation slice (Youden's J) — never the test tail. HPO is a compact Optuna
search over TFT capacity/regularisation, selected on validation direction ROC-AUC.

Heavyweight + version-sensitive (pytorch-forecasting + lightning); the leaderboard guards
this so any failure skips just the TFT row. Install (server-side):
    uv pip install pytorch-forecasting lightning   (both MIT/Apache-2.0)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

MAX_ENCODER_LEN = 30
SEED = 42


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


def _train_one(frame, cov_cols, params, train_max_idx, *, max_epochs, enc):
    """Train one TFT on rows with time_idx < ``train_max_idx``; return (model, training_ds)."""
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss

    pl.seed_everything(SEED, workers=True)
    training = TimeSeriesDataSet(
        frame[frame.time_idx < train_max_idx], time_idx="time_idx", target="r", group_ids=["group"],
        max_encoder_length=enc, min_encoder_length=max(enc // 2, 5), max_prediction_length=1,
        time_varying_unknown_reals=["r", *cov_cols], time_varying_known_reals=["time_idx"],
        add_relative_time_idx=True, add_target_scales=True, allow_missing_timesteps=True)
    train_dl = training.to_dataloader(train=True, batch_size=params["batch_size"])
    import torch
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="auto", devices=1, enable_progress_bar=False,
                         enable_checkpointing=False, logger=False, gradient_clip_val=0.1,
                         callbacks=[EarlyStopping(monitor="train_loss", patience=5, mode="min")])
    tft = TemporalFusionTransformer.from_dataset(
        training, learning_rate=params["learning_rate"], hidden_size=params["hidden_size"],
        attention_head_size=params["attention_head_size"], dropout=params["dropout"],
        hidden_continuous_size=min(params["hidden_continuous_size"], params["hidden_size"]),
        loss=QuantileLoss(), optimizer="adam")
    trainer.fit(tft, train_dataloaders=train_dl)
    return tft, training


def tft_directions(price: pd.Series, cutoff, *, covariate_frame: pd.DataFrame | None = None,
                   n_trials: int = 8, enc: int = MAX_ENCODER_LEN, max_epochs: int = 30):
    """HPO a TFT and return ``(scores, labels, threshold)`` on the held-out test tail.

    Splits the return series 70/15/15. Optuna maximises validation direction ROC-AUC over
    TFT hyperparameters; the winner is retrained on train+val and evaluated on the last 15%
    (same OOS window as the other models). Returns ``None`` if pytorch-forecasting is absent.
    """
    try:
        import optuna
        from sentisense.models.backtest import forecast_to_proba
        from sklearn.metrics import roc_auc_score, roc_curve
    except ImportError as exc:
        raise ImportError(_install_hint()) from exc

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
        params = {
            "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64]),
            "attention_head_size": trial.suggest_categorical("attention_head_size", [1, 2, 4]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.4),
            "hidden_continuous_size": trial.suggest_categorical("hidden_continuous_size", [8, 16, 32]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        }
        try:
            model, training = _train_one(frame, cov_cols, params, v0, max_epochs=max_epochs, enc=enc)
            s, lab = _eval(model, training, v0, v1)
            if len(s) and len(np.unique(lab)) > 1:
                return float(roc_auc_score(lab, s))
        except Exception as exc:  # noqa: BLE001 — bad config shouldn't kill the study
            logger.warning("TFT trial failed: {}", str(exc)[:120])
        return 0.5

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("TFT HPO: best val ROC-AUC {:.4f} | {}", study.best_value, study.best_params)

    best = {**study.best_params}
    model, training = _train_one(frame, cov_cols, best, v1, max_epochs=max_epochs, enc=enc)
    # Tune threshold on validation, score the test tail.
    sv, lv = _eval(model, training, v0, v1)
    thr = 0.5
    if len(sv) and lv.nunique() > 1:
        fpr, tpr, t = roc_curve(lv.to_numpy(), sv.to_numpy())
        thr = float(t[int(np.argmax(tpr - fpr))])
    st, lt = _eval(model, training, v1, n)
    if not len(st):
        return None
    return st, lt, thr
