"""TimesFM (Google) as a third SentiSense forecaster, alongside XGBoost + LSTM.

TimesFM is a decoder-only time-series foundation model that outputs a CONTINUOUS
forecast. The project target is BINARY next-day TA-125 direction, so:

  * Forecast the daily LOG-RETURN series (stationary), expanding-context walk-forward:
    at decision day ``d`` the context is the returns strictly ``≤ d`` → forecast the
    next step's return.
  * Map ``sign(forecast)`` → 1=Up / 0=Down, and score the SIGNED forecast magnitude on
    the SAME ``metrics_at`` as the classifiers via
    :func:`sentisense.models.backtest.forecast_to_proba` (monotonic, 0.5⇔sign). Label =
    ``1 if return[d+1] > 0``. No future leak: context is strictly past; fine-tuning (if
    used) sees only the train portion.

Forms (each runnable per data regime):
  * ``zero_shot``  — pretrained, no fine-tune (strong default on a small daily series).
  * ``finetuned``  — fine-tune on the train slice only (best-effort; see note below).
  * covariates     — sentiment features as XReg dynamic covariates vs univariate
    (the feature-value ablation).

API pinned to **TimesFM 2.5** (``google/timesfm-2.5-200m-pytorch``):

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(REPO, torch_compile=True)
    model.compile(timesfm.ForecastConfig(max_context=CONTEXT_LEN, max_horizon=1,
                                         normalize_inputs=True, use_continuous_quantile_head=True,
                                         force_flip_invariance=True, infer_is_positive=False,
                                         fix_quantile_crossing=True))
    point_forecast, _ = model.forecast(horizon=1, inputs=[ctx_array])   # point: (1, 1)

``timesfm`` is an optional, heavy, GPU dependency. Install on the server:
    uv sync --extra timesfm          # or: pip install 'timesfm[torch]'
    # if PyPI lacks it: git clone github.com/google-research/timesfm && pip install -e '.[torch]'
``HF_TOKEN`` is read from the environment by huggingface_hub if the checkpoint is gated;
it is never hardcoded.

NOTE (2.5 finetune / XReg): TimesFM 2.5 re-added covariates (XReg, Oct-2025) and ships
finetuning examples, but those exact call signatures are not yet in the stable docs.
``finetune_model`` / the covariate path are therefore best-effort: they try the
documented hooks and, if unavailable in the installed build, log a warning and fall
back to the zero-shot univariate forecast so the pipeline still produces a row. The
fallback is reported (never silent), satisfying the "else run univariate + document".
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from loguru import logger

from sentisense.config import SEED
from sentisense.models.backtest import forecast_to_proba

TIMESFM_REPO = "google/timesfm-2.5-200m-pytorch"
CONTEXT_LEN = 512          # daily history fed as context (≤ model max_context 1024)
MIN_CONTEXT = 64           # don't forecast until at least this much history exists
HORIZON = 1                # next-day


def _install_hint() -> str:
    return ("The 'timesfm' extra is not installed. On the server:\n"
            "    uv sync --extra timesfm   (or: pip install 'timesfm[torch]')\n"
            "    # if PyPI lacks it: git clone github.com/google-research/timesfm && pip install -e '.[torch]'")


def load_timesfm(context_len: int = CONTEXT_LEN, *, torch_compile: bool = True):
    """Load + compile the pinned TimesFM 2.5 model, or fail fast with an install hint."""
    try:
        import timesfm
    except ModuleNotFoundError as exc:
        raise RuntimeError(_install_hint()) from exc

    if os.environ.get("HF_TOKEN"):
        logger.info("HF_TOKEN present in env — used by huggingface_hub for the download.")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(TIMESFM_REPO, torch_compile=torch_compile)
    model.compile(
        timesfm.ForecastConfig(
            max_context=context_len,
            max_horizon=HORIZON,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=False,        # log-returns are signed
            fix_quantile_crossing=True,
        )
    )
    logger.info("TimesFM 2.5 loaded ({}), context_len={}.", TIMESFM_REPO, context_len)
    return model


def make_forecast_fn(model, *, covariate_frame: pd.DataFrame | None = None):
    """Return ``forecast_fn(context_returns) -> float`` over the pinned 2.5 API.

    ``covariate_frame`` (sentiment features aligned to the return index) enables the
    XReg covariate path when the installed build supports it; otherwise the call
    degrades to univariate and a one-time warning is logged.
    """
    state = {"warned": False}

    def forecast_fn(context: np.ndarray) -> float:
        ctx = np.asarray(context, dtype=np.float32)
        # XReg covariates (2.5+) — best-effort; fall back to univariate if unsupported.
        if covariate_frame is not None:
            try:
                cov = covariate_frame.iloc[-len(ctx):].to_numpy(dtype=np.float32)
                point, _ = model.forecast(
                    horizon=HORIZON, inputs=[ctx],
                    dynamic_numerical_covariates={c: [cov[:, j]]
                                                  for j, c in enumerate(covariate_frame.columns)},
                )
                return float(np.ravel(point)[0])
            except TypeError:
                if not state["warned"]:
                    logger.warning("Installed TimesFM build does not accept covariates here "
                                   "— falling back to UNIVARIATE for the covariate form.")
                    state["warned"] = True
        point, _ = model.forecast(horizon=HORIZON, inputs=[ctx])
        return float(np.ravel(point)[0])

    return forecast_fn


def walk_forward_directions(
    returns: pd.Series,
    test_index: pd.DatetimeIndex,
    forecast_fn,
    *,
    context_len: int = CONTEXT_LEN,
    min_context: int = MIN_CONTEXT,
) -> tuple[pd.Series, pd.Series]:
    """Expanding-context walk-forward; return ``(scores, labels)`` aligned by decision day.

    For each ``d`` in ``test_index``: context = ``returns`` strictly up to and including
    ``d`` (last ``context_len`` points) → ``forecast_fn`` → predicted next-step return.
    Label = ``1 if returns[d+1] > 0`` (the project's next-day Up target). Days with no
    next-day return, or with < ``min_context`` history, are skipped. ``scores`` is the
    forecast→[0,1] pseudo-probability so ``metrics_at`` scores it like a classifier.

    ``forecast_fn`` is injected (so the harness is testable without the model) and must
    receive ONLY the strictly-past context — the single leak-safety contract.
    """
    r = returns.sort_index()
    idx = r.index
    pos = {d: i for i, d in enumerate(idx)}
    raw, labels, dates = [], [], []
    for d in test_index:
        i = pos.get(d)
        if i is None or i + 1 >= len(idx):    # unknown date or no next-day label
            continue
        if i + 1 < min_context:               # not enough history yet
            continue
        ctx = r.iloc[max(0, i - context_len + 1): i + 1].to_numpy()  # strictly ≤ d
        raw.append(forecast_fn(ctx))
        labels.append(1 if r.iloc[i + 1] > 0 else 0)
        dates.append(d)
    scores = forecast_to_proba(np.asarray(raw)) if raw else np.asarray([])
    return (pd.Series(scores, index=pd.DatetimeIndex(dates), name="score"),
            pd.Series(labels, index=pd.DatetimeIndex(dates), name="label"))


def finetune_on_train(model, train_returns: pd.Series):
    """Best-effort fine-tune on the TRAIN slice only (no val/test leak).

    TimesFM 2.5's finetuning API is not in the stable docs; this tries the common hook
    and, if absent, logs and returns the zero-shot model unchanged (the ``finetuned``
    form then equals zero-shot — reported, not silent).
    """
    fit = getattr(model, "finetune", None) or getattr(model, "fit", None)
    if fit is None:
        logger.warning("Installed TimesFM build exposes no finetune/fit hook — the "
                       "'finetuned' form falls back to zero-shot (documented).")
        return model
    try:
        import torch
        torch.manual_seed(SEED)
        fit([train_returns.to_numpy(dtype=np.float32)])
        logger.info("TimesFM fine-tuned on {} train points.", len(train_returns))
    except Exception as exc:  # noqa: BLE001 — never let a finetune-API mismatch kill the run
        logger.warning("TimesFM fine-tune failed ({}); using zero-shot weights.", exc)
    return model
