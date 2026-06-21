"""Chronos (Amazon) zero-shot time-series foundation model → next-day direction.

Like TimesFM: a frozen pretrained forecaster, so HPO is the context length (+ the decision
threshold, tuned on validation). Univariate — it forecasts the next RETURN from the return
history and ignores covariates. Reuses the leak-safe walk-forward bridge in
:mod:`sentisense.models.timesfm_forecaster` (forecast_fn sees only strictly-past context).

Install (server-side): ``uv pip install chronos-forecasting``  (Apache-2.0).
"""

from __future__ import annotations

import numpy as np
from loguru import logger

CHRONOS_MODEL = "amazon/chronos-bolt-small"   # fast default; -base/-large also work
CONTEXT_LEN = 512


def _install_hint() -> str:
    return ("Chronos not installed. Run:  uv pip install chronos-forecasting\n"
            "    (Apache-2.0; pulls torch + transformers.)")


def load_chronos(model_name: str = CHRONOS_MODEL):
    """Load a Chronos pipeline onto GPU if available (bfloat16) else CPU (float32)."""
    try:
        from chronos import BaseChronosPipeline
    except ImportError as exc:
        raise ImportError(_install_hint()) from exc
    import torch
    cuda = torch.cuda.is_available()
    pipe = BaseChronosPipeline.from_pretrained(
        model_name, device_map="cuda" if cuda else "cpu",
        torch_dtype=torch.bfloat16 if cuda else torch.float32)
    logger.info("Chronos '{}' loaded on {}.", model_name, "cuda" if cuda else "cpu")
    return pipe


def make_chronos_forecast_fn(pipe):
    """Return ``forecast_fn(context, cov_window=None) -> float`` (median next-step return).

    Matches the TimesFM forecast_fn contract so it drops into ``walk_forward_directions``.
    Covariates are ignored (Chronos is univariate). Handles both the Bolt
    (``predict_quantiles``) and T5 (``predict`` samples) APIs.
    """
    import torch

    def forecast_fn(context: np.ndarray, cov_window=None) -> float:   # noqa: ARG001 — univariate
        ctx = torch.tensor(np.asarray(context, dtype=np.float32))
        try:
            quantiles, _ = pipe.predict_quantiles(
                context=ctx, prediction_length=1, quantile_levels=[0.5])
            return float(np.asarray(quantiles)[0, 0, 0])      # median of the 1-step horizon
        except AttributeError:
            fc = pipe.predict(ctx, prediction_length=1)        # (1, num_samples, 1)
            return float(np.median(np.asarray(fc)[0, :, 0]))

    return forecast_fn
