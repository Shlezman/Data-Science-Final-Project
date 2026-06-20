"""Shared scoring + backtest helpers — one source for every forecaster.

Reuses :func:`sentisense.models.train.metrics_at` for the classification metrics
(accuracy / balanced-accuracy / F1 / ROC-AUC / MCC) and adds the equity-curve /
Sharpe / max-drawdown helpers that were previously inlined in the analysis notebook,
so XGBoost, LSTM, and TimesFM are all scored identically (apples-to-apples).

The forecast→direction bridge (:func:`forecast_to_proba`) maps a continuous forecast
(e.g. a TimesFM predicted return) onto a [0, 1] pseudo-probability whose 0.5 threshold
is exactly ``forecast > 0`` and whose ordering is preserved — so ``metrics_at`` gives a
valid ROC-AUC on the signed forecast magnitude and the SAME thresholded acc/F1 as the
classifiers, with no metric re-implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentisense.models.train import metrics_at

TRADING_DAYS = 252


def forecast_to_proba(forecast: np.ndarray, scale: float | None = None) -> np.ndarray:
    """Map a continuous forecast to a [0, 1] pseudo-probability for direction scoring.

    ``p = 0.5 + 0.5 * tanh(forecast / scale)`` — monotonic in ``forecast`` (so ROC-AUC is
    unchanged vs scoring the raw forecast) and ``p > 0.5 ⇔ forecast > 0`` (so a 0.5
    threshold reproduces the sign decision). ``scale`` defaults to the forecast std
    (robust to the series' units); a degenerate (zero) scale falls back to 1.0.
    """
    f = np.asarray(forecast, dtype=float)
    if scale is None:
        s = float(np.std(f))
        scale = s if s > 1e-12 else 1.0
    return 0.5 + 0.5 * np.tanh(f / scale)


def direction_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    """Classification metrics for a direction call — thin reuse of ``metrics_at``."""
    return metrics_at(np.asarray(scores), np.asarray(labels), threshold)


def sharpe(returns: np.ndarray, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe of a per-period return series (0 mean/vol → 0)."""
    r = np.asarray(returns, dtype=float)
    sd = r.std()
    return float(np.sqrt(periods) * r.mean() / sd) if sd > 1e-12 else 0.0


def max_drawdown(equity: np.ndarray) -> float:
    """Maximum drawdown of an equity curve (negative fraction, e.g. -0.23)."""
    eq = np.asarray(equity, dtype=float)
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min())


def equity_curve(signal: np.ndarray, next_returns: np.ndarray) -> np.ndarray:
    """Cumulative equity of ``signal`` (0/1 long-flat or weight) × next-period return."""
    strat = np.asarray(signal, dtype=float) * np.asarray(next_returns, dtype=float)
    return np.cumprod(1.0 + strat)


def strategy_stats(signal: np.ndarray, next_returns: np.ndarray,
                   periods: int = TRADING_DAYS) -> dict:
    """Cumulative return, Sharpe, and max drawdown for a long/flat strategy."""
    strat = np.asarray(signal, dtype=float) * np.asarray(next_returns, dtype=float)
    eq = np.cumprod(1.0 + strat)
    return {
        "cum_return": float(eq[-1] - 1.0) if eq.size else 0.0,
        "sharpe": sharpe(strat, periods),
        "max_drawdown": max_drawdown(eq),
    }


def next_day_returns(price: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """Realised NEXT-day simple return for each date in ``dates`` (0 where unknown)."""
    ret = price.sort_index().pct_change()
    return ret.reindex(dates).shift(-1).fillna(0.0).to_numpy()


def directions_from_price(price: pd.Series) -> pd.Series:
    """Date-indexed next-day direction label (1 = next close up); last row dropped."""
    p = price.sort_index().astype(float)
    nxt = p.shift(-1)
    d = (nxt > p).astype("Int64")
    d[nxt.isna()] = pd.NA
    return d[d.notna()].astype(int).rename("Target")
