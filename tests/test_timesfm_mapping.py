"""TimesFM forecast→direction mapping + walk-forward no-leak tests (no model needed).

Exercises the pure harness in sentisense.models.timesfm_forecaster + the scoring bridge
in sentisense.models.backtest with an injected stub forecaster — so the leak-safety and
the continuous→binary mapping are verified without downloading TimesFM.
Run: uv run pytest tests/test_timesfm_mapping.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from sentisense.models.backtest import (
    direction_metrics,
    forecast_to_proba,
    max_drawdown,
    sharpe,
    strategy_stats,
)
from sentisense.models.timesfm_forecaster import walk_forward_directions


def test_forecast_to_proba_monotonic_and_signed():
    f = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    p = forecast_to_proba(f, scale=1.0)
    assert np.isclose(p[2], 0.5)                 # zero forecast → 0.5
    assert (p[:2] < 0.5).all()                   # negative → Down
    assert (p[3:] > 0.5).all()                   # positive → Up
    assert np.all(np.diff(p) > 0)                # strictly monotonic → ROC-AUC preserved
    assert (p >= 0).all() and (p <= 1).all()


def test_forecast_to_proba_default_scale_is_finite():
    p = forecast_to_proba(np.zeros(5))           # degenerate (zero std) → scale=1, all 0.5
    assert np.allclose(p, 0.5)


def test_walk_forward_no_leak_and_alignment():
    idx = pd.date_range("2020-01-01", periods=20, freq="D")
    r = pd.Series(np.linspace(-0.05, 0.05, 20), index=idx)   # strictly increasing returns
    seen: list[np.ndarray] = []

    def stub(ctx: np.ndarray) -> float:
        seen.append(np.array(ctx, copy=True))
        return float(ctx[-1])                    # "forecast" = last STRICTLY-PAST return

    test_index = idx[5:18]
    scores, labels = walk_forward_directions(r, test_index, stub, context_len=4, min_context=1)

    assert len(scores) == len(labels) == len(seen) > 0
    order = list(idx)
    for k, d in enumerate(scores.index):
        i = order.index(d)
        # label is the NEXT-day direction (the project target)
        assert labels[d] == int(r.iloc[i + 1] > 0)
        # leak-safety: context ends at the DECISION day r[i], never the future r[i+1]
        assert np.isclose(seen[k][-1], r.iloc[i])
        assert not np.isclose(seen[k][-1], r.iloc[i + 1])
        # context window respected (≤ context_len) and strictly past
        assert len(seen[k]) <= 4
        # mapping: score > 0.5  ⇔  forecast(=r[i]) > 0
        assert (scores[d] > 0.5) == (r.iloc[i] > 0)


def test_walk_forward_drops_last_day_without_future():
    idx = pd.date_range("2021-01-01", periods=10, freq="D")
    r = pd.Series(np.arange(10) - 4.5, index=idx)
    scores, labels = walk_forward_directions(r, idx, lambda c: float(c[-1]),
                                             context_len=5, min_context=1)
    assert idx[-1] not in scores.index          # last day has no next-day return → dropped


def test_direction_metrics_reuses_metrics_at():
    labels = np.array([0, 1, 0, 1, 1, 0])
    scores = np.array([0.2, 0.8, 0.4, 0.7, 0.9, 0.3])
    m = direction_metrics(scores, labels, 0.5)
    assert set(m) == {"accuracy", "balanced_accuracy", "f1", "roc_auc", "mcc"}
    assert m["accuracy"] == 1.0 and m["roc_auc"] == 1.0


def test_strategy_stats_shapes():
    sig = np.array([1, 0, 1, 1, 0], dtype=float)
    nxt = np.array([0.01, -0.02, 0.03, -0.01, 0.0])
    st = strategy_stats(sig, nxt)
    assert set(st) == {"cum_return", "sharpe", "max_drawdown"}
    assert st["max_drawdown"] <= 0.0
    assert np.isfinite(st["sharpe"])
