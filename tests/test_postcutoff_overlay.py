"""Post-cutoff buy-overlay scorecard tests (pure metric logic; finance mocked).

Exercises sentisense.hpo.optuna_lstm.postcutoff_buy_overlay — pre-cutoff model
decisions + forced-BUY post-cutoff, combined confusion/accuracy. No DB / no torch.
Run: uv run pytest tests/test_postcutoff_overlay.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from sentisense.hpo import optuna_lstm


def test_buy_overlay_combines_segments(monkeypatch):
    # Post-cutoff "truth" forced to a known direction series (finance loader mocked).
    post = pd.Series([1, 1, 0], name="Target",
                     index=pd.to_datetime(["2023-10-10", "2023-10-11", "2023-10-12"]))
    monkeypatch.setattr("sentisense.features.dataset.postcutoff_directions", lambda: post)

    # Pre-cutoff sacred test: probs [0.6, 0.3] @0.5 → decisions [1, 0]; truth [1, 0] → both right.
    proba = pd.Series([0.6, 0.3], index=pd.to_datetime(["2023-10-05", "2023-10-06"]))
    labels = pd.Series([1, 0], index=proba.index)

    out = optuna_lstm.postcutoff_buy_overlay(proba, labels, threshold=0.5)

    # Combined truth=[1,0,1,1,0], decision=[1,0,1,1,1]: tp=3, tn=1, fp=1, fn=0.
    assert (out["tp"], out["tn"], out["fp"], out["fn"]) == (3, 1, 1, 0)
    assert out["n_pre"] == 2 and out["n_post"] == 3
    assert out["combined_accuracy"] == pytest.approx(4 / 5)
    assert out["pre_cutoff_accuracy"] == pytest.approx(1.0)
    # post-cutoff days are FORCED buy → "buy-only accuracy" == real up-rate of post days.
    assert out["postcutoff_buy_accuracy"] == pytest.approx(2 / 3)


def test_buy_overlay_all_post_up_is_perfect_on_post(monkeypatch):
    post = pd.Series([1, 1, 1], name="Target",
                     index=pd.to_datetime(["2023-10-10", "2023-10-11", "2023-10-12"]))
    monkeypatch.setattr("sentisense.features.dataset.postcutoff_directions", lambda: post)
    proba = pd.Series([0.9], index=pd.to_datetime(["2023-10-05"]))
    labels = pd.Series([1], index=proba.index)

    out = optuna_lstm.postcutoff_buy_overlay(proba, labels, threshold=0.5)
    # All post days up + forced buy → no fp/fn on the post segment.
    assert out["fp"] == 0 and out["fn"] == 0
    assert out["postcutoff_buy_accuracy"] == pytest.approx(1.0)
    assert out["combined_accuracy"] == pytest.approx(1.0)
