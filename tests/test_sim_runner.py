"""MiroFish runner seed-window leak-safety + source-balanced seed (pure; no DB/service)."""

from __future__ import annotations

import pandas as pd

from sentisense.sim.runner import _balance_by_source, _compose_seed, seed_window


def _df(rows):
    return pd.DataFrame(rows, columns=["date", "source", "hour", "headline"])


def test_seed_window_strictly_past_and_inclusive_of_T():
    lo, hi = seed_window("2024-03-15", lookback=7)
    assert hi == pd.Timestamp("2024-03-15")        # T included (close-of-T news is known)
    assert lo == pd.Timestamp("2024-03-08")        # T - lookback
    assert lo < hi                                 # window never reaches > T (no future leak)


def test_seed_window_normalizes_time():
    lo, hi = seed_window(pd.Timestamp("2024-03-15 13:45"), lookback=3)
    assert hi == pd.Timestamp("2024-03-15")        # time component dropped
    assert (hi - lo).days == 3


def test_balance_by_source_caps_per_source_then_total():
    rows = [("2024-03-10", "Globes", h, f"g{h}") for h in range(6)]      # prolific outlet
    rows += [("2024-03-10", "Calcalist", h, f"c{h}") for h in range(2)]  # sparse outlet
    out = _balance_by_source(_df(rows), per_source_cap=3, total_cap=10)
    counts = out["source"].value_counts()
    assert counts["Globes"] == 3        # prolific outlet capped — volume skew killed
    assert counts["Calcalist"] == 2     # sparse outlet preserved in full
    # newest kept for the capped source (hours 3,4,5), not the oldest
    assert set(out[out.source == "Globes"]["hour"]) == {3, 4, 5}


def test_balance_by_source_applies_total_cap():
    rows = [("2024-03-10", "A", h, f"a{h}") for h in range(4)]
    rows += [("2024-03-10", "B", h, f"b{h}") for h in range(4)]
    out = _balance_by_source(_df(rows), per_source_cap=4, total_cap=5)
    assert len(out) == 5                # overall cap enforced after per-source


def test_compose_seed_emits_per_source_sections():
    df = _df([("2024-03-10", "Globes", 9, "rates up"),
              ("2024-03-10", "Calcalist", 8, "tech rally")])
    seed = _compose_seed(df, pd.Timestamp("2024-03-04"), pd.Timestamp("2024-03-10"))
    assert "distinct voice" in seed                  # perspective-aware preamble present
    assert "### Source: Globes (1 headlines)" in seed
    assert "### Source: Calcalist (1 headlines)" in seed
