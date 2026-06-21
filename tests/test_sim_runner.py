"""MiroFish runner seed-window leak-safety (pure; no DB/service)."""

from __future__ import annotations

import pandas as pd

from sentisense.sim.runner import seed_window


def test_seed_window_strictly_past_and_inclusive_of_T():
    lo, hi = seed_window("2024-03-15", lookback=7)
    assert hi == pd.Timestamp("2024-03-15")        # T included (close-of-T news is known)
    assert lo == pd.Timestamp("2024-03-08")        # T - lookback
    assert lo < hi                                 # window never reaches > T (no future leak)


def test_seed_window_normalizes_time():
    lo, hi = seed_window(pd.Timestamp("2024-03-15 13:45"), lookback=3)
    assert hi == pd.Timestamp("2024-03-15")        # time component dropped
    assert (hi - lo).days == 3
