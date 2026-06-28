"""H-day-ahead target generalisation + the horizon-aware block-bootstrap CI."""

from __future__ import annotations

import importlib.util as _u
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sqlalchemy")


def test_finalize_horizon_target_and_trailing_drop():
    from sentisense.features.dataset import _finalize

    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    df = pd.DataFrame({"TA125_Price": [100, 90, 110, 95, 105, 80, 120, 100]}, index=idx, dtype=float)
    out = _finalize(df.copy(), cutoff=pd.Timestamp("2100-01-01"), horizon=2)
    # Target[T] = close(T+2) > close(T); the trailing 2 rows (no future price) are dropped.
    assert len(out) == 8 - 2
    assert out["Target"].tolist() == [1, 1, 0, 0, 1, 1]


def test_finalize_default_horizon_is_next_day():
    from sentisense.features.dataset import _finalize

    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    df = pd.DataFrame({"TA125_Price": [100.0, 101, 100, 102]}, index=idx)
    out = _finalize(df.copy(), cutoff=pd.Timestamp("2100-01-01"))   # horizon defaults to 1
    assert out["Target"].tolist() == [1, 0, 1]                       # last row dropped


def _load_sweep():
    pytest.importorskip("sklearn")
    p = Path(__file__).resolve().parent.parent / "scripts" / "horizon_sweep.py"
    spec = _u.spec_from_file_location("hs", p)
    m = _u.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_block_bootstrap_random_straddles_half():
    m = _load_sweep()
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    s = rng.random(400)                              # random scores → no skill
    lo, hi = m._auc_ci_block(s, y, block=3, n_boot=300)
    assert lo < 0.5 < hi


def test_block_bootstrap_single_class_is_nan():
    m = _load_sweep()
    lo, hi = m._auc_ci_block(np.array([0.3, 0.6, 0.4]), np.array([1, 1, 1]))
    assert math.isnan(lo) and math.isnan(hi)
