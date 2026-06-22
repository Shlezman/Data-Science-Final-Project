"""Leaderboard rigor helpers: bootstrap ROC-AUC CI (pure)."""

from __future__ import annotations

import importlib.util as _u
import math
from pathlib import Path

import numpy as np


def _load_pc():
    p = Path(__file__).resolve().parent.parent / "scripts" / "pipeline_compare.py"
    spec = _u.spec_from_file_location("pc", p)
    m = _u.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_auc_ci_random_straddles_half():
    m = _load_pc()
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    s = rng.random(400)                      # random scores → no skill
    lo, hi = m._auc_ci(s, y, n_boot=300)
    assert lo < 0.5 < hi                      # CI must straddle chance


def test_auc_ci_single_class_is_nan():
    m = _load_pc()
    lo, hi = m._auc_ci(np.array([0.3, 0.6, 0.4]), np.array([1, 1, 1]))
    assert math.isnan(lo) and math.isnan(hi)


def test_auc_ci_columns_in_cols():
    m = _load_pc()
    assert "auc_lo" in m._COLS and "auc_hi" in m._COLS


def test_track_of_parses_tag():
    m = _load_pc()
    assert m._track_of("GRU [scored/FULL+ovn]") == "FULL+ovn"
    assert m._track_of("Chronos-zeroshot [CUT]") == "CUT"
    assert m._track_of("Buy&Hold [FULL]") == "FULL"
    assert m._track_of("noBracket") is None


def test_abstention_lifts_accuracy_on_confident_subset():
    import pandas as pd
    m = _load_pc()
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    s = pd.Series([0.9, 0.1, 0.8, 0.2, 0.55, 0.45, 0.52, 0.48], index=idx)   # 4 high-conf, 4 low-conf
    y = pd.Series([1, 0, 1, 0, 0, 1, 0, 1], index=idx)                       # high-conf correct, low-conf wrong
    ab = m._abstention(s, y)
    assert ab[1.0] == 0.5           # acting on all → 50%
    assert ab[0.5] == 1.0           # acting on the most-confident half → 100%
