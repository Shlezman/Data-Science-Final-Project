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
