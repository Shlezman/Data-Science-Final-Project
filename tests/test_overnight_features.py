"""Overnight global-feature block: unshifted day-T return, leak-safe vs the close(T) baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentisense.features.dataset import (
    _CROSS_ASSETS,
    _OVERNIGHT_ASSETS,
    add_cross_asset_features,
    add_overnight_features,
)


def test_overnight_is_unshifted_day_T_return():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    s = pd.Series([100.0, 101, 103, 102, 104, 105], index=idx)
    df = pd.DataFrame({"Market_SP500": s})
    lr = np.log(s / s.shift(1))
    ovn = add_overnight_features(df)["ovn_SP500_ret"]
    base = add_cross_asset_features(df)["SP500_logret_lag1"]
    # overnight = the day-T global return (no extra shift) — valid at open(T+1)
    assert np.allclose(ovn.dropna().values, lr.dropna().values)
    # the close(T)-safe baseline is exactly ONE more shift back (so it never uses day-T US)
    assert np.allclose(base.dropna().values, lr.shift(1).dropna().values)


def test_overnight_excludes_local_vta35_includes_nasdaq():
    assert "Nasdaq" in _CROSS_ASSETS            # new tech-heavy overnight driver
    assert "VTA35" not in _OVERNIGHT_ASSETS     # local same-day → not "overnight"
    assert {"SP500", "Nasdaq", "VIX"} <= set(_OVERNIGHT_ASSETS)


def test_overnight_emits_block_only_for_present_cols():
    df = pd.DataFrame({"Market_VIX": [20.0, 21, 19, 22]},
                      index=pd.date_range("2024-01-01", periods=4, freq="D"))
    cols = add_overnight_features(df).columns
    assert "ovn_VIX_ret" in cols and "ovn_VIX_2dret" in cols
    assert not any(c.startswith("ovn_SP500") for c in cols)   # absent asset → no column


def test_finalize_preserves_overnight_columns():
    from sentisense.features.dataset import _finalize
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    df = pd.DataFrame({"TA125_Price": [100.0, 101, 102, 103, 104, 105],
                       "ovn_SP500_ret": [0.01, -0.02, 0.0, 0.03, -0.01, 0.02]}, index=idx)
    out = _finalize(df, cutoff=pd.Timestamp("2100-01-01"))
    assert "Target" in out.columns and "ovn_SP500_ret" in out.columns   # ovn_ survives finalize


def test_cov_cols_includes_ovn_for_scored_only():
    import importlib.util as u
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "scripts" / "pipeline_compare.py"
    spec = u.spec_from_file_location("pc", p); m = u.module_from_spec(spec); spec.loader.exec_module(m)
    df = pd.DataFrame({"mean_a": [1.0], "ix_b": [2.0], "ovn_SP500_ret": [0.1],
                       "embc_000": [3.0], "emb_dispersion": [0.2], "emb_count": [4.0], "Target": [1]})
    assert "ovn_SP500_ret" in m._cov_cols(df, "scored").columns          # overnight reaches forecasters
    assert "embc_000" not in m._cov_cols(df, "scored").columns           # never the 768-d centroid
    assert "ovn_SP500_ret" not in m._cov_cols(df, "embedded").columns    # embedded cov = summaries only
