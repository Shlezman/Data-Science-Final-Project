"""Phase 3 feature tests — trading-calendar rollover, leak-free features, cutoff.

Exercises the pure helpers in sentisense.features.dataset (no DB / no finance deps).
Run: uv run pytest tests/test_features.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentisense.constants import CUTOFF_DATE
from sentisense.features import dataset as ds


# Sun–Thu trading week: 2023-10-01 (Sun) … 2023-10-05 (Thu), next 2023-10-08 (Sun).
TRADING = pd.DatetimeIndex([
    "2023-10-01", "2023-10-02", "2023-10-03", "2023-10-04", "2023-10-05",
    "2023-10-08", "2023-10-09",
])


def test_rollover_weekend_news_rolls_to_next_sunday():
    # Fri 2023-10-06 + Sat 2023-10-07 news must land on Sun 2023-10-08.
    news = pd.DataFrame(
        {"v": [1, 1, 1]},
        index=pd.DatetimeIndex(["2023-10-05", "2023-10-06", "2023-10-07"]),
    )
    rolled = ds._roll_to_trading_days(news, TRADING, agg="sum")
    assert rolled.loc["2023-10-05", "v"] == 1          # Thu stays Thu
    assert rolled.loc["2023-10-08", "v"] == 2          # Fri+Sat → next Sun
    # No backward leak: the Fri/Sat values never touch the prior trading day.
    assert rolled.loc["2023-10-04", "v"] == 0


def test_rollover_trading_day_maps_to_itself():
    news = pd.DataFrame({"v": [5]}, index=pd.DatetimeIndex(["2023-10-02"]))
    rolled = ds._roll_to_trading_days(news, TRADING, agg="sum")
    assert rolled.loc["2023-10-02", "v"] == 5


def test_rollover_after_last_trading_day_is_dropped():
    # News after the last trading day has nowhere to roll forward to → dropped.
    news = pd.DataFrame({"v": [9]}, index=pd.DatetimeIndex(["2023-10-20"]))
    rolled = ds._roll_to_trading_days(news, TRADING, agg="sum")
    assert rolled["v"].sum() == 0
    assert len(rolled) == len(TRADING)


def test_rollover_empty_days_are_zero_filled():
    news = pd.DataFrame({"v": [1]}, index=pd.DatetimeIndex(["2023-10-01"]))
    rolled = ds._roll_to_trading_days(news, TRADING, agg="sum")
    # Reindexed to the full trading calendar with fill_value=0.
    assert (rolled.loc[["2023-10-02", "2023-10-03"], "v"] == 0).all()


def test_add_ta125_features_is_leak_free():
    idx = pd.date_range("2023-01-01", periods=40, freq="D")
    price = pd.Series(np.linspace(100, 140, 40), index=idx)
    df = pd.DataFrame({"TA125_Volume": np.arange(1, 41)}, index=idx)
    out = ds.add_ta125_features(df, price)

    # lag1 log-return at row t must equal log(p[t-1]/p[t-2]) — strictly past data.
    expected = np.log(price.shift(1) / price.shift(2))
    got = out["TA125_logret_lag1"]
    mask = expected.notna() & got.notna()
    assert mask.sum() > 0
    assert np.allclose(got[mask].values, expected[mask].values)
    for col in ("TA125_RSI14", "TA125_logret_5d_mean", "TA125_volume_z20d"):
        assert col in out.columns


def test_finalize_applies_cutoff_and_drops_price():
    idx = pd.date_range("2023-10-01", periods=12, freq="D")
    df = pd.DataFrame({"TA125_Price": np.arange(100, 112), "feat": np.arange(12)}, index=idx)
    out = ds._finalize(df)
    assert out.index.max() <= pd.Timestamp(CUTOFF_DATE)   # cutoff held
    assert "TA125_Price" not in out.columns               # price dropped (leak guard)
    assert "Target" in out.columns
    assert set(out["Target"].unique()) <= {0, 1}


def test_finalize_target_is_next_day_direction():
    idx = pd.date_range("2023-09-20", periods=8, freq="D")
    # strictly increasing price → every next day is "up" (1), except the dropped last row
    df = pd.DataFrame({"TA125_Price": [10, 11, 12, 13, 14, 15, 16, 17]}, index=idx)
    out = ds._finalize(df)
    assert (out["Target"] == 1).all()


def test_chronological_split_scaler_fit_on_train_only():
    pytest.importorskip("sklearn")
    idx = pd.date_range("2022-01-01", periods=200, freq="D")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(5, 2, size=(200, 4)), index=idx,
                      columns=[f"f{i}" for i in range(4)])
    df["Target"] = rng.integers(0, 2, size=200)
    X_tr, y_tr, X_va, y_va, X_te, y_te, nf = ds_split(df)
    # Train slice standardised → ~zero mean; val/test use train stats so means drift.
    assert abs(X_tr.mean()) < 1e-6
    assert nf == 4
    assert len(X_tr) + len(X_va) + len(X_te) == 200


def ds_split(df):
    # Torch-free split — exercises the train-only scaler without importing torch.
    return ds.chronological_split(df)


def test_add_cross_asset_features_is_leak_free():
    idx = pd.date_range("2022-01-01", periods=30, freq="D")
    s = pd.Series(np.linspace(100, 130, 30), index=idx)
    df = pd.DataFrame({"Market_SP500": s, "Market_VIX": s * 0.2,
                       "Market_Brent_Oil": s * 0.8, "FX_USD_ILS": s * 0.03,
                       "VTA35_Price": s * 0.1}, index=idx)
    out = ds.add_cross_asset_features(df)
    logret = np.log(df["Market_SP500"] / df["Market_SP500"].shift(1))
    expected = logret.shift(1)            # lag1 uses strictly past data
    got = out["SP500_logret_lag1"]
    m = expected.notna() & got.notna()
    assert m.sum() > 0 and np.allclose(got[m].values, expected[m].values)
    for col in ("VIX_logret_lag1", "Brent_logret_5d_mean", "USDILS_logret_lag3", "VTA35_logret_lag1"):
        assert col in out.columns


def test_pca_prefix_only_reduces_centroid_block():
    pytest.importorskip("sklearn")
    idx = pd.date_range("2021-01-01", periods=200, freq="D")
    rng = np.random.default_rng(1)
    cols = {f"embc_{i:03d}": rng.normal(size=200) for i in range(40)}
    cols.update({f"fin_{i}": rng.normal(size=200) for i in range(6)})
    df = pd.DataFrame(cols, index=idx)
    df["Target"] = rng.integers(0, 2, 200)
    _, _, _, _, _, _, nf = ds.chronological_split(df, pca_components=10, pca_prefix="embc_")
    assert nf == 10 + 6   # 40 centroid → 10 PCA comps, 6 finance passthrough
