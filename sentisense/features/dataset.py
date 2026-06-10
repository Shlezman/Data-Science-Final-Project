"""Build the leakage-safe daily modeling frames (daily-mean + per-source) ≤ cutoff.

Faithful port of transformer_forecaster.ipynb cells 6/8/9/10/12, with three
deliberate hardening changes documented inline:
  1. The news cutoff is pushed into the SQL (`rh.date <= :cutoff`) AND re-applied
     after the calendar merge — defense in depth.
  2. The news query is parameterized via SQLAlchemy (model + cutoff bound), not an
     inline literal.
  3. The VTA-35 MinMaxScaler that the notebook fit on the WHOLE frame (a leak) is
     replaced by a leak-free fillna(0.0) + a `VTA35_missing` indicator; the
     train-only StandardScaler in :mod:`sentisense.models.sequence` does the scaling.

Returns two frames, both indexed by trading day with a `Target` column:
  * ``mt`` — daily-MEAN strategy (~tree-model shape)
  * ``ml`` — per-source SUM pivot (~LSTM shape)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

from sentisense.config import TOP_N_SOURCES
from sentisense.constants import (
    CUTOFF_DATE,
    CUTOFF_DATE_ISO,
    ACTIVE_MODEL_NAME,
    SCORE_COLUMNS,
    TA125_CSV,
    VTA35_CSV,
    VTA35_INCEPTION,
)
from sentisense.db import get_engine

_SCORE_COLS = list(SCORE_COLUMNS)

# Cutoff pushed into SQL; model + validation filters match the standardised dataset.
_RAW_SCORES_SQL = text(
    """
    SELECT rh.date::date AS date,
           rh.source,
           nv.relevance_politics,
           nv.relevance_economy,
           nv.relevance_security,
           nv.relevance_health,
           nv.relevance_science,
           nv.relevance_technology,
           nv.global_sentiment
    FROM raw_headlines rh
    JOIN nlp_vectors nv ON nv.headline_id = rh.id
    WHERE nv.validation_passed = TRUE
      AND nv.model_name = :model
      AND rh.date <= :cutoff
    """
)


def _load_raw_scores(engine) -> pd.DataFrame:
    """Load validated, cutoff-bound, single-model news scores into a DataFrame."""
    with engine.connect() as conn:
        df = pd.read_sql(_RAW_SCORES_SQL, conn,
                         params={"model": ACTIVE_MODEL_NAME, "cutoff": CUTOFF_DATE})
    df["date"] = pd.to_datetime(df["date"])
    logger.info("Loaded {:,} validated rows (<= {}), {} sources",
                len(df), CUTOFF_DATE_ISO, df["source"].nunique())
    return df


def _safe_col(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in str(name))


def _build_daily_mean(raw: pd.DataFrame) -> pd.DataFrame:
    dm = raw.groupby("date", observed=True)[_SCORE_COLS].mean().add_prefix("mean_")
    dm["n_headlines"] = raw.groupby("date", observed=True).size()
    return dm


def _build_per_source_wide(raw: pd.DataFrame, top_n: int) -> pd.DataFrame:
    top_sources = raw["source"].value_counts().head(top_n).index.tolist()
    long = raw.groupby(["date", "source"], observed=True)[_SCORE_COLS].sum().reset_index()
    long["count"] = (
        raw.groupby(["date", "source"], observed=True).size().reset_index(name="count")["count"]
    )
    long["source_group"] = long["source"].apply(
        lambda s: _safe_col(s) if s in top_sources else "_other"
    )
    grouped = long.groupby(["date", "source_group"], observed=True)[[*_SCORE_COLS, "count"]].sum().reset_index()
    pivots = []
    for col in [*_SCORE_COLS, "count"]:
        p = grouped.pivot(index="date", columns="source_group", values=col).fillna(0)
        p.columns = [f"{col}_{s}" for s in p.columns]
        pivots.append(p)
    return pd.concat(pivots, axis=1).sort_index()


def _load_finance() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load TA-125 + VTA-35 (CSV) and S&P/VIX/Brent (yfinance) + USD/ILS (Frankfurter).

    Lazy-imports yfinance/requests (the ``finance`` extra) so this module imports
    cleanly in the Phase 0/1 base env.
    """
    import requests
    import yfinance as yf

    def convert_volume(val):
        if pd.isna(val):
            return 0.0
        s = str(val).upper().replace(",", "")
        if s.endswith("M"):
            return float(s[:-1]) * 1e6
        if s.endswith("B"):
            return float(s[:-1]) * 1e9
        if s.endswith("K"):
            return float(s[:-1]) * 1e3
        try:
            return float(s)
        except ValueError:
            return 0.0

    def to_float(s):
        return (s.astype(float) if pd.api.types.is_numeric_dtype(s)
                else s.astype(str).str.replace(",", "", regex=False).astype(float))

    ta125 = pd.read_csv(TA125_CSV)
    ta125["Date"] = pd.to_datetime(ta125["Date"])
    ta125 = ta125.set_index("Date").sort_index()
    ta125_clean = pd.DataFrame({
        "TA125_Price": to_float(ta125["Price"]),
        "TA125_Volume": ta125["Vol."].apply(convert_volume),
    })

    vta35 = pd.read_csv(VTA35_CSV)
    vta35["Date"] = pd.to_datetime(vta35["Date"])
    vta35 = vta35.set_index("Date").sort_index()
    vta35_clean = pd.DataFrame({"VTA35_Price": to_float(vta35["Price"])})
    vta35_clean.loc[vta35_clean.index < pd.Timestamp(VTA35_INCEPTION), "VTA35_Price"] = np.nan

    start = "2015-12-17"
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    market = yf.download(["^GSPC", "^VIX", "BZ=F"], start=start, end=end, progress=False)["Close"]
    # yfinance returns multi-ticker Close columns alphabetically (BZ=F, ^GSPC, ^VIX).
    market.columns = ["Brent_Oil", "SP500", "VIX"]
    market_clean = market.add_prefix("Market_")

    resp = requests.get(f"https://api.frankfurter.app/{start}..?from=USD&to=ILS", timeout=30)
    resp.raise_for_status()
    fx = pd.DataFrame.from_dict(resp.json()["rates"], orient="index")
    fx.index = pd.to_datetime(fx.index)
    fx.columns = ["FX_USD_ILS"]
    fx_clean = fx.sort_index()
    return ta125_clean, vta35_clean, market_clean, fx_clean


def _roll_to_trading_days(df: pd.DataFrame, trading_days: pd.DatetimeIndex, agg: str) -> pd.DataFrame:
    """Roll calendar dates forward to the next trading day (Fri/Sat → Sun), then aggregate."""
    arr = np.asarray(trading_days)
    pos = np.searchsorted(arr, df.index.values, side="left")
    mask = pos < len(arr)
    attached = df.iloc[mask].copy()
    attached["_td"] = arr[pos[mask]]
    result = attached.groupby("_td").agg(agg)
    result.index.name = "date"
    return result.reindex(trading_days, fill_value=0)


def _roll_mean_and_count(dm: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    arr = np.asarray(trading_days)
    pos = np.searchsorted(arr, dm.index.values, side="left")
    mask = pos < len(arr)
    attached = dm.iloc[mask].copy()
    attached["_td"] = arr[pos[mask]]
    score_cols = [c for c in attached.columns if c not in ("n_headlines", "_td")]
    mean_part = attached.groupby("_td")[score_cols].mean()
    result = mean_part
    if "n_headlines" in attached.columns:
        result = result.join(attached.groupby("_td")["n_headlines"].sum())
    result.index.name = "date"
    return result.reindex(trading_days)


def add_ta125_features(df: pd.DataFrame, price_series: pd.Series) -> pd.DataFrame:
    """Append leak-free TA-125 technical features (all use .shift(>=1))."""
    df = df.copy()
    p = price_series.reindex(df.index).astype(float)

    logret = np.log(p / p.shift(1))
    for lag in range(1, 8):
        df[f"TA125_logret_lag{lag}"] = logret.shift(lag)
    df["TA125_logret_5d_mean"] = logret.shift(1).rolling(5).mean()
    df["TA125_logret_5d_std"] = logret.shift(1).rolling(5).std()
    df["TA125_logret_20d_std"] = logret.shift(1).rolling(20).std()

    delta = p.diff()
    gain = (delta.clip(lower=0)).shift(1).rolling(14).mean()
    loss = (-delta.clip(upper=0)).shift(1).rolling(14).mean()
    rs = gain / loss
    df["TA125_RSI14"] = 100 - (100 / (1 + rs))

    if "TA125_Volume" in df.columns:
        v = df["TA125_Volume"]
        df["TA125_volume_z20d"] = (v - v.shift(1).rolling(20).mean()) / v.shift(1).rolling(20).std()

    dow = pd.get_dummies(df.index.dayofweek, prefix="DoW", drop_first=True).astype(int)
    dow.index = df.index
    return df.join(dow)


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Compute target, leak-free VTA-35 handling, NaN/inf cleanup, cutoff slice."""
    df = df.copy()
    next_price = df["TA125_Price"].shift(-1)
    df["Target"] = (next_price > df["TA125_Price"]).astype("Int64")
    # `NaN > x` yields False (not NA) in pandas, so the trailing row with no next-day
    # price would get a definitive WRONG 0 label. Explicitly NA those rows so the
    # notna() filter below drops them — no fabricated label, no leak.
    df.loc[next_price.isna(), "Target"] = pd.NA
    df = df.drop(columns=["TA125_Price"])

    # Leak-free VTA-35: no full-frame MinMax (the notebook's leak). Indicator + 0-fill;
    # the train-only StandardScaler in prepare_data() does the actual scaling.
    if "VTA35_Price" in df.columns:
        df["VTA35_missing"] = df["VTA35_Price"].isna().astype(int)
        df["VTA35_Price"] = df["VTA35_Price"].fillna(0.0)

    # Drop the final row (no next-day target) BEFORE the frame-level fillna —
    # otherwise the Int64 NA target is filled to a bogus 0 (a wrong label / leak).
    df = df[df["Target"].notna()].copy()
    df = df.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    df["Target"] = df["Target"].astype(int)

    # Hard cutoff (defense in depth — SQL already bounds news, but trading_days come
    # from the CSV which extends past the cutoff).
    df = df[df.index <= pd.Timestamp(CUTOFF_DATE)]
    return df


def chronological_split(df: pd.DataFrame, *, val_frac: float = 0.15, test_frac: float = 0.15):
    """Train-only-scaled chronological split (torch-free; sklearn lazy-imported).

    Returns ``(X_tr, y_tr, X_va, y_va, X_te, y_te, n_features)`` with the
    StandardScaler fit on the training slice only — no future leak. Lives here (not
    in models.sequence) so it can be used without importing torch.
    """
    from sklearn.preprocessing import StandardScaler

    y = df["Target"].values.astype(np.float32)
    X = df.drop(columns=["Target"]).values.astype(np.float32)
    n = len(df)
    n_val, n_test = int(n * val_frac), int(n * test_frac)
    n_train = n - n_val - n_test

    X_tr, X_va, X_te = X[:n_train], X[n_train:n_train + n_val], X[n_train + n_val:]
    y_tr, y_va, y_te = y[:n_train], y[n_train:n_train + n_val], y[n_train + n_val:]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)
    return X_tr, y_tr, X_va, y_va, X_te, y_te, X.shape[1]


def build_datasets(
    engine=None,
    *,
    top_n: int = TOP_N_SOURCES,
    extra_daily_features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble the (daily-mean, per-source) modeling frames, cutoff-applied.

    Args:
        engine: SQLAlchemy engine; created from env if None.
        top_n: Number of highest-volume sources kept per-source; rest → ``_other``.
        extra_daily_features: Optional frame indexed by trading day (e.g. the
            narrative ``dominant_cluster_ratio`` from Phase 4) joined into BOTH
            frames before feature engineering. Must already be leakage-safe.

    Returns:
        ``(mt, ml)`` — daily-mean and per-source frames, each with a ``Target`` column.
    """
    engine = engine or get_engine()
    raw = _load_raw_scores(engine)
    daily_mean = _build_daily_mean(raw)
    per_source = _build_per_source_wide(raw, top_n)

    ta125, vta35, market, fx = _load_finance()
    trading_days = pd.DatetimeIndex(ta125.index).sort_values()

    base = (
        pd.DataFrame(index=trading_days)
        .join(ta125, how="left").join(vta35, how="left")
        .join(market, how="left").join(fx, how="left")
    )
    base.index.name = "date"
    ffill_cols = [c for c in base.columns if c.startswith(("Market_", "FX_", "VTA35_"))]
    base[ffill_cols] = base[ffill_cols].ffill()

    if extra_daily_features is not None and not extra_daily_features.empty:
        extra = extra_daily_features.reindex(trading_days)
        base = base.join(extra, how="left")

    dm_td = _roll_mean_and_count(daily_mean, trading_days)
    ps_td = _roll_to_trading_days(per_source, trading_days, agg="sum")

    merged_trees = base.join(dm_td, how="left")
    merged_lstm = base.join(ps_td, how="left")

    price_full = ta125["TA125_Price"].reindex(trading_days)
    mt = _finalize(add_ta125_features(merged_trees, price_full))
    ml = _finalize(add_ta125_features(merged_lstm, price_full))

    logger.info("Datasets built (<= {}): mt={}, ml={}", CUTOFF_DATE_ISO, mt.shape, ml.shape)
    logger.info("  trading-day rows: mt={:,}, ml={:,}  (LSTM-viability bar ~750)", len(mt), len(ml))
    return mt, ml
