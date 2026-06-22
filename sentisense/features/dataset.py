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
    SCORE_COLUMNS,
    TA125_CSV,
    VTA35_CSV,
    VTA35_INCEPTION,
)
from sentisense.db import get_engine

_SCORE_COLS = list(SCORE_COLUMNS)

# Combine ALL validated scores regardless of model_name, one row per headline (the
# latest by created_at). The corpus mixes models on disjoint date ranges
# (mistral-small-4 recent + locally-backfilled mistral-small3.2 olds); this uses the
# whole corpus. DISTINCT ON (rh.id) + ORDER BY created_at DESC = latest score per
# headline. Cutoff pushed into SQL.
_RAW_SCORES_SQL = text(
    """
    SELECT DISTINCT ON (rh.id)
           rh.id          AS headline_id,
           rh.date::date  AS date,
           rh.source,
           nv.model_name,
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
      AND rh.date <= :cutoff
    ORDER BY rh.id, nv.created_at DESC, nv.id DESC
    """
)


def _load_raw_scores(engine, cutoff=CUTOFF_DATE) -> pd.DataFrame:
    """Load validated news scores up to ``cutoff`` — all models, one row per headline.

    ``cutoff`` defaults to the project cutoff (the leak-safe modeling bound); pass a
    later date to include more history (e.g. the full-history comparison pipeline).

    NOTE: scores from different LLM models (mistral-small-4 vs mistral-small3.2) may
    differ in scale/quality, so the feature distribution can shift at the date where
    the scoring model changes. This is an accepted trade-off for using the full
    backfilled corpus (operator chose 'combine all models').
    """
    with engine.connect() as conn:
        df = pd.read_sql(_RAW_SCORES_SQL, conn, params={"cutoff": cutoff})
    df["date"] = pd.to_datetime(df["date"])
    models = sorted(df["model_name"].unique()) if not df.empty else []
    logger.info("Loaded {:,} validated headlines (<= {}), {} sources, models={}",
                len(df), pd.Timestamp(cutoff).date(), df["source"].nunique(), models)
    # model_name was only for dedupe/logging; drop before aggregation.
    return df.drop(columns=["model_name", "headline_id"])


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
    # ^IXIC (Nasdaq) added — tech-heavy, strong overnight driver of TA-125. Rename BY TICKER
    # (not positional) so column order from yfinance can't silently mis-map.
    _tk = {"^GSPC": "SP500", "^IXIC": "Nasdaq", "^VIX": "VIX", "BZ=F": "Brent_Oil"}
    market = yf.download(list(_tk), start=start, end=end, progress=False)["Close"].rename(columns=_tk)
    present = [c for c in _tk.values() if c in market.columns]   # degrade gracefully if a ticker is missing
    if "SP500" not in present:
        raise RuntimeError("yfinance returned no S&P 500 (^GSPC) — finance load failed; retry.")
    if len(present) < len(_tk):
        logger.warning("yfinance missing {} — proceeding without them.",
                       sorted(set(_tk.values()) - set(present)))
    market_clean = market[present].add_prefix("Market_")

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
    """Append TA-125 technical features for next-day prediction.

    All price-derived features use ``.shift(>=1)`` (no future leak). NOTE: the raw
    ``TA125_Volume`` column and the ``TA125_volume_z20d`` numerator are intentional
    SAME-DAY (close-of-T) features — standard for next-day direction models, since the
    target is ``close(T+1) > close(T)`` and day-T close-derived signals are known at
    prediction time. They are not future-into-past leaks.
    """
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


# Cross-asset price columns → leak-free lagged-return features. Index direction is
# often driven more by global moves than local news, so these usually add real signal.
_CROSS_ASSETS = {
    "SP500": "Market_SP500",
    "Nasdaq": "Market_Nasdaq",
    "VIX": "Market_VIX",
    "Brent": "Market_Brent_Oil",
    "USDILS": "FX_USD_ILS",
    "VTA35": "VTA35_Price",
}
# Global assets whose close(T) lands AFTER TA-125 close(T) but BEFORE open(T+1) → the
# genuine "overnight" signal (VTA-35 is local same-day, so it's excluded here).
_OVERNIGHT_ASSETS = {k: v for k, v in _CROSS_ASSETS.items() if k != "VTA35"}


def add_cross_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lagged log-returns + short rolling stats for S&P/VIX/Brent/USDILS/VTA-35.

    All use ``.shift(>=1)`` (causal). 0-sentinels (e.g. pre-inception VTA-35) are
    treated as missing so we never take log(0); _finalize fills the resulting NaNs.
    """
    df = df.copy()
    for name, col in _CROSS_ASSETS.items():
        if col not in df.columns:
            continue
        s = df[col].astype(float).replace(0.0, np.nan)  # 0 = missing sentinel → avoid log(0)
        logret = np.log(s / s.shift(1))
        df[f"{name}_logret_lag1"] = logret.shift(1)
        df[f"{name}_logret_lag2"] = logret.shift(2)
        df[f"{name}_logret_lag3"] = logret.shift(3)
        df[f"{name}_logret_5d_mean"] = logret.shift(1).rolling(5).mean()
        df[f"{name}_logret_5d_std"] = logret.shift(1).rolling(5).std()
    return df


def add_overnight_features(df: pd.DataFrame) -> pd.DataFrame:
    """Overnight global-close returns — for the OPEN(T+1) decision contract ONLY.

    add_cross_asset_features shifts ≥1 so it's safe for a CLOSE(T) decision. These use the
    day-T close-to-close return with NO extra shift: the most recent global close BEFORE
    TA-125's open(T+1) (US/Nasdaq/VIX/Brent/USD-ILS all settle after TA close(T) but before
    its next open), which is exactly the gap-driving signal. Known at open(T+1); a LEAK for a
    close(T) decision — hence the separate ``ovn_`` block + the build_datasets(overnight=)
    flag. Never derived from TA-125 itself, so it cannot peek at the target.
    """
    df = df.copy()
    for name, col in _OVERNIGHT_ASSETS.items():
        if col not in df.columns:
            continue
        s = df[col].astype(float).replace(0.0, np.nan)
        logret = np.log(s / s.shift(1))
        df[f"ovn_{name}_ret"] = logret                      # day-T global move (known at open T+1)
        df[f"ovn_{name}_2dret"] = logret.rolling(2).sum()   # 2-day momentum into the open
    return df


def _build_interactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Global daily sentiment×relevance interaction features (per raw date).

    Domain prior: sentiment matters more when economy/security relevance is high, and
    headline *volume* + sentiment *intensity* carry signal beyond the means. Joined into
    both modeling frames so every model can use them.
    """
    relevance_cols = [c for c in SCORE_COLUMNS if c != "global_sentiment"]
    g = raw.groupby("date", observed=True)
    econ = g["relevance_economy"].mean()
    sec = g["relevance_security"].mean()
    pol = g["relevance_politics"].mean()
    sent = g["global_sentiment"].mean()
    out = pd.DataFrame({
        "ix_econ_sent": econ * sent,
        "ix_sec_sent": sec * sent,
        "ix_pol_sent": pol * sent,
        "ix_total_relevance": g[relevance_cols].mean().sum(axis=1),
        "ix_sent_intensity": g["global_sentiment"].apply(lambda s: s.abs().mean()),
        "ix_sent_dispersion": g["global_sentiment"].std(),
    })
    return out


def _finalize(df: pd.DataFrame, cutoff=CUTOFF_DATE) -> pd.DataFrame:
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
    # from the CSV which extends past the cutoff). Parameterised so the full-history
    # comparison can build the same frame over every date.
    df = df[df.index <= pd.Timestamp(cutoff)]
    return df


def chronological_split(df: pd.DataFrame, *, val_frac: float = 0.15, test_frac: float = 0.15,
                        pca_components: int | None = None, pca_prefix: str | None = None):
    """Train-only-scaled chronological split (torch-free; sklearn lazy-imported).

    StandardScaler (and optional PCA) is fit on the TRAIN slice only — no future leak.

    ``pca_components`` reduces dimensionality after scaling. ``pca_prefix`` scopes PCA
    to only the columns whose name starts with that prefix (e.g. 'embc_' — the embedding
    centroid block), passing finance/TA-125/dispersion features through un-reduced.
    Without a prefix, PCA applies to all features.
    """
    from sklearn.preprocessing import StandardScaler

    feat_cols = df.drop(columns=["Target"]).columns
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

    if pca_components:
        import numpy as _np
        from sklearn.decomposition import PCA

        if pca_prefix:
            mask = _np.array([c.startswith(pca_prefix) for c in feat_cols])
        else:
            mask = _np.ones(len(feat_cols), dtype=bool)
        if mask.sum() > pca_components:
            pca = PCA(n_components=pca_components, random_state=0).fit(X_tr[:, mask])  # TRAIN-only
            def _reduce(a):
                return _np.hstack([pca.transform(a[:, mask]), a[:, ~mask]])
            X_tr, X_va, X_te = _reduce(X_tr), _reduce(X_va), _reduce(X_te)
        else:
            logger.warning("PCA skipped: block to reduce ({}) <= pca_components ({}) — "
                           "running un-reduced.", int(mask.sum()), pca_components)

    return (X_tr.astype(np.float32), y_tr, X_va.astype(np.float32), y_va,
            X_te.astype(np.float32), y_te, X_tr.shape[1])


def build_datasets(
    engine=None,
    *,
    top_n: int = TOP_N_SOURCES,
    extra_daily_features: pd.DataFrame | None = None,
    cutoff=CUTOFF_DATE,
    overnight: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble the (daily-mean, per-source) modeling frames, cutoff-applied.

    Args:
        engine: SQLAlchemy engine; created from env if None.
        top_n: Number of highest-volume sources kept per-source; rest → ``_other``.
        extra_daily_features: Optional frame indexed by trading day (e.g. the
            narrative ``dominant_cluster_ratio`` from Phase 4) joined into BOTH
            frames before feature engineering. Must already be leakage-safe.
        cutoff: Upper date bound (default = project cutoff). Pass a later date to
            build over more history (full-history comparison pipeline).

    Returns:
        ``(mt, ml)`` — daily-mean and per-source frames, each with a ``Target`` column.
    """
    engine = engine or get_engine()
    raw = _load_raw_scores(engine, cutoff)
    daily_mean = _build_daily_mean(raw)
    per_source = _build_per_source_wide(raw, top_n)

    base, trading_days, price_full = _finance_base(extra_daily_features)

    # Global sentiment×relevance interactions → both frames (rolled mean to trading days).
    interactions_td = _roll_to_trading_days(_build_interactions(raw), trading_days, agg="mean")
    base = base.join(interactions_td, how="left")

    dm_td = _roll_mean_and_count(daily_mean, trading_days)
    ps_td = _roll_to_trading_days(per_source, trading_days, agg="sum")

    def _assemble(news_td):
        feat = add_cross_asset_features(add_ta125_features(base.join(news_td, how="left"), price_full))
        if overnight:                       # open(T+1)-decision overnight global block
            feat = add_overnight_features(feat)
        return _finalize(feat, cutoff)

    mt = _assemble(dm_td)
    ml = _assemble(ps_td)

    logger.info("Datasets built (<= {}, overnight={}): mt={}, ml={}",
                pd.Timestamp(cutoff).date(), overnight, mt.shape, ml.shape)
    logger.info("  trading-day rows: mt={:,}, ml={:,}  (LSTM-viability bar ~750)", len(mt), len(ml))
    return mt, ml


def _finance_base(extra_daily_features: pd.DataFrame | None = None):
    """Build the finance/market base frame on the TA-125 trading calendar.

    Returns ``(base, trading_days, price_full)`` — shared by the score dataset and
    the embedding dataset so both sit on the identical calendar + finance block.
    """
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
        base = base.join(extra_daily_features.reindex(trading_days), how="left")

    price_full = ta125["TA125_Price"].reindex(trading_days)
    return base, trading_days, price_full


def build_embedding_dataset(engine=None, *, cutoff=CUTOFF_DATE, overnight: bool = False) -> pd.DataFrame:
    """Daily e5-centroid dataset for the 'embedded data' LSTM (PCA applied at train time).

    Per trading day: the MEAN of that day's headline embeddings (rolled Fri/Sat → Sun
    like the news), giving an ``emb_000..emb_NNN`` daily centroid, merged with the same
    finance/calendar block + leak-free TA-125 features + next-day target + cutoff.

    ``cutoff`` bounds the history (default = project cutoff; pass a later date for the
    full-history comparison). Dimensionality reduction (PCA → ~50-d) is NOT done here; it
    is fit on the TRAIN fold only inside the HPO/eval split to stay leakage-safe. Returns
    an empty frame if no embeddings are cached.
    """
    engine = engine or get_engine()
    from sentisense.embed import daily_embedding_centroid

    # Streamed per-date centroid: never materialises the full ~3M×768 matrix (OOM-safe on
    # the full-history corpus). 'embc_*' so PCA (pca_prefix='embc_') reduces ONLY the
    # centroid block — the scalar dispersion/count + finance/TA-125 features stay raw.
    cen = daily_embedding_centroid(engine, cutoff)
    if cen.empty:
        logger.warning("No embeddings cached — run the 'embed' stage first. "
                       "Returning empty embedding dataset.")
        return pd.DataFrame()
    centroid_by_date = cen[[c for c in cen.columns if c.startswith("embc_")]]
    extras = cen[["emb_dispersion", "emb_count"]]
    dim = centroid_by_date.shape[1]

    base, trading_days, price_full = _finance_base()
    emb_td = _roll_to_trading_days(centroid_by_date, trading_days, agg="mean")
    extras_td = _roll_to_trading_days(extras, trading_days, agg="mean")
    merged = base.join(emb_td, how="left").join(extras_td, how="left")
    feat = add_cross_asset_features(add_ta125_features(merged, price_full))
    if overnight:
        feat = add_overnight_features(feat)
    df = _finalize(feat, cutoff)
    logger.info("Embedding dataset built (<= {}, overnight={}): {} ({}-d centroid + finance)",
                pd.Timestamp(cutoff).date(), overnight, df.shape, dim)
    return df


def postcutoff_directions() -> pd.Series:
    """Real next-day TA-125 direction for trading days AFTER the cutoff.

    Used only by the post-cutoff "buy-only" metric overlay — these days are NEVER
    fed to any model (training stays ≤ cutoff). Returns a date-indexed int Series
    (1 = next close up) for dates in ``(CUTOFF_DATE, last available)``; the final row
    with no next-day price is dropped (no fabricated label).
    """
    ta125, _, _, _ = _load_finance()
    price = ta125["TA125_Price"].astype(float).sort_index()
    nxt = price.shift(-1)
    direction = (nxt > price).astype("Int64")
    direction[nxt.isna()] = pd.NA
    direction = direction[direction.notna()].astype(int)
    post = direction[direction.index > pd.Timestamp(CUTOFF_DATE)]
    logger.info("Post-cutoff directions: {} trading days ({} … {}), real up-rate {:.3f}",
                len(post), post.index.min().date() if len(post) else "—",
                post.index.max().date() if len(post) else "—",
                float(post.mean()) if len(post) else float("nan"))
    return post.rename("Target")


def build_fused_dataset(engine=None, *, top_n: int = TOP_N_SOURCES, cutoff=CUTOFF_DATE,
                        overnight: bool = False) -> pd.DataFrame:
    """Fused dataset: per-source SCORE features ⊕ daily embedding CENTROID, one calendar.

    Combines the per-source score pivot (``ml`` shape), the sentiment×relevance
    interactions, AND the daily e5 centroid (``embc_*``) + dispersion/count onto the
    shared finance/TA-125 base. Lets a single LSTM see both signal families at once.

    The ``embc_`` centroid block is the only part meant for PCA (pass ``pca_prefix=
    'embc_'`` downstream); per-source scores, interactions, and finance pass through
    un-reduced. Returns an empty frame if no embeddings are cached (fused needs both).
    """
    engine = engine or get_engine()
    from sentisense.embed import daily_embedding_centroid

    # Streamed per-date centroid (OOM-safe on the full-history corpus — see build_embedding_dataset).
    cen = daily_embedding_centroid(engine, cutoff)
    if cen.empty:
        logger.warning("No embeddings cached — fused dataset needs both scores AND "
                       "embeddings. Returning empty frame (run the 'embed' stage).")
        return pd.DataFrame()

    raw = _load_raw_scores(engine, cutoff)
    per_source = _build_per_source_wide(raw, top_n)
    interactions = _build_interactions(raw)

    centroid_by_date = cen[[c for c in cen.columns if c.startswith("embc_")]]
    extras = cen[["emb_dispersion", "emb_count"]]
    dim = centroid_by_date.shape[1]

    base, trading_days, price_full = _finance_base()
    base = base.join(_roll_to_trading_days(interactions, trading_days, agg="mean"), how="left")
    ps_td = _roll_to_trading_days(per_source, trading_days, agg="sum")
    emb_td = _roll_to_trading_days(centroid_by_date, trading_days, agg="mean")
    extras_td = _roll_to_trading_days(extras, trading_days, agg="mean")

    merged = base.join(ps_td, how="left").join(emb_td, how="left").join(extras_td, how="left")
    feat = add_cross_asset_features(add_ta125_features(merged, price_full))
    if overnight:
        feat = add_overnight_features(feat)
    df = _finalize(feat, cutoff)
    logger.info("Fused dataset built (<= {}, overnight={}): {} (scores + {}-d centroid + finance)",
                pd.Timestamp(cutoff).date(), overnight, df.shape, dim)
    return df
