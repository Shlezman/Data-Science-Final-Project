"""One leaderboard across every forecaster × both data regimes — out-of-sample only.

Rows : Buy&Hold · XGBoost · LSTM · TimesFM-zeroshot · TimesFM-finetuned ·
       TimesFM-cov · TimesFM-nocov            (per regime)
Regimes : CUT  (data ≤ 2023-10-07, the regime-break guard)
          FULL (entire timeline)              ← the CUT-vs-FULL delta is the deliverable
Cols : ROC-AUC · F1 · MCC · accuracy · cumulative return · Sharpe · max drawdown

Every model is reduced to a uniform ``(scores: Series, labels: Series)`` on its held-out
window, then scored by the EXISTING ``metrics_at`` + the shared backtest helpers — no
metric is re-implemented here. TimesFM rows need the ``timesfm`` extra + GPU; each model
is guarded so a missing dep / failure skips just that row.

Run (server-side, from repo root):
    uv run python scripts/pipeline_compare.py
    uv run python scripts/pipeline_compare.py --regimes CUT          # one regime
    uv run python scripts/pipeline_compare.py --no-timesfm           # XGB/LSTM/B&H only
"""

from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd
from loguru import logger

from sentisense.constants import CUTOFF_DATE, TA125_CSV
from sentisense.models.backtest import (
    direction_metrics,
    next_day_returns,
    strategy_stats,
)

_FAR_FUTURE = dt.date(2100, 1, 1)
_REGIMES = {"CUT": CUTOFF_DATE, "FULL": _FAR_FUTURE}
_COLS = ["roc_auc", "f1", "mcc", "accuracy", "cum_return", "sharpe", "max_drawdown", "n"]
SEED = 42


def _load_price() -> pd.Series:
    """TA-125 close, date-indexed (same parse as the analysis notebook)."""
    ta = pd.read_csv(TA125_CSV)
    ta["Date"] = pd.to_datetime(ta["Date"], errors="coerce")
    ta = ta.dropna(subset=["Date"]).set_index("Date").sort_index()
    return ta["Price"].astype(str).str.replace(",", "", regex=False).astype(float)


def _row(scores: pd.Series, labels: pd.Series, price: pd.Series) -> dict:
    """Uniform scorecard for a model's (scores, labels) — reuses metrics_at + backtest."""
    m = direction_metrics(scores.to_numpy(), labels.to_numpy(), 0.5)
    nxt = next_day_returns(price, scores.index)
    st = strategy_stats((scores.to_numpy() > 0.5).astype(float), nxt)
    return {**m, **st, "n": int(len(scores))}


# ── per-model adapters → (scores, labels) on an out-of-sample window ──────────────
def _xgb(mt: pd.DataFrame):
    import xgboost as xgb
    y = mt["Target"].to_numpy().astype(int)
    X = mt.drop(columns=["Target"])
    n = len(mt); ntr = int(n * 0.7); nva = int(n * 0.15)
    Xtr, ytr = X.iloc[:ntr], y[:ntr]
    Xte, yte = X.iloc[ntr + nva:], y[ntr + nva:]
    pos = max(int(ytr.sum()), 1); neg = max(len(ytr) - int(ytr.sum()), 1)
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=neg / pos,
                            eval_metric="logloss", random_state=SEED, verbosity=0)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return pd.Series(p, index=Xte.index), pd.Series(yte, index=Xte.index)


def _lstm(ml: pd.DataFrame):
    import optuna
    from sentisense.db import get_connection_url
    from sentisense.hpo import final_holdout_eval
    from sentisense.hpo.optuna_lstm import STUDY_SCORES, has_completed_trials
    study = optuna.load_study(study_name=STUDY_SCORES, storage=get_connection_url())
    if not has_completed_trials(study):
        raise RuntimeError("score-LSTM study has no completed trials — run tune first.")
    _, proba, labels = final_holdout_eval(ml, study.best_params, n_seeds=2)
    return proba, labels


def _timesfm_forms(mt: pd.DataFrame, ml: pd.DataFrame, price: pd.Series, cutoff) -> dict:
    """Return {form_name: (scores, labels)} for the TimesFM forms in this regime."""
    from sentisense.models.timesfm_forecaster import (
        CONTEXT_LEN, finetune_on_train, load_timesfm, make_forecast_fn,
        walk_forward_directions,
    )
    # Log-return series within the regime; test window = last 15% (out-of-sample).
    p = price[price.index <= pd.Timestamp(cutoff)].sort_index()
    returns = np.log(p / p.shift(1)).dropna()
    n = len(returns); split = int(n * 0.85)
    train_returns = returns.iloc[:split]
    test_index = returns.index[split:]

    model = load_timesfm(CONTEXT_LEN)
    out: dict = {}

    def run(name, covariate_frame=None):
        # walk_forward slices the covariate frame to each day's context window (aligned,
        # strictly past) — so the covariate form has no future leak.
        s, lab = walk_forward_directions(returns, test_index, make_forecast_fn(model),
                                         context_len=CONTEXT_LEN, covariate_frame=covariate_frame)
        if len(s):
            out[name] = (s, lab)

    run("TimesFM-zeroshot")

    finetune_on_train(model, train_returns)   # best-effort; falls back to zero-shot
    run("TimesFM-finetuned")

    # Covariate ablation: sentiment news features (daily-mean) as XReg.
    news_cols = [c for c in mt.columns if c.startswith(("mean_", "ix_"))]
    cov = mt[news_cols] if news_cols else None
    run("TimesFM-cov", covariate_frame=cov)
    run("TimesFM-nocov")
    return out


def _buy_and_hold(price: pd.Series, ref_index: pd.DatetimeIndex):
    """Always-long benchmark on the same out-of-sample dates (score 1.0 → always Up)."""
    from sentisense.models.backtest import directions_from_price
    d = directions_from_price(price)
    common = d.index.intersection(ref_index)
    return pd.Series(1.0, index=common), d.reindex(common)


def build_leaderboard(regimes: list[str], use_timesfm: bool) -> pd.DataFrame:
    from sentisense.features import build_datasets
    price = _load_price()
    rows: dict[tuple, dict] = {}

    for regime in regimes:
        cutoff = _REGIMES[regime]
        logger.info("══ regime {} (cutoff {}) ══", regime, pd.Timestamp(cutoff).date())
        mt, ml = build_datasets(cutoff=cutoff)

        def add(model_name, scores, labels):
            rows[(model_name, regime)] = _row(scores, labels, price)

        for name, fn in [("XGBoost", lambda: _xgb(mt)), ("LSTM", lambda: _lstm(ml))]:
            try:
                s, lab = fn()
                add(name, s, lab)
            except Exception as exc:  # noqa: BLE001
                logger.warning("{} [{}] skipped: {}", name, regime, str(exc)[:120])

        # Buy&Hold on the LSTM/XGB out-of-sample window (last 15% of mt).
        try:
            ref = mt.index[int(len(mt) * 0.85):]
            s, lab = _buy_and_hold(price, ref)
            add("Buy&Hold", s, lab)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Buy&Hold [{}] skipped: {}", regime, str(exc)[:120])

        if use_timesfm:
            try:
                for name, (s, lab) in _timesfm_forms(mt, ml, price, cutoff).items():
                    add(name, s, lab)
            except Exception as exc:  # noqa: BLE001
                logger.warning("TimesFM [{}] skipped: {}", regime, str(exc)[:160])

    board = pd.DataFrame(
        {f"{model} [{reg}]": {c: r.get(c) for c in _COLS} for (model, reg), r in rows.items()}
    ).T
    return board[_COLS]


def _to_markdown(board: pd.DataFrame) -> str:
    """GitHub-flavoured markdown table — no `tabulate` dependency (pandas.to_markdown needs it)."""
    df = board.round(4)
    cols = [str(c) for c in df.columns]
    head = "| model [regime] | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    out = [head, sep]
    for idx, row in df.iterrows():
        cells = ["" if pd.isna(v) else (f"{v:g}" if isinstance(v, float) else str(v)) for v in row]
        out.append("| " + str(idx) + " | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-model × two-regime leaderboard.")
    parser.add_argument("--regimes", default="CUT,FULL", help="Comma list of CUT,FULL.")
    parser.add_argument("--no-timesfm", action="store_true", help="Skip TimesFM rows.")
    parser.add_argument("--out", default="", help="Optional path to write the markdown table.")
    args = parser.parse_args()
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip() in _REGIMES]

    board = build_leaderboard(regimes, use_timesfm=not args.no_timesfm)
    md = _to_markdown(board)
    logger.info("\n=== LEADERBOARD (out-of-sample) ===\n{}", md)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("# SentiSense model leaderboard (out-of-sample)\n\n" + md + "\n")
        logger.info("Wrote {}", args.out)


if __name__ == "__main__":
    main()
