"""One leaderboard across every forecaster × both data regimes — out-of-sample only.

Rows : Buy&Hold · XGBoost · LSTM · GRU · TCN · PatchTST · TFT ·
       Chronos-{zeroshot,tuned} · TimesFM-{zeroshot,tuned,finetuned,cov,nocov}  (per regime)
Regimes : CUT  (data ≤ 2023-10-07, the regime-break guard)
          FULL (entire timeline)              ← the CUT-vs-FULL delta is the deliverable
Cols : ROC-AUC · F1 · MCC · accuracy · cumulative return · Sharpe · max drawdown

Every model is reduced to a uniform ``(scores: Series, labels: Series)`` on its held-out
window, then scored by the EXISTING ``metrics_at`` + the shared backtest helpers — no
metric is re-implemented here. Each model is HPO'd (GRU/TCN/PatchTST + TFT via Optuna;
Chronos/TimesFM sweep context length + threshold). Heavy rows need extra deps + GPU
(``ml`` for the torch zoo, ``tft``/``chronos``/``timesfm`` extras); each model is guarded
so a missing dep / failure skips just that row. Writes leaderboard.md by default.

Run (server-side, from repo root):
    uv run python scripts/pipeline_compare.py                        # full board → leaderboard.md
    uv run python scripts/pipeline_compare.py --regimes CUT          # one regime
    uv run python scripts/pipeline_compare.py --no-timesfm --no-tft  # subset
    uv run python scripts/pipeline_compare.py --seq-trials 40 --tft-trials 15
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


# TimesFM hyperparameter sweep: context length (its main knob — frozen foundation model,
# so no architecture HPO). The decision threshold is tuned alongside, on validation only.
_TIMESFM_CTX_GRID = [128, 256, 512]


def _youden(labels: np.ndarray, scores: np.ndarray) -> float:
    """Decision threshold maximising Youden's J on a (validation) slice; 0.5 if single-class."""
    from sklearn.metrics import roc_curve
    if len(np.unique(labels)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(labels, scores)
    return float(thr[int(np.argmax(tpr - fpr))])


def _row(scores: pd.Series, labels: pd.Series, price: pd.Series, threshold: float = 0.5) -> dict:
    """Uniform scorecard for a model's (scores, labels) at ``threshold`` — reuses metrics_at + backtest."""
    m = direction_metrics(scores.to_numpy(), labels.to_numpy(), threshold)
    nxt = next_day_returns(price, scores.index)
    st = strategy_stats((scores.to_numpy() > threshold).astype(float), nxt)
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


def _seq(ml: pd.DataFrame, arch: str, *, tune_trials: int = 0):
    """GRU/TCN/PatchTST classifier via the generic arch-parametric Optuna HPO → (scores, labels).

    Resumes a per-arch study from the project DB; if none exists and ``tune_trials`` > 0,
    runs HPO first (these architectures have no pre-tuned study, unlike LSTM)."""
    import optuna
    from sentisense.db import get_connection_url
    from sentisense.hpo.optuna_lstm import has_completed_trials
    from sentisense.hpo.optuna_seq import run_seq_hpo, seq_holdout_eval, study_name_for
    name = study_name_for(arch)
    try:
        study = optuna.load_study(study_name=name, storage=get_connection_url())
    except Exception:  # noqa: BLE001
        study = None
    if study is None or not has_completed_trials(study):
        if tune_trials <= 0:
            raise RuntimeError(f"{arch} study '{name}' has no trials — pass --seq-trials > 0.")
        study = run_seq_hpo(ml, arch, n_trials=tune_trials, study_name=name)
    return seq_holdout_eval(ml, arch, study.best_params)


def _chronos_forms(price: pd.Series, cutoff, ctx_grid=None) -> dict:
    """{form: (scores, labels, threshold)} for Chronos — zero-shot + context-len-tuned.

    Same forecast→direction bridge as TimesFM (univariate foundation model). HPO sweeps the
    context length and tunes the decision threshold on the VALIDATION slice only (no leak)."""
    from sentisense.models.chronos_forecaster import load_chronos, make_chronos_forecast_fn
    from sentisense.models.timesfm_forecaster import walk_forward_directions
    ctx_grid = ctx_grid or _TIMESFM_CTX_GRID
    p = price[price.index <= pd.Timestamp(cutoff)].sort_index()
    returns = np.log(p / p.shift(1)).dropna()
    n = len(returns); v0 = int(n * 0.70); v1 = int(n * 0.85)
    val_index, test_index = returns.index[v0:v1], returns.index[v1:]
    fn = make_chronos_forecast_fn(load_chronos())
    out: dict = {}
    s, lab = walk_forward_directions(returns, test_index, fn, context_len=max(ctx_grid))
    if len(s):
        out["Chronos-zeroshot"] = (s, lab, 0.5)
    best = None
    for cl in ctx_grid:
        sv, lv = walk_forward_directions(returns, val_index, fn, context_len=cl)
        if not len(sv) or len(np.unique(lv.to_numpy())) < 2:
            continue
        thr = _youden(lv.to_numpy(), sv.to_numpy())
        auc = direction_metrics(sv.to_numpy(), lv.to_numpy(), thr)["roc_auc"]
        if best is None or auc > best["auc"]:
            best = {"cl": cl, "thr": thr, "auc": auc}
    if best:
        logger.info("Chronos HPO → context_len={} threshold={:.3f} (val ROC-AUC={:.4f})",
                    best["cl"], best["thr"], best["auc"])
        s, lab = walk_forward_directions(returns, test_index, fn, context_len=best["cl"])
        if len(s):
            out["Chronos-tuned"] = (s, lab, best["thr"])
    return out


def _tft(mt: pd.DataFrame, price: pd.Series, cutoff, *, n_trials: int = 8):
    """TFT (pytorch-forecasting) with sentiment covariates → (scores, labels, threshold) or None."""
    from sentisense.models.tft_forecaster import tft_directions
    news_cols = [c for c in mt.columns if c.startswith(("mean_", "ix_"))]
    cov = mt[news_cols] if news_cols else None
    return tft_directions(price, cutoff, covariate_frame=cov, n_trials=n_trials)


def _timesfm_forms(mt: pd.DataFrame, ml: pd.DataFrame, price: pd.Series, cutoff,
                   ctx_grid=None) -> dict:
    """Return {form_name: (scores, labels, threshold)} for the TimesFM forms in this regime.

    Includes ``TimesFM-tuned`` — the best (context_len, decision-threshold) chosen on a
    VALIDATION slice [70%,85%) by val ROC-AUC, then evaluated on the held-out test
    window [85%,100%]. The sweep + threshold never see the test slice (no leak), and the
    test window matches XGB/LSTM/Buy&Hold (last 15%).
    """
    from sentisense.models.timesfm_forecaster import (
        CONTEXT_LEN, finetune_on_train, load_timesfm, make_forecast_fn,
        walk_forward_directions,
    )
    ctx_grid = ctx_grid or _TIMESFM_CTX_GRID
    p = price[price.index <= pd.Timestamp(cutoff)].sort_index()
    returns = np.log(p / p.shift(1)).dropna()
    n = len(returns); v0 = int(n * 0.70); v1 = int(n * 0.85)
    train_returns = returns.iloc[:v0]
    val_index = returns.index[v0:v1]
    test_index = returns.index[v1:]                 # last 15% — same OOS window as XGB/LSTM

    model = load_timesfm(max(CONTEXT_LEN, max(ctx_grid)))   # compile once at the largest context
    fn = make_forecast_fn(model)
    out: dict = {}

    def run(name, *, covariate_frame=None, context_len=CONTEXT_LEN, threshold=0.5):
        s, lab = walk_forward_directions(returns, test_index, fn,
                                         context_len=context_len, covariate_frame=covariate_frame)
        if len(s):
            out[name] = (s, lab, threshold)

    run("TimesFM-zeroshot")

    # ── HPO: sweep context_len, tune the decision threshold — on VALIDATION only ──
    best = None
    for cl in ctx_grid:
        sv, lv = walk_forward_directions(returns, val_index, fn, context_len=cl)
        if not len(sv) or len(np.unique(lv.to_numpy())) < 2:
            continue
        thr = _youden(lv.to_numpy(), sv.to_numpy())
        auc = direction_metrics(sv.to_numpy(), lv.to_numpy(), thr)["roc_auc"]
        logger.info("  TimesFM HPO trial: context_len={:>4} thr={:.3f} → val ROC-AUC {:.4f}", cl, thr, auc)
        if best is None or auc > best["val_auc"]:
            best = {"context_len": cl, "threshold": thr, "val_auc": auc}
    if best:
        logger.info("TimesFM HPO → ultimate: context_len={} threshold={:.3f} (val ROC-AUC={:.4f})",
                    best["context_len"], best["threshold"], best["val_auc"])
        run("TimesFM-tuned", context_len=best["context_len"], threshold=best["threshold"])

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


def build_leaderboard(regimes: list[str], use_timesfm: bool, *, use_seq: bool = True,
                      use_chronos: bool = True, use_tft: bool = True,
                      seq_trials: int = 20, tft_trials: int = 8) -> pd.DataFrame:
    from sentisense.features import build_datasets
    price = _load_price()
    rows: dict[tuple, dict] = {}

    for regime in regimes:
        cutoff = _REGIMES[regime]
        logger.info("══ regime {} (cutoff {}) ══", regime, pd.Timestamp(cutoff).date())
        mt, ml = build_datasets(cutoff=cutoff)

        def add(model_name, scores, labels, threshold=0.5):
            rows[(model_name, regime)] = _row(scores, labels, price, threshold)

        # Classifier rows: XGBoost, LSTM (pre-tuned), + the HPO'd sequence zoo (GRU/TCN/PatchTST).
        classifiers = [("XGBoost", lambda: _xgb(mt)), ("LSTM", lambda: _lstm(ml))]
        if use_seq:
            classifiers += [(arch, lambda a=arch: _seq(ml, a, tune_trials=seq_trials))
                            for arch in ("GRU", "TCN", "PatchTST")]
        for name, fn in classifiers:
            try:
                s, lab = fn()
                add(name, s, lab)
            except Exception as exc:  # noqa: BLE001
                logger.warning("{} [{}] skipped: {}", name, regime, str(exc)[:160])

        # Buy&Hold on the LSTM/XGB out-of-sample window (last 15% of mt).
        try:
            ref = mt.index[int(len(mt) * 0.85):]
            s, lab = _buy_and_hold(price, ref)
            add("Buy&Hold", s, lab)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Buy&Hold [{}] skipped: {}", regime, str(exc)[:120])

        if use_timesfm:
            try:
                for name, (s, lab, thr) in _timesfm_forms(mt, ml, price, cutoff).items():
                    add(name, s, lab, thr)
            except Exception as exc:  # noqa: BLE001
                logger.warning("TimesFM [{}] skipped: {}", regime, str(exc)[:160])

        if use_chronos:
            try:
                for name, (s, lab, thr) in _chronos_forms(price, cutoff).items():
                    add(name, s, lab, thr)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Chronos [{}] skipped: {}", regime, str(exc)[:160])

        if use_tft:
            try:
                res = _tft(mt, price, cutoff, n_trials=tft_trials)
                if res:
                    add("TFT", *res)
            except Exception as exc:  # noqa: BLE001
                logger.warning("TFT [{}] skipped: {}", regime, str(exc)[:160])

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
    parser.add_argument("--no-seq", action="store_true", help="Skip the sequence zoo (GRU/TCN/PatchTST).")
    parser.add_argument("--no-chronos", action="store_true", help="Skip Chronos rows.")
    parser.add_argument("--no-tft", action="store_true", help="Skip the TFT row.")
    parser.add_argument("--seq-trials", type=int, default=20,
                        help="Optuna trials for each new classifier (GRU/TCN/PatchTST) if untuned.")
    parser.add_argument("--tft-trials", type=int, default=8, help="Optuna trials for TFT.")
    parser.add_argument("--out", default="leaderboard.md", help="Path to write the markdown table.")
    args = parser.parse_args()
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip() in _REGIMES]

    board = build_leaderboard(
        regimes, use_timesfm=not args.no_timesfm, use_seq=not args.no_seq,
        use_chronos=not args.no_chronos, use_tft=not args.no_tft,
        seq_trials=args.seq_trials, tft_trials=args.tft_trials)
    md = _to_markdown(board)

    # The "ultimate" model = best out-of-sample ROC-AUC across the whole board
    # (TimesFM-tuned carries its swept context_len/threshold — logged during the run).
    best_line = ""
    if not board.empty and board["roc_auc"].notna().any():
        bi = board["roc_auc"].astype(float).idxmax()
        parts = ", ".join(f"{c}={board.loc[bi, c]:.4f}"
                          for c in ("roc_auc", "f1", "mcc", "sharpe", "cum_return")
                          if c in board.columns and pd.notna(board.loc[bi, c]))
        best_line = f"**Ultimate model (best out-of-sample ROC-AUC):** `{bi}` — {parts}"

    logger.info("\n=== LEADERBOARD (out-of-sample) ===\n{}\n\n{}", md, best_line)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("# SentiSense model leaderboard (out-of-sample)\n\n" + md + "\n\n" + best_line + "\n")
        logger.info("Wrote {}", args.out)


if __name__ == "__main__":
    main()
