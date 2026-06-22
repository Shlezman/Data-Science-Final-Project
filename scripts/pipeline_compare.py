"""One leaderboard: every model × data-type × regime — out-of-sample only.

Grid axes:
  Models     : XGBoost · LSTM · GRU · TCN · PatchTST (classifiers) ·
               TFT · NHiTS · NBEATS · Chronos · TimesFM (forecasters) · Buy&Hold
  Data-type  : scored (LLM news scores) · embedded (e5 centroid) · fused (both)
               — classifiers run on all three; forecasters use scored covariates +
               univariate (the 768-d centroid is NOT fed as TFT covariates).
  Regime     : CUT (≤ 2023-10-07, the regime-break guard) · FULL (entire timeline)
Cols : ROC-AUC · F1 · MCC · accuracy · cumulative return · Sharpe · max drawdown

Row labels: ``model [datatype/regime]`` (classifiers) / ``model [cov=…/regime]`` /
``model [regime]`` (forecasters). Every cell self-reports ran/skipped in the Coverage
section of leaderboard.md — no silent skips.

Every model is reduced to a uniform ``(scores: Series, labels: Series)`` on its held-out
window, then scored by the EXISTING ``metrics_at`` + the shared backtest helpers — no
metric is re-implemented here. Each model is HPO'd (GRU/TCN/PatchTST + TFT via Optuna;
Chronos/TimesFM sweep context length + threshold). Heavy rows need extra deps + GPU
(``ml`` for the torch zoo, ``tft``/``chronos``/``timesfm`` extras); each model is guarded
so a missing dep / failure skips just that row. Writes leaderboard.md by default.

Resumable: each finished cell's metrics are cached to leaderboard_cache.json the moment it
completes, so a re-run (or crash-resume) reuses done cells and only computes new/changed
ones — `--fresh` ignores the cache.

Run (server-side, from repo root):
    uv run python scripts/pipeline_compare.py                        # full board → leaderboard.md
    uv run python scripts/pipeline_compare.py --regimes CUT          # one regime
    uv run python scripts/pipeline_compare.py --no-timesfm --no-tft  # subset
    uv run python scripts/pipeline_compare.py --fresh                # recompute everything
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os

import numpy as np
import pandas as pd
from loguru import logger

from sentisense.constants import CUTOFF_DATE, TA125_CSV

try:
    from sentisense.models.backtest import (
        direction_metrics,
        next_day_returns,
        strategy_stats,
    )
except ModuleNotFoundError as exc:  # e.g. sklearn missing → wrong/slim env (the LLM 'processing-engine' venv)
    raise SystemExit(
        f"Missing modeling dependency ({exc.name}). This script runs in the ROOT 'sentisense' "
        "project, NOT --project processing_engine. From the repo root:\n"
        "    uv run --extra finance --extra ml --extra tft --extra chronos \\\n"
        "        python scripts/pipeline_compare.py [args]"
    ) from exc

_FAR_FUTURE = dt.date(2100, 1, 1)
_REGIMES = {"CUT": CUTOFF_DATE, "FULL": _FAR_FUTURE}
_COLS = ["roc_auc", "auc_lo", "auc_hi", "f1", "mcc", "accuracy", "cum_return", "sharpe", "max_drawdown", "n"]
SEED = 42


def _load_price() -> pd.Series:
    """TA-125 close, date-indexed (same parse as the analysis notebook)."""
    ta = pd.read_csv(TA125_CSV)
    ta["Date"] = pd.to_datetime(ta["Date"], errors="coerce")
    ta = ta.dropna(subset=["Date"]).set_index("Date").sort_index()
    return ta["Price"].astype(str).str.replace(",", "", regex=False).astype(float)


# Context-length sweep for the frozen foundation forecasters (TimesFM, Chronos) — their
# main knob. The decision threshold is tuned alongside, on validation only. Wide grid.
_TIMESFM_CTX_GRID = [64, 128, 256, 384, 512, 768, 1024]


def _youden(labels: np.ndarray, scores: np.ndarray) -> float:
    """Decision threshold maximising Youden's J on a (validation) slice; 0.5 if single-class."""
    from sklearn.metrics import roc_curve
    if len(np.unique(labels)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(labels, scores)
    return float(thr[int(np.argmax(tpr - fpr))])


def _auc_ci(scores: np.ndarray, labels: np.ndarray, n_boot: int = 500) -> tuple[float, float]:
    """95% bootstrap CI for ROC-AUC. If the CI straddles 0.5, the cell is not distinguishable
    from chance — the honest read on a near-EMH task. Fixed seed → reproducible."""
    from sklearn.metrics import roc_auc_score
    if len(np.unique(labels)) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(SEED)
    n = len(labels)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(labels[idx])) > 1:
            aucs.append(roc_auc_score(labels[idx], scores[idx]))
    if not aucs:
        return (float("nan"), float("nan"))
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


def _row(scores: pd.Series, labels: pd.Series, price: pd.Series, threshold: float = 0.5) -> dict:
    """Uniform scorecard for a model's (scores, labels) at ``threshold`` — reuses metrics_at +
    backtest, plus a bootstrap ROC-AUC CI (auc_lo/auc_hi) so the board shows whether the cell
    is distinguishable from chance."""
    s, y = scores.to_numpy(), labels.to_numpy()
    m = direction_metrics(s, y, threshold)
    nxt = next_day_returns(price, scores.index)
    st = strategy_stats((s > threshold).astype(float), nxt)
    lo, hi = _auc_ci(s, y)
    return {**m, **st, "n": int(len(scores)), "auc_lo": lo, "auc_hi": hi}


# ── per-model adapters → (scores, labels) on an out-of-sample window ──────────────
def _xgb(mt: pd.DataFrame, *, n_trials: int = 40):
    """XGBoost via wide Optuna HPO (val-tuned, OOS test) → (scores, labels)."""
    from sentisense.models.xgb_hpo import xgb_hpo
    _, scores, labels = xgb_hpo(mt, n_trials=n_trials)
    return scores, labels


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


def _seq(ml: pd.DataFrame, arch: str, *, tune_trials: int = 0, study_name: str | None = None):
    """LSTM/GRU/TCN/PatchTST classifier via the generic arch-parametric Optuna HPO → (scores, labels).

    Resumes the study ``study_name`` (unique per data-type×regime cell) from the project DB;
    if none exists and ``tune_trials`` > 0, runs HPO first."""
    import optuna
    from sentisense.db import get_connection_url
    from sentisense.hpo.optuna_lstm import has_completed_trials
    from sentisense.hpo.optuna_seq import run_seq_hpo, seq_holdout_eval, study_name_for
    name = study_name or study_name_for(arch)
    try:
        study = optuna.load_study(study_name=name, storage=get_connection_url())
    except Exception:  # noqa: BLE001
        study = None
    if study is None or not has_completed_trials(study):
        if tune_trials <= 0:
            raise RuntimeError(f"{arch} study '{name}' has no trials — pass --seq-trials > 0.")
        study = run_seq_hpo(ml, arch, n_trials=tune_trials, study_name=name)
    return seq_holdout_eval(ml, arch, study.best_params)


def _chronos_forms(price: pd.Series, cutoff, ctx_grid=None, tag: str = "") -> dict:
    """{form [tag]: (scores, labels, threshold)} for Chronos — zero-shot + context-len-tuned.
    ``tag`` namespaces rows by regime/track so they don't collide in the board/cache.

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
    _lbl = lambda name: f"{name} [{tag}]" if tag else name
    s, lab = walk_forward_directions(returns, test_index, fn, context_len=max(ctx_grid))
    if len(s):
        out[_lbl("Chronos-zeroshot")] = (s, lab, 0.5)
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
            out[_lbl("Chronos-tuned")] = (s, lab, best["thr"])
    return out


def _pf(arch: str, price: pd.Series, cutoff, *, cov: pd.DataFrame | None = None,
        n_trials: int = 8, max_epochs: int = 30):
    """pytorch-forecasting model (TFT/NHiTS/NBEATS) → (scores, labels, threshold) or None.

    ``cov`` is the covariate frame for this cell (None = univariate; NBEATS always
    univariate inside ``pf_directions``)."""
    from sentisense.models.tft_forecaster import pf_directions
    return pf_directions(arch, price, cutoff, covariate_frame=cov, n_trials=n_trials,
                         max_epochs=max_epochs)


def _timesfm_forms(mt: pd.DataFrame, ml: pd.DataFrame, price: pd.Series, cutoff,
                   ctx_grid=None, tag: str = "") -> dict:
    """Return {form_name [tag]: (scores, labels, threshold)} for the TimesFM forms.

    ``tag`` namespaces the row labels by regime/track (e.g. 'FULL', 'FULL+ovn') so the
    multiple forms don't collide across regimes/the overnight track in the board/cache.

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
            out[f"{name} [{tag}]" if tag else name] = (s, lab, threshold)

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
    news_cols = [c for c in mt.columns if c.startswith(("mean_", "ix_", "ovn_"))]  # ovn_ on the overnight track
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


_DATA_TYPES = ("scored", "embedded", "fused")
_SEQ_ARCHS = ("LSTM", "GRU", "TCN", "PatchTST")


def _load_cache(path: str, fresh: bool) -> dict:
    """Per-cell result cache {attempt_label: {row_label: metrics}}; {} if fresh/absent/bad."""
    if fresh or not path or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001 — a corrupt cache shouldn't block a run
        logger.warning("ignoring unreadable cache {}: {}", path, str(exc)[:80])
        return {}


def _save_cache(path: str, cache: dict) -> None:
    """Atomically persist the cache after each cell (crash-safe resume)."""
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp, path)


def _classifier_labels(dtype: str, regime: str, use_seq: bool) -> list[str]:
    """Deterministic cell labels a data-type frame feeds — to skip its (heavy) build when cached."""
    sfx = f"{dtype}/{regime}"
    labs = [f"XGBoost [{sfx}]"]
    if use_seq:
        labs += [f"{a} [{sfx}]" for a in _SEQ_ARCHS]
    return labs


def _track_of(label: str) -> str | None:
    """The regime/track tag inside a cell label, e.g. 'GRU [scored/FULL+ovn]' → 'FULL+ovn'."""
    if "[" not in label:
        return None
    return label.rsplit("[", 1)[1].rstrip("]").rsplit("/", 1)[-1]


def _abstention(scores: pd.Series, labels: pd.Series,
                coverages=(1.0, 0.75, 0.5, 0.25)) -> dict:
    """Accuracy when acting only on the most-confident fraction (|p-0.5| highest). Selective
    prediction: trade coverage for accuracy. Returns {coverage: accuracy}."""
    conf = (scores - 0.5).abs().sort_values(ascending=False)
    out = {}
    for cov in coverages:
        sel = conf.index[: max(int(len(conf) * cov), 1)]
        dec = (scores.reindex(sel) > 0.5).astype(int).to_numpy()
        y = labels.reindex(sel).to_numpy()
        out[cov] = float((dec == y).mean()) if len(y) else float("nan")
    return out


def _cov_cols(df: pd.DataFrame, dtype: str) -> pd.DataFrame | None:
    """Forecaster covariate frame for ``dtype`` (low-dim signal cols only — never the 768-d
    embedding centroid, which would blow up a TFT). Returns None if no usable cols."""
    if dtype == "scored":
        # news daily-mean + interactions, plus the overnight global block when present (ovn_)
        cols = [c for c in df.columns if c.startswith(("mean_", "ix_", "ovn_"))]
    else:                       # embedded/fused → the cheap embedding summaries, not embc_*
        cols = [c for c in df.columns if c in ("emb_dispersion", "emb_count")]
    return df[cols] if cols else None


def build_leaderboard(regimes: list[str], use_timesfm: bool, *, data_types=_DATA_TYPES,
                      use_seq: bool = True, use_chronos: bool = True, use_tft: bool = True,
                      use_nhits: bool = True, use_nbeats: bool = True, seq_trials: int = 20,
                      pf_trials: int = 8, pf_epochs: int = 30, xgb_trials: int = 40,
                      overnight: bool = False,
                      cache_path: str = "leaderboard_cache.json", fresh: bool = False):
    """Grid: model × data-type (scored/embedded/fused) × regime (CUT/FULL).

    Returns ``(board, status)`` — the metrics DataFrame and a per-cell ran/cached/skip map.
    Each completed cell's metrics are cached to ``cache_path`` immediately, so a re-run (or a
    crash-resume) reuses finished cells and only recomputes new/changed ones; ``fresh`` ignores
    the cache. Classifiers run on every data type; forecasters use scored covariates (+
    univariate) — the 768-d embedding centroid is NOT fed as covariates (it would overwhelm TFT)."""
    from sentisense.features import build_datasets, build_embedding_dataset, build_fused_dataset
    price = _load_price()
    rows: dict[str, dict] = {}
    status: dict[str, str] = {}
    preds: dict[str, tuple] = {}     # row_label → (scores, labels) for the soft-vote ensemble
    cache = _load_cache(cache_path, fresh)

    def _restore_preds(rl, row):
        if "_proba" in row and "_dates" in row and "_labels" in row:
            idx = pd.to_datetime(row["_dates"])
            preds[rl] = (pd.Series(row["_proba"], index=idx), pd.Series(row["_labels"], index=idx))

    def restore(label) -> bool:
        """Load a cell's cached rows (+ proba for the ensemble) into the board. True if cached."""
        if fresh or label not in cache:
            return False
        for rl, row in cache[label].items():
            rows[rl] = row
            status[rl] = "cached"
            _restore_preds(rl, row)
        return True

    def attempt(label, thunk):
        """Cached → reuse; else run, emit row(s), persist. thunk → (s,lab)|(s,lab,thr)|{name:(...)}|None."""
        if restore(label):
            logger.info("{} — from cache", label)
            return
        produced: dict[str, dict] = {}

        def emit(rl, s, lab, thr=0.5):
            row = _row(s, lab, price, thr)
            row["_proba"] = [float(x) for x in s.to_numpy()]       # retained for the ensemble +
            row["_dates"] = [str(d) for d in s.index]              # cached so a resume can ensemble too
            row["_labels"] = [int(x) for x in lab.to_numpy()]
            rows[rl] = row
            produced[rl] = row
            status[rl] = "ran"
            preds[rl] = (s, lab)

        try:
            res = thunk()
        except Exception as exc:  # noqa: BLE001 — one cell must not sink the grid
            status[label] = f"skip: {str(exc)[:130]}"
            logger.warning("{} skipped: {}", label, str(exc)[:160])
            return
        if res is None:
            status[label] = "skip: no output"
            logger.warning("{} skipped: no output", label)
        elif isinstance(res, dict):
            for nm, (s, lab, thr) in res.items():
                emit(nm, s, lab, thr)
        elif len(res) == 3:
            emit(label, res[0], res[1], res[2])
        else:
            emit(label, res[0], res[1])
        if produced:                       # persist immediately → crash-safe resume
            cache[label] = produced
            _save_cache(cache_path, cache)

    def _all_cached(labels) -> bool:
        return bool(labels) and not fresh and all(label in cache for label in labels)

    for regime in regimes:
        cutoff = _REGIMES[regime]
        rtag = f"{regime}+ovn" if overnight else regime   # label/cache/study tag (overnight track)
        logger.info("══ regime {} (cutoff {}, overnight={}) ══", regime, pd.Timestamp(cutoff).date(), overnight)

        # Build each data-type's frames for this regime: dtype → (tabular_df, sequence_df).
        # Skip the (heavy, esp. embedded/fused 3M-vector) build when all its cells are cached.
        frames: dict[str, tuple] = {}
        if "scored" in data_types:
            mt, ml = build_datasets(cutoff=cutoff, overnight=overnight)
            frames["scored"] = (mt, ml)
        for dt_name, builder in (("embedded", build_embedding_dataset), ("fused", build_fused_dataset)):
            if dt_name not in data_types:
                continue
            if _all_cached(_classifier_labels(dt_name, rtag, use_seq)):
                logger.info("{}/{}: all cells cached — skipping dataset build", dt_name, rtag)
                for lab in _classifier_labels(dt_name, rtag, use_seq):
                    restore(lab)
                continue
            df = builder(cutoff=cutoff, overnight=overnight)
            if df.empty:
                status[f"[{dt_name}/{rtag}]"] = "skip: no embeddings cached (run embed stage)"
            else:
                frames[dt_name] = (df, df)

        # ── Classifiers: every model × every available data type ──────────────────
        for dtype, (tab, seq) in frames.items():
            sfx = f"{dtype}/{rtag}"
            attempt(f"XGBoost [{sfx}]", lambda tab=tab: _xgb(tab, n_trials=xgb_trials))
            if use_seq:
                for arch in _SEQ_ARCHS:
                    study = f"sentisense_{arch.lower()}_{dtype}_{regime.lower()}{'_ovn' if overnight else ''}"
                    attempt(f"{arch} [{sfx}]",
                            lambda seq=seq, arch=arch, study=study:
                                _seq(seq, arch, tune_trials=seq_trials, study_name=study))

        # Buy&Hold benchmark (data-type-independent) on the last-15% window.
        if "scored" in frames:
            mt = frames["scored"][0]
            attempt(f"Buy&Hold [{rtag}]",
                    lambda mt=mt: _buy_and_hold(price, mt.index[int(len(mt) * 0.85):]))

        # ── Forecasters (price→direction); scored (+overnight) covariates + univariate ──
        scored = frames.get("scored")
        if use_timesfm and scored is not None:
            attempt(f"TimesFM [{rtag}]",
                    lambda: _timesfm_forms(scored[0], scored[1], price, cutoff, tag=rtag))
        if use_chronos:
            attempt(f"Chronos [{rtag}]", lambda: _chronos_forms(price, cutoff, tag=rtag))
        cov = _cov_cols(scored[0], "scored") if scored is not None else None   # incl. ovn_ when overnight
        for arch, on in [("TFT", use_tft), ("NHiTS", use_nhits)]:
            if not on:
                continue
            attempt(f"{arch} [cov=scored/{rtag}]",
                    lambda arch=arch: _pf(arch, price, cutoff, cov=cov, n_trials=pf_trials, max_epochs=pf_epochs))
            if not overnight:   # cov=none is univariate → identical to baseline; skip on the +ovn track
                attempt(f"{arch} [cov=none/{rtag}]",
                        lambda arch=arch: _pf(arch, price, cutoff, cov=None, n_trials=pf_trials, max_epochs=pf_epochs))
        if use_nbeats:
            attempt(f"NBEATS [{rtag}]",
                    lambda: _pf("NBEATS", price, cutoff, cov=None, n_trials=pf_trials, max_epochs=pf_epochs))

    # ── Soft-vote ensemble per track + abstention curves ──────────────────────────
    # Rank-normalise each member's scores (keeps each model's ROC-AUC, makes scales
    # comparable across heterogeneous models) then average. An ensemble of chance-level
    # members stays ~chance — this is a fair-comparison row, not a magic lift.
    from collections import defaultdict
    by_track: dict[str, list] = defaultdict(list)
    for lbl, (s, lab) in preds.items():
        if lbl.startswith(("Ensemble", "Buy&Hold")):
            continue
        t = _track_of(lbl)
        if t:
            by_track[t].append((s, lab))
    notes: list[str] = []
    for track, members in sorted(by_track.items()):
        if len(members) < 3:
            continue
        common = sorted(set.intersection(*[set(s.index) for s, _ in members]))
        if len(common) < 30:
            logger.warning("Ensemble [{}] skipped: only {} common dates across members", track, len(common))
            continue
        ens = pd.concat([s.reindex(common).rank(pct=True) for s, _ in members], axis=1).mean(axis=1)
        lab0 = members[0][1].reindex(common)
        rows[f"Ensemble [{track}]"] = _row(ens, lab0, price, 0.5)
        status[f"Ensemble [{track}]"] = "ran"
        ab = _abstention(ens, lab0)
        notes.append(f"- `Ensemble [{track}]` ({len(members)} models, {len(common)} days) "
                     "acc@coverage: " + ", ".join(f"{int(c * 100)}%={a:.3f}" for c, a in ab.items()))

    board = pd.DataFrame({label: {c: r.get(c) for c in _COLS} for label, r in rows.items()}).T
    board = board[_COLS] if not board.empty else board
    return board, status, notes


def _to_markdown(board: pd.DataFrame) -> str:
    """GitHub-flavoured markdown table — no `tabulate` dependency (pandas.to_markdown needs it)."""
    df = board.round(4)
    cols = [str(c) for c in df.columns]
    head = "| model [datatype/regime] | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    out = [head, sep]
    for idx, row in df.iterrows():
        cells = ["" if pd.isna(v) else (f"{v:g}" if isinstance(v, float) else str(v)) for v in row]
        out.append("| " + str(idx) + " | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-model × two-regime leaderboard.")
    parser.add_argument("--regimes", default="CUT,FULL", help="Comma list of CUT,FULL.")
    parser.add_argument("--data-types", default="scored,embedded,fused",
                        help="Comma list of scored,embedded,fused (the feature-set axis).")
    parser.add_argument("--no-timesfm", action="store_true", help="Skip TimesFM rows.")
    parser.add_argument("--no-seq", action="store_true", help="Skip the sequence zoo (GRU/TCN/PatchTST).")
    parser.add_argument("--no-chronos", action="store_true", help="Skip Chronos rows.")
    parser.add_argument("--no-tft", action="store_true", help="Skip the TFT row.")
    parser.add_argument("--no-nhits", action="store_true", help="Skip the N-HiTS row.")
    parser.add_argument("--no-nbeats", action="store_true", help="Skip the N-BEATS row.")
    parser.add_argument("--seq-trials", type=int, default=20,
                        help="Optuna trials for each new classifier (GRU/TCN/PatchTST) if untuned.")
    parser.add_argument("--pf-trials", "--tft-trials", dest="pf_trials", type=int, default=8,
                        help="Optuna trials for each pytorch-forecasting model (TFT/N-HiTS/N-BEATS).")
    parser.add_argument("--pf-epochs", type=int, default=30,
                        help="Max epochs per pytorch-forecasting fit (lower = faster smoke checks).")
    parser.add_argument("--xgb-trials", type=int, default=40, help="Optuna trials for XGBoost.")
    parser.add_argument("--overnight", action="store_true",
                        help="Add the open(T+1) overnight global-feature track (cells tagged +ovn).")
    parser.add_argument("--cache", default="leaderboard_cache.json",
                        help="Per-cell result cache (resumes finished cells across runs).")
    parser.add_argument("--fresh", action="store_true", help="Ignore the cache; recompute every cell.")
    parser.add_argument("--out", default="leaderboard.md", help="Path to write the markdown table.")
    args = parser.parse_args()
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip() in _REGIMES]
    data_types = tuple(d.strip() for d in args.data_types.split(",") if d.strip() in _DATA_TYPES)

    board, status, notes = build_leaderboard(
        regimes, use_timesfm=not args.no_timesfm, data_types=data_types, use_seq=not args.no_seq,
        use_chronos=not args.no_chronos, use_tft=not args.no_tft,
        use_nhits=not args.no_nhits, use_nbeats=not args.no_nbeats,
        seq_trials=args.seq_trials, pf_trials=args.pf_trials, pf_epochs=args.pf_epochs,
        xgb_trials=args.xgb_trials, overnight=args.overnight, cache_path=args.cache, fresh=args.fresh)
    md = _to_markdown(board) if not board.empty else "_(no cells produced output)_"

    best_line = ""
    if not board.empty and board["roc_auc"].notna().any():
        bi = board["roc_auc"].astype(float).idxmax()
        parts = ", ".join(f"{c}={board.loc[bi, c]:.4f}"
                          for c in ("roc_auc", "f1", "mcc", "sharpe", "cum_return")
                          if c in board.columns and pd.notna(board.loc[bi, c]))
        best_line = f"**Ultimate model (best out-of-sample ROC-AUC):** `{bi}` — {parts}"

    # Coverage report — every cell self-reports ran / cached / skipped(+reason). No silent skips.
    ran = sorted(k for k, v in status.items() if v == "ran")
    cached = sorted(k for k, v in status.items() if v == "cached")
    skipped = sorted((k, v) for k, v in status.items() if v not in ("ran", "cached"))
    cov_lines = [f"## Coverage — {len(ran)} ran, {len(cached)} cached, {len(skipped)} skipped", "",
                 f"**Ran ({len(ran)}):** " + (", ".join(f"`{r}`" for r in ran) or "—"), "",
                 f"**Cached ({len(cached)}):** " + (", ".join(f"`{r}`" for r in cached) or "—"), ""]
    if skipped:
        cov_lines.append("**Skipped (why):**")
        cov_lines += [f"- `{k}` — {v.removeprefix('skip: ')}" for k, v in skipped]
    if notes:
        cov_lines += ["", "## Ensemble abstention (accuracy when acting on the most-confident fraction)", *notes]
    coverage = "\n".join(cov_lines)

    logger.info("\n=== LEADERBOARD (out-of-sample) ===\n{}\n\n{}\n\n{}", md, best_line, coverage)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("# SentiSense model leaderboard (out-of-sample)\n\n"
                    + md + "\n\n" + best_line + "\n\n" + coverage + "\n")
        logger.info("Wrote {}", args.out)


if __name__ == "__main__":
    main()
