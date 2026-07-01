"""Optional gated challenger — train a fresh model on the latest data; promote ONLY if it
beats the champion out-of-sample by a margin. OFF by default (champion serving never depends
on this; it only runs when you invoke it).

Both champion (pinned params) and challenger (fresh Optuna search) are scored on the SAME
chronological last-15% out-of-sample tail of the fused/FULL dataset, with the EXISTING
financial metrics (``direction_metrics`` → ROC-AUC/MCC). The promotion gate is deliberately
strict — daily data is noisy and direction is ~chance, so a tiny win is almost certainly luck:

    promote  ⇔  ΔROC-AUC ≥ min_auc_gain  AND  challenger MCC ≥ champion MCC  AND  n_oos ≥ min_n

On promotion, ``models/champion.json`` is overwritten (version bumped) and the decision is
appended to ``logs/promotions.jsonl``. Every evaluation (promoted or not) is logged.

Run (server-side, after the daily collection; never auto-scheduled by default):
    uv run --extra finance --extra ml python scripts/challenger_hpo.py --xgb-trials 80
    uv run --extra finance --extra ml python scripts/challenger_hpo.py --dry-run
    uv run --extra finance --extra ml python scripts/challenger_hpo.py --min-auc-gain 0.03
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger

from sentisense.constants import REPO_ROOT
from sentisense.db import get_engine
from sentisense.serve.champion import load_champion, save_champion

_IL_TZ = ZoneInfo("Asia/Jerusalem")
_PROMOTIONS_LOG = REPO_ROOT / "logs" / "promotions.jsonl"
_FAR_FUTURE = dt.date(2100, 1, 1)


def should_promote(champ: dict, chal: dict, *, min_auc_gain: float = 0.02,
                   min_n: int = 200) -> tuple[bool, str]:
    """Decide promotion from out-of-sample metrics (pure → unit-tested).

    Promote only if the challenger clears the champion's ROC-AUC by ``min_auc_gain``, does not
    regress MCC, and the OOS window is at least ``min_n`` days. Returns ``(promote, reason)``.
    """
    n = int(chal.get("n", 0))
    if n < min_n:
        return False, f"insufficient OOS window (n={n} < {min_n})"
    auc_gain = chal["roc_auc"] - champ["roc_auc"]
    if auc_gain < min_auc_gain:
        return False, f"ROC-AUC gain {auc_gain:+.4f} < margin {min_auc_gain}"
    if chal["mcc"] < champ["mcc"]:
        return False, f"MCC regressed ({chal['mcc']:.4f} < {champ['mcc']:.4f})"
    return True, f"ROC-AUC +{auc_gain:.4f}, MCC {chal['mcc']:.4f} ≥ {champ['mcc']:.4f}, n={n}"


def _eval_params(df: pd.DataFrame, params: dict) -> dict:
    """Train ``params`` on the 70/15 train+val, score the last-15% OOS tail (champion path)."""
    from sentisense.models.backtest import direction_metrics
    from sentisense.models.xgb_hpo import _fit_predict

    y = df["Target"].to_numpy().astype(int)
    X = df.drop(columns=["Target"])
    n = len(df); ntr = int(n * 0.70); nva = int(n * 0.15)
    proba = _fit_predict(params, X.iloc[:ntr + nva], y[:ntr + nva], X.iloc[ntr + nva:])
    m = direction_metrics(proba, y[ntr + nva:], 0.5)
    return {"roc_auc": m["roc_auc"], "mcc": m.get("mcc", 0.0), "n": int(n - ntr - nva)}


def run_challenger(engine=None, *, n_trials: int = 80, min_auc_gain: float = 0.02,
                   min_n: int = 200, dry_run: bool = False) -> dict:
    """Evaluate champion vs a fresh challenger on the same OOS tail; promote iff the gate passes."""
    engine = engine or get_engine()
    from sentisense.features import build_fused_dataset
    from sentisense.models.backtest import direction_metrics
    from sentisense.models.xgb_hpo import xgb_hpo

    df = build_fused_dataset(engine, cutoff=_FAR_FUTURE, overnight=True)
    if df.empty or len(df) < min_n + 50:
        raise RuntimeError(f"too few labeled rows ({0 if df.empty else len(df)}) for a fair gate.")

    champ = load_champion()
    champ_m = _eval_params(df, champ.get("params", {}))
    best_params, chal_scores, chal_labels = xgb_hpo(df, n_trials=n_trials)
    chal_m = direction_metrics(chal_scores.to_numpy(), chal_labels.to_numpy(), 0.5)
    chal_m = {"roc_auc": chal_m["roc_auc"], "mcc": chal_m.get("mcc", 0.0), "n": int(len(chal_labels))}

    promote, reason = should_promote(champ_m, chal_m, min_auc_gain=min_auc_gain, min_n=min_n)
    decision = {"ts": str(dt.datetime.now(_IL_TZ)), "champion_version": champ["version"],
                "champion": champ_m, "challenger": chal_m, "promote": promote, "reason": reason,
                "dry_run": dry_run}
    logger.info("Champion {} ROC-AUC {:.4f}/MCC {:.4f} vs challenger {:.4f}/{:.4f} → {} ({})",
                champ["version"], champ_m["roc_auc"], champ_m["mcc"],
                chal_m["roc_auc"], chal_m["mcc"], "PROMOTE" if promote else "keep", reason)

    if promote and not dry_run:
        new_version = f"xgb-fused-full-{dt.datetime.now(_IL_TZ):%Y%m%d-%H%M}"
        new_cfg = {**champ, "version": new_version, "params": best_params,
                   "promoted_at": decision["ts"], "prev_version": champ["version"],
                   "oos_metrics": chal_m}
        save_champion(new_cfg)
        decision["new_version"] = new_version

    if not dry_run:
        _PROMOTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_PROMOTIONS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(decision) + "\n")
    return decision


def main() -> int:
    """CLI entry. Exit 0 regardless of promotion (a non-promotion is not a failure)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xgb-trials", type=int, default=80, help="Optuna trials for the challenger.")
    ap.add_argument("--min-auc-gain", type=float, default=0.02, help="Required ROC-AUC margin.")
    ap.add_argument("--min-n", type=int, default=200, help="Min OOS window to trust the gate.")
    ap.add_argument("--dry-run", action="store_true", help="Evaluate + log decision; never promote.")
    args = ap.parse_args()
    run_challenger(n_trials=args.xgb_trials, min_auc_gain=args.min_auc_gain,
                   min_n=args.min_n, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
