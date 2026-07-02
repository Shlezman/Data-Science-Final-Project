"""Full-history in-sample eval of the served champion → ``champion_full_eval``.

Fits the pinned champion on ALL labeled fused days and predicts those same days (in-sample),
storing per-day prediction/proba/actual so the light UI box can render the "all days" confusion
matrix + colour the 3D centroids without carrying the ml extra. Needs ``ml`` (XGBoost); run on
the GPU container, daily after ``daily_live.py`` (or manually to backfill now).

Run (server-side, has ml):
    uv run --extra finance --extra ml python scripts/compute_full_eval.py
    uv run --extra finance --extra ml python scripts/compute_full_eval.py --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

from sentisense.constants import REPO_ROOT
from sentisense.db import get_engine

_FAR_FUTURE = dt.date(2100, 1, 1)
_MIGRATION = REPO_ROOT / "sentisense" / "db" / "migrations" / "006_champion_full_eval.sql"

_UPSERT = text(
    """
    INSERT INTO champion_full_eval (model_version, date, prediction, proba, actual)
    VALUES (:version, :date, :prediction, :proba, :actual)
    ON CONFLICT (model_version, date) DO UPDATE
        SET prediction = EXCLUDED.prediction, proba = EXCLUDED.proba,
            actual = EXCLUDED.actual, created_at = NOW()
    """
)


def _ensure_table(engine) -> None:
    """Apply migration 006 (idempotent) so the table exists before upsert."""
    import re

    ddl = re.sub(r"--[^\n]*", "", _MIGRATION.read_text(encoding="utf-8"))  # strip comments ('; ' inside)
    with engine.begin() as conn:
        for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
            conn.execute(text(stmt))


def _in_sample_eval(engine, cfg: dict) -> pd.DataFrame:
    """Fit the champion on all labeled fused days, predict them in-sample.

    Returns a frame indexed by date with columns ``prediction, proba, actual``.
    """
    import xgboost as xgb

    from sentisense.features import build_fused_dataset
    from sentisense.models.xgb_hpo import _xgb_device

    df = build_fused_dataset(engine, cutoff=_FAR_FUTURE,
                             overnight=bool(cfg.get("overnight", True)), keep_unlabeled=False)
    if df.empty:
        return df
    feat_cols = df.columns.drop("Target")
    X = df[feat_cols].to_numpy(np.float32)
    y = df["Target"].to_numpy(int)
    pos, neg = max(int(y.sum()), 1), max(int((y == 0).sum()), 1)
    clf = xgb.XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0,
                            scale_pos_weight=neg / pos, tree_method="hist",
                            device=_xgb_device(), **cfg.get("params", {}))
    clf.fit(X, y)
    proba = np.clip(clf.predict_proba(X)[:, 1], 0.0, 1.0)
    return pd.DataFrame({"proba": proba, "prediction": proba > 0.5, "actual": y.astype(bool)},
                        index=df.index)


def main() -> int:
    """Compute + upsert the champion's full in-sample eval."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Compute but do not write to the DB.")
    args = ap.parse_args()

    from sentisense.serve.champion import load_champion

    engine = get_engine()
    cfg = load_champion()
    out = _in_sample_eval(engine, cfg)
    if out.empty:
        raise SystemExit("No labeled fused data — run scrape/score/embed/derived first.")

    acc = float((out["prediction"] == out["actual"]).mean())
    logger.info("Champion {} in-sample over {} days: accuracy={:.4f}",
                cfg["version"], len(out), acc)
    if args.dry_run:
        logger.info("[dry-run] computed {} rows; wrote nothing.", len(out))
        return 0

    _ensure_table(engine)
    rows = [{"version": cfg["version"], "date": pd.Timestamp(d).date(),
             "prediction": bool(r.prediction), "proba": float(r.proba), "actual": bool(r.actual)}
            for d, r in out.iterrows()]
    with engine.begin() as conn:
        conn.execute(_UPSERT, rows)
    logger.info("Upserted {} rows into champion_full_eval (version={}).", len(rows), cfg["version"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
