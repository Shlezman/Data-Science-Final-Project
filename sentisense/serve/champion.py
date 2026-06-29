"""Pinned champion: train on labeled history, forward-predict the next TA-125 move.

The leaderboard verdict is that next-day direction is ~chance (best FULL-regime OOS ROC-AUC
≈ 0.53–0.56, CIs spanning 0.5). The served champion is therefore the most ROBUST, cheap-to-
retrain cell — XGBoost on the ``fused`` dataset (every feature family), FULL regime, overnight
features on — NOT a skillful model. Its config is a versioned artifact (``models/champion.json``)
that the optional challenger HPO (WS2) overwrites only after passing an out-of-sample gate.

Serving contract (leak-safe): build the fused dataset with ``keep_unlabeled=True`` so the
latest trading day(s) with an unknown next-day close are retained as ``Target == -1``. Train on
``Target in {0,1}`` (real labels only), predict the ``Target == -1`` rows, and upsert them into
``model_predictions`` (``actual`` stays NULL until settled post-close).
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

from sentisense.constants import REPO_ROOT
from sentisense.db import get_engine

CHAMPION_PATH = REPO_ROOT / "models" / "champion.json"
_FAR_FUTURE = dt.date(2100, 1, 1)

# Pinned default champion. Params are a sane XGBoost config (not daily-re-tuned — that's the
# challenger's job). datatype/regime/overnight define the feature frame it serves on.
DEFAULT_CHAMPION = {
    "version": "xgb-fused-full-v1",
    "model": "xgboost",
    "datatype": "fused",
    "regime": "FULL",
    "overnight": True,
    "params": {
        "n_estimators": 600,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 5,
        "reg_lambda": 2.0,
        "reg_alpha": 0.1,
        "gamma": 0.5,
    },
}

_ENSURE_TABLE = text(
    """
    CREATE TABLE IF NOT EXISTS model_predictions (
        id            BIGSERIAL PRIMARY KEY,
        date          DATE          NOT NULL,
        model_version VARCHAR(100)  NOT NULL,
        prediction    BOOLEAN       NOT NULL,
        confidence    REAL          NOT NULL,
        actual        BOOLEAN,
        created_at    TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
        CONSTRAINT uq_model_predictions UNIQUE (date, model_version)
    )
    """
)

_UPSERT = text(
    """
    INSERT INTO model_predictions (date, model_version, prediction, confidence)
    VALUES (:date, :version, :prediction, :confidence)
    ON CONFLICT (date, model_version) DO UPDATE
        SET prediction = EXCLUDED.prediction,
            confidence = EXCLUDED.confidence,
            created_at = NOW()
    """
)


def load_champion() -> dict:
    """Load the pinned champion config (``models/champion.json``), or the built-in default."""
    if CHAMPION_PATH.exists():
        try:
            return json.loads(CHAMPION_PATH.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — a corrupt artifact shouldn't break serving
            logger.warning("Unreadable champion.json ({}); using default.", str(exc)[:80])
    return dict(DEFAULT_CHAMPION)


def save_champion(cfg: dict) -> None:
    """Persist a champion config as the versioned artifact (used by WS2 promotion)."""
    CHAMPION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHAMPION_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    logger.info("Champion artifact written: {} (version={})", CHAMPION_PATH, cfg.get("version"))


def _serving_frames(engine, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the fused serving frame → ``(labeled, to_predict)`` split on the -1 sentinel."""
    from sentisense.features import build_fused_dataset

    df = build_fused_dataset(engine, cutoff=_FAR_FUTURE, overnight=bool(cfg.get("overnight", True)),
                             keep_unlabeled=True)
    if df.empty:
        return df, df
    labeled = df[df["Target"] != -1].copy()
    to_predict = df[df["Target"] == -1].copy()
    return labeled, to_predict


def _train_predict(labeled: pd.DataFrame, to_predict: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Fit XGBoost on all labeled history; return per-date predicted up-probability."""
    import xgboost as xgb

    from sentisense.models.xgb_hpo import _xgb_device

    feat_cols = labeled.columns.drop("Target")
    X, y = labeled[feat_cols].to_numpy(np.float32), labeled["Target"].to_numpy(int)
    pos, neg = max(int(y.sum()), 1), max(int((y == 0).sum()), 1)
    clf = xgb.XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0,
                            scale_pos_weight=neg / pos, tree_method="hist",
                            device=_xgb_device(), **cfg.get("params", {}))
    clf.fit(X, y)
    proba = clf.predict_proba(to_predict[feat_cols].to_numpy(np.float32))[:, 1]
    return pd.DataFrame({"date": to_predict.index, "proba": np.clip(proba, 0.0, 1.0)})


def ensure_predictions_table(engine=None) -> None:
    """Create ``model_predictions`` if absent (idempotent)."""
    engine = engine or get_engine()
    with engine.begin() as conn:
        conn.execute(_ENSURE_TABLE)


def train_and_predict(engine=None, *, dry_run: bool = False) -> dict:
    """Train the champion on history, predict the latest unlabeled day(s), upsert predictions.

    Args:
        engine: SQLAlchemy engine; created from env if None.
        dry_run: compute predictions but do not write to the DB.

    Returns:
        Summary dict: champion version, train-row count, and the predicted ``{date: (up, conf)}``.
    """
    engine = engine or get_engine()
    cfg = load_champion()
    labeled, to_predict = _serving_frames(engine, cfg)
    if labeled.empty:
        raise RuntimeError("No labeled fused data — run scrape/score/embed/derived first.")
    if to_predict.empty:
        logger.info("No unlabeled day to predict — predictions already current.")
        return {"version": cfg["version"], "n_train": int(len(labeled)), "predicted": {}}

    preds = _train_predict(labeled, to_predict, cfg)
    out = {str(pd.Timestamp(r.date).date()): (bool(r.proba > 0.5), round(float(r.proba), 4))
           for r in preds.itertuples()}
    logger.info("Champion {} trained on {} days → predicted {} day(s): {}",
                cfg["version"], len(labeled), len(out), out)

    if dry_run:
        return {"version": cfg["version"], "n_train": int(len(labeled)), "predicted": out,
                "dry_run": True}

    ensure_predictions_table(engine)
    rows = [{"date": pd.Timestamp(r.date).date(), "version": cfg["version"],
             "prediction": bool(r.proba > 0.5), "confidence": float(r.proba)}
            for r in preds.itertuples()]
    with engine.begin() as conn:
        conn.execute(_UPSERT, rows)
    return {"version": cfg["version"], "n_train": int(len(labeled)), "predicted": out}


def predict_today(engine=None) -> dict:
    """Convenience alias for the daily orchestrator: train + predict + persist."""
    return train_and_predict(engine)
