"""Model registry — persist trained models + OOS metrics in Postgres; pick/serve the active one.

Pure DB layer (no torch/xgboost here): stores each model's serialized ``artifact`` bytes +
``artifact_format`` + ``feature_cols``; the trainer serializes and the serving path deserializes
(they carry the ml extra). Selection = **auto-best with a sticky manual override**: a row with
``activated_by='manual'`` is never auto-replaced; otherwise the highest-OOS candidate is served.
At most one row is active (DB partial-unique index).
"""

from __future__ import annotations

import datetime as dt
import json

from loguru import logger
from sqlalchemy import text

from sentisense.constants import REPO_ROOT
from sentisense.db import get_engine

_MIGRATION = REPO_ROOT / "sentisense" / "db" / "migrations" / "005_model_registry.sql"

_META_COLS = ("id", "version", "name", "model_type", "datatype", "regime", "overnight",
              "oos_roc_auc", "oos_auc_lo", "oos_auc_hi", "oos_mcc", "oos_accuracy", "oos_n",
              "artifact_format", "members", "feature_cols", "trained_rows", "trained_at",
              "is_active", "activated_by", "activated_at")


def ensure_registry_table(engine=None) -> None:
    """Apply the registry migration (idempotent)."""
    engine = engine or get_engine()
    ddl = _MIGRATION.read_text(encoding="utf-8")
    with engine.begin() as conn:
        for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
            conn.execute(text(stmt))


_UPSERT = text(
    """
    INSERT INTO model_registry
        (version, name, model_type, datatype, regime, overnight, params,
         oos_roc_auc, oos_auc_lo, oos_auc_hi, oos_mcc, oos_accuracy, oos_n,
         artifact, artifact_format, members, feature_cols, trained_rows)
    VALUES
        (:version, :name, :model_type, :datatype, :regime, :overnight, :params,
         :roc_auc, :auc_lo, :auc_hi, :mcc, :accuracy, :n,
         :artifact, :artifact_format, :members, :feature_cols, :trained_rows)
    ON CONFLICT (version) DO UPDATE SET
        name=EXCLUDED.name, model_type=EXCLUDED.model_type, params=EXCLUDED.params,
        oos_roc_auc=EXCLUDED.oos_roc_auc, oos_auc_lo=EXCLUDED.oos_auc_lo,
        oos_auc_hi=EXCLUDED.oos_auc_hi, oos_mcc=EXCLUDED.oos_mcc,
        oos_accuracy=EXCLUDED.oos_accuracy, oos_n=EXCLUDED.oos_n,
        artifact=EXCLUDED.artifact, artifact_format=EXCLUDED.artifact_format,
        members=EXCLUDED.members, feature_cols=EXCLUDED.feature_cols,
        trained_rows=EXCLUDED.trained_rows, trained_at=NOW()
    """
)


def register_model(engine=None, *, version: str, name: str, model_type: str, params: dict,
                   metrics: dict, artifact: bytes | None, artifact_format: str = "joblib",
                   members: list | None = None, feature_cols: list | None = None,
                   trained_rows: int | None = None, datatype: str = "fused",
                   regime: str = "FULL", overnight: bool = True) -> None:
    """Upsert a trained model (by ``version``). Metrics keys: roc_auc/auc_lo/auc_hi/mcc/accuracy/n."""
    engine = engine or get_engine()
    ensure_registry_table(engine)
    with engine.begin() as conn:
        conn.execute(_UPSERT, {
            "version": version, "name": name, "model_type": model_type, "datatype": datatype,
            "regime": regime, "overnight": overnight, "params": json.dumps(params or {}),
            "roc_auc": metrics.get("roc_auc"), "auc_lo": metrics.get("auc_lo"),
            "auc_hi": metrics.get("auc_hi"), "mcc": metrics.get("mcc"),
            "accuracy": metrics.get("accuracy"), "n": metrics.get("n"),
            "artifact": artifact, "artifact_format": artifact_format,
            "members": json.dumps(members) if members is not None else None,
            "feature_cols": json.dumps(feature_cols) if feature_cols is not None else None,
            "trained_rows": trained_rows,
        })
    logger.info("Registered model {} ({}) roc_auc={} mcc={}", version, model_type,
                metrics.get("roc_auc"), metrics.get("mcc"))


def _row_to_dict(row) -> dict | None:
    if row is None:
        return None
    d = dict(row._mapping)
    for k in ("members", "feature_cols"):
        if isinstance(d.get(k), str):
            d[k] = json.loads(d[k])
    return d


def list_models(engine=None) -> list[dict]:
    """All registered models (metadata + metrics, no artifact bytes), newest first."""
    engine = engine or get_engine()
    cols = ", ".join(_META_COLS)
    with engine.connect() as conn:
        rows = conn.execute(text(f"SELECT {cols} FROM model_registry ORDER BY trained_at DESC")).all()
    return [_row_to_dict(r) for r in rows]


def get_by_version(engine=None, version: str = "") -> dict | None:
    """Full row (including artifact bytes) for a version."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        row = conn.execute(text("SELECT * FROM model_registry WHERE version=:v"),
                           {"v": version}).first()
    return _row_to_dict(row)


def get_active(engine=None) -> dict | None:
    """The active model's full row (including artifact bytes), or None if none active."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        row = conn.execute(text("SELECT * FROM model_registry WHERE is_active")).first()
    return _row_to_dict(row)


def set_active(engine=None, version: str = "", by: str = "manual") -> bool:
    """Make ``version`` the sole active model (by='manual'|'auto'). Returns False if not found."""
    engine = engine or get_engine()
    with engine.begin() as conn:
        exists = conn.execute(text("SELECT 1 FROM model_registry WHERE version=:v"),
                              {"v": version}).first()
        if not exists:
            return False
        conn.execute(text("UPDATE model_registry SET is_active=FALSE WHERE is_active"))
        conn.execute(text("UPDATE model_registry SET is_active=TRUE, activated_by=:by, "
                          "activated_at=NOW() WHERE version=:v"), {"by": by, "v": version})
    logger.info("Active model → {} ({})", version, by)
    return True


def auto_select_best(engine=None, *, metric: str = "oos_roc_auc", respect_manual: bool = True) -> str | None:
    """Activate the highest-``metric`` model — unless a manual pick is active (sticky).

    Returns the version now active, or None if the registry is empty.
    """
    engine = engine or get_engine()
    active = get_active(engine)
    if respect_manual and active and active.get("activated_by") == "manual":
        logger.info("Keeping manual active model {} (auto-select skipped).", active["version"])
        return active["version"]
    with engine.connect() as conn:
        best = conn.execute(text(
            f"SELECT version FROM model_registry WHERE {metric} IS NOT NULL "
            f"ORDER BY {metric} DESC NULLS LAST LIMIT 1")).first()
    if not best:
        return active["version"] if active else None
    set_active(engine, best[0], by="auto")
    return best[0]
