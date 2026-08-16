"""Backfill held-out classification metrics (precision/recall/F1) for a registry model.

Reproduces the EXACT out-of-sample evaluation ``train_registry.py`` ran at registration
time — same fused frame, same chronological 70/15/15 split, same fixed seeds, same
``seq_holdout_eval`` mean-over-seeds probabilities on the sacred test tail — then computes
the positive-class ("up") precision, recall, and F1 at the 0.5 threshold and stores them
on the model's registry row (migration 009). The recomputed accuracy is checked against
the stored ``oos_accuracy`` so a drifted reproduction fails loudly instead of writing
metrics that belong to a different evaluation.

Run on the GPU node (torch + the fused frame build):
    uv run --extra finance --extra ml python scripts/backfill_oos_classification.py            # active model
    uv run --extra finance --extra ml python scripts/backfill_oos_classification.py --version patchtst-20260702-1351

# ponytail: torch sequence families only — trees/forecasters need a different
# reproduction path; add it when a non-seq champion needs the panel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentisense.db import get_engine  # noqa: E402

_SEQ_ARCHS = {"lstm": "LSTM", "gru": "GRU", "tcn": "TCN", "patchtst": "PatchTST"}
_FAR_FUTURE = "2100-01-01"
_ACC_TOLERANCE = 0.005

_MIGRATION = Path(__file__).resolve().parents[1] / "sentisense" / "db" / "migrations" / "009_oos_classification.sql"

_ROW = text("""
    SELECT version, model_type, params, oos_accuracy, oos_n
    FROM model_registry
    WHERE (CAST(:version AS text) IS NULL AND is_active) OR version = CAST(:version AS text)
    LIMIT 1
""")

_UPDATE = text("""
    UPDATE model_registry
    SET oos_precision = :precision, oos_recall = :recall, oos_f1 = :f1
    WHERE version = :version
""")


def main() -> int:
    """Recompute the held-out eval for one registry model and store precision/recall/F1."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default=None, help="Registry version (default: the active model).")
    ap.add_argument("--seeds", type=int, default=3, help="Seeds to average (match the registration run).")
    ap.add_argument("--cutoff", default=_FAR_FUTURE,
                    help="Frame cutoff date — set to the model's training date so the "
                         "70/15/15 split reproduces the ORIGINAL test tail (the frame has "
                         "grown since registration; e.g. --cutoff 2026-07-02).")
    args = ap.parse_args()

    engine = get_engine()
    with engine.begin() as conn:
        for stmt in _MIGRATION.read_text(encoding="utf-8").split(";"):
            if stmt.strip() and not stmt.strip().startswith("--"):
                conn.execute(text(stmt))
        row = conn.execute(_ROW, {"version": args.version}).mappings().first()
    if not row:
        raise SystemExit(f"no registry row found (version={args.version or 'active'}).")

    arch = _SEQ_ARCHS.get(row["model_type"])
    if arch is None:
        raise SystemExit(f"{row['version']} is a '{row['model_type']}' model — only torch "
                         f"sequence families {sorted(_SEQ_ARCHS)} are supported.")
    params = row["params"] if isinstance(row["params"], dict) else __import__("json").loads(row["params"])

    from sklearn.metrics import f1_score, precision_score, recall_score

    from sentisense.features import build_fused_dataset
    from sentisense.hpo.optuna_seq import seq_holdout_eval

    logger.info("Rebuilding fused frame and reproducing the OOS eval for {} ({} seeds)…",
                row["version"], args.seeds)
    df = build_fused_dataset(engine, cutoff=args.cutoff, overnight=True)
    proba_s, label_s = seq_holdout_eval(df, arch, params, n_seeds=args.seeds)
    proba, labels = proba_s.to_numpy(), label_s.to_numpy().astype(int)
    preds = (proba > 0.5).astype(int)

    acc = float((preds == labels).mean())
    stored = float(row["oos_accuracy"]) if row["oos_accuracy"] is not None else None
    logger.info("Reproduced n={} accuracy={:.4f} (stored {}).", len(labels), acc,
                f"{stored:.4f}" if stored is not None else "—")
    if stored is not None and abs(acc - stored) > _ACC_TOLERANCE:
        raise SystemExit(f"reproduced accuracy {acc:.4f} differs from stored {stored:.4f} "
                         f"by > {_ACC_TOLERANCE} — refusing to write mismatched metrics.")

    metrics = {"precision": float(precision_score(labels, preds, zero_division=0)),
               "recall": float(recall_score(labels, preds, zero_division=0)),
               "f1": float(f1_score(labels, preds, zero_division=0))}
    with engine.begin() as conn:
        conn.execute(_UPDATE, {"version": row["version"], **metrics})
    logger.info("Stored for {}: precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}.",
                row["version"], **metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
