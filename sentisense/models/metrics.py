"""Torch-free classification metrics (single source).

Lives apart from ``train.py`` (which imports torch) so the metric set can be reused by
the leaderboard / backtest / non-LSTM models without dragging torch + CUDA. ``train.py``
re-exports ``metrics_at`` for backward compatibility.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def metrics_at(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Classification metrics at a given decision threshold (+ ROC-AUC, threshold-free)."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    preds = (probs > threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="macro")),
        "roc_auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.5,
        "mcc": float(matthews_corrcoef(labels, preds)),
    }
