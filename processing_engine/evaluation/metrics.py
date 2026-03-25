"""
processing_engine.evaluation.metrics
=====================================
Pure metric computation functions for the SentiSense relevance evaluation.

All functions operate on plain Python lists/floats — no pandas required —
so they can be used in unit tests without heavy dependencies.

Metrics implemented
-------------------
- mae(predicted, gold)              Mean Absolute Error
- within_n_accuracy(predicted, gold, n)  % of predictions within n of gold
- pearson_r(predicted, gold)        Pearson correlation coefficient
- composite_score(per_category_within1)  Average Within-1 Accuracy across categories

Evaluation scope
----------------
Only the 6 relevance categories are evaluated.
global_sentiment is produced by the pipeline but is NOT evaluated here.
"""

from __future__ import annotations

import math
from typing import Sequence


# ═══════════════════════════════════════════════════════════════════════
# Core metric functions
# ═══════════════════════════════════════════════════════════════════════


def mae(predicted: Sequence[float], gold: Sequence[float]) -> float:
    """
    Mean Absolute Error between predicted and gold scores.

    Parameters
    ----------
    predicted : sequence of float
        Model-predicted scores.
    gold : sequence of float
        Human gold-label scores.

    Returns
    -------
    float
        Mean absolute error. Lower is better. 0.0 = perfect agreement.

    Raises
    ------
    ValueError
        If sequences have different lengths or are empty.
    """
    if len(predicted) != len(gold):
        raise ValueError(
            f"Length mismatch: predicted={len(predicted)}, gold={len(gold)}"
        )
    if len(predicted) == 0:
        raise ValueError("Cannot compute MAE on empty sequences.")
    return sum(abs(p - g) for p, g in zip(predicted, gold)) / len(predicted)


def within_n_accuracy(
    predicted: Sequence[float],
    gold: Sequence[float],
    n: int,
) -> float:
    """
    Percentage of predictions within ``n`` points of the gold label.

    Parameters
    ----------
    predicted : sequence of float
        Model-predicted scores.
    gold : sequence of float
        Human gold-label scores.
    n : int
        Tolerance threshold (e.g., 1 for Within-1, 2 for Within-2).

    Returns
    -------
    float
        Accuracy in [0.0, 1.0]. Higher is better. 1.0 = all predictions
        within n of gold.

    Raises
    ------
    ValueError
        If sequences have different lengths or are empty.
    """
    if len(predicted) != len(gold):
        raise ValueError(
            f"Length mismatch: predicted={len(predicted)}, gold={len(gold)}"
        )
    if len(predicted) == 0:
        raise ValueError("Cannot compute Within-N Accuracy on empty sequences.")
    hits = sum(1 for p, g in zip(predicted, gold) if abs(p - g) <= n)
    return hits / len(predicted)


def pearson_r(predicted: Sequence[float], gold: Sequence[float]) -> float:
    """
    Pearson correlation coefficient between predicted and gold scores.

    Measures whether the model ranks headlines in the same order as the
    gold labels, regardless of absolute scale.

    Parameters
    ----------
    predicted : sequence of float
        Model-predicted scores.
    gold : sequence of float
        Human gold-label scores.

    Returns
    -------
    float
        Pearson r in [-1.0, 1.0]. Higher is better.
        Returns 0.0 if either sequence has zero variance (constant values).

    Raises
    ------
    ValueError
        If sequences have different lengths or are empty.
    """
    if len(predicted) != len(gold):
        raise ValueError(
            f"Length mismatch: predicted={len(predicted)}, gold={len(gold)}"
        )
    if len(predicted) == 0:
        raise ValueError("Cannot compute Pearson r on empty sequences.")

    n = len(predicted)
    mean_p = sum(predicted) / n
    mean_g = sum(gold) / n

    cov = sum((p - mean_p) * (g - mean_g) for p, g in zip(predicted, gold))
    std_p = math.sqrt(sum((p - mean_p) ** 2 for p in predicted))
    std_g = math.sqrt(sum((g - mean_g) ** 2 for g in gold))

    if std_p == 0.0 or std_g == 0.0:
        # One sequence is constant — correlation is undefined; return 0.
        return 0.0

    return cov / (std_p * std_g)


def composite_score(per_category_within1: Sequence[float]) -> float:
    """
    Composite model score: average Within-1 Accuracy across all 6 categories.

    This is the primary metric used to rank models on the leaderboard.

    Parameters
    ----------
    per_category_within1 : sequence of float
        Within-1 Accuracy for each category (values in [0.0, 1.0]).
        Must have exactly 6 elements.

    Returns
    -------
    float
        Composite score in [0.0, 1.0]. Higher is better.

    Raises
    ------
    ValueError
        If the sequence does not have exactly 6 elements.
    """
    if len(per_category_within1) != 6:
        raise ValueError(
            f"Expected 6 category scores, got {len(per_category_within1)}."
        )
    return sum(per_category_within1) / 6


# ═══════════════════════════════════════════════════════════════════════
# Per-category metric bundle
# ═══════════════════════════════════════════════════════════════════════

CATEGORY_NAMES: list[str] = [
    "Politics & Government",
    "Economy & Finance",
    "Security & Military",
    "Health & Medicine",
    "Science & Climate",
    "Technology",
]

# These match the exact column names in the golden dataset CSV and the
# pipeline output (relevance_category_1…6 maps to these in order).
CATEGORY_COLUMNS: list[str] = [
    "politics_government",
    "economy_finance",
    "security_military",
    "health_medicine",
    "science_climate",
    "technology",
]


def compute_category_metrics(
    predicted: Sequence[float],
    gold: Sequence[float],
) -> dict[str, float]:
    """
    Compute all metrics for a single category.

    Parameters
    ----------
    predicted : sequence of float
        Model-predicted scores for this category.
    gold : sequence of float
        Gold-label scores for this category.

    Returns
    -------
    dict with keys:
        mae          float  Mean Absolute Error
        within1      float  Within-1 Accuracy (0.0–1.0)
        within2      float  Within-2 Accuracy (0.0–1.0)
        pearson_r    float  Pearson correlation coefficient
    """
    return {
        "mae": mae(predicted, gold),
        "within1": within_n_accuracy(predicted, gold, n=1),
        "within2": within_n_accuracy(predicted, gold, n=2),
        "pearson_r": pearson_r(predicted, gold),
    }


def compute_all_metrics(
    predictions: dict[str, list[float]],
    gold_labels: dict[str, list[float]],
) -> dict[str, dict[str, float]]:
    """
    Compute metrics for all 6 categories and the aggregate.

    Parameters
    ----------
    predictions : dict
        Keys: ``"cat_1"`` … ``"cat_6"``.
        Values: list of predicted scores (one per headline).
    gold_labels : dict
        Keys: ``"cat_1"`` … ``"cat_6"``.
        Values: list of gold scores (one per headline).

    Returns
    -------
    dict
        Keys: ``"cat_1"`` … ``"cat_6"`` + ``"average"``.
        Values: dicts with keys ``mae``, ``within1``, ``within2``,
        ``pearson_r``.
        The ``"average"`` entry also includes ``"composite_score"``.
    """
    results: dict[str, dict[str, float]] = {}

    for col in CATEGORY_COLUMNS:
        results[col] = compute_category_metrics(
            predictions[col], gold_labels[col]
        )

    # Aggregate across categories
    avg_mae = sum(results[c]["mae"] for c in CATEGORY_COLUMNS) / 6
    avg_within1 = sum(results[c]["within1"] for c in CATEGORY_COLUMNS) / 6
    avg_within2 = sum(results[c]["within2"] for c in CATEGORY_COLUMNS) / 6
    avg_pearson = sum(results[c]["pearson_r"] for c in CATEGORY_COLUMNS) / 6

    results["average"] = {
        "mae": avg_mae,
        "within1": avg_within1,
        "within2": avg_within2,
        "pearson_r": avg_pearson,
        "composite_score": composite_score(
            [results[c]["within1"] for c in CATEGORY_COLUMNS]
        ),
    }

    return results
