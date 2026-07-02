"""Dashboard-v2 backend: full-history confusion math + new query surface (no DB needed)."""

from __future__ import annotations

import pytest

pytest.importorskip("sqlalchemy")


def test_full_eval_confusion_all_settled():
    """champion_full_eval rows are always settled (actual non-null) → pending=0, n=all."""
    from ui import queries

    rows = [
        {"date": "2026-01-01", "prediction": True, "actual": True},    # tp
        {"date": "2026-01-02", "prediction": True, "actual": False},   # fp
        {"date": "2026-01-03", "prediction": False, "actual": False},  # tn
        {"date": "2026-01-04", "prediction": False, "actual": True},   # fn
        {"date": "2026-01-05", "prediction": True, "actual": True},    # tp
    ]
    cm = queries.confusion_matrix(rows)
    assert cm["tp"] == 2 and cm["fp"] == 1 and cm["tn"] == 1 and cm["fn"] == 1
    assert cm["n"] == 5 and cm["pending"] == 0
    assert cm["accuracy"] == round(3 / 5, 4)


def test_new_query_functions_exist():
    """The dashboard-v2 read paths are importable + callable."""
    from ui import queries

    for name in ("full_eval_rows", "today_prediction", "eda_aggregates", "centroid_points"):
        assert callable(getattr(queries, name)), f"missing queries.{name}"


def test_corr_sql_has_15_pairs():
    """The category-correlation query covers all 15 unordered relevance pairs (6 choose 2)."""
    from ui import queries

    assert str(queries._EDA_CORR).count("corr(") == 15
    assert len(queries._CORR_LABELS) == 6
