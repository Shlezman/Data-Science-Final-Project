"""UI confusion matrix — counts + derived metrics from model_predictions (predicted vs actual)."""

from __future__ import annotations

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("sentisense")

from ui.queries import confusion_matrix


def _rows(tp, tn, fp, fn, pending):
    rows = []
    rows += [{"prediction": True, "actual": True}] * tp
    rows += [{"prediction": False, "actual": False}] * tn
    rows += [{"prediction": True, "actual": False}] * fp
    rows += [{"prediction": False, "actual": True}] * fn
    rows += [{"prediction": True, "actual": None}] * pending
    return rows


def test_counts_and_pending():
    cm = confusion_matrix(_rows(2, 3, 1, 1, 2))
    assert (cm["tp"], cm["tn"], cm["fp"], cm["fn"]) == (2, 3, 1, 1)
    assert cm["n"] == 7 and cm["pending"] == 2          # pending excluded from the matrix


def test_derived_metrics_match_hand_computation():
    cm = confusion_matrix(_rows(2, 3, 1, 1, 0))
    # acc=5/7, prec=2/3, rec=2/3, f1=2/3, mcc=(6-1)/sqrt(3*3*4*4)=5/12
    assert cm["accuracy"] == round(5 / 7, 4)
    assert cm["precision"] == round(2 / 3, 4)
    assert cm["recall"] == round(2 / 3, 4)
    assert cm["f1"] == round(2 / 3, 4)
    assert cm["mcc"] == round(5 / 12, 4)


def test_matches_sklearn():
    skl = pytest.importorskip("sklearn.metrics")
    rows = _rows(5, 4, 2, 3, 0)
    y_true = [int(r["actual"]) for r in rows]
    y_pred = [int(r["prediction"]) for r in rows]
    cm = confusion_matrix(rows)
    assert cm["accuracy"] == round(skl.accuracy_score(y_true, y_pred), 4)
    assert cm["mcc"] == round(skl.matthews_corrcoef(y_true, y_pred), 4)
    assert cm["f1"] == round(skl.f1_score(y_true, y_pred), 4)


def test_empty_is_safe():
    cm = confusion_matrix([])
    assert cm["n"] == 0 and cm["accuracy"] == 0.0 and cm["mcc"] == 0.0
