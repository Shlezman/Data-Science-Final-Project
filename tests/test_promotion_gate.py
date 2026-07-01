"""WS2 promotion gate — challenger promotes ONLY past the OOS margin + guards."""

from __future__ import annotations

import importlib.util as _u
from pathlib import Path

import pytest

pytest.importorskip("loguru")
pytest.importorskip("sentisense")


def _gate():
    p = Path(__file__).resolve().parent.parent / "scripts" / "challenger_hpo.py"
    spec = _u.spec_from_file_location("challenger_hpo", p)
    m = _u.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.should_promote


def test_worse_challenger_is_not_promoted():
    should_promote = _gate()
    champ = {"roc_auc": 0.56, "mcc": 0.10, "n": 380}
    chal = {"roc_auc": 0.54, "mcc": 0.08, "n": 380}      # lower AUC
    ok, reason = should_promote(champ, chal)
    assert ok is False and "ROC-AUC gain" in reason


def test_better_challenger_is_promoted():
    should_promote = _gate()
    champ = {"roc_auc": 0.52, "mcc": 0.02, "n": 380}
    chal = {"roc_auc": 0.56, "mcc": 0.05, "n": 380}      # +0.04 AUC, MCC up
    ok, reason = should_promote(champ, chal, min_auc_gain=0.02)
    assert ok is True


def test_tiny_gain_below_margin_is_not_promoted():
    should_promote = _gate()
    champ = {"roc_auc": 0.540, "mcc": 0.02, "n": 380}
    chal = {"roc_auc": 0.545, "mcc": 0.03, "n": 380}     # +0.005 < 0.02 margin
    ok, _ = should_promote(champ, chal, min_auc_gain=0.02)
    assert ok is False


def test_small_oos_window_blocks_promotion():
    should_promote = _gate()
    champ = {"roc_auc": 0.52, "mcc": 0.02, "n": 380}
    chal = {"roc_auc": 0.60, "mcc": 0.20, "n": 120}      # great, but n < min_n
    ok, reason = should_promote(champ, chal, min_n=200)
    assert ok is False and "insufficient" in reason


def test_mcc_regression_blocks_promotion():
    should_promote = _gate()
    champ = {"roc_auc": 0.52, "mcc": 0.12, "n": 380}
    chal = {"roc_auc": 0.56, "mcc": 0.05, "n": 380}      # AUC up but MCC worse
    ok, reason = should_promote(champ, chal)
    assert ok is False and "MCC regressed" in reason
