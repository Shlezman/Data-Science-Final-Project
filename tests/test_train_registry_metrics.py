"""train_registry._metrics must honor a tuned threshold (else forecaster oos_accuracy is wrong)."""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("numpy")

_TR = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "train_registry.py"


def _load():
    spec = importlib.util.spec_from_file_location("train_registry", _TR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_metrics_threshold_is_threaded():
    """Accuracy must change with the threshold — proving forecasters aren't scored at a fixed 0.5."""
    mod = _load()
    proba = [0.6, 0.6, 0.4, 0.4]
    labels = [1, 1, 0, 0]
    at_050 = mod._metrics(proba, labels, threshold=0.5)["accuracy"]
    at_070 = mod._metrics(proba, labels, threshold=0.7)["accuracy"]
    assert at_050 == pytest.approx(1.0)          # 0.6→up, 0.4→down → all correct
    assert at_070 < at_050                        # everything below 0.7 → all "down" → worse
    # roc_auc is threshold-free → identical regardless of threshold
    assert mod._metrics(proba, labels, threshold=0.5)["roc_auc"] == \
        mod._metrics(proba, labels, threshold=0.7)["roc_auc"]


def test_pf_arch_map():
    """The pytorch-forecasting family map covers TFT / NHiTS / NBEATS."""
    mod = _load()
    assert mod._PF_ARCHS == {"tft": "TFT", "nhits": "NHiTS", "nbeats": "NBEATS"}
