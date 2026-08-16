"""Reforecast serve path (champion._predict_reforecast) — pure/offline, heavy bits stubbed.

Covers: threshold folding into the proba contract, HPO-params write-back, and the
family guard that keeps Chronos/TimesFM rows on the pinned-fallback path.
"""

import numpy as np
import pandas as pd
import pytest

from sentisense.serve import champion


def _active(params: dict) -> dict:
    return {"version": "TFT-test", "model_type": "pf", "artifact_format": "reforecast",
            "params": params}


def _frames():
    idx = pd.DatetimeIndex(["2026-08-13"])
    to_predict = pd.DataFrame({"Target": [-1], "mean_s": [0.2]}, index=idx)
    full = to_predict.copy()
    return to_predict, full


def test_threshold_folds_into_proba(monkeypatch):
    """p=0.65 with val-tuned thr=0.60 must serve as 0.55 (> 0.5 ⇔ > thr contract)."""
    to_predict, full = _frames()
    monkeypatch.setattr("sentisense.features.dataset._load_finance",
                        lambda: ({"TA125_Price": pd.Series([1.0])},) * 4)
    monkeypatch.setattr(
        "sentisense.models.tft_forecaster.pf_serve_forecast",
        lambda *a, **k: {"proba": {pd.Timestamp("2026-08-13"): 0.65}, "hpo_params": {}})
    out = champion._predict_reforecast(None, _active({"family": "pf", "arch": "TFT",
                                                      "threshold": 0.60, "hpo": {"x": 1}}),
                                       to_predict, full)
    assert list(out.columns) == ["date", "proba"]
    assert np.isclose(out["proba"].iloc[0], 0.55)


def test_below_threshold_maps_below_half(monkeypatch):
    to_predict, full = _frames()
    monkeypatch.setattr("sentisense.features.dataset._load_finance",
                        lambda: ({"TA125_Price": pd.Series([1.0])},) * 4)
    monkeypatch.setattr(
        "sentisense.models.tft_forecaster.pf_serve_forecast",
        lambda *a, **k: {"proba": {pd.Timestamp("2026-08-13"): 0.55}, "hpo_params": {}})
    out = champion._predict_reforecast(None, _active({"family": "pf", "threshold": 0.60,
                                                      "hpo": {"x": 1}}), to_predict, full)
    assert out["proba"].iloc[0] < 0.5


def test_hpo_params_written_back(monkeypatch):
    """First serve (no cached hpo) must persist the search winner on the registry row."""
    to_predict, full = _frames()
    saved = {}
    monkeypatch.setattr("sentisense.features.dataset._load_finance",
                        lambda: ({"TA125_Price": pd.Series([1.0])},) * 4)
    monkeypatch.setattr(
        "sentisense.models.tft_forecaster.pf_serve_forecast",
        lambda *a, **k: {"proba": {pd.Timestamp("2026-08-13"): 0.5},
                         "hpo_params": {"hidden_size": 32}})
    monkeypatch.setattr("sentisense.serve.registry.update_params",
                        lambda engine, *, version, params: saved.update(params) or True)
    champion._predict_reforecast(None, _active({"family": "pf", "threshold": 0.5}),
                                 to_predict, full)
    assert saved.get("hpo") == {"hidden_size": 32}


def test_non_pf_family_raises(monkeypatch):
    to_predict, full = _frames()
    with pytest.raises(NotImplementedError):
        champion._predict_reforecast(None, _active({"family": "chronos", "threshold": 0.5}),
                                     to_predict, full)
