"""Registry serve path: joblib deserialize + feature-column alignment (no DB needed)."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("joblib")
pytest.importorskip("sklearn")


def _joblib_model():
    import joblib
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = rng.random((80, 3))
    y = (X[:, 0] > 0.5).astype(int)
    clf = LogisticRegression().fit(X, y)
    buf = io.BytesIO()
    joblib.dump(clf, buf)
    return {"artifact_format": "joblib", "artifact": buf.getvalue(), "version": "t1",
            "model_type": "xgboost", "feature_cols": ["a", "b", "c"]}


def _row(cols):
    return pd.DataFrame({**{c: [0.7] for c in cols}, "Target": [-1]},
                        index=pd.to_datetime(["2026-07-01"]))


def test_joblib_predict_shape_and_range():
    from sentisense.serve import champion
    out = champion._predict_from_registry(None, _joblib_model(), _row(["a", "b", "c"]))
    assert list(out.columns) == ["date", "proba"]
    assert out.shape[0] == 1 and 0.0 <= float(out["proba"].iloc[0]) <= 1.0


def test_feature_alignment_fills_missing_and_drops_extra():
    from sentisense.serve import champion
    # frame missing 'b' and carrying an extra 'zzz' → aligned to the model's feature_cols (b→0)
    frame = _row(["a", "c", "zzz"])
    out = champion._predict_from_registry(None, _joblib_model(), frame)
    assert out.shape[0] == 1 and np.isfinite(out["proba"].iloc[0])


def test_missing_artifact_raises():
    from sentisense.serve import champion
    bad = {"artifact_format": "joblib", "artifact": None, "version": "x", "model_type": "xgboost"}
    with pytest.raises(RuntimeError):
        champion._predict_from_registry(None, bad, _row(["a", "b", "c"]))


def _torch_bundle(cols, window):
    """A weights_only-safe seq bundle for a freshly-built GRU (untrained — we test I/O, not skill)."""
    import io

    import torch
    from sklearn.preprocessing import StandardScaler

    from sentisense.hpo.optuna_seq import _build

    params = {"dropout": 0.1, "dense_act": "relu", "d_dense": 16, "units": 8, "n_layers": 1,
              "recurrent_dropout": 0.0, "pooling": "last", "bidirectional": False}
    model = _build("GRU", len(cols), params)
    rng = np.random.default_rng(0)
    scaler = StandardScaler().fit(rng.random((50, len(cols))))
    bundle = {"arch": "GRU", "params": params, "window": window, "feature_cols": list(cols),
              "scaler_mean": scaler.mean_.astype(np.float32).tolist(),
              "scaler_scale": scaler.scale_.astype(np.float32).tolist(),
              "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}}
    buf = io.BytesIO(); torch.save(bundle, buf)
    return {"artifact_format": "torch", "artifact": buf.getvalue(), "version": "gru1",
            "model_type": "gru", "feature_cols": list(cols)}


def test_torch_serve_roundtrip_shape_and_range():
    pytest.importorskip("torch")
    from sentisense.serve import champion

    cols, window = ["a", "b", "c"], 3
    dates = pd.to_datetime(["2026-06-25", "2026-06-28", "2026-06-29", "2026-06-30", "2026-07-01"])
    full = pd.DataFrame({c: np.linspace(0, 1, len(dates)) for c in cols}, index=dates)
    full["Target"] = [0, 1, 0, 1, -1]
    to_predict = full[full["Target"] == -1]
    out = champion._predict_from_registry(None, _torch_bundle(cols, window), to_predict, full=full)
    assert list(out.columns) == ["date", "proba"]
    assert out.shape[0] == 1 and 0.0 <= float(out["proba"].iloc[0]) <= 1.0


def test_torch_serve_abstains_without_enough_history():
    pytest.importorskip("torch")
    from sentisense.serve import champion

    cols, window = ["a", "b", "c"], 10          # window longer than the frame → abstain at 0.5
    dates = pd.to_datetime(["2026-06-30", "2026-07-01"])
    full = pd.DataFrame({c: [0.3, 0.6] for c in cols}, index=dates)
    full["Target"] = [1, -1]
    to_predict = full[full["Target"] == -1]
    out = champion._predict_from_registry(None, _torch_bundle(cols, window), to_predict, full=full)
    assert float(out["proba"].iloc[0]) == 0.5
