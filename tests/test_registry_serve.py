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
