"""The UI's numpy projection must equal sklearn's scaler→PCA transform (same embpca space)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")


def test_projection_formula_matches_sklearn():
    """((x - scaler_mean)/scaler_scale - pca_mean) @ components.T == pca.transform(scaler.transform(x))."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 32)).astype(np.float64)
    scaler = StandardScaler().fit(X)
    pca = PCA(n_components=8, random_state=42).fit(scaler.transform(X))

    mean = scaler.mean_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)
    pmean = pca.mean_.astype(np.float32)
    comps = pca.components_.astype(np.float32)

    Xq = rng.normal(size=(10, 32)).astype(np.float32)
    ours = ((Xq - mean) / scale - pmean) @ comps.T
    ref = pca.transform(scaler.transform(Xq.astype(np.float64)))
    np.testing.assert_allclose(ours, ref, rtol=1e-3, atol=1e-3)


def test_basis_roundtrip_through_bytes():
    """float32 → tobytes → frombuffer reshape is lossless (the BYTEA storage contract)."""
    rng = np.random.default_rng(0)
    comps = rng.normal(size=(16, 768)).astype(np.float32)
    back = np.frombuffer(comps.tobytes(), dtype=np.float32).reshape(16, 768)
    np.testing.assert_array_equal(comps, back)


def test_persona_vote_thresholds():
    """Stance mapping: ≥ +0.5 up, ≤ −0.5 down, else neutral."""
    from ui.queries import _vote

    assert _vote(1.2) == "up"
    assert _vote(0.5) == "up"
    assert _vote(0.49) == "neutral"
    assert _vote(-0.49) == "neutral"
    assert _vote(-0.5) == "down"
