"""Derived embedding features: leak-safe PCA + cluster-distance basis (fit train-only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("sqlalchemy")

from sentisense.embed.derived import fit_transform_derived


def _centroids(n_days=200, dim=32, seed=3):
    idx = pd.date_range("2020-01-01", periods=n_days, freq="D")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((n_days, dim)), index=idx,
                        columns=[f"embc_{i:03d}" for i in range(dim)])


def test_shape_and_column_names():
    cen = _centroids()
    fit_cutoff = cen.index[int(len(cen) * 0.85) - 1]
    out = fit_transform_derived(cen, fit_cutoff=fit_cutoff, n_pca=16, n_clusters=8)
    assert out.shape == (len(cen), 16 + 8)
    assert list(out.index) == list(cen.index)
    assert [c for c in out.columns if c.startswith("embpca_")] == [f"embpca_{i:03d}" for i in range(16)]
    assert [c for c in out.columns if c.startswith("embclus_dist_")] == [f"embclus_dist_{i}" for i in range(8)]
    # derived cols must NOT collide with the raw 'embc_' prefix (downstream PCA scoping relies on it)
    assert not any(c.startswith("embc_0") for c in out.columns)


def test_basis_is_train_only_no_leak():
    cen = _centroids()
    fit_cutoff = cen.index[int(len(cen) * 0.85) - 1]
    full = fit_transform_derived(cen, fit_cutoff=fit_cutoff, n_pca=16, n_clusters=8)

    # Re-deriving with the test rows DELETED must not change the train rows' features —
    # i.e. the OOS rows never influenced their own transform basis.
    train_only_input = cen[cen.index <= fit_cutoff]
    refit = fit_transform_derived(train_only_input, fit_cutoff=fit_cutoff, n_pca=16, n_clusters=8)
    np.testing.assert_allclose(full.loc[refit.index].to_numpy(), refit.to_numpy(),
                               rtol=1e-6, atol=1e-6)


def test_too_few_train_days_raises():
    cen = _centroids(n_days=10, dim=32)
    early = cen.index[2]                       # only 3 train days, far below n_clusters
    with pytest.raises(ValueError):
        fit_transform_derived(cen, fit_cutoff=early, n_pca=16, n_clusters=8)


def test_pca_capped_to_available_dims():
    cen = _centroids(n_days=120, dim=8)        # ask for more PCA comps than features
    fit_cutoff = cen.index[int(len(cen) * 0.85) - 1]
    out = fit_transform_derived(cen, fit_cutoff=fit_cutoff, n_pca=16, n_clusters=4)
    assert len([c for c in out.columns if c.startswith("embpca_")]) == 8   # capped to n_features
