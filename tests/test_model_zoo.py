"""Model-zoo additions: pure logic offline; torch/dep forward passes guarded."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentisense.hpo.optuna_seq import _model_kwargs, study_name_for


def test_study_name_per_arch():
    assert study_name_for("TCN") == "sentisense_tcn_scores"
    assert study_name_for("PatchTST") == "sentisense_patchtst_scores"


@pytest.mark.parametrize("arch,params,expect", [
    ("GRU", {"dropout": 0.2, "dense_act": "relu", "d_dense": 32, "units": 48, "n_layers": 2,
             "recurrent_dropout": 0.1, "pooling": "attn", "bidirectional": True},
     {"hidden": 48, "n_layers": 2, "bidirectional": True}),
    ("TCN", {"dropout": 0.2, "dense_act": "elu", "d_dense": 16, "channels": 64, "levels": 4,
             "kernel_size": 3, "pooling": "avg"},
     {"channels": 64, "levels": 4, "kernel_size": 3}),
    ("PatchTST", {"dropout": 0.2, "dense_act": "gelu", "d_dense": 64, "d_model": 64, "n_heads": 4,
                  "depth": 2, "patch_len": 8, "stride": 4},
     {"d_model": 64, "n_heads": 4, "patch_len": 8, "stride": 4}),
])
def test_model_kwargs_maps_per_arch(arch, params, expect):
    kw = _model_kwargs(arch, params)
    for k, v in expect.items():
        assert kw[k] == v
    assert kw["dropout"] == 0.2 and "d_dense" in kw


def test_model_kwargs_unknown_arch_raises():
    with pytest.raises(ValueError, match="unknown arch"):
        _model_kwargs("Nope", {"dropout": 0.1, "dense_act": "relu"})


def test_tft_make_frame_time_idx_and_dates():
    from sentisense.models.tft_forecaster import _make_frame
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    price = pd.Series(np.linspace(100, 110, 10), index=dates)
    cov = pd.DataFrame({"mean_x": np.arange(10.0)}, index=dates)
    frame, cols, didx = _make_frame(price, dt_cut := pd.Timestamp("2024-01-31"), cov)
    assert list(frame["time_idx"]) == list(range(len(frame)))   # contiguous
    assert (frame["group"] == "ta125").all()
    assert cols == ["mean_x"] and len(didx) == len(frame)        # one return dropped (diff)


@pytest.mark.parametrize("arch,fixed,expect", [
    ("TFT", {"learning_rate": 1e-3, "dropout": 0.2, "batch_size": 64, "hidden_size": 32,
             "attention_head_size": 2, "hidden_continuous_size": 16},
     ["hidden_size", "attention_head_size", "hidden_continuous_size"]),
    ("NHiTS", {"learning_rate": 1e-3, "dropout": 0.2, "batch_size": 64, "hidden_size": 128},
     ["hidden_size"]),
    ("NBEATS", {"learning_rate": 1e-3, "dropout": 0.2, "batch_size": 64, "widths": "32x512",
                "backcast_loss_ratio": 0.1},
     ["widths", "backcast_loss_ratio"]),
])
def test_pf_param_space_per_arch(arch, fixed, expect):
    optuna = pytest.importorskip("optuna")
    from sentisense.models.tft_forecaster import PF_ARCHS, _param_space
    assert set(PF_ARCHS) == {"TFT", "NHiTS", "NBEATS"}
    p = _param_space(optuna.trial.FixedTrial(fixed), arch)
    assert {"learning_rate", "dropout", "batch_size"} <= set(p)   # shared knobs
    assert all(k in p for k in expect)                            # arch-specific body


def test_nbeats_is_univariate_marker():
    from sentisense.models.tft_forecaster import _UNIVARIATE
    assert "NBEATS" in _UNIVARIATE and "TFT" not in _UNIVARIATE


def test_chronos_load_raises_without_dep():
    pytest.importorskip  # noqa: B018
    try:
        import chronos  # noqa: F401
    except ImportError:
        from sentisense.models.chronos_forecaster import load_chronos
        with pytest.raises(ImportError, match="chronos-forecasting"):
            load_chronos()
    else:
        pytest.skip("chronos installed — skip the missing-dep path")


# ── torch forward-pass shape checks (skip if torch absent) ───────────────────
def test_tcn_forward_shape():
    torch = pytest.importorskip("torch")
    from sentisense.models.seq_zoo import TCNClassifier
    m = TCNClassifier(n_features=6, channels=16, levels=3, kernel_size=3)
    out = m(torch.randn(4, 12, 6))
    assert out.shape == (4,)


def test_patchtst_forward_shape_and_short_window():
    torch = pytest.importorskip("torch")
    from sentisense.models.seq_zoo import PatchTSTClassifier
    m = PatchTSTClassifier(n_features=5, patch_len=8, stride=4, d_model=32, n_heads=4, depth=1)
    assert m(torch.randn(3, 20, 5)).shape == (3,)
    assert m(torch.randn(3, 4, 5)).shape == (3,)    # window < patch_len → left-padded, no crash


def test_zoo_registry_has_all_archs():
    pytest.importorskip("torch")
    from sentisense.models.seq_zoo import ARCHITECTURES
    assert {"LSTM", "GRU", "TCN", "PatchTST"} <= set(ARCHITECTURES)
