"""pgvector deploy/fill helper — pure conversion logic (no DB)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_path = Path(__file__).resolve().parent.parent / "scripts" / "deploy_vectordb.py"
_spec = importlib.util.spec_from_file_location("deploy_vectordb", _path)
deploy_vectordb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(deploy_vectordb)
_vec_literal = deploy_vectordb._vec_literal


def test_vec_literal_roundtrips_float32():
    v = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32)
    lit = _vec_literal(v.tobytes(), dim=4)
    assert lit.startswith("[") and lit.endswith("]")
    parsed = np.array([float(x) for x in lit[1:-1].split(",")], dtype=np.float32)
    assert np.allclose(parsed, v, atol=1e-6)


def test_vec_literal_dim_mismatch_raises():
    v = np.zeros(3, dtype=np.float32)
    with pytest.raises(ValueError, match="!= dim"):
        _vec_literal(v.tobytes(), dim=4)
