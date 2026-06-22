"""Streamed daily centroid == old full-matrix groupby-mean (the OOM fix must not change results)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentisense.embed import embeddings as E


class _FakeConn:
    """Minimal context-manager that mimics engine.connect().execution_options(...)."""

    def execution_options(self, **_):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeEngine:
    def connect(self):
        return _FakeConn()


def _make_rows(rng, n_per_day, dim, days):
    """Build a (meta, vectors) corpus + the row-dicts the SQL would return."""
    rows = []
    for d in days:
        for _ in range(n_per_day):
            v = rng.standard_normal(dim).astype(np.float32)
            rows.append({"headline_id": len(rows), "date": d, "dim": dim, "embedding": v.tobytes()})
    return pd.DataFrame(rows)


def test_streamed_centroid_matches_full_groupby(monkeypatch):
    rng = np.random.default_rng(7)
    dim = 16
    days = ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03"]  # uneven counts per day
    # 3 distinct dates, 5 headlines each
    full = _make_rows(rng, 5, dim, ["2024-01-01", "2024-01-02", "2024-01-03"])

    # Reference: the OLD path — materialise everything, groupby(date).mean() + dispersion.
    vecs = np.vstack([np.frombuffer(b, dtype=np.float32) for b in full["embedding"]]).reshape(-1, dim)
    cdf = pd.DataFrame(vecs, index=pd.to_datetime(full["date"]),
                       columns=[f"embc_{i:03d}" for i in range(dim)])
    grp = cdf.groupby(level=0)
    ref_mean = grp.mean()

    # Streamed path: feed chunks that straddle date boundaries (chunksize < a day's rows).
    def fake_read_sql(_sql, _conn, params=None, chunksize=None):
        for start in range(0, len(full), 4):                  # 4-row chunks across 15 rows
            yield full.iloc[start:start + 4].reset_index(drop=True)

    monkeypatch.setattr(E.pd, "read_sql", fake_read_sql)
    out = E.daily_embedding_centroid(_FakeEngine(), cutoff="2100-01-01", chunksize=4)

    emb_cols = [c for c in out.columns if c.startswith("embc_")]
    assert len(emb_cols) == dim
    assert list(out.index) == list(ref_mean.index)
    np.testing.assert_allclose(out[emb_cols].to_numpy(), ref_mean.to_numpy(), rtol=1e-5, atol=1e-5)
    assert (out["emb_count"] == 5).all()                      # 5 headlines per date
    assert (out["emb_dispersion"] >= 0).all()


def test_empty_corpus_returns_empty_frame(monkeypatch):
    monkeypatch.setattr(E.pd, "read_sql", lambda *a, **k: iter(()))
    out = E.daily_embedding_centroid(_FakeEngine(), cutoff="2100-01-01")
    assert out.empty
