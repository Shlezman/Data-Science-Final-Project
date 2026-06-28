"""Keyset-paginated daily centroid == full groupby-mean (the OOM fix must not change results)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentisense.embed import embeddings as E


class _FakeConn:
    """Minimal context-manager standing in for engine.connect()."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeEngine:
    def connect(self):
        return _FakeConn()


def _make_rows(rng, n_per_day, dim, dates):
    """Build the row-dicts the SQL would return: ascending headline_id, BYTEA embeddings."""
    rows = []
    for d in dates:
        for _ in range(n_per_day):
            v = rng.standard_normal(dim).astype(np.float32)
            rows.append({"headline_id": len(rows), "date": d, "dim": dim, "embedding": v.tobytes()})
    return pd.DataFrame(rows)


def _keyset_reader(full: pd.DataFrame):
    """Return a pd.read_sql stand-in honouring the (last_id, page) keyset contract."""

    def fake_read_sql(_sql, _conn, params=None):
        sub = full[full["headline_id"] > params["last_id"]].sort_values("headline_id")
        return sub.head(params["page"]).reset_index(drop=True)

    return fake_read_sql


def test_keyset_centroid_matches_full_groupby(monkeypatch):
    rng = np.random.default_rng(7)
    dim = 16
    full = _make_rows(rng, 5, dim, ["2024-01-01", "2024-01-02", "2024-01-03"])  # 15 rows, 3 dates

    # Reference: the old full-matrix path — groupby(date).mean().
    vecs = np.vstack([np.frombuffer(b, dtype=np.float32) for b in full["embedding"]]).reshape(-1, dim)
    cdf = pd.DataFrame(vecs, index=pd.to_datetime(full["date"]),
                       columns=[f"embc_{i:03d}" for i in range(dim)])
    ref_mean = cdf.groupby(level=0).mean()

    # Paged path with page=4 → multiple pages straddling date boundaries (order-independent accum).
    monkeypatch.setattr(E.pd, "read_sql", _keyset_reader(full))
    out = E.daily_embedding_centroid(_FakeEngine(), cutoff="2100-01-01", page=4)

    emb_cols = [c for c in out.columns if c.startswith("embc_")]
    assert len(emb_cols) == dim
    assert list(out.index) == list(ref_mean.index)
    np.testing.assert_allclose(out[emb_cols].to_numpy(), ref_mean.to_numpy(), rtol=1e-5, atol=1e-5)
    assert (out["emb_count"] == 5).all()
    assert (out["emb_dispersion"] >= 0).all()


def test_exact_page_multiple_terminates(monkeypatch):
    # 12 rows with page=4 → 3 full pages then an empty 4th; loop must stop, not spin.
    rng = np.random.default_rng(1)
    full = _make_rows(rng, 4, 8, ["2024-02-01", "2024-02-02", "2024-02-03"])
    monkeypatch.setattr(E.pd, "read_sql", _keyset_reader(full))
    out = E.daily_embedding_centroid(_FakeEngine(), cutoff="2100-01-01", page=4)
    assert len(out) == 3 and (out["emb_count"] == 4).all()


def test_empty_corpus_returns_empty_frame(monkeypatch):
    monkeypatch.setattr(E.pd, "read_sql", lambda *a, **k: pd.DataFrame(
        columns=["headline_id", "date", "dim", "embedding"]))
    out = E.daily_embedding_centroid(_FakeEngine(), cutoff="2100-01-01")
    assert out.empty
