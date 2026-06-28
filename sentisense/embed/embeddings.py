"""Embed Hebrew headlines (≤ cutoff) and cache to Postgres — idempotent + resumable.

Uses the configured multilingual model (default ``intfloat/multilingual-e5-base``,
which handles Hebrew well). e5 expects a ``passage: `` prefix for documents. Vectors
are L2-normalised and stored as float32 BYTEA in ``headline_embeddings`` keyed by
(headline_id, embed_model), so re-runs skip already-embedded rows.

Run (server-side):
    uv sync --extra embed
    uv run python -m sentisense.embed.embeddings --dry-run
    uv run python -m sentisense.embed.embeddings --batch 128
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

from sentisense.config import EMBED_BATCH, EMBED_MODEL
from sentisense.constants import CUTOFF_DATE, REPO_ROOT
from sentisense.db import get_engine

_MIGRATION = REPO_ROOT / "sentisense" / "db" / "migrations" / "001_headline_embeddings.sql"

# Date predicate templated by scope so one indexed join serves three modes:
#   precutoff  → rh.date <= :cutoff   (default; the modeling corpus)
#   postcutoff → rh.date >  :cutoff   (Phase-2 buy-overlay / forward use)
#   all        → no date bound
_UNEMBEDDED_SQL_TMPL = """
    SELECT rh.id AS headline_id, rh.headline
    FROM raw_headlines rh
    LEFT JOIN headline_embeddings he
        ON he.headline_id = rh.id AND he.embed_model = :model
    WHERE he.headline_id IS NULL
      {date_clause}
    ORDER BY rh.id
"""
_DATE_CLAUSES = {
    "precutoff": "AND rh.date <= :cutoff",
    "postcutoff": "AND rh.date > :cutoff",
    "all": "",
}

_INSERT_SQL = text(
    """
    INSERT INTO headline_embeddings (headline_id, embed_model, dim, embedding)
    VALUES (:headline_id, :embed_model, :dim, :embedding)
    ON CONFLICT (headline_id, embed_model) DO NOTHING
    """
)

_LOAD_SQL = text(
    """
    SELECT he.headline_id, rh.date::date AS date, he.dim, he.embedding
    FROM headline_embeddings he
    JOIN raw_headlines rh ON rh.id = he.headline_id
    WHERE he.embed_model = :model
      AND rh.date <= :cutoff
    ORDER BY rh.date, he.headline_id
    """
)


# Keyset-paginated variant for the streaming centroid: bounded by ``headline_id`` (the
# leading column of the PK ``(headline_id, embed_model)``), so each page is an index range
# scan with NO global sort, and peak client RAM is one page regardless of corpus size or
# whether the driver honours server-side streaming.
_PAGE_SQL = text(
    """
    SELECT he.headline_id, rh.date::date AS date, he.dim, he.embedding
    FROM headline_embeddings he
    JOIN raw_headlines rh ON rh.id = he.headline_id
    WHERE he.embed_model = :model
      AND rh.date <= :cutoff
      AND he.headline_id > :last_id
    ORDER BY he.headline_id
    LIMIT :page
    """
)


def ensure_table(engine) -> None:
    """Apply the embeddings-cache migration (idempotent CREATE TABLE IF NOT EXISTS)."""
    ddl = _MIGRATION.read_text(encoding="utf-8")
    with engine.begin() as conn:
        for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
            conn.execute(text(stmt))


def _load_embedder():
    """Import + construct the sentence-transformer, or fail fast with an install hint.

    Checked up front (before the expensive unembedded-count query) so a missing
    optional dep doesn't surface as an opaque ModuleNotFoundError mid-run.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The 'embed' extra is not installed (sentence-transformers/torch). "
            "Install it before the embed stage:\n"
            "    uv sync --extra ml --extra embed --extra finance\n"
            "then re-run. (Embeddings need GPU/CPU torch — see docs/RUNBOOK.md.)"
        ) from exc
    return SentenceTransformer(EMBED_MODEL)


def embed_missing(engine=None, *, batch: int = EMBED_BATCH, dry_run: bool = False,
                  scope: str = "precutoff") -> int:
    """Embed headlines lacking a vector for the active model, scoped by date.

    Args:
        scope: 'precutoff' (≤ cutoff, the modeling corpus — default), 'postcutoff'
            (> cutoff — for the buy-overlay / forward use), or 'all'.

    Returns:
        The number of new embeddings written (0 on dry-run).
    """
    if scope not in _DATE_CLAUSES:
        raise ValueError(f"scope must be one of {sorted(_DATE_CLAUSES)}, got {scope!r}")
    engine = engine or get_engine()
    ensure_table(engine)

    # Fail fast on a missing 'embed' extra BEFORE the long count query / dry-run.
    model = None if dry_run else _load_embedder()

    query = text(_UNEMBEDDED_SQL_TMPL.format(date_clause=_DATE_CLAUSES[scope]))
    params = {"model": EMBED_MODEL}
    if scope != "all":
        params["cutoff"] = CUTOFF_DATE
    with engine.connect() as conn:
        todo = pd.read_sql(query, conn, params=params)
    logger.info("{:,} headlines need embedding under {} (scope={}, cutoff {})",
                len(todo), EMBED_MODEL, scope, CUTOFF_DATE.isoformat())
    if dry_run or todo.empty:
        return 0
    written = 0
    for start in range(0, len(todo), batch):
        chunk = todo.iloc[start:start + batch]
        texts = [f"passage: {h}" for h in chunk["headline"].tolist()]
        vecs = model.encode(texts, normalize_embeddings=True,
                            batch_size=batch, show_progress_bar=False).astype(np.float32)
        rows = [
            {"headline_id": int(hid), "embed_model": EMBED_MODEL,
             "dim": int(vecs.shape[1]), "embedding": vecs[i].tobytes()}
            for i, hid in enumerate(chunk["headline_id"].tolist())
        ]
        with engine.begin() as conn:
            conn.execute(_INSERT_SQL, rows)
        written += len(rows)
        logger.info("  embedded {:,}/{:,}", min(start + batch, len(todo)), len(todo))
    logger.info("Done — wrote {:,} embeddings.", written)
    return written


def daily_embedding_centroid(engine=None, cutoff=CUTOFF_DATE, *, page: int = 100_000) -> pd.DataFrame:
    """Per-date e5 centroid (+ dispersion + count), keyset-paginated so RAM stays bounded.

    Accumulates per-date sum and sum-of-squares over fixed-size pages → mean + per-dim std,
    instead of materialising the full ~3M×768 matrix at once (which OOMs on the full-history
    corpus). Pages are keyset-bounded by ``headline_id`` (PK leading column) so each query is
    an index range scan with no global sort and peak client RAM is one page (~``page``×768×4
    bytes), regardless of corpus size or whether the driver honours server-side streaming. The
    sum/sum-of-squares/count accumulation is order-independent, so pages need no date ordering.

    Returns a date-indexed frame: ``embc_000..NNN`` (mean), ``emb_dispersion`` (mean per-dim
    std), ``emb_count``. Empty frame if no embeddings cached.
    """
    engine = engine or get_engine()
    sums: dict = {}
    sqsums: dict = {}
    counts: dict = {}
    dim = None
    last_id = -1
    while True:
        with engine.connect() as conn:
            chunk = pd.read_sql(_PAGE_SQL, conn, params={
                "model": EMBED_MODEL, "cutoff": cutoff, "last_id": last_id, "page": page})
        if chunk.empty:
            break
        if dim is None:
            dim = int(chunk["dim"].iloc[0])
        vecs = np.vstack([np.frombuffer(b, dtype=np.float32) for b in chunk["embedding"]]).reshape(-1, dim)
        dates = pd.to_datetime(chunk["date"]).to_numpy()
        for d in np.unique(dates):
            v = vecs[dates == d].astype(np.float64)
            key = pd.Timestamp(d)
            if key not in sums:
                sums[key] = np.zeros(dim); sqsums[key] = np.zeros(dim); counts[key] = 0
            sums[key] += v.sum(0); sqsums[key] += (v ** 2).sum(0); counts[key] += len(v)
        last_id = int(chunk["headline_id"].iloc[-1])
        if len(chunk) < page:
            break
    if dim is None or not counts:
        return pd.DataFrame()
    order = sorted(sums)
    mean = np.vstack([sums[d] / counts[d] for d in order])
    var = np.clip(np.vstack([sqsums[d] / counts[d] - (sums[d] / counts[d]) ** 2 for d in order]), 0.0, None)
    idx = pd.DatetimeIndex(order)
    out = pd.DataFrame(mean, index=idx, columns=[f"embc_{i:03d}" for i in range(dim)])
    out["emb_dispersion"] = np.sqrt(var).mean(axis=1)
    out["emb_count"] = [counts[d] for d in order]
    logger.info("Daily centroid (keyset-paged, <= {}): {} days × {}-d", pd.Timestamp(cutoff).date(), len(out), dim)
    return out


def load_embeddings(engine=None, cutoff=CUTOFF_DATE) -> tuple[pd.DataFrame, np.ndarray]:
    """Load all cached embeddings ≤ ``cutoff`` for the active model.

    ``cutoff`` defaults to the project cutoff (the modeling corpus). Pass a later date
    (e.g. far-future) to include post-cutoff embeddings — used by the full-date
    visualizations, never by the leak-safe modeling path.

    Returns:
        ``(meta, vectors)`` where ``meta`` has columns [headline_id, date] aligned
        row-for-row with ``vectors`` (float32 matrix, L2-normalised).
    """
    engine = engine or get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(_LOAD_SQL, conn, params={"model": EMBED_MODEL, "cutoff": cutoff})
    if df.empty:
        return df[["headline_id", "date"]], np.empty((0, 0), dtype=np.float32)
    dim = int(df["dim"].iloc[0])
    vectors = np.vstack([np.frombuffer(b, dtype=np.float32) for b in df["embedding"]]).reshape(-1, dim)
    df["date"] = pd.to_datetime(df["date"])
    return df[["headline_id", "date"]], vectors


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 — embed headlines into the cache.")
    parser.add_argument("--batch", type=int, default=EMBED_BATCH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--scope", choices=["precutoff", "postcutoff", "all"], default="precutoff",
                        help="Which headlines to embed by date (default: precutoff = modeling corpus).")
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch must be >= 1")
    embed_missing(batch=args.batch, dry_run=args.dry_run, scope=args.scope)


if __name__ == "__main__":
    main()
