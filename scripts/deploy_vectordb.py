"""Deploy a pgvector vector DB in the existing Postgres and fill it from the embedding cache.

No Docker, no new service: enables the ``vector`` extension in the native Postgres the
project already uses, creates ``headline_vectors`` (one row per embedded headline with an
``vector(dim)`` column + HNSW cosine index), and copies every cached embedding from
``headline_embeddings`` (BYTEA float32) joined to ``raw_headlines`` for date/source/text.

Idempotent + resumable (keyset pagination + ON CONFLICT DO NOTHING). Embeddings are
L2-normalised, so cosine distance is the right metric.

Run (server-side; SENTISENSE_DATABASE_URL points at the native PG):
    uv run python scripts/deploy_vectordb.py --dry-run
    uv run python scripts/deploy_vectordb.py --batch 2000
    uv run python scripts/deploy_vectordb.py --query 12345 --k 5     # demo: nearest headlines
"""

from __future__ import annotations

import argparse
import re

import numpy as np
from loguru import logger
from sqlalchemy import text

from sentisense.config import EMBED_MODEL
from sentisense.db import get_engine

_EXT_HINT = (
    "pgvector extension unavailable in this Postgres. Install it for YOUR PG major (drop\n"
    "`sudo` if already root), then re-run (this script issues CREATE EXTENSION vector):\n"
    "  PGVER=$(psql -tAc 'SHOW server_version_num' | cut -c1-2)   # e.g. 14\n"
    "  apt-get update && apt-get install -y postgresql-${PGVER}-pgvector\n"
    "  # if no apt package — build from source:\n"
    "  apt-get install -y build-essential postgresql-server-dev-${PGVER} git\n"
    "  git clone --depth 1 https://github.com/pgvector/pgvector /tmp/pgvector\n"
    "  make -C /tmp/pgvector && make -C /tmp/pgvector install"
)


def _active_dim(engine, model: str) -> int:
    """Embedding dimensionality recorded for ``model`` in the cache (e.g. 768 for e5-base)."""
    with engine.connect() as conn:
        row = conn.execute(text("SELECT dim FROM headline_embeddings WHERE embed_model=:m LIMIT 1"),
                           {"m": model}).first()
    if not row:
        raise SystemExit(f"No embeddings for model {model!r} — run `python -m sentisense.embed.embeddings` first.")
    return int(row.dim)


def ensure_extension(engine) -> None:
    """Enable the pgvector extension. No-op if already present (so a non-superuser run works
    after a superuser has created it once — CREATE EXTENSION needs superuser/owner)."""
    with engine.connect() as conn:
        if conn.execute(text("SELECT 1 FROM pg_extension WHERE extname='vector'")).first():
            return
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except Exception as exc:  # noqa: BLE001 — binary not installed, or not superuser
        raise SystemExit(
            f"{_EXT_HINT}\n  (error: {str(exc)[:160]})\n"
            "  Already built the .so but hit a privilege error? Create it once as the PG\n"
            "  superuser, then re-run this (it will skip the create):\n"
            "    su postgres -c \"psql -d sentisense -c 'CREATE EXTENSION IF NOT EXISTS vector;'\"")


def ensure_table(engine, dim: int, *, rebuild: bool = False) -> None:
    """Create ``headline_vectors`` (vector(dim) + a cheap date index). ``rebuild`` drops first.

    The HNSW index is built AFTER the bulk load (see :func:`build_index`) — inserting millions
    of rows into a live HNSW index is far slower than load-then-index."""
    with engine.begin() as conn:
        if rebuild:
            conn.execute(text("DROP TABLE IF EXISTS headline_vectors"))
        conn.execute(text(
            f"""
            CREATE TABLE IF NOT EXISTS headline_vectors (
                headline_id BIGINT       NOT NULL,
                embed_model VARCHAR(100) NOT NULL,
                date        DATE,
                source      TEXT,
                headline    TEXT,
                embedding   vector({dim}) NOT NULL,
                PRIMARY KEY (headline_id, embed_model)
            )
            """))   # dim is an int read from the DB, not user input — safe to interpolate
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_headline_vectors_date "
                          "ON headline_vectors (date)"))


_MEM_RE = re.compile(r"^\d+(kB|MB|GB)$")


def build_index(engine, *, method: str = "hnsw", mem: str = "512MB") -> None:
    """Build the cosine index after the bulk load (idempotent), single-threaded.

    Parallel index builds allocate ~maintenance_work_mem of POSIX shared memory per worker,
    which overflows a container's small /dev/shm (the DiskFull error). We force
    ``max_parallel_maintenance_workers = 0`` so the build uses regular memory only.
    ``method``: 'hnsw' (best recall, heaviest), 'ivfflat' (lighter/faster build), 'none'."""
    if method == "none":
        logger.info("Index build skipped (--index none) — queries fall back to a brute-force scan.")
        return
    if not _MEM_RE.match(mem):
        raise SystemExit(f"--index-mem {mem!r} invalid (expected like 512MB / 1GB).")
    lists = None
    if method == "ivfflat":
        with engine.connect() as conn:
            n = int(conn.execute(text("SELECT COUNT(*) FROM headline_vectors")).scalar())
        # pgvector heuristic: rows/1000 up to 1M, else sqrt(rows) (fewer lists ⇒ lower build mem).
        lists = int(n ** 0.5) if n > 1_000_000 else min(max(n // 1000, 100), 5000)
        lists = max(lists, 100)
    logger.info("Building {} index (cosine), single-threaded — can take a while on millions of rows…", method)
    with engine.begin() as conn:
        conn.execute(text("SET max_parallel_maintenance_workers = 0"))   # no /dev/shm DSM segment
        conn.execute(text(f"SET maintenance_work_mem = '{mem}'"))        # validated above
        if method == "ivfflat":
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_headline_vectors_ivf "
                              f"ON headline_vectors USING ivfflat (embedding vector_cosine_ops) "
                              f"WITH (lists = {lists})"))
        else:
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_headline_vectors_hnsw "
                              "ON headline_vectors USING hnsw (embedding vector_cosine_ops)"))
    logger.info("{} index ready.", method)


def _vec_literal(blob: bytes, dim: int) -> str:
    """BYTEA float32 → pgvector text literal '[v0,v1,...]' (cast to vector on insert)."""
    v = np.frombuffer(blob, dtype=np.float32)
    if v.shape[0] != dim:
        raise ValueError(f"embedding length {v.shape[0]} != dim {dim}")
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


_PAGE = text(
    """
    SELECT he.headline_id, he.embedding, rh.date::date AS date, rh.source, rh.headline
    FROM headline_embeddings he
    JOIN raw_headlines rh ON rh.id = he.headline_id
    WHERE he.embed_model = :m AND he.headline_id > :last
      AND NOT EXISTS (SELECT 1 FROM headline_vectors hv
                      WHERE hv.headline_id = he.headline_id AND hv.embed_model = :m)
    ORDER BY he.headline_id
    LIMIT :lim
    """)
_INSERT = text(
    """
    INSERT INTO headline_vectors (headline_id, embed_model, date, source, headline, embedding)
    VALUES (:headline_id, :embed_model, :date, :source, :headline, CAST(:embedding AS vector))
    ON CONFLICT (headline_id, embed_model) DO NOTHING
    """)


def _count_source(engine, model: str) -> int:
    """How many cached embeddings exist for ``model`` (upper bound on what a fill would load)."""
    with engine.connect() as conn:
        return int(conn.execute(
            text("SELECT COUNT(*) FROM headline_embeddings WHERE embed_model=:m"), {"m": model}).scalar())


def fill(engine, model: str, dim: int, *, batch: int = 2000) -> int:
    """Copy not-yet-loaded embeddings into headline_vectors (keyset paginated). Returns count."""
    last, written = -1, 0
    while True:
        with engine.connect() as conn:
            rows = conn.execute(_PAGE, {"m": model, "last": last, "lim": batch}).fetchall()
        if not rows:
            break
        last = int(rows[-1].headline_id)
        payload = [{"headline_id": int(r.headline_id), "embed_model": model, "date": r.date,
                    "source": r.source, "headline": r.headline,
                    "embedding": _vec_literal(r.embedding, dim)} for r in rows]
        with engine.begin() as conn:
            conn.execute(_INSERT, payload)
        written += len(rows)
        logger.info("upserted {:,} (through headline_id {})", written, last)
    return written


def similar(engine, headline_id: int, model: str, k: int = 5) -> list[dict]:
    """Top-k nearest headlines to ``headline_id`` by cosine distance (HNSW)."""
    q = text(
        """
        SELECT hv.headline_id, hv.date, hv.source, hv.headline,
               hv.embedding <=> (SELECT embedding FROM headline_vectors
                                 WHERE headline_id = :id AND embed_model = :m) AS distance
        FROM headline_vectors hv
        WHERE hv.embed_model = :m AND hv.headline_id <> :id
        ORDER BY distance ASC
        LIMIT :k
        """)
    with engine.connect() as conn:
        return [dict(r._mapping) for r in conn.execute(q, {"id": headline_id, "m": model, "k": k})]


def main() -> None:
    p = argparse.ArgumentParser(description="Deploy + fill the pgvector embedding store.")
    p.add_argument("--model", default=EMBED_MODEL, help="Embedding model name (default: active EMBED_MODEL).")
    p.add_argument("--batch", type=int, default=2000)
    p.add_argument("--rebuild", action="store_true", help="Drop + recreate headline_vectors first.")
    p.add_argument("--dry-run", action="store_true", help="Report how many would load; write nothing.")
    p.add_argument("--query", type=int, default=0, help="Demo: print k nearest headlines to this headline_id.")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--index", choices=["hnsw", "ivfflat", "none"], default="hnsw",
                   help="ANN index to build after the load (ivfflat is lighter than hnsw).")
    p.add_argument("--index-mem", default="2GB", help="maintenance_work_mem for the index build.")
    p.add_argument("--index-only", action="store_true",
                   help="Skip the fill (data already loaded) — just build the index.")
    args = p.parse_args()

    engine = get_engine()
    dim = _active_dim(engine, args.model)
    logger.info("Embedding model {} — dim {}", args.model, dim)

    if args.query:
        for r in similar(engine, args.query, args.model, args.k):
            logger.info("  {:.4f}  [{}] {} — {}", r["distance"], r["date"], r["source"], r["headline"][:80])
        return

    if args.dry_run:   # read-only: don't touch the (maybe-absent) target table
        logger.info("Would load up to {:,} vectors (cached embeddings for {}).",
                    _count_source(engine, args.model), args.model)
        return

    if args.index_only:
        build_index(engine, method=args.index, mem=args.index_mem)
        return

    ensure_extension(engine)
    ensure_table(engine, dim, rebuild=args.rebuild)
    n = fill(engine, args.model, dim, batch=args.batch)
    logger.info("Loaded {:,} vectors into headline_vectors.", n)
    build_index(engine, method=args.index, mem=args.index_mem)   # after the bulk load


if __name__ == "__main__":
    main()
