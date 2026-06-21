# pgvector embedding store

A vector DB for the headline embeddings, deployed **inside the existing native Postgres**
(no Docker, no new service) via the `pgvector` extension. `scripts/deploy_vectordb.py`
enables the extension, creates `headline_vectors`, and fills it from the embedding cache.

## What it builds
- Table `headline_vectors(headline_id, embed_model, date, source, headline, embedding vector(dim))`
  — one row per cached embedding, with metadata for filtering.
- `dim` is read from `headline_embeddings.dim` (e.g. 768 for `intfloat/multilingual-e5-base`).
- **HNSW** index with `vector_cosine_ops` (embeddings are L2-normalised → cosine is correct),
  plus a `date` index for time-filtered search.
- Source data: `headline_embeddings` (float32 BYTEA) ⋈ `raw_headlines` (date/source/text).

## One-time: install the extension (no Docker)
The `vector` extension must exist in the PG server. Match YOUR PG major (drop `sudo` if
already root, e.g. in a container):
```bash
PGVER=$(psql "$SENTISENSE_DATABASE_URL" -tAc 'SHOW server_version_num' | cut -c1-2)   # e.g. 14
apt-get update && apt-get install -y postgresql-${PGVER}-pgvector
# if no apt package, build from source:
apt-get install -y build-essential postgresql-server-dev-${PGVER} git
git clone --depth 1 https://github.com/pgvector/pgvector /tmp/pgvector
make -C /tmp/pgvector && make -C /tmp/pgvector install
```
The script then runs `CREATE EXTENSION IF NOT EXISTS vector` itself.

## Deploy + fill
```bash
export SENTISENSE_DATABASE_URL=postgresql://sentisense:...@localhost:5432/sentisense
uv run python scripts/deploy_vectordb.py --dry-run        # how many would load
uv run python scripts/deploy_vectordb.py --batch 2000     # create + fill (idempotent, resumable)
uv run python scripts/deploy_vectordb.py --rebuild        # drop + recreate, then refill
```
Idempotent: keyset-paginated, `ON CONFLICT DO NOTHING` — safe to re-run after more headlines
are embedded (`python -m sentisense.embed.embeddings`).

## Query (demo)
```bash
uv run python scripts/deploy_vectordb.py --query 12345 --k 5   # nearest headlines by cosine
```
Or in SQL — nearest neighbours of a headline, optionally date-filtered:
```sql
SELECT headline_id, date, source, headline,
       embedding <=> (SELECT embedding FROM headline_vectors WHERE headline_id = 12345) AS distance
FROM headline_vectors
WHERE date >= '2024-01-01'
ORDER BY distance ASC
LIMIT 10;
```

## Notes
- No new Python dependency — the script casts a text literal to `vector` (`CAST(:v AS vector)`),
  so only the PG-side extension is required.
- All writes are parameterised; the only interpolated value is the integer `dim` (read from the DB).
