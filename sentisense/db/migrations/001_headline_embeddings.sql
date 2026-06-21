-- Phase 4 — narrative-detection embedding cache.
-- Idempotent (IF NOT EXISTS). Keyed by (headline_id, embed_model) so re-runs skip
-- already-embedded rows and multiple embedding models can coexist.
-- Embeddings stored as raw float32 bytes (BYTEA) → no pgvector dependency required.
CREATE TABLE IF NOT EXISTS headline_embeddings (
    headline_id BIGINT       NOT NULL REFERENCES raw_headlines(id) ON DELETE CASCADE,
    embed_model VARCHAR(100) NOT NULL,
    dim         INTEGER      NOT NULL,
    embedding   BYTEA        NOT NULL,   -- np.float32 .tobytes(), length = dim * 4
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_headline_embeddings PRIMARY KEY (headline_id, embed_model)
);

CREATE INDEX IF NOT EXISTS idx_headline_embeddings_model
    ON headline_embeddings (embed_model);
