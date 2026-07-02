-- Persisted embedding-derived transform basis (StandardScaler → PCA) so the UI can project
-- per-headline 768-d embeddings into the SAME 16-d embpca space the dataset features use
-- (per-day centroid drawer "day view"). Written by scripts/build_embedding_derived.py at fit
-- time. One row per embed_model (upserted in place). Vectors are raw float32 bytes (BYTEA),
-- matching headline_embeddings.embedding, so the light UI box needs only numpy to project.
CREATE TABLE IF NOT EXISTS embedding_pca_basis (
    embed_model    VARCHAR(100) NOT NULL,
    n_features     INTEGER      NOT NULL,   -- input dim (768)
    n_pca          INTEGER      NOT NULL,   -- output dim (16)
    fit_cutoff     DATE         NOT NULL,   -- leakage boundary the basis was fit on
    scaler_mean    BYTEA        NOT NULL,   -- float32[n_features]
    scaler_scale   BYTEA        NOT NULL,   -- float32[n_features]
    pca_mean       BYTEA        NOT NULL,   -- float32[n_features] (post-scaler PCA mean)
    pca_components BYTEA        NOT NULL,   -- float32[n_pca * n_features], row-major
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_embedding_pca_basis PRIMARY KEY (embed_model)
);
