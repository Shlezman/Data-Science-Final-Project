-- Derived embedding features — leak-safe PCA + cluster-distance block built from the daily
-- e5 centroid. One JSONB row per (date, embed_model): {embpca_000.., embclus_dist_0..}.
-- The transform basis (scaler→PCA→KMeans) is fit ONCE on dates <= fit_cutoff (a train window
-- preceding every out-of-sample tail) and applied to all dates, so the features for a later
-- OOS window never see their own basis. fit_cutoff records that leakage boundary in-band.
-- Idempotent (IF NOT EXISTS), keyed by (date, embed_model) so a rebuild upserts in place.
CREATE TABLE IF NOT EXISTS daily_embedding_derived (
    date        DATE          NOT NULL,
    embed_model VARCHAR(100)  NOT NULL,
    features    JSONB         NOT NULL,   -- {embpca_000:.., embclus_dist_0:.., ...}
    n_pca       INTEGER       NOT NULL,
    n_clusters  INTEGER       NOT NULL,
    fit_cutoff  DATE          NOT NULL,   -- leakage boundary: basis fit on dates <= this
    created_at  TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_daily_embedding_derived PRIMARY KEY (date, embed_model)
);

CREATE INDEX IF NOT EXISTS idx_daily_embedding_derived_model
    ON daily_embedding_derived (embed_model);
