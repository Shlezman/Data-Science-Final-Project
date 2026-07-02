-- Full-history in-sample evaluation of the served champion: one row per labeled trading day,
-- the champion (fit on ALL labeled days) predicting that same day. Powers the dashboard's
-- "all days" confusion matrix + the 3D centroid colouring. Computed by scripts/compute_full_eval.py
-- (needs the ml extra) so the light UI box only READS this table. Idempotent; upserts in place.
CREATE TABLE IF NOT EXISTS champion_full_eval (
    model_version VARCHAR(100) NOT NULL,
    date          DATE         NOT NULL,
    prediction    BOOLEAN      NOT NULL,   -- predicted UP (proba > 0.5)
    proba         REAL         NOT NULL,   -- predicted up-probability
    actual        BOOLEAN      NOT NULL,   -- realised next-day direction (fused Target)
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_champion_full_eval PRIMARY KEY (model_version, date)
);

CREATE INDEX IF NOT EXISTS idx_champion_full_eval_date ON champion_full_eval (date);
