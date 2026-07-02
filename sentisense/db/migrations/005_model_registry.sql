-- Model registry — every trained candidate model, its OOS metrics, and its serialized artifact.
-- The daily/periodic multi-model HPO registers one row per model (+ a soft-vote ensemble row);
-- serving loads the row where is_active. Selection is auto-best with a STICKY manual override
-- (activated_by='manual' rows are not auto-replaced). At most one row is active (partial unique
-- index). Idempotent (IF NOT EXISTS); version is the natural upsert key.
CREATE TABLE IF NOT EXISTS model_registry (
    id              BIGSERIAL     PRIMARY KEY,
    version         VARCHAR(120)  NOT NULL UNIQUE,     -- e.g. xgboost-20260701-0927
    name            VARCHAR(80)   NOT NULL,            -- display name
    model_type      VARCHAR(40)   NOT NULL,            -- xgboost|lgbm|catboost|lstm|gru|tcn|patchtst|ensemble|chronos|timesfm|pf
    datatype        VARCHAR(20)   NOT NULL DEFAULT 'fused',
    regime          VARCHAR(10)   NOT NULL DEFAULT 'FULL',
    overnight       BOOLEAN       NOT NULL DEFAULT TRUE,
    params          JSONB         NOT NULL DEFAULT '{}',
    oos_roc_auc     REAL,
    oos_auc_lo      REAL,
    oos_auc_hi      REAL,
    oos_mcc         REAL,
    oos_accuracy    REAL,
    oos_n           INTEGER,
    artifact        BYTEA,                             -- serialized model bytes; NULL for ensemble
    artifact_format VARCHAR(20)   NOT NULL DEFAULT 'joblib',   -- joblib|torch|ensemble|reforecast (zero-shot: no artifact, live re-forecast)
    members         JSONB,                             -- ensemble: [member version strings]
    feature_cols    JSONB,                             -- exact feature column order the model expects
    trained_rows    INTEGER,
    trained_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    is_active       BOOLEAN       NOT NULL DEFAULT FALSE,
    activated_by    VARCHAR(10),                       -- auto|manual
    activated_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_model_registry_type ON model_registry (model_type);
-- Enforce at most ONE active model at a time.
CREATE UNIQUE INDEX IF NOT EXISTS uq_model_registry_one_active ON model_registry (is_active) WHERE is_active;
