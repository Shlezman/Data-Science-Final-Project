-- SentiSense — Database Schema Initialization
-- ==============================================
-- Auto-executed by Postgres on first container start
-- (mounted via docker-compose as /docker-entrypoint-initdb.d/01_init.sql)

-- ─────────────────────────────────────────────────
-- 1. Raw headlines collected from mivzakim_scraper
-- ─────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS raw_headlines (
    id              BIGSERIAL PRIMARY KEY,
    date            DATE         NOT NULL,
    source          TEXT         NOT NULL,
    hour            TIME,
    popularity      VARCHAR(10),          -- p1, p2 etc.
    headline        TEXT         NOT NULL,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    -- Deduplicate on (date, source, hour, headline) to prevent re-imports
    CONSTRAINT uq_headline UNIQUE (date, source, hour, headline)
);

-- Index for fast lookups by date (daily aggregation queries)
CREATE INDEX IF NOT EXISTS idx_raw_headlines_date ON raw_headlines (date);

-- Index for fast lookups by source
CREATE INDEX IF NOT EXISTS idx_raw_headlines_source ON raw_headlines (source);

-- ─────────────────────────────────────────────────
-- 2. NLP vectors produced by the processing_engine
-- ─────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS nlp_vectors (
    id                      BIGSERIAL PRIMARY KEY,
    headline_id             BIGINT       NOT NULL REFERENCES raw_headlines(id) ON DELETE CASCADE,
    model_name              VARCHAR(100) NOT NULL,
    relevance_politics      SMALLINT     CHECK (relevance_politics      BETWEEN 0 AND 10),
    relevance_economy       SMALLINT     CHECK (relevance_economy       BETWEEN 0 AND 10),
    relevance_security      SMALLINT     CHECK (relevance_security      BETWEEN 0 AND 10),
    relevance_health        SMALLINT     CHECK (relevance_health        BETWEEN 0 AND 10),
    relevance_science       SMALLINT     CHECK (relevance_science       BETWEEN 0 AND 10),
    relevance_technology    SMALLINT     CHECK (relevance_technology    BETWEEN 0 AND 10),
    global_sentiment        SMALLINT     CHECK (global_sentiment        BETWEEN -10 AND 10),
    validation_passed       BOOLEAN      NOT NULL DEFAULT FALSE,
    processing_time_seconds REAL,
    errors                  TEXT[],
    created_at              TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_nlp_headline_model UNIQUE (headline_id, model_name)
);

CREATE INDEX IF NOT EXISTS idx_nlp_vectors_headline ON nlp_vectors (headline_id);

-- ─────────────────────────────────────────────────
-- 3. Daily aggregated features (for LSTM model)
-- ─────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS daily_features (
    date                DATE PRIMARY KEY,
    politics_avg        REAL,
    economy_avg         REAL,
    security_avg        REAL,
    health_avg          REAL,
    science_avg         REAL,
    technology_avg      REAL,
    sentiment_avg       REAL,
    headline_count      INTEGER      NOT NULL DEFAULT 0,
    usd_nis             REAL,
    sp500_close         REAL,
    sp500_change        REAL,
    nasdaq_close        REAL,
    nasdaq_change       REAL,
    ta125_up            BOOLEAN,             -- target variable (NULL = not yet known)
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ─────────────────────────────────────────────────
-- 4. Model predictions (inference log)
-- ─────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS model_predictions (
    id              BIGSERIAL PRIMARY KEY,
    date            DATE         NOT NULL,
    model_version   VARCHAR(100) NOT NULL,
    prediction      BOOLEAN      NOT NULL,   -- TRUE = up, FALSE = down
    confidence      REAL         CHECK (confidence BETWEEN 0.0 AND 1.0),
    actual          BOOLEAN,                 -- filled in after market close
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_prediction_date_model UNIQUE (date, model_version)
);
