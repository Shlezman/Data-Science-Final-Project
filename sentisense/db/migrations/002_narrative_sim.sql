-- MiroFish narrative-simulation cache (modes A/B/C). Idempotent (IF NOT EXISTS).
-- Each row = one causal day-sim (seed = headlines strictly ≤ that trading day). Keyed so
-- re-runs skip done work and multiple seeds / LLMs / lookbacks can coexist.

-- A: numeric per-day feature fed to the forecasters (one row per (date, seed, llm)).
CREATE TABLE IF NOT EXISTS narrative_sim (
    sim_date           DATE         NOT NULL,   -- decision day T (seed ≤ T; question is T+1)
    seed_hash          VARCHAR(64)  NOT NULL,   -- hash of the exact seed → cache key + reproducibility
    llm_model          VARCHAR(100) NOT NULL,   -- e.g. gemma-4
    seed_idx           INTEGER      NOT NULL DEFAULT 0,  -- multi-seed run index (variance → mean±std)
    dir_score          REAL,                    -- agent consensus in [-1, 1] (bull − bear)
    confidence         REAL,                    -- [0, 1]
    disagreement       REAL,                    -- agent-vote entropy / variance
    magnitude          REAL,                    -- predicted move size (optional)
    dominant_narrative TEXT,                     -- short label of the leading narrative
    n_agents           INTEGER,
    n_steps            INTEGER,
    simulation_id      VARCHAR(120),            -- MiroFish simulation_id (link to report/graph)
    created_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_narrative_sim PRIMARY KEY (sim_date, seed_hash, llm_model, seed_idx)
);
CREATE INDEX IF NOT EXISTS idx_narrative_sim_date ON narrative_sim (sim_date);

-- B/C: the agent / knowledge graph for the future UI (nodes + edges per sim).
CREATE TABLE IF NOT EXISTS narrative_sim_graph (
    sim_run_id  VARCHAR(120) NOT NULL,          -- MiroFish simulation_id (or graph_id)
    sim_date    DATE         NOT NULL,
    graph_id    VARCHAR(120),
    nodes       JSONB        NOT NULL,          -- [{id, type, label, attrs}]
    edges       JSONB        NOT NULL,          -- [{src, dst, type, weight}]
    meta        JSONB,                          -- {question, consensus, n_agents, ...}
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_narrative_sim_graph PRIMARY KEY (sim_run_id)
);
CREATE INDEX IF NOT EXISTS idx_narrative_sim_graph_date ON narrative_sim_graph (sim_date);

-- B: the qualitative report (markdown sections) for explainability.
CREATE TABLE IF NOT EXISTS narrative_sim_report (
    sim_run_id  VARCHAR(120) NOT NULL,
    sim_date    DATE         NOT NULL,
    report_id   VARCHAR(120),
    sections    JSONB,                          -- [{filename, section_index, content}]
    report_md   TEXT,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_narrative_sim_report PRIMARY KEY (sim_run_id)
);
CREATE INDEX IF NOT EXISTS idx_narrative_sim_report_date ON narrative_sim_report (sim_date);
