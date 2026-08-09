-- LLM request queue — the DB is the transport between the UI host and the GPU box
-- (the firewall only passes Postgres between them, so no direct HTTP to Ollama).
-- The UI inserts a row; scripts/llm_worker.py on the GPU box polls, calls the local
-- Ollama model, and writes the answer back. Idempotent.
CREATE TABLE IF NOT EXISTS llm_requests (
    id          BIGSERIAL    PRIMARY KEY,
    kind        VARCHAR(20)  NOT NULL DEFAULT 'ask',     -- narrate | ask
    date        DATE,                                    -- day the question is about
    question    TEXT,                                    -- user question (NULL for narrate)
    status      VARCHAR(10)  NOT NULL DEFAULT 'pending', -- pending | done | error
    answer      TEXT,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    answered_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_llm_requests_status ON llm_requests (status, id);
