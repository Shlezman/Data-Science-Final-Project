-- Add a `mode` tag to the narrative-sim cache so two views can coexist per day:
--   'source' — per-provider voices (one agent per news outlet)
--   'flat'   — whole-day news pooled, provider-agnostic (deduped, no source attribution)
-- The seed text already differs per mode, so seed_hash keeps rows distinct (no PK change);
-- `mode` is an explicit tag used to pivot features into sim_<mode>_* columns. Idempotent.

ALTER TABLE narrative_sim       ADD COLUMN IF NOT EXISTS mode VARCHAR(16) NOT NULL DEFAULT 'source';
ALTER TABLE narrative_sim_graph ADD COLUMN IF NOT EXISTS mode VARCHAR(16) NOT NULL DEFAULT 'source';
ALTER TABLE narrative_sim_report ADD COLUMN IF NOT EXISTS mode VARCHAR(16) NOT NULL DEFAULT 'source';

CREATE INDEX IF NOT EXISTS idx_narrative_sim_date_mode ON narrative_sim (sim_date, mode);
