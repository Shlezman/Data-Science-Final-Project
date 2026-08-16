-- 009: per-model held-out classification metrics (positive class = "up").
-- Computed by scripts/backfill_oos_classification.py from the SAME sacred
-- test-tail evaluation that produced oos_accuracy/oos_roc_auc, so the
-- dashboard can show precision/recall/F1 with honest eval provenance.
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS oos_precision DOUBLE PRECISION;
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS oos_recall    DOUBLE PRECISION;
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS oos_f1        DOUBLE PRECISION;
