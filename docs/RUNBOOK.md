# SentiSense Phase 2&3 — Operator Runbook

The implementer has no DB/server access. Every step below is run **by the operator
on the server**, against branch `GENAI-sentisense-phase23`. Paste the named artifact
back at each gate.

## Environment (once per shell)

```bash
# Required for every DB-touching step. connection.py FAILS FAST if unset (no default DSN).
export SENTISENSE_DATABASE_URL='postgresql+psycopg://sentisense:****@localhost:5432/sentisense'

# Required ONLY for Phase 1.2 scoring (production vLLM mistral-small-4, completions-only):
export SENTISENSE_LLM_BACKEND=openai
export SENTISENSE_OPENAI_BASE_URL=https://10.10.248.21/v1
export SENTISENSE_OPENAI_MODEL=mistral-small-4
export SENTISENSE_OPENAI_HOST_HEADER=mistral-small-4-119b-nvfp4-runai-model-120b.cs.colman.ac.il
export SENTISENSE_FORCE_COMPLETIONS_API=true
export SENTISENSE_COMPLETIONS_MAX_TOKENS=8192   # size for --headlines-per-call; see prior OOM notes

# Provision the sentisense base env (sqlalchemy, psycopg v3, pandas, loguru):
uv sync                       # at repo root
```

## Tests (no DB needed)

```bash
uv run pytest tests/test_phase0_phase1.py -v
```

## Phase 1 — backfill, score, coverage (→ Gate A)

```bash
# 1.1 (optional) extend history backwards. Dry-run first to see the plan.
uv run python -m sentisense.ingest.backfill --window 7 --dry-run
uv run python -m sentisense.ingest.backfill --window 7            # real run

# 1.2 score unscored headlines, HARD-capped at <= 2023-10-07. Dry-run shows the count.
uv run python -m sentisense.ingest.score --dry-run
uv run python -m sentisense.ingest.score --headlines-per-call 20 --concurrency 50   # real run

# 1.3 GATE A artifact — coverage report. Paste the printed report (and the file) back.
uv run python -m sentisense.ingest.coverage_report
#   → writes sentisense_reports/phase1_coverage_report.md
```

### Gate A — paste back `sentisense_reports/phase1_coverage_report.md`
Implementer verifies: cutoff held (latest date ≤ 2023-10-07), backfill saturated
(earliest date as far back as the source allows), scored % high, and the distinct
news-date count (LSTM-viability signal — true TASE trading-day count comes in Phase 2).

## Notes
- `python -m sentisense.X` runs from the **repo root** (the root `pyproject.toml`
  defines the `sentisense` package). Scoring internally delegates to
  `uv run --project processing_engine ...` so it uses the scoring env + LLM deps.
- Do NOT run two ingestion jobs in parallel (shared `headlines.csv` temp file).
- A previously *failed* score row is not overwritten by `score` (happy-path
  `ON CONFLICT DO NOTHING`). To re-score failures the operator runs the existing
  `scripts/retry_failed_headlines.py` (documented in the main CLAUDE.md).

## Gate sequence (whole project)
- **Gate A** (here): Phase 1 coverage report.
- **Gate B**: Phase 2 schema/feasibility report (trading-day count, class balance).
- **Gate C**: Phase 6 — review the Optuna study + launch command BEFORE the 7-day HPO run.
- **Phase 7**: final out-of-sample metrics from the untouched held-out window.
