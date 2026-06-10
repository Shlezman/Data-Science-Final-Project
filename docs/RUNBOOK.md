# SentiSense Phase 2&3 — Operator Runbook

The implementer has no DB/server access. Every step below is run **by the operator
on the server**, against branch `GENAI-sentisense-phase23`. Paste the named artifact
back at each gate.

## Environment (once)

Config is read from a repo-root **`.env`** (auto-loaded by the `sentisense` package on
import — and the vars propagate to the scoring subprocess). Copy the provided `env`:

```bash
cp env .env          # or: cp .env.example .env  — then edit
uv sync              # provision the sentisense base env at repo root
```

### Local LLM (default — Ollama)
The `.env` already selects local Ollama. `ACTIVE_MODEL_NAME` automatically tracks the
backend, so locally-scored rows (`model_name=qwen2.5:14b`) are exactly the ones every
downstream query reads — no manual override needed.

```ini
# .env (local)
SENTISENSE_DATABASE_URL=postgresql://sentisense:sentisense_dev@localhost:5432/sentisense
SENTISENSE_LLM_BACKEND=ollama
SENTISENSE_OLLAMA_MODEL=qwen2.5:14b
SENTISENSE_OLLAMA_BASE_URL=http://localhost:11434
```
Make sure Ollama is up with the model pulled: `ollama pull qwen2.5:14b`.
**Concurrency:** a single local Ollama serves ~1 request at a time — use
`--concurrency 4` (not 50) for scoring, or it just queues.

### Production LLM (alternative — vLLM mistral-small-4)
Set instead in `.env`: `SENTISENSE_LLM_BACKEND=openai`, `SENTISENSE_OPENAI_MODEL=mistral-small-4`,
`SENTISENSE_OPENAI_BASE_URL=…`, `SENTISENSE_OPENAI_HOST_HEADER=…`,
`SENTISENSE_FORCE_COMPLETIONS_API=true`, `SENTISENSE_COMPLETIONS_MAX_TOKENS=8192`.

### Live ETA
`python -m sentisense.pipeline` prints an up-front per-stage estimate + total ETA
(from live DB counts × rate priors), then a running ETA after each stage. The two long
stages stream their own progress — scoring (`~Ns remaining`) and HPO (`trial k/N … ETA`).
Tune the priors via `SENTISENSE_ETA_SCORE_SECS` / `_EMBED_SECS` / `_HPO_TRIAL_SECS`.

## Tests (no DB needed)

```bash
uv run pytest tests/test_phase0_phase1.py -v
```

## Phase 1 — backfill, score, coverage (→ Gate A)

```bash
# 1.1 (optional) extend history backwards. Dry-run first to see the plan.
uv run python -m sentisense.ingest.backfill --window 7 --dry-run
uv run python -m sentisense.ingest.backfill --window 7            # real run

# 1.2 score ONLY truly-unscored headlines (no validated row from ANY model),
#     HARD-capped at <= 2023-10-07. New rows are written under your local backend
#     model (SENTISENSE_OLLAMA_MODEL). It will NOT re-score your existing
#     mistral-small-4 corpus — only the freshly-backfilled olds.
uv run python -m sentisense.ingest.score --dry-run
uv run python -m sentisense.ingest.score --headlines-per-call 20 --concurrency 4    # local Ollama
# (--rescore-all-under-model forces re-scoring the whole corpus under the local model)
#
# The score stage AUTO-manages Ollama on the ollama backend: it runs
# `ollama serve > ollama_server.log 2>&1` before scoring and `pkill -9 ollama`
# after, freeing the GPU for the embedding / LSTM stages. Pass --no-manage-ollama
# if Ollama is already running or remote.

# 1.3 GATE A artifact — coverage report. Paste the printed report (and the file) back.
uv run python -m sentisense.ingest.coverage_report
#   → writes sentisense_reports/phase1_coverage_report.md
```

### Gate A — paste back `sentisense_reports/phase1_coverage_report.md`
The report's **scored-model breakdown** shows every model_name + its validated count
(e.g. `mistral-small-4` 1.3M, `mistral-small3.2` N). Modeling combines ALL of them
(one score per headline, latest). Implementer verifies: cutoff held, backfill
saturated, validated-% high, distinct news-date count vs the ~750 LSTM bar.

### Your flow (already-scored corpus + local backfill)
```
backfill (scrape olds) → score (fills only the new olds, locally) → embed (ALL
headlines) → cluster → features → baselines → tune (scores + embeddings) → final
```
`uv run python -m sentisense.pipeline` runs it all with the live ETA. Because the
score stage is truly-unscored-only, it won't touch your mistral-small-4 rows.

## Notes
- `python -m sentisense.X` runs from the **repo root** (the root `pyproject.toml`
  defines the `sentisense` package). Scoring internally delegates to
  `uv run --project processing_engine ...` so it uses the scoring env + LLM deps.
- Do NOT run two ingestion jobs in parallel (shared `headlines.csv` temp file).
- A previously *failed* score row is not overwritten by `score` (happy-path
  `ON CONFLICT DO NOTHING`). To re-score failures the operator runs the existing
  `scripts/retry_failed_headlines.py` (documented in the main CLAUDE.md).

## Full pipeline (Phases 4–7) — the "run all features" orchestrator

```bash
uv sync --extra ml --extra embed --extra finance     # heavy deps for modeling
uv run --extra dev --with pytest pytest tests/ -q     # all tests green

# Dry-run the whole chain (prints plan; ingest/embed stages no-op):
uv run python -m sentisense.pipeline --dry-run

# Run individual stages or ranges:
uv run python -m sentisense.pipeline --only embed                 # Phase 4 embed
uv run python -m sentisense.pipeline --only cluster,features,baselines
uv run python -m sentisense.pipeline --from features              # skip ingest+embed

# GATE C — review BEFORE the long HPO run, then launch under tmux/nohup (resumable):
tmux new -s hpo
uv run python -m sentisense.hpo.optuna_lstm --trials 100          # resumes if killed
# Phase 7 sacred-holdout eval of the best trial:
uv run python -m sentisense.pipeline --only final
```

Stage order: `backfill → score → coverage → embed → cluster → features → baselines → tune → final`.

The `tune` stage runs **two** Optuna studies (compared on the sacred holdout in `final`):
  * `sentisense_lstm_scores` — LSTM on the per-source SCORE features.
  * `sentisense_lstm_emb` — LSTM on the daily e5-centroid EMBEDDING features (PCA→`SENTISENSE_EMBED_PCA`, default 50, train-fit). Skipped if no embeddings cached.
Both persist to the project DB (RDBStorage via `SENTISENSE_DATABASE_URL`), so a killed
run resumes on relaunch (`create_study(load_if_exists=True)`).

Migration: `sentisense/db/migrations/001_headline_embeddings.sql` (the embedding cache
table) is applied automatically by the embed stage (`ensure_table`, idempotent).

## Gate sequence (whole project)
- **Gate A** (here): Phase 1 coverage report.
- **Gate B**: Phase 2 schema/feasibility report (trading-day count, class balance).
- **Gate C**: Phase 6 — review the Optuna study + launch command BEFORE the 7-day HPO run.
- **Phase 7**: final out-of-sample metrics from the untouched held-out window.
