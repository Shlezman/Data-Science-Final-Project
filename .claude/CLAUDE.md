# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SentiSense** — An AI-driven predictive pipeline that forecasts the daily rise/fall of the **TA-125** (Tel Aviv Stock Exchange 125 Index) based on real-time Hebrew news sentiment and macroeconomic indicators.

The full pipeline spans five modules (see `.claude/ROADMAP.md` for the complete build agenda):
1. **Ingestion** — scrape Hebrew news headlines ✅
2. **NLP Processing** — multi-agent LLM scoring ✅
3. **Feature Engineering** — daily vectorization + financial data injection 🔜
4. **Forecasting Engine** — LSTM/GRU deep learning model 🔜
5. **Orchestration, Dashboard & DevOps** — scheduling, DB, UI, K8s 🔜

## Module Layout

- `mivzakim_scraper/` — Playwright-based scraper for Hebrew news from mivzakim.net
- `processing_engine/` — LangGraph multi-agent pipeline; the core ML component
- `evaluation/` — Model evaluation harness (golden dataset, metrics, leaderboard)
- `scripts/` — Data pipeline: DB schema, CSV migration, daily scraper-to-DB cronjob
- `docker-compose.yml` — PostgreSQL 16 + optional pgAdmin

Both modules are managed with **uv** (lockfiles at `mivzakim_scraper/uv.lock` and `processing_engine/uv.lock`).

## Common Commands

### Installation
```bash
# mivzakim_scraper
cd mivzakim_scraper && uv sync && uv run playwright install firefox

# processing_engine (includes psycopg for DB scripts)
cd processing_engine && uv sync
```

### Running the pipeline (smoke test)
```bash
cd processing_engine
uv run python -m processing_engine
```

### Evaluation harness
```bash
cd processing_engine

# Validate golden dataset only (no LLM calls)
uv run python -m evaluation.evaluate \
    --golden evaluation/golden_dataset.csv \
    --dry-run

# Run evaluation against one or more Ollama models
uv run python -m evaluation.evaluate \
    --golden evaluation/golden_dataset.csv \
    --models qwen2.5:14b llama3.1:8b \
    --output evaluation/results/

# Generate leaderboard from saved results
uv run python -m evaluation.report \
    --results evaluation/results/ \
    --output evaluation/results/leaderboard.md
```

### Install Ollama models
```bash
bash evaluation/install_llms.sh
```

### Database & Data Pipeline
```bash
# Start PostgreSQL (schema auto-initializes on first start via scripts/init_db.sql)
docker compose up -d

# Optional: start pgAdmin web UI on port 5050
docker compose --profile admin up -d

# One-time: migrate data.csv into PostgreSQL
cd processing_engine && uv run python ../scripts/migrate_csv_to_db.py

# Update data.csv with newly scraped headlines
cd processing_engine && uv run python ../scripts/update_data_csv.py

# Daily cronjob: scrape today's headlines directly into DB
cd processing_engine && uv run python ../scripts/daily_scrape_to_db.py

# Backfill history: scrape backwards from the oldest stored date until
# the site returns no more data (stops after 2 consecutive empty windows)
cd processing_engine && uv run python ../scripts/backfill_history.py

# Backfill with custom window / cap
cd processing_engine && uv run python ../scripts/backfill_history.py \
    --window 30 --max-days 365

# Retry headlines that previously failed NLP processing
# (those with validation_passed=FALSE in nlp_vectors for the active model).
# Deletes the stale failure rows, then re-runs through the same pipeline.
cd processing_engine && uv run python ../scripts/retry_failed_headlines.py \
    --fast --headlines-per-call 50 --concurrency 50

# Retry including never-processed headlines (superset of unprocessed + failed)
cd processing_engine && uv run python ../scripts/retry_failed_headlines.py \
    --fast --headlines-per-call 50 --include-missing

# Dry run: show how many would be retried without touching the DB
cd processing_engine && uv run python ../scripts/retry_failed_headlines.py --dry-run

# Standardise the dataset on a single 'latest' model: any headline that
# was only scored by an older model (e.g. mistral-large-2) or that
# failed under the latest model gets re-processed under the active
# model.  Legacy non-latest rows are deleted unless --keep-old-rows.
cd processing_engine && uv run python ../scripts/standardize_to_latest_model.py \
    --fast --headlines-per-call 50 --concurrency 50

# Same but preserve the multi-model history (still re-scores under latest)
cd processing_engine && uv run python ../scripts/standardize_to_latest_model.py \
    --fast --headlines-per-call 50 --keep-old-rows

# Pin the latest model name explicitly
cd processing_engine && uv run python ../scripts/standardize_to_latest_model.py \
    --latest-model mistral-small-4 --fast --headlines-per-call 50

# Force-rescore EVERY headline that any non-latest model touched, even
# if a successful latest-model row already exists.  Existing latest
# rows for those headlines are deleted before re-scoring — use this
# after rolling out a new model when you want a uniform re-scoring of
# every legacy headline regardless of whether the latest already covered it.
cd processing_engine && uv run python ../scripts/standardize_to_latest_model.py \
    --fast --headlines-per-call 50 --concurrency 50 --rescore-legacy
```

### Exploratory data analysis (`eda.ipynb`)
```bash
# One-time: install jupyter + pandas/matplotlib/seaborn into processing_engine's venv.
# These are an optional extra so the production processing image stays slim.
cd processing_engine && uv sync --extra notebook

# Launch the notebook from the repo root (relative DB URL works there).
cd .. && uv run --project processing_engine jupyter lab eda.ipynb
```

The notebook joins `raw_headlines` ⨝ `nlp_vectors` for a configurable `MODEL_NAME` and explores volume, validation health, score distributions, correlations, temporal patterns, and a preview of the daily aggregation that will feed the forecaster. Set `SAMPLE_LIMIT = None` in the first code cell to load the full dataset (≈ 1.9M rows; needs a few GB of RAM).

### Full pipeline init (Ubuntu)
```bash
chmod +x scripts/init_pipeline.sh
./scripts/init_pipeline.sh
```

## Architecture: processing_engine

This is the most complex component. Understanding it requires reading several files together.

**LangGraph graph topology:**
```
START → ingestion → [7 parallel agent nodes] → validation → aggregation → END
```

**7 parallel ReAct agents** (each built with `langgraph.prebuilt.create_react_agent`):
- 6 relevancy agents — one per category (Politics, Economy, Security, Health, Science, Technology)
- 1 sentiment agent — global tone scoring

**Scoring contract:**
- Relevancy per category: integer 0–10
- Global sentiment: integer -10 to +10

**Output per headline (7 score columns + metadata):**
```
relevance_category_1..6, global_sentiment, validation_passed, errors, processing_time_seconds
```

**LLM backend:** Local Ollama (default model `qwen2.5:14b`, temperature 0.1). Configurable via `SENTISENSE_*` environment variables (base_url, model, temperature, timeout, context window, recursion limit, retry settings).

**Production (vLLM mistral-small-4, completions-only):**
```bash
export SENTISENSE_LLM_BACKEND=openai
export SENTISENSE_OPENAI_BASE_URL=https://10.10.248.21/v1
export SENTISENSE_OPENAI_MODEL=mistral-small-4
export SENTISENSE_OPENAI_HOST_HEADER=mistral-small-4-119b-nvfp4-runai-model-120b.cs.colman.ac.il
export SENTISENSE_FORCE_COMPLETIONS_API=true
export SENTISENSE_COMPLETIONS_MAX_TOKENS=32768   # safe for --headlines-per-call up to ~100
```
With these set, `get_active_model_name()` returns `mistral-small-4` and both `process_headlines.py` and `retry_failed_headlines.py` target the correct `nlp_vectors.model_name` rows.

**Tools per agent:** Each agent has shared Hebrew text utilities (`clean_hebrew_text`, `transliterate_hebrew`, `detect_urgency_signals`, etc.) plus category-specific keyword scanners.

**Entry point for programmatic use:**
```python
from processing_engine import process_single_observation
result = await process_single_observation(observation_dict)
```

## Architecture: mivzakim_scraper

- Two scraper classes: `Scraper` (by date) and `SearchScraper` (by keyword)
- Uses Playwright (headless Firefox) with session/cookie persistence
- Anti-detection: random user agents, viewports, mouse movements
- Extracts: date, time, source, importance level, headline text

## Evaluation Metrics

Located in `evaluation/metrics.py`:
- **MAE** — mean absolute error in score points
- **Within-1 Accuracy** — % of predictions within 1 point of gold (primary ranking metric)
- **Within-2 Accuracy** — looser tolerance
- **Pearson r** — ranking correlation
- **Composite Score** — average Within-1 across 6 categories

## Database Schema

PostgreSQL tables defined in `scripts/init_db.sql`:

- **`raw_headlines`** — scraped headlines (date, source, hour, popularity, headline). Unique constraint on `(date, source, hour, headline)`.
- **`nlp_vectors`** — per-headline LLM scores (6 relevance + 1 sentiment). FK to `raw_headlines`.
- **`daily_features`** — aggregated daily feature vector for model training. PK on date.
- **`model_predictions`** — inference log (prediction, confidence, actual outcome).

Connection: `SENTISENSE_DATABASE_URL` env var (default: `postgresql://sentisense:sentisense_dev@localhost:5432/sentisense`).

## Data Pipeline Scripts

Located in `scripts/`:
- **`update_data_csv.py`** — scrapes new dates, merges into `data.csv`, renames legacy columns, logs summary
- **`migrate_csv_to_db.py`** — one-time bulk import of `data.csv` → `raw_headlines` table (idempotent)
- **`daily_scrape_to_db.py`** — cronjob: scrape today + yesterday → insert directly to DB
- **`backfill_history.py`** — scrapes backwards from the oldest stored date until the site returns no more data
- **`process_headlines.py`** — runs unprocessed `raw_headlines` through the LLM pipeline, writing to `nlp_vectors`
- **`retry_failed_headlines.py`** — re-runs headlines whose previous `nlp_vectors` row was marked `validation_passed=FALSE` (DELETE + re-INSERT, since `ON CONFLICT DO NOTHING` would otherwise skip them)
- **`standardize_to_latest_model.py`** — broader version of retry: re-scores every headline that lacks a successful row under the latest model (whether it was scored by an older model or failed under the latest). Optionally deletes legacy non-latest rows so the dataset ends up uniform on one model

All scripts use loguru (stderr + `logs/` directory) and support `--dry-run`.

## Golden Dataset Schema

Required CSV columns: `headline`, `politics_government`, `economy_finance`, `security_military`, `health_medicine`, `science_climate`, `technology`
