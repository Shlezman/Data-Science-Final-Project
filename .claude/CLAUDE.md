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

`mivzakim_scraper` is managed with **uv** (lockfile at `mivzakim_scraper/uv.lock`).
`processing_engine` is managed with plain **pip** (`pyproject.toml`).

## Common Commands

### Installation
```bash
# mivzakim_scraper (uv)
cd mivzakim_scraper && uv sync && uv run playwright install firefox

# processing_engine (pip)
cd processing_engine && pip install -e .
```

### Running the pipeline (smoke test)
```bash
cd processing_engine
python -m processing_engine
```

### Evaluation harness
```bash
# Validate golden dataset only (no LLM calls)
python -m evaluation.evaluate \
    --golden evaluation/golden_dataset.csv \
    --dry-run

# Run evaluation against one or more Ollama models
python -m evaluation.evaluate \
    --golden evaluation/golden_dataset.csv \
    --models qwen2.5:14b llama3.1:8b \
    --output evaluation/results/

# Generate leaderboard from saved results
python -m evaluation.report \
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
python scripts/migrate_csv_to_db.py

# Update data.csv with newly scraped headlines
python scripts/update_data_csv.py

# Daily cronjob: scrape today's headlines directly into DB
python scripts/daily_scrape_to_db.py
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

All scripts use loguru (stderr + `logs/` directory) and support `--dry-run`.

## Golden Dataset Schema

Required CSV columns: `headline`, `politics_government`, `economy_finance`, `security_military`, `health_medicine`, `science_climate`, `technology`
