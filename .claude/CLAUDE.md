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
python -m processing_engine.evaluation.evaluate \
    --golden processing_engine/evaluation/golden_dataset.csv \
    --dry-run

# Run evaluation against one or more Ollama models
python -m processing_engine.evaluation.evaluate \
    --golden processing_engine/evaluation/golden_dataset.csv \
    --models qwen2.5:14b llama3.1:8b \
    --output processing_engine/evaluation/results/

# Generate leaderboard from saved results
python -m processing_engine.evaluation.report \
    --results processing_engine/evaluation/results/ \
    --output processing_engine/evaluation/results/leaderboard.md
```

### Install Ollama models
```bash
bash processing_engine/evaluation/install_llms.sh
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

Located in `processing_engine/evaluation/metrics.py`:
- **MAE** — mean absolute error in score points
- **Within-1 Accuracy** — % of predictions within 1 point of gold (primary ranking metric)
- **Within-2 Accuracy** — looser tolerance
- **Pearson r** — ranking correlation
- **Composite Score** — average Within-1 across 6 categories

## Golden Dataset Schema

Required CSV columns: `headline`, `politics_government`, `economy_finance`, `security_military`, `health_medicine`, `science_climate`, `technology`
