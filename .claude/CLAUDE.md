.# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SentiSense** — A Hebrew news headline analysis pipeline that combines web scraping, multi-agent AI reasoning, structured evaluation, and a manual annotation UI.

## Module Layout

- `mivzakim_scraper/` — Playwright-based scraper for Hebrew news from mivzakim.net
- `processing_engine/` — LangGraph multi-agent pipeline; the core ML component
- `self_ranking_platform/` — Streamlit annotation UI for human labeling

Each module has its own `pyproject.toml` and is managed independently.

## Common Commands

### Installation (each module separately)
```bash
cd processing_engine && pip install -e .
cd mivzakim_scraper && pip install -e .
cd self_ranking_platform && pip install -r requirements.txt
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

# Auto-discover ALL installed Ollama models and benchmark them (recommended)
python -m processing_engine.evaluation.evaluate \
    --all-models \
    --output processing_engine/evaluation/results/

# Run evaluation against an explicit list of models
python -m processing_engine.evaluation.evaluate \
    --golden processing_engine/evaluation/golden_dataset.csv \
    --models qwen2.5:14b llama3.1:8b \
    --output processing_engine/evaluation/results/

# Generate leaderboard from saved results
python -m processing_engine.evaluation.report \
    --results processing_engine/evaluation/results/ \
    --output processing_engine/evaluation/results/leaderboard.md
```

**Model resolution order** (when running `evaluate`):
1. `--all-models` flag → runs `ollama list`, evaluates every installed model
2. `--models <name ...>` → explicit list
3. `SENTISENSE_OLLAMA_MODEL` env var → single model from env
4. Auto-discovery fallback → same as `--all-models`; if Ollama is unreachable, defaults to `qwen2.5:14b`

### Annotation UI
```bash
cd self_ranking_platform
streamlit run ranking_script.py --server.headless true
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

## Architecture: self_ranking_platform

- Streamlit UI for manual annotation of CSV-loaded headlines
- Annotators score 6 relevance categories (0–10) and global sentiment (-10 to +10)
- Supports CSV upload, random sampling without replacement, and export

## Evaluation Metrics

Located in `processing_engine/evaluation/metrics.py`:
- **MAE** — mean absolute error in score points
- **Within-1 Accuracy** — % of predictions within 1 point of gold (primary ranking metric)
- **Within-2 Accuracy** — looser tolerance
- **Pearson r** — ranking correlation
- **Composite Score** — average Within-1 across 6 categories

## Golden Dataset Schema

Required CSV columns: `headline`, `politics_government`, `economy_finance`, `security_military`, `health_medicine`, `science_climate`, `technology`
