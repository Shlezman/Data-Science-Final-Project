# SentiSense

A Hebrew news headline analysis pipeline that scrapes Israeli news and scores each headline across six topic categories using a multi-agent LLM system.

## System Overview

```
┌─────────────────────┐     CSV      ┌──────────────────────┐
│  mivzakim_scraper   │ ──────────▶  │  processing_engine   │
│                     │              │                       │
│  Playwright scraper │              │  LangGraph multi-     │
│  for mivzakim.net   │              │  agent LLM pipeline   │
│  (Hebrew news)      │              │  (7 parallel agents)  │
└─────────────────────┘              └──────────────────────┘
                                               │
                                               ▼
                                     ┌──────────────────────┐
                                     │  evaluation/         │
                                     │                      │
                                     │  Harness to compare  │
                                     │  Ollama models       │
                                     │  against gold labels │
                                     └──────────────────────┘
```

## Modules

| Module | Purpose | Entry Point |
|--------|---------|-------------|
| [`mivzakim_scraper/`](mivzakim_scraper/) | Scrape Hebrew news headlines by date or keyword | `python main.py` |
| [`processing_engine/`](processing_engine/) | Score headlines with 7 parallel LLM agents | `from processing_engine import process_single_observation` |

## Output Schema

Each headline is scored by the pipeline and produces:

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `relevance_category_1` | int | 0–10 | Politics & Government |
| `relevance_category_2` | int | 0–10 | Economy & Finance |
| `relevance_category_3` | int | 0–10 | Security & Military |
| `relevance_category_4` | int | 0–10 | Health & Medicine |
| `relevance_category_5` | int | 0–10 | Science & Climate |
| `relevance_category_6` | int | 0–10 | Technology |
| `global_sentiment` | int | −10–+10 | Overall tone (negative → positive) |
| `validation_passed` | bool | — | Whether all agent outputs passed validation |
| `errors` | list | — | Any agent-level errors |
| `processing_time_seconds` | float | — | Wall-clock time for the full pipeline |

## Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- [Ollama](https://ollama.com/) running locally with `qwen2.5:14b` pulled

### 1 — Scrape headlines
```bash
cd mivzakim_scraper
uv sync
uv run playwright install firefox
uv run python main.py      # scrapes ~3450 days → headlines.csv
```

### 2 — Run the pipeline
```bash
cd processing_engine
pip install -e .
python -m processing_engine   # smoke test with a sample headline
```

### 3 — Evaluate models
```bash
# Validate the golden dataset (no LLM calls)
python -m processing_engine.evaluation.evaluate \
    --golden processing_engine/evaluation/golden_dataset.csv \
    --dry-run

# Benchmark one or more models
python -m processing_engine.evaluation.evaluate \
    --golden processing_engine/evaluation/golden_dataset.csv \
    --models qwen2.5:14b llama3.1:8b \
    --output processing_engine/evaluation/results/

# Generate leaderboard from saved results
python -m processing_engine.evaluation.report \
    --results processing_engine/evaluation/results/ \
    --output processing_engine/evaluation/results/leaderboard.md
```

## Configuration

All `processing_engine` settings can be overridden with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SENTISENSE_OLLAMA_MODEL` | `qwen2.5:14b` | Ollama model name |
| `SENTISENSE_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `SENTISENSE_OLLAMA_TEMPERATURE` | `0.1` | LLM temperature |
| `SENTISENSE_OLLAMA_NUM_CTX` | `8192` | Context window size |
| `SENTISENSE_OLLAMA_TIMEOUT` | `120` | Request timeout (seconds) |
| `SENTISENSE_AGENT_RECURSION_LIMIT` | `10` | LangGraph recursion limit |

## Repository Structure

```
├── mivzakim_scraper/               # Web scraper module
│   ├── mivzakim_scraper.py         # Scraper class (date-based)
│   ├── mivzakim_search_scraper.py  # SearchScraper class (keyword-based)
│   ├── scrape.py                   # Batch orchestration logic
│   ├── main.py                     # CLI entry point
│   ├── utils.py                    # Session, cookies, anti-detection helpers
│   ├── pyproject.toml              # Package definition
│   └── uv.lock                     # Pinned dependency lockfile
│
└── processing_engine/              # LLM pipeline module
    ├── config.py                   # Centralized config + env vars
    ├── models.py                   # Pydantic data models
    ├── agents.py                   # ReAct agent constructors
    ├── tools.py                    # Hebrew text tools per agent
    ├── prompts.py                  # System prompts + LLM factory
    ├── graph.py                    # LangGraph state graph
    ├── nodes.py                    # Ingestion, validation, aggregation nodes
    ├── engine.py                   # Public async API
    └── evaluation/                 # Evaluation harness
        ├── golden_dataset.csv      # 26 hand-labeled headlines
        ├── evaluate.py             # CLI: run models against golden dataset
        ├── metrics.py              # MAE, Within-N Accuracy, Pearson r
        └── report.py               # Leaderboard generator
```
