# SentiSense

An end-to-end pipeline that scrapes Hebrew news, scores every headline across six
topic categories + global sentiment with an LLM, and forecasts the **next-day
direction of the TA-125** (Tel Aviv 125 index) from those signals plus market data.

## Pipeline at a glance

```
 ┌──────────────────┐   headlines   ┌────────────────────┐   7 scores    ┌──────────────────┐
 │ mivzakim_scraper │ ───────────▶  │  processing_engine │ ───────────▶  │   PostgreSQL     │
 │  Playwright /    │               │  LLM scoring       │   per headline│  raw_headlines   │
 │  mivzakim.net    │               │  (fast or 7-agent) │               │  nlp_vectors     │
 └──────────────────┘               └────────────────────┘               └────────┬─────────┘
                                                                                   │
        ┌──────────────────────────────────────────────────────────────────────── ┘
        ▼
 ┌────────────────────────────┐     features      ┌─────────────────────────────┐
 │  sentisense/ (Phase 2&3)   │ ───────────────▶  │  Forecasting                │
 │  features · embed · cluster│                   │  trees / LSTM / GRU +        │
 │  baselines · LSTM HPO      │                   │  Optuna HPO → next-day TA-125│
 └────────────────────────────┘                   └─────────────────────────────┘
```

Five modules (see [`.claude/ROADMAP.md`](.claude/ROADMAP.md) for the full agenda):
**1. Ingestion** ✅ · **2. NLP scoring** ✅ · **3. Feature engineering** ✅ ·
**4. Forecasting (trees + LSTM/GRU + HPO)** ✅ · **5. Orchestration / dashboard / DevOps** 🔜

## Modules

| Module | Purpose | Entry point |
|--------|---------|-------------|
| [`mivzakim_scraper/`](mivzakim_scraper/) | Scrape Hebrew headlines by date or keyword | `python main.py` |
| [`processing_engine/`](processing_engine/) | Score headlines (6 relevance + sentiment) via LLM | `from processing_engine import process_single_observation` |
| [`scripts/`](scripts/) | Data pipeline: schema, backfill, scoring, retry, standardise | `python scripts/<name>.py` |
| [`sentisense/`](sentisense/) | Phase 2&3 forecasting: features, embeddings, clustering, baselines, LSTM HPO | `python -m sentisense.pipeline` |
| [`evaluation/`](evaluation/) | Benchmark Ollama models against a golden dataset | `python -m evaluation.evaluate` |

## Notebooks (repo root)

| Notebook | Purpose |
|----------|---------|
| [`eda.ipynb`](eda.ipynb) | Exploratory analysis — volume, validation health, score distributions, correlations |
| [`poc.ipynb`](poc.ipynb) | Tree-model PoC (XGB/LGBM/CatBoost) for next-day direction + statistical tests |
| [`lstm_forecaster.ipynb`](lstm_forecaster.ipynb) | LSTM next-day predictor on the per-source feature shape |
| [`tuning.ipynb`](tuning.ipynb) | Long-running Optuna + isotonic-calibration tuning across model classes |
| [`transformer_forecaster.ipynb`](transformer_forecaster.ipynb) | Transformer model zoo (vanilla / PatchTST / two-tower / Informer) |

## Quick start

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker (PostgreSQL),
and an LLM backend (local [Ollama](https://ollama.com/) `qwen2.5:14b`, or a vLLM /
OpenAI-compatible endpoint for `mistral-small-4`).

```bash
# 0 — database (schema auto-initialises from scripts/init_db.sql)
docker compose up -d

# 1 — scrape headlines
cd mivzakim_scraper && uv sync && uv run playwright install firefox && uv run python main.py

# 2 — score unscored headlines into nlp_vectors
cd processing_engine && uv sync
uv run python ../scripts/process_headlines.py --fast --headlines-per-call 50 --concurrency 50

# 3 — forecast (Phase 2&3) — run the full chain or individual stages
uv sync --extra ml --extra embed --extra finance        # at repo root
uv run python -m sentisense.pipeline --dry-run           # preview the stage plan
uv run python -m sentisense.pipeline --from features     # features → baselines → tune → final
```

The forecasting pipeline enforces a hard **`<= 2023-10-07` cutoff** (regime break) and is
leakage-safe end to end. Full operator runbook + gate sequence:
[`docs/RUNBOOK.md`](docs/RUNBOOK.md).

## Output schema (`nlp_vectors`)

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `relevance_politics` | int | 0–10 | Politics & Government |
| `relevance_economy` | int | 0–10 | Economy & Finance |
| `relevance_security` | int | 0–10 | Security & Military |
| `relevance_health` | int | 0–10 | Health & Medicine |
| `relevance_science` | int | 0–10 | Science & Climate |
| `relevance_technology` | int | 0–10 | Technology |
| `global_sentiment` | int | −10–+10 | Overall tone (negative → positive) |
| `validation_passed` | bool | — | Whether the LLM output passed validation |

(The scoring pipeline emits these as `relevance_category_1..6`; the DB stores the named
columns above. See [`DATA_HANDOFF.md`](DATA_HANDOFF.md) for the full data dictionary.)

## Repository structure

```
├── mivzakim_scraper/          # Playwright scraper for mivzakim.net (Hebrew news)
├── processing_engine/         # LLM scoring pipeline (fast single-prompt + 7-agent LangGraph)
├── sentisense/                # Phase 2&3 forecasting package (run via `python -m sentisense.X`)
│   ├── constants.py           #   cutoff, model name, score contract
│   ├── config.py              #   modeling/HPO knobs (env-overridable)
│   ├── db/                    #   SQLAlchemy engine (env-only DSN) + migrations
│   ├── ingest/                #   backfill · score · coverage report (Gate A)
│   ├── features/              #   leak-safe daily dataset assembly
│   ├── embed/                 #   multilingual-e5 headline embeddings + cache
│   ├── cluster/               #   causal expanding-window narrative clustering
│   ├── models/                #   sequence datasets, train harness, LSTM/GRU, baselines
│   ├── hpo/                   #   resumable Optuna LSTM HPO + sacred-holdout eval
│   └── pipeline.py            #   end-to-end orchestrator
├── evaluation/                # Model benchmark harness (golden dataset, metrics, leaderboard)
├── scripts/                   # init_db.sql · backfill · process/retry/standardize · daily cron
├── tests/                     # pytest — cutoff, leakage, calendar rollover, connection
├── docs/                      # RUNBOOK.md · sentisense-understanding.md
├── *.ipynb                    # eda · poc · lstm_forecaster · tuning · transformer_forecaster
├── docker-compose.yml         # PostgreSQL 16 + optional pgAdmin
├── DATA_HANDOFF.md            # consumer-level data dictionary
└── pyproject.toml             # sentisense package (base + ml/embed/finance/dev extras)
```

## Documentation

| Doc | Audience |
|-----|----------|
| [`.claude/CLAUDE.md`](.claude/CLAUDE.md) | Operator reference for running the whole pipeline |
| [`DATA_HANDOFF.md`](DATA_HANDOFF.md) | Consumer reference for working with the scored dataset |
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | Phase 2&3 server-side run commands + gate sequence |
| [`docs/sentisense-understanding.md`](docs/sentisense-understanding.md) | Schema + pipeline ground truth |
| [`.claude/ROADMAP.md`](.claude/ROADMAP.md) | Full five-module build agenda |

## Configuration

`processing_engine` and `sentisense` read `SENTISENSE_*` environment variables. Key ones:

| Variable | Default | Description |
|----------|---------|-------------|
| `SENTISENSE_DATABASE_URL` | `postgresql://sentisense:…@localhost:5432/sentisense` | DB connection (required by `sentisense`; no embedded default) |
| `SENTISENSE_LLM_BACKEND` | `ollama` | `ollama` or `openai` (vLLM / OpenAI-compatible) |
| `SENTISENSE_OPENAI_MODEL` | `mistral-large-2` | Production model (set to `mistral-small-4`) |
| `SENTISENSE_OPTUNA_TRIALS` | `100` | Optuna trials per HPO run |
| `SENTISENSE_EMBED_MODEL` | `intfloat/multilingual-e5-base` | Hebrew-aware embedding model |

A full list lives in [`processing_engine/config.py`](processing_engine/config.py) and
[`sentisense/config.py`](sentisense/config.py).
