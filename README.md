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
 │  sentisense/ (Phase 2&3)   │ ───────────────▶  │  Forecasting + serving      │
 │  features · embed · cluster│                   │  model zoo + Optuna HPO →    │
 │  models · hpo · serve · sim│                   │  registry champion → nightly │
 └────────────────────────────┘                   │  prediction + live dashboard │
                                                  └─────────────────────────────┘
```

Five modules:
**1. Ingestion** ✅ · **2. NLP scoring** ✅ · **3. Feature engineering** ✅ ·
**4. Forecasting (model zoo + HPO + registry)** ✅ · **5. Orchestration / dashboard / DevOps** ✅

## Modules

| Module | Purpose | Entry point |
|--------|---------|-------------|
| [`mivzakim_scraper/`](mivzakim_scraper/) | Scrape Hebrew headlines by date or keyword | `python main.py` |
| [`processing_engine/`](processing_engine/) | Score headlines (6 relevance + sentiment) via LLM | `from processing_engine import process_single_observation` |
| [`scripts/`](scripts/) | Data pipeline: schema, backfill, scoring, retry, standardise | `python scripts/<name>.py` |
| [`sentisense/`](sentisense/) | Phase 2&3 forecasting: features, embeddings, clustering, model zoo, HPO, registry serving, narrative sims | `python -m sentisense.pipeline` |
| [`ui/`](ui/) | Live dashboard — FastAPI backend + React SPA (login-gated, served behind nginx TLS) | `uvicorn ui.app:app` |
| [`evaluation/`](evaluation/) | Benchmark Ollama models against a golden dataset | `python -m evaluation.evaluate` |
| [`ops/`](ops/) | Deployment artifacts: crontab, pm2 config, nginx reverse-proxy config | — |
| [`external/`](external/) | MiroFish narrative-simulation submodule (AGPL — isolated as a separate local-only service) | — |

## Notebooks ([`notebooks/`](notebooks/))

| Notebook | Purpose |
|----------|---------|
| [`eda.ipynb`](notebooks/eda.ipynb) | Exploratory analysis — volume, validation health, score distributions, correlations |
| [`poc.ipynb`](notebooks/poc.ipynb) | Tree-model PoC (XGB/LGBM/CatBoost) for next-day direction + statistical tests |
| [`lstm_forecaster.ipynb`](notebooks/lstm_forecaster.ipynb) | LSTM next-day predictor on the per-source feature shape |
| [`tuning.ipynb`](notebooks/tuning.ipynb) | Long-running Optuna + isotonic-calibration tuning across model classes |
| [`transformer_forecaster.ipynb`](notebooks/transformer_forecaster.ipynb) | Transformer model zoo (vanilla / PatchTST / two-tower / Informer) |
| [`sentisense_analysis.ipynb`](notebooks/sentisense_analysis.ipynb) | Package-driven analysis over the full pipeline outputs |
| [`compare_lstm_features_with_poc.ipynb`](notebooks/compare_lstm_features_with_poc.ipynb) | Feature-shape ablation: per-source LSTM features vs the PoC daily-mean shape |
| [`timesfm_explainability.ipynb`](notebooks/timesfm_explainability.ipynb) | TimesFM foundation-forecaster explainability |
| [`miro_explainability.ipynb`](notebooks/miro_explainability.ipynb) | Narrative-simulation (MiroFish) feature explainability |

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
columns above. See [`docs/DATA_HANDOFF.md`](docs/DATA_HANDOFF.md) for the full data dictionary.)

## Repository structure

```
├── mivzakim_scraper/          # Playwright scraper for mivzakim.net (Hebrew news)
├── processing_engine/         # LLM scoring pipeline (fast single-prompt + 7-agent LangGraph)
├── sentisense/                # Phase 2&3 forecasting package (run via `python -m sentisense.X`)
│   ├── constants.py           #   cutoff, model name, score contract, data paths
│   ├── config.py              #   modeling/HPO knobs (env-overridable)
│   ├── db/                    #   SQLAlchemy engine (env-only DSN) + migrations 001-008
│   ├── ingest/                #   backfill · score · coverage report (Gate A)
│   ├── features/              #   leak-safe daily dataset assembly (+ overnight block)
│   ├── embed/                 #   multilingual-e5 headline embeddings + cache
│   ├── cluster/               #   causal expanding-window narrative clustering
│   ├── models/                #   model zoo: trees · LSTM/GRU/TCN/PatchTST · TFT/N-HiTS/N-BEATS · Chronos/TimesFM
│   ├── hpo/                   #   resumable Optuna HPO + sacred-holdout eval
│   ├── serve/                 #   model registry + champion serving (nightly prediction)
│   ├── sim/                   #   narrative simulations (local-LLM personas, MiroFish client)
│   └── pipeline.py            #   end-to-end orchestrator
├── ui/                        # Live dashboard: FastAPI + React SPA (login gate, websocket)
├── evaluation/                # LLM benchmark harness (golden dataset) + finance CSVs
├── scripts/                   # init_db.sql · backfill · scoring · daily_live · sims · registry training
├── ops/                       # crontab · pm2 config · nginx TLS reverse-proxy config
├── notebooks/                 # eda · poc · lstm · tuning · transformer zoo · explainability
├── external/                  # MiroFish submodule (AGPL, isolated local-only service)
├── tests/                     # pytest — offline: leakage, calendar, serving, promotion gate
├── docs/                      # runbooks · MODEL_ZOO · DATA_HANDOFF · leaderboard · miro/
├── docker-compose.yml         # PostgreSQL 16 + optional pgAdmin
└── pyproject.toml             # sentisense package (base + ml/embed/finance/tft/chronos/ui/... extras)
```

## Documentation

| Doc | Audience |
|-----|----------|
| [`docs/LIVE_RUNBOOK.md`](docs/LIVE_RUNBOOK.md) | Operator reference for the live two-machine deployment |
| [`docs/DATA_HANDOFF.md`](docs/DATA_HANDOFF.md) | Consumer reference for working with the scored dataset |
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | Phase 2&3 server-side run commands + gate sequence |
| [`docs/MODEL_ZOO.md`](docs/MODEL_ZOO.md) | Model grid: families × data types × regimes |
| [`docs/sentisense-understanding.md`](docs/sentisense-understanding.md) | Schema + pipeline ground truth |

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
