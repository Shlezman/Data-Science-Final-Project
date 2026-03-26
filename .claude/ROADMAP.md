# SentiSense — Build Roadmap

> **How to read this file**
> Each module is broken into discrete tasks. Tasks are intentionally kept at a description level — no code yet. When a module is ready to be implemented, tasks will be expanded into concrete specs and assigned to a branch. Statuses and task details **will change** as the project evolves.

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Done / exists in repo |
| 🔜 | Next up |
| ⏸ | Planned but not started |
| 🔄 | In progress |
| ❓ | Decision still open |

---

## Already Completed

### Module 0 — Ingestion Service (`mivzakim_scraper/`) ✅
- Playwright-based scraper for mivzakim.net (date + keyword modes)
- Anti-detection layer (random UA, viewport, mouse movements)
- Session / cookie persistence
- Output: CSV of `{date, time, source, importance, headline}`

### Module 0b — NLP Processing Engine (`processing_engine/`) ✅
- LangGraph graph: `ingestion → 7 parallel ReAct agents → validation → aggregation`
- 6 category relevancy agents (Politics, Economy, Security, Health, Science, Technology) — score 0–10
- 1 global sentiment agent — score −10 to +10
- LLM backend: Ollama (`qwen2.5:14b`, configurable via env vars)
- Hebrew text utilities per agent (clean, transliterate, urgency signals, keyword scanners)
- Evaluation harness with golden dataset, MAE / Within-N / Pearson metrics

---

## Module 1 — Data Aggregation & Feature Engineering 🔜

> **Goal:** Turn the raw per-headline score vectors into a clean daily time-series dataset ready for model training.

### Tasks

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1.1 | Design DB schema for raw news rows and processed AI vectors | ⏸ | PostgreSQL for structured, MongoDB for raw JSON |
| 1.2 | Build daily aggregator: group by `(date, news_provider)`, compute mean of 7 scores | ⏸ | One row = one trading day |
| 1.3 | Integrate USD/NIS exchange rate (daily close) | ⏸ | Source TBD ❓ (e.g. Yahoo Finance, Bank of Israel API) |
| 1.4 | Integrate S&P 500 daily close + day-over-day change | ⏸ | Source TBD ❓ |
| 1.5 | Integrate NASDAQ daily close + day-over-day change | ⏸ | Source TBD ❓ |
| 1.6 | Compute binary target variable: `ta125_up` = 1 if TA-125 closed higher than previous day, else 0 | ⏸ | Source TBD ❓ (TASE data API) |
| 1.7 | Handle missing trading days (weekends, holidays) and align NLP dates to market calendar | ⏸ | |
| 1.8 | Output: versioned `daily_features.parquet` (or DB table `daily_features`) | ⏸ | |
| 1.9 | Write unit tests for aggregator + feature builder | ⏸ | |

**Expected output schema per trading day:**
```
date | politics_avg | economy_avg | security_avg | health_avg | science_avg | technology_avg
     | sentiment_avg | usd_nis | sp500_close | sp500_change | nasdaq_close | nasdaq_change
     | ta125_up (target)
```

---

## Module 2 — Deep Learning Forecasting Engine ⏸

> **Goal:** A time-series classifier that ingests a sliding window of daily feature vectors and predicts next-day TA-125 direction.

### Tasks

| # | Task | Status | Notes |
|---|------|--------|-------|
| 2.1 | Decide architecture: LSTM vs GRU ❓ | ⏸ | Start with LSTM; GRU as ablation |
| 2.2 | Define input tensor shape: `(batch, window_size, n_features)` | ⏸ | Window = 5–10 days, n_features = 13 |
| 2.3 | Build PyTorch `SentiSenseLSTM` model class (configurable layers, hidden dim, dropout) | ⏸ | |
| 2.4 | Build training loop (AdamW, BCEWithLogitsLoss, LR scheduler, early stopping) | ⏸ | |
| 2.5 | Build evaluation loop (accuracy, precision, recall, F1, ROC-AUC) | ⏸ | |
| 2.6 | Implement inference function: returns `{prediction: 0|1, confidence: float}` | ⏸ | |
| 2.7 | Feature scaling / normalization strategy (fit on train, apply on val/test) | ⏸ | |
| 2.8 | Train/val/test split respecting temporal order (no leakage) | ⏸ | |
| 2.9 | Model checkpointing — save best weights to `models/` | ⏸ | |
| 2.10 | Experiment tracking (MLflow or Weights & Biases) | ⏸ | ❓ Tool to be decided |
| 2.11 | Write tests for model forward pass and inference shape | ⏸ | |

---

## Module 3 — System Orchestration & Databases ⏸

> **Goal:** Fully automated, scheduled daily data flow from scrape → NLP → features → inference.

### Tasks

| # | Task | Status | Notes |
|---|------|--------|-------|
| 3.1 | Design PostgreSQL schema: `raw_headlines`, `nlp_vectors`, `daily_features`, `model_predictions` | ⏸ | |
| 3.2 | Design MongoDB collections: `raw_scrape_jobs` (full page dumps, metadata) | ⏸ | |
| 3.3 | Build DB migration system (Alembic for Postgres) | ⏸ | |
| 3.4 | Decide orchestration tool: Airflow vs Prefect vs K8s CronJobs ❓ | ⏸ | K8s CronJobs for simplicity |
| 3.5 | Define daily DAG / job sequence: `scrape → nlp → aggregate → inference → store` | ⏸ | |
| 3.6 | Dead-letter queue / retry logic for failed NLP observations | ⏸ | |
| 3.7 | Alerting on pipeline failures (email / Slack webhook) | ⏸ | |
| 3.8 | Write integration test that runs the full pipeline end-to-end with mock data | ⏸ | |

---

## Module 4 — Frontend Dashboard ⏸

> **Goal:** A web UI that shows the latest sentiment vectors, financial features, and tomorrow's prediction.

### Tasks

| # | Task | Status | Notes |
|---|------|--------|-------|
| 4.1 | Decide framework: React + FastAPI backend vs Streamlit vs Dash ❓ | ⏸ | |
| 4.2 | Prediction panel: next-day TA-125 direction + confidence gauge | ⏸ | |
| 4.3 | Sentiment trend chart: 7 score time-series (last 30 days) | ⏸ | |
| 4.4 | Financial features panel: USD/NIS, S&P 500, NASDAQ (latest + sparkline) | ⏸ | |
| 4.5 | News feed: today's top headlines with individual scores | ⏸ | |
| 4.6 | Historical prediction accuracy table | ⏸ | |
| 4.7 | Auth / access control (basic, for internal use) | ⏸ | ❓ |
| 4.8 | REST API layer (FastAPI) that the frontend consumes | ⏸ | |

---

## Module 5 — DevOps & Infrastructure as Code ⏸

> **Goal:** Deploy the full system on a Linux server with Kubernetes; secrets managed properly.

### Tasks

| # | Task | Status | Notes |
|---|------|--------|-------|
| 5.1 | Dockerize each service: `mivzakim_scraper`, `processing_engine`, `feature_engineering`, `forecasting_engine`, `dashboard` | ⏸ | One Dockerfile per service |
| 5.2 | K8s `Deployment` manifest for the inference / processing API | ⏸ | |
| 5.3 | K8s `CronJob` manifest for daily scraping | ⏸ | |
| 5.4 | K8s `CronJob` manifest for daily NLP + feature build + inference | ⏸ | |
| 5.5 | K8s `Service` + `Ingress` for dashboard and API | ⏸ | |
| 5.6 | `Secret` manifests (DB credentials, API keys, Ollama URL) — never hardcoded | ⏸ | |
| 5.7 | `ConfigMap` for non-sensitive environment config | ⏸ | |
| 5.8 | Persistent volume claims for model weights and Postgres data | ⏸ | |
| 5.9 | Horizontal Pod Autoscaler for the processing API | ⏸ | |
| 5.10 | CI/CD pipeline (GitHub Actions): lint → test → build → push image → deploy | ⏸ | |
| 5.11 | Monitoring stack: Prometheus + Grafana (or lightweight alternative) | ⏸ | ❓ |

---

## Open Decisions Log

| ID | Question | Options | Decision |
|----|----------|---------|----------|
| D1 | Financial data source (USD/NIS, S&P, NASDAQ, TA-125) | Yahoo Finance / Bank of Israel API / TASE API / Alpha Vantage | ❓ |
| D2 | ML framework for forecasting engine | PyTorch vs TensorFlow/Keras | ❓ |
| D3 | Orchestration tool | K8s CronJobs vs Airflow vs Prefect | ❓ |
| D4 | Dashboard framework | React+FastAPI vs Streamlit vs Dash | ❓ |
| D5 | Experiment tracking | MLflow vs W&B vs none | ❓ |
| D6 | Monitoring stack | Prometheus+Grafana vs Datadog vs lightweight | ❓ |
