# SentiSense — Pipeline Understanding (Phase 0 ground truth)

> Authored from a read-only map of `origin/main` @ `0d4727f` (PR #17 merged).
> This is the operator-verified ground truth for the Phase 2 & 3 financial-modeling
> work, since the implementer has no live DB access. Every claim below is sourced
> from a file in the repo at that commit.

---

## 1. Database schema (`scripts/init_db.sql`) — 4 tables, PostgreSQL 16

### `raw_headlines` — scraped Hebrew headlines (source of truth)
| column | type | notes |
|---|---|---|
| `id` | BIGSERIAL PK | join key → `nlp_vectors.headline_id` |
| `date` | DATE NOT NULL | **the only event-date for the news cutoff** |
| `source` | TEXT NOT NULL | Hebrew outlet name; indexed |
| `hour` | TIME (nullable) | part of dedup key |
| `popularity` | VARCHAR(10) | importance CSS class `p1`/`p2`… — **not numeric** |
| `headline` | TEXT NOT NULL | Hebrew, UTF-8 |
| `created_at` | TIMESTAMPTZ | **ingestion time, NOT event date — never use for cutoff/splits** |
| `headline_hash` | TEXT GENERATED `md5(headline)` STORED | backs dedup |

- UNIQUE `(date, source, hour, headline_hash)`; indexes on `date`, `source`.
- Dedup is on `md5(headline)`, not raw text (B-tree index size limit on long Hebrew strings).

### `nlp_vectors` — per-(headline, model) LLM scores
- `id` PK, `headline_id` FK→`raw_headlines(id)` ON DELETE CASCADE, `model_name` VARCHAR(100).
- Score columns (all **SMALLINT, NULLABLE**): `relevance_politics`, `relevance_economy`,
  `relevance_security`, `relevance_health`, `relevance_science`, `relevance_technology`
  (CHECK 0–10), `global_sentiment` (CHECK −10..+10).
- `validation_passed` BOOLEAN DEFAULT FALSE, `processing_time_seconds` REAL, `errors` TEXT[].
- UNIQUE `(headline_id, model_name)`. **No date column** — inherits date only via JOIN.

### `daily_features` — per-day aggregate vector (**EXISTS, EMPTY — no script writes it yet**)
- `date` DATE **PK**.
- News aggregates: `politics_avg`, `economy_avg`, `security_avg`, `health_avg`,
  `science_avg`, `technology_avg`, `sentiment_avg` (REAL), `headline_count` INT.
- Finance/market: `usd_nis`, `sp500_close`, `sp500_change`, `nasdaq_close`, `nasdaq_change`.
- **`ta125_up` BOOLEAN (nullable) = the only target. NULL = not yet labeled.**
- `created_at`, `updated_at` TIMESTAMPTZ.

### `model_predictions` — inference log (**EXISTS, EMPTY**)
- `date`, `model_version`, `prediction` BOOLEAN, `confidence` REAL (0–1), `actual` BOOLEAN.
- UNIQUE `(date, model_version)`.

### ⚠️ Schema facts that constrain Phase 2 & 3
1. **No continuous TA-125 price/return is stored — only the boolean `ta125_up`.**
   Open-gap, intraday-return, magnitude-weighting, or any regression target is
   **impossible from the DB alone**. TA-125 OHLC exists only in the
   `TA 125 Historical Data.csv` (Investing.com format: Price/Open/High/Low/Vol/Change%).
2. **Cutoff column = `raw_headlines.date` (and `daily_features.date`).** `created_at`
   is ingestion time; using it for the ≤ 2023-10-07 cutoff or for train/test splits leaks.
3. **`ta125_up` is nullable** — modeling must `WHERE ta125_up IS NOT NULL` or NULLs
   masquerade as a class.
4. **Score-column naming has 3 conventions** — map explicitly:
   | DB (`nlp_vectors`) | engine result dict | golden CSV |
   |---|---|---|
   | `relevance_politics` | `relevance_category_1` | `politics_government` |
   | `relevance_economy` | `relevance_category_2` | `economy_finance` |
   | `relevance_security` | `relevance_category_3` | `security_military` |
   | `relevance_health` | `relevance_category_4` | `health_medicine` |
   | `relevance_science` | `relevance_category_5` | `science_climate` |
   | `relevance_technology` | `relevance_category_6` | `technology` |
   | `global_sentiment` | `global_sentiment` | — |

---

## 2. Connection contract (what `db/connection.py` must implement)
- Env var: **`SENTISENSE_DATABASE_URL`**. Default DSN used by all scripts:
  `postgresql://sentisense:sentisense_dev@localhost:5432/sentisense`.
- Driver: **psycopg v3** is the only installed driver (`psycopg[binary]>=3.1`).
  **SQLAlchemy is NOT a dependency yet** and must be added. For SQLAlchemy the URL
  must use the **`postgresql+psycopg://`** dialect (plain `postgresql://` defaults to
  psycopg2, which is absent → engine creation fails).
- `processing_engine/config.py` defines **no** DB config — `db/connection.py` is greenfield.
- **Security**: do NOT copy the existing `os.environ.get(..., "<dsn-with-dev-password>")`
  fallback. `connection.py` must read the env var and **fail fast (raise)** if unset.
  Never embed the password in source (org policy + task spec).

---

## 3. LLM scoring pipeline (`processing_engine/`) — reuse, do not reinvent
- Two paths. **Fast** (production): `precompute_tool_evidence` → ONE structured LLM call
  → all 7 scores. Works on `/v1/chat/completions` AND raw `/v1/completions` (vLLM).
  **Multi-agent** (LangGraph, 7 ReAct agents): Ollama-only legacy, **incompatible with
  `SENTISENSE_FORCE_COMPLETIONS_API=true`** (production vLLM `mistral-small-4`).
- Public entry points:
  - Fast: `processing_engine.fast_pipeline.score_headlines_batch(...)` /
    `score_headlines_concurrent(...)` / `score_headline(...)`.
  - Multi-agent: `processing_engine.process_single_observation(obs)` (avoid for backfill).
- Model name: `get_active_model_name()` lives in **`scripts/process_headlines.py`** (not
  config) → returns `OpenAIConfig().model` (`openai` backend) else `OllamaConfig().model`.
  Production = **`mistral-small-4`**.
- Standard result dict (path-agnostic): `relevance_category_1..6`, `global_sentiment`,
  `validation_passed`, `errors`, `processing_time_seconds`.

---

## 4. Data-pipeline scripts (`scripts/*.py`)
| script | purpose | DB |
|---|---|---|
| `backfill_history.py` | scrape raw_headlines **BACKWARDS** from oldest date until exhausted | W raw_headlines |
| `process_headlines.py` | **THE scoring-entry script** — scores unscored rows → nlp_vectors | R/W |
| `daily_scrape_to_db.py` | cron: scrape today+yesterday; **canonical `get_connection`/`scrape_dates`/`insert_rows`** | W |
| `migrate_csv_to_db.py` | one-time bulk CSV import | W |
| `retry_failed_headlines.py` | DELETE+re-score `validation_passed=FALSE` rows | R/W/D |
| `standardize_to_latest_model.py` | re-score every headline lacking a success under latest model | R/W/D |
| `update_data_csv.py` | CSV-only, no DB | — |

- Connection helper: `get_connection(db_url)` → psycopg v3 first, psycopg2 fallback,
  `autocommit=False`. Reuse `process_headlines.get_connection` (most robust).
- Unscored query: `LEFT JOIN nlp_vectors ON headline_id AND model_name=%s WHERE nv.id IS NULL`.
- `ON CONFLICT (headline_id, model_name) DO NOTHING` → a naive re-run will NOT overwrite a
  failed row; re-scoring requires DELETE first (that's why `retry_*`/`standardize_*` exist).
- **`daily_features` and `model_predictions` are written by NO script** — Phase 2 feature
  engineering is their first writer.

---

## 5. Headline source for backfill (`mivzakim_scraper/`)
- Scrapes **https://mivzakim.net** (Hebrew breaking-news), Playwright + headless Firefox.
- **Backward scraping CONFIRMED**: `scrape.get_data(start_date, days, …)` builds dates
  going backward; `main.py` drives 3450 days back to ~2015. `backfill_history.py` is
  purpose-built to extend before any cutoff incl. 2023-10-07. Limit = whatever the site
  still serves; exhaustion = N consecutive empty/all-dup windows (`--empty-streak`, def 2).
- **Phase 1 entry point**: `scripts/backfill_history.py --start-before 2023-10-08` (or auto
  `MIN(date)` lookup). Reuses `daily_scrape_to_db.scrape_dates` + `insert_rows`.
- Brittle hardcoded XPaths → if site markup changes the scraper silently returns empty,
  which the backfill loop misreads as "history exhausted". Verify non-empty before trusting
  an empty-streak stop. TZ-anchored to Asia/Jerusalem. Do not run two ingestion jobs in
  parallel (shared `headlines.csv` temp file).

---

## 6. Existing modeling notebooks (heavy overlap with the Phase 2&3 ask)
- **`transformer_forecaster.ipynb` (NEW, teammate, PR #17)** — already contains a working
  end-to-end pipeline that the Phase 2&3 work largely duplicates:
  - News load SQL (hard-codes `model_name='mistral-small-4' AND validation_passed=TRUE`).
  - Finance loader: `convert_volume`, `to_float`, TA-125/VTA-35 CSV parse,
    `VTA35_INCEPTION='2019-07-17'` NaN-masking, yfinance `^GSPC/^VIX/BZ=F`, Frankfurter USD/ILS.
  - **Trading-calendar rollover**: `trading_days` from TA-125; `np.searchsorted(side='left')`
    rolls Fri/Sat news to the next Sunday; market/FX/VTA forward-filled.
  - **Leak-free features** `add_ta125_features`: lagged log-returns (1–7), 5d/20d roll stats,
    Wilder RSI-14, 20d volume z-score, day-of-week one-hots — all `.shift(≥1)`.
  - **Target** = `(TA125_Price.shift(-1) > TA125_Price)` = genuine next-day **close-to-close** rise.
  - **Cutoff already applied**: `mt = mt[mt.index <= '2023-10-07']`.
  - Chronological 70/15/15 split, `StandardScaler` fit on train only, `SequenceDataset`,
    `train_model`/`evaluate_on_test`/`compute_class_weights`.
  - 5 transformer architectures + tree baselines (XGB/LGBM/CatBoost) + ElasticNet + MajorityClass.
  - **Optuna scaffold wired but DISABLED** (`OPTUNA_TRIALS=50`, run skipped) → no real HPO done yet.
  - `ModelD_MultiResolutionHierarchical` is a **stub** (runs day-level, not true per-headline).
  > As of the merge of PR #15/#17 into `main`, `lstm_forecaster.ipynb` and
  > `tuning.ipynb` also live at the repo root (per-source LSTM + a long-running
  > Optuna/calibration tuning notebook). The `sentisense/` package generalises this
  > same machinery into importable, server-runnable modules — see §7 and `RUNBOOK.md`.
- **`poc.ipynb`** — tree-model proof-of-concept (resolved on `main`; earlier had a
  leaky `StratifiedKFold(shuffle=True)` + `shift(-1)` LastDay features). The
  `sentisense` package deliberately does **not** port those leaky features; it uses
  the chronological, leak-free path from `transformer_forecaster.ipynb`.
- **`lstm_forecaster.ipynb` / `tuning.ipynb`** — per-source LSTM and the Optuna +
  isotonic-calibration tuning notebook (on `main`). The package's `models/` + `hpo/`
  cover the same ground as importable modules with leakage/cutoff hardening.
- **`eda.ipynb`** — exploratory only; no finance, no target, no training.

### Implication for Phase 6 (LSTM HPO)
The feature engineering, calendar, split, training harness, tree/linear baselines, and the
≤2023-10-07 cutoff **already exist** in `transformer_forecaster.ipynb`. The LSTM HPO is **not**
duplicating completed transformer tuning (that HPO was never run), but it **should reuse** that
notebook's machinery (extracted into the package) rather than rebuild it. Adding an LSTM/GRU to
the existing `ARCHITECTURES` registry + re-enabling the disabled Optuna scaffold is the minimal,
non-divergent path.

---

## 7. Open conflicts the prompt's plan must resolve (decisions pending)
1. **Targets**: prompt wants *Open Gap* (overnight) + *Intraday Return* buckets, but the DB
   has only close-to-close `ta125_up`, and TA-125 OHLC lives only in the CSV. Either persist
   OHLC via a migration (enables open/intraday targets) or keep close-to-close (matches the
   existing transformer notebook + schema).
2. **Build strategy**: reuse/extract the transformer notebook's proven machinery into a
   package vs build the prompt's standalone pipeline fresh (duplication + divergence risk).
3. **Package layout**: prompt implies a top-level `sentisense/` package run via
   `python -m sentisense.X`; the existing repo convention is `processing_engine/` + `scripts/`
   run via `cd processing_engine && uv run python ../scripts/X.py`.
4. **Phase 4 embeddings**: prompt's default `all-MiniLM-L6-v2` is English-centric — poor on
   Hebrew. A multilingual/Hebrew model (e.g. `intfloat/multilingual-e5-base`, AlephBERT) is
   strongly preferable. Also needs a NEW embeddings-cache table (migration).

---

## 8. Hard constraints (carried from the task spec)
- Data cutoff: **only `raw_headlines.date` ≤ 2023-10-07** in every ingest/query/feature step.
- No DB access for the implementer — every runnable artifact is a committed script run
  server-side by the operator; verification is paste-back.
- No hardcoded secrets — `SENTISENSE_DATABASE_URL` from env, fail fast if unset.
- No leakage — fit scaler/KMeans/imputer on TRAIN fold only; `TimeSeriesSplit` everywhere;
  sacred held-out window untouched until Phase 7.
- Calendar Sun–Thu; weekend + holiday news rolls forward into the next trading day.
- Reproducibility — fixed seeds; ablation ≥3 seeds (mean±std).
