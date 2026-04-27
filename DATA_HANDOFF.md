# SentiSense — Data Handoff Guide

A practical onboarding document for someone joining this project to do
additional processing on top of the existing dataset (feature
engineering, model training, deeper EDA, alternative analyses, etc.).

If you have the local repo and want a 30-second orientation, the rest
of this file is for you.

---

## What's in the project

**SentiSense** scores Hebrew news headlines on six relevance categories
plus a global sentiment, with the eventual goal of forecasting daily
TA-125 (Tel Aviv 125 stock index) movements. Three things have already
been built:

1. **Scraper** (`mivzakim_scraper/`) — pulls Hebrew breaking-news
   headlines from `mivzakim.net` via a headless Firefox.
2. **Processing engine** (`processing_engine/`) — sends each headline
   through an LLM (currently `mistral-small-4` on a vLLM server) and
   produces a 7-dimension score vector.
3. **PostgreSQL database** — stores the scraped headlines and the
   resulting score vectors.

What you'll most likely care about: **the database**. The pipeline has
already produced ~1.9M scored headlines across many years.

---

## The data, in detail

Two tables matter for downstream work. (Schema lives in
`scripts/init_db.sql` if you ever want the canonical source.)

### `raw_headlines`

One row per scraped headline. The "source of truth" — never modified
once written.

| Column | Type | Meaning |
|---|---|---|
| `id` | `BIGSERIAL` | Primary key |
| `date` | `DATE` | Day the headline was published |
| `source` | `VARCHAR(255)` | Hebrew news outlet that published it |
| `hour` | `TIME` | Time of day |
| `popularity` | `VARCHAR(50)` | One of `regular`, `important`, `breaking` |
| `headline` | `TEXT` | The Hebrew headline text |
| `created_at` | `TIMESTAMPTZ` | When the row was inserted into our DB |

Unique constraint on `(date, source, hour, headline)`.

### `nlp_vectors`

One row per `(headline, model)` pair. This is where the LLM scores live.

| Column | Type | Range | Meaning |
|---|---|---|---|
| `id` | `BIGSERIAL` | — | Primary key |
| `headline_id` | `BIGINT` | — | FK to `raw_headlines.id` |
| `model_name` | `VARCHAR(100)` | — | Which LLM scored it (currently `mistral-small-4`) |
| `relevance_politics` | `SMALLINT` | 0–10 | How relevant to politics & government |
| `relevance_economy` | `SMALLINT` | 0–10 | …to economy & finance |
| `relevance_security` | `SMALLINT` | 0–10 | …to security & military |
| `relevance_health` | `SMALLINT` | 0–10 | …to health & medicine |
| `relevance_science` | `SMALLINT` | 0–10 | …to science & climate |
| `relevance_technology` | `SMALLINT` | 0–10 | …to technology |
| `global_sentiment` | `SMALLINT` | -10 to +10 | Tone (-10 very negative, +10 very positive) |
| `validation_passed` | `BOOLEAN` | — | `TRUE` = LLM output parsed cleanly; `FALSE` = pipeline failure |
| `processing_time_seconds` | `REAL` | — | Wall-clock time for the LLM call |
| `errors` | `TEXT[]` | — | Error messages if `validation_passed=FALSE` |
| `created_at` | `TIMESTAMPTZ` | — | When this score was written |

Unique on `(headline_id, model_name)`.

### Two more tables (mostly empty for now)

- `daily_features` — empty placeholder for the per-day feature vector
  the forecaster will eventually consume. Schema is there, no rows yet.
- `model_predictions` — empty placeholder for forecasting outputs.

You'll probably end up populating one or both of these.

---

## Getting set up

### Option A — share the live DB (fastest if you're on the same network)

If you can reach the existing Postgres host, you only need:

```bash
git clone <this-repo-url>
cd Data-Science-Final-Project
cd processing_engine && uv sync --extra notebook   # Python deps + jupyter
```

Then point your tools at:
```
postgresql://sentisense:sentisense_dev@<host>:5432/sentisense
```

Or set it once via env var:
```bash
export SENTISENSE_DATABASE_URL="postgresql://sentisense:sentisense_dev@<host>:5432/sentisense"
```

### Option B — get a dump (recommended if you'll work offline)

Ask whoever runs the live DB to produce one of these. Both are
self-contained — no internet required after that.

**A full Postgres dump** (preserves schema + indexes; ~5GB compressed):
```bash
# Run on the host with the live DB:
PGPASSWORD=sentisense_dev pg_dump \
    -h localhost -U sentisense -d sentisense -F c \
    -f sentisense_$(date +%F).dump

# On your machine, import:
docker compose up -d           # starts an empty PostgreSQL
PGPASSWORD=sentisense_dev pg_restore \
    -h localhost -U sentisense -d sentisense -c \
    sentisense_<date>.dump
```

**Or a CSV/Parquet export** (much smaller, easier to share, works
without Postgres on the receiving end):
```bash
# Run on the host:
PGPASSWORD=sentisense_dev psql -h localhost -U sentisense -d sentisense -c "
\\COPY (
    SELECT rh.id          AS headline_id,
           rh.date,
           rh.source,
           rh.hour,
           rh.popularity,
           rh.headline,
           nv.relevance_politics,
           nv.relevance_economy,
           nv.relevance_security,
           nv.relevance_health,
           nv.relevance_science,
           nv.relevance_technology,
           nv.global_sentiment,
           nv.validation_passed,
           nv.created_at AS scored_at
    FROM raw_headlines rh
    JOIN nlp_vectors nv ON nv.headline_id = rh.id
    WHERE nv.model_name = 'mistral-small-4'
) TO STDOUT WITH CSV HEADER" > sentisense_dataset.csv

# On your machine:
import pandas as pd
df = pd.read_csv('sentisense_dataset.csv', encoding='utf-8', parse_dates=['date', 'scored_at'])
```

For Parquet (1/3 the size, faster to load):
```python
df.to_parquet('sentisense_dataset.parquet', index=False)
```

### Option C — bootstrap from CSVs (last resort)

The repo has the original scraped CSVs. Spin up Postgres locally and
run the migration script:

```bash
docker compose up -d
cd processing_engine
uv run python ../scripts/migrate_csv_to_db.py
```

Then re-score from scratch — but that requires access to the LLM, which
will likely cost you. Ask first.

---

## Loading into Python

Two patterns covering 90% of cases.

### One-off interactive query

```python
import os
import psycopg
import pandas as pd

DB = os.getenv("SENTISENSE_DATABASE_URL",
               "postgresql://sentisense:sentisense_dev@localhost:5432/sentisense")

with psycopg.connect(DB) as conn, conn.cursor() as cur:
    cur.execute("""
        SELECT rh.date, rh.headline, nv.global_sentiment, nv.relevance_economy
        FROM raw_headlines rh
        JOIN nlp_vectors nv ON nv.headline_id = rh.id
        WHERE nv.model_name = 'mistral-small-4'
          AND nv.validation_passed = TRUE
          AND rh.date >= '2024-01-01'
        ORDER BY rh.date
        LIMIT 1000;
    """)
    cols = [c[0] for c in cur.description]
    df = pd.DataFrame(cur.fetchall(), columns=cols)
```

### A ready-made notebook

The repo ships with `eda.ipynb` at the root. It loads the joined
dataset, does data-quality checks, score-distribution plots, time
series, and a daily-aggregation preview. Use it as a starting point:

```bash
cd processing_engine && uv sync --extra notebook
cd .. && uv run --project processing_engine jupyter lab eda.ipynb
```

The notebook's first config cell has two knobs worth knowing:

```python
MODEL_NAME: str | None = None      # None = load every model; "mistral-small-4" = filter to one
SAMPLE_LIMIT: int | None = 200_000 # None = load everything (≈ 1.9M rows)
```

---

## Score scale reference (memorize this)

* **Relevance** (six columns, `relevance_*`): integer 0–10. Higher = more
  relevant to that category. 0 means the headline has nothing to do with
  the category. 10 means it's squarely about that category.
* **Sentiment** (`global_sentiment`): integer **-10 to +10**. Negative =
  bad news / pessimistic tone, positive = good news / optimistic tone, 0
  = neutral or mixed.
* **`validation_passed`**: `TRUE` means the LLM emitted a parseable JSON
  with valid scores in the right ranges. `FALSE` means the pipeline
  failed somewhere (timeout, JSON syntax error, missing field, server
  error). **Always filter to `validation_passed = TRUE` for analysis
  unless you're explicitly investigating failures.**

---

## Known data quirks (read this before you trust the scores)

### 1. All-zero "validated" rows

A non-trivial chunk of `validation_passed=TRUE` rows have **0 across
every column**. The LLM occasionally emits zeros for everything when
it can't categorise a headline, and the validator accepts it because
all values are technically in range. Quick check:

```sql
SELECT COUNT(*) AS all_zero_rows
FROM nlp_vectors
WHERE relevance_politics   = 0 AND relevance_economy   = 0
  AND relevance_security   = 0 AND relevance_health    = 0
  AND relevance_science    = 0 AND relevance_technology= 0
  AND global_sentiment     = 0
  AND validation_passed    = TRUE;
```

Treat these like missing data unless you're specifically modelling LLM
failure modes. The EDA notebook (`eda.ipynb` §3) already does a
per-source breakdown that surfaces them.

### 2. The dataset has been standardised on `mistral-small-4`

Earlier in the project, headlines were scored by `mistral-large-2` and
`mistral-small3.2` as well. Those legacy rows were re-processed under
`mistral-small-4` to make the dataset uniform. So:

- Filter `WHERE model_name = 'mistral-small-4'` for analysis.
- If you see other model names in the data, those are residual rows
  that didn't get standardised — confirm with whoever is operating the
  DB before treating them as canonical.

### 3. Possible orphans

A small number of `raw_headlines` rows might have **no** corresponding
`nlp_vectors` row. Check with:

```sql
SELECT COUNT(*) FROM raw_headlines rh
LEFT JOIN nlp_vectors nv ON nv.headline_id = rh.id
WHERE nv.id IS NULL;
```

Should be 0 in a healthy state. Non-zero usually means a recent
processing run got interrupted; the operator can recover with
`scripts/retry_failed_headlines.py --include-missing`.

### 4. Date range and gaps

Backfill should cover roughly the past few years up to "yesterday".
Saturday volume is typically much lower (Israeli weekend); that's not a
data gap, it's real. Genuine gaps look like *all* days in a stretch
having ~0 headlines.

### 5. Hebrew text and encoding

All `headline` strings are **UTF-8 Hebrew**. If you save to CSV or load
in pandas, make sure to specify `encoding='utf-8'` explicitly —
Windows-default `cp1252` will mangle them silently.

---

## Common queries

Pre-baked SQL for the most likely questions you'll have. Run via
`psql` or paste into the notebook.

### Daily volume

```sql
SELECT date, COUNT(*) AS n_headlines
FROM raw_headlines
GROUP BY date
ORDER BY date;
```

### Daily mean sentiment (post-validation)

```sql
SELECT rh.date,
       AVG(nv.global_sentiment) AS mean_sentiment,
       COUNT(*)                 AS n_scored
FROM raw_headlines rh
JOIN nlp_vectors nv ON nv.headline_id = rh.id
WHERE nv.model_name        = 'mistral-small-4'
  AND nv.validation_passed = TRUE
GROUP BY rh.date
ORDER BY rh.date;
```

### Daily aggregated feature vector (the schema for `daily_features`)

```sql
SELECT
    rh.date,
    COUNT(*)                          AS n_headlines,
    AVG(nv.relevance_politics)        AS mean_politics,
    AVG(nv.relevance_economy)         AS mean_economy,
    AVG(nv.relevance_security)        AS mean_security,
    AVG(nv.relevance_health)          AS mean_health,
    AVG(nv.relevance_science)         AS mean_science,
    AVG(nv.relevance_technology)      AS mean_technology,
    AVG(nv.global_sentiment)          AS mean_sentiment,
    STDDEV(nv.global_sentiment)       AS std_sentiment,
    AVG((nv.global_sentiment < 0)::int)::float AS pct_negative,
    AVG((nv.global_sentiment > 0)::int)::float AS pct_positive
FROM raw_headlines rh
JOIN nlp_vectors nv ON nv.headline_id = rh.id
WHERE nv.model_name        = 'mistral-small-4'
  AND nv.validation_passed = TRUE
GROUP BY rh.date
ORDER BY rh.date;
```

### Top sources by volume

```sql
SELECT source, COUNT(*) AS n_headlines
FROM raw_headlines
GROUP BY source
ORDER BY n_headlines DESC
LIMIT 20;
```

### High-impact headlines (e.g. very negative sentiment + security relevance)

```sql
SELECT rh.date, rh.source, rh.headline,
       nv.global_sentiment, nv.relevance_security
FROM raw_headlines rh
JOIN nlp_vectors nv ON nv.headline_id = rh.id
WHERE nv.model_name        = 'mistral-small-4'
  AND nv.validation_passed = TRUE
  AND nv.global_sentiment   <= -7
  AND nv.relevance_security >= 8
ORDER BY rh.date DESC
LIMIT 50;
```

---

## What scripts exist (and which you'll likely never need)

If you're staying on the analysis side and won't re-process the
pipeline, you can ignore these. They're listed for awareness only.

| Script | What it does |
|---|---|
| `mivzakim_scraper/` | Scraping (don't run unless coordinating) |
| `scripts/daily_scrape_to_db.py` | Daily cron — scrapes today's headlines |
| `scripts/backfill_history.py` | Walks backwards to fill historical dates |
| `scripts/process_headlines.py` | Sends unprocessed headlines through the LLM |
| `scripts/retry_failed_headlines.py` | Re-runs `validation_passed=FALSE` rows |
| `scripts/standardize_to_latest_model.py` | Re-scores legacy rows under the latest model |
| `scripts/migrate_csv_to_db.py` | One-time bulk import (already run) |

Full command examples are in `.claude/CLAUDE.md` if you ever do need
to run them.

---

## What NOT to do (mistakes to avoid)

1. **Don't run any script in `scripts/` against a shared DB without
   asking** — most of them write to `nlp_vectors` and a stray
   `--rescore-legacy` would re-process millions of rows.
2. **Don't `DELETE FROM nlp_vectors` without a fresh dump.** Recovery
   means waiting for hours of LLM time.
3. **Don't trust an all-zero score row** — see "Known data quirks" §1.
4. **Don't filter on `validation_passed` alone** when computing
   per-headline aggregates — also filter on `model_name = 'mistral-small-4'`,
   otherwise you may double-count headlines that were scored under
   multiple models historically.
5. **Don't assume timezone.** All dates / hours are recorded as Asia /
   Jerusalem (Israel time), but `created_at` is `TIMESTAMPTZ` and is
   in UTC at the wire format. Keep that in mind for any time-of-day
   analysis.

---

## Where to look next

* **`eda.ipynb`** — fully wired exploratory notebook covering volume,
  quality, distributions, correlations, time series, and a preview of
  the daily aggregation. Run this first.
* **`.claude/CLAUDE.md`** — operator-level guide for the whole
  project: full architecture, all scripts, environment variables,
  database schema, evaluation harness, and runtime configuration. The
  most authoritative reference if you're going deeper.
* **`scripts/init_db.sql`** — canonical schema definitions.
* **`processing_engine/`** — the LLM scoring pipeline. Read this if
  you want to understand or tune *how* the scores were produced rather
  than just consume them.
* **`evaluation/`** (under `processing_engine/`) — golden dataset and
  metrics that benchmark the scoring quality. Useful if you want to
  validate before trusting the scores.

If anything in the data looks weird, the first place to check is the
`errors` array on `nlp_vectors` for rows where `validation_passed=FALSE`
— it usually contains the LLM's actual failure message and explains
the issue in one line.

Good luck. Treat the dataset as immutable from your side, and you can't
break anything.
