# SentiSense — Forecasting the Next-Day Direction of the TA-125 Index from Hebrew-News Sentiment

by
[Your Name]

Approved by the supervisor: Dr. Eliav Menachi

Submitted to the Computer Science Faculty of College of Management
Rishon LeZion
[Month, Year]

---

## Acknowledgments

We would like to express our gratitude to our supervisor, Dr. Eliav Menachi,
for his guidance throughout this project. We would also like to thank our
families for their support, and the open-source community whose tools
(PyTorch, scikit-learn, Optuna, pytorch-forecasting, and the Hugging Face
ecosystem) made this work possible.

---

## Executive Summary

SentiSense is an end-to-end data-science system that asks a focused question:
**can the daily flow of Hebrew breaking-news, distilled by a large language
model (LLM) into a structured sentiment signal, help predict whether the
Tel-Aviv 125 (TA-125) stock index will close higher the next trading day?**

The system spans five stages. (1) A **scraper** collects Hebrew breaking-news
headlines from `mivzakim.net` going back to ~2015. (2) A **processing engine**
sends every headline through an LLM, which scores it on six relevance
categories (politics, economy, security, health, science, technology) and one
global sentiment value (−10…+10), producing a corpus of roughly **3 million
scored headlines** in PostgreSQL. (3) A **feature-engineering layer**
aggregates the per-headline scores into leakage-safe daily feature vectors,
joined with market data (TA-125 OHLC, the VTA-35 volatility index, S&P 500,
VIX, Brent crude, USD/ILS), with multilingual headline **embeddings**, a
leak-safe **PCA/clustering block** derived from the daily embedding centroid,
and causal **narrative-clustering** features. (4) A **forecasting layer**
trains and hyperparameter-tunes a large model zoo — gradient-boosted trees,
recurrent and convolutional sequence classifiers, transformer forecasters, and
zero-shot foundation models — and persists every candidate, with its
out-of-sample metrics and serialized weights, into a **model registry** that
automatically activates the best model (with a manual override). (5) An
**operations layer** runs the whole chain as a nightly job on a GPU node —
scrape → score → embed → derive → predict → settle — and serves the results
through a **live web dashboard** (prediction hero, model metrics, exploratory
analytics, a 3-D news-centroid explorer, per-source "persona" votes, and a
narrative simulator).

Every research stage is engineered to be **leakage-safe**: a hard data cutoff
of `2023-10-07` (the regime break preceding a major market shock) is enforced
in SQL and re-applied after feature assembly; all scalers, PCA, and clustering
are fit on the training fold only; and splits are strictly chronological.

The central empirical finding is sobering and honest, and it is corroborated
across **multiple independent experiment tracks** — a tree-model proof of
concept, an extensive feature-set comparison with walk-forward and multi-seed
robustness checks, a nine-model transformer zoo, sequence-model HPO, a
hardened end-to-end package run of **40+ tuned model × data-type × regime
cells**, and finally the productionized registry run over the full zoo. In
every track, out-of-sample performance hovers close to the no-skill baseline:
on the unified grid the best model by ROC-AUC reaches only **0.576** and the
best by accuracy **0.592**; no model meets the transformer track's
pre-registered ≥58% success criterion or clears a permutation/binomial
significance test. The project's contribution is therefore threefold: a
**reusable, reproducible, leakage-hardened research pipeline** for news-driven
financial forecasting; a rigorous, negative-leaning result that quantifies how
little next-day directional signal the LLM-scored Hebrew-news stream carries
on its own; and a **complete production system** — registry, nightly
orchestration, and dashboard — that keeps measuring that signal honestly on
live data, one settled trading day at a time.

---

## Table of Contents

1. Introduction
   - 1.1 Background
   - 1.2 Problem Statement
   - 1.3 Objectives
   - 1.4 Scope and Limitations
   - 1.5 Methodology
   - 1.6 Organization of the Project Book
2. Literature Review
   - 2.1 Overview of Relevant Literature
3. System Design and Implementation
   - 3.1 System Architecture
   - 3.2 Data Collection and Preprocessing
   - 3.3 Feature Engineering
   - 3.4 Modeling, Hyper-Parameter Optimization, and the Model Registry
   - 3.5 Live Operation: Orchestration, Serving, and the Dashboard
   - 3.6 Implementation Details
   - 3.7 Evaluation Metrics
4. Results and Analysis
   - 4.1 Experimental Setup
   - 4.2 Presentation of Results
     - 4.2.1 Tree-model proof-of-concept (`poc.ipynb`) — daily-mean
     - 4.2.2 LSTM feature-set vs PoC study (`compare_lstm_features_with_poc.ipynb`) — per-source
     - 4.2.3 LSTM base forecaster (`lstm_forecaster.ipynb`) — per-source
     - 4.2.4 Transformer model zoo + ablations (`transformer_forecaster.ipynb`)
     - 4.2.5 Sequence-model tuning & robustness (`tuning.ipynb`)
     - 4.2.6 Hardened-package analysis (`sentisense_analysis.ipynb`)
     - 4.2.7 Foundation-model explainability (`timesfm_explainability.ipynb`)
     - 4.2.8 Unified out-of-sample grid (`leaderboard.md`)
     - 4.2.9 Production registry run and the live champion
   - 4.3 Data Analysis and Interpretation
   - 4.4 Comparison with Existing Approaches
   - 4.5 Discussion of Findings
5. Conclusion and Future Work
6. References
7. Appendix A — Data Dictionary, Schema, and Commands
8. Appendix B — Live Deployment Runbook (summary)

---

## List of Figures

- Figure 1: SentiSense end-to-end pipeline (§1.6)
- Figure 2: System architecture — modules and data flow (§3.1) *(placeholder)*
- Figure 3: Two-host deployment topology (§3.5) *(placeholder)*
- Figure 4: Leakage-safe chronological split and cutoff (§3.3) *(placeholder)*
- Figure 5: Model registry lifecycle — train → register → select → serve (§3.4) *(placeholder)*
- Figure 6: Dashboard — prediction hero and model-performance panel (§3.5) *(screenshot placeholder)*
- Figure 7: Dashboard — exploratory data-analysis panels (§3.5) *(screenshot placeholder)*
- Figure 8: 3-D daily news centroids, colored by KMeans cluster (§3.5) *(screenshot placeholder)*
- Figure 9: Single-day headline cloud in the shared PCA space (§3.5) *(screenshot placeholder)*
- Figure 10: Per-source persona votes vs the model's call (§3.5) *(screenshot placeholder)*
- Figure 11: Unified leaderboard — ROC-AUC vs accuracy scatter (§4.2.8) *(placeholder)*
- Figure 12: Models panel — registry leaderboard with the active champion (§4.2.9) *(screenshot placeholder)*

## List of Tables

- Table 1: Best result per experiment track vs its baseline (§4.2)
- Table 2: PoC tree-model 5-fold cross-validation accuracy (§4.2.1)
- Table 3: PoC chronological 80/20 holdout accuracy (§4.2.1)
- Table 4: PoC significance tests vs majority-class baseline (§4.2.1)
- Table 5: Per-source feature-set holdout comparison (tree models) (§4.2.2)
- Table 6: Feature-group ablation (CatBoost) (§4.2.2)
- Table 7: LSTM base forecaster holdout classification report (§4.2.3)
- Table 8: Transformer zoo final leaderboard vs baselines (§4.2.4)
- Table 9: Tuning notebook — threshold-optimized validation (Youden's J) (§4.2.5)
- Table 10: Hardened-package score-LSTM final holdout (§4.2.6)
- Table 11: Unified out-of-sample leaderboard (§4.2.8)
- Table 12: Registry validation run — tree zoo OOS metrics (§4.2.9)
- Table 13: Active production champion — held-out evaluation (§4.2.9)

---

## Table of Abbreviations

| Abbreviation | Meaning |
|---|---|
| TA-125 | Tel-Aviv 125 stock index |
| VTA-35 | Tel-Aviv 35 Volatility Index |
| LLM | Large Language Model |
| NLP | Natural Language Processing |
| HPO | Hyper-Parameter Optimization |
| OOS | Out-Of-Sample |
| ROC-AUC | Area Under the Receiver Operating Characteristic Curve |
| F1 | F1 score (harmonic mean of precision and recall) |
| MCC | Matthews Correlation Coefficient |
| CI | Confidence Interval |
| LSTM | Long Short-Term Memory network |
| GRU | Gated Recurrent Unit |
| TCN | Temporal Convolutional Network |
| TFT | Temporal Fusion Transformer |
| PCA | Principal Component Analysis |
| OHLC | Open / High / Low / Close (price data) |
| FX | Foreign Exchange rate |
| DSN | Database Source Name (connection string) |
| API | Application Programming Interface |
| SPA | Single-Page Application (browser front-end) |
| UI | User Interface |

---

## 1. Introduction

This chapter provides the background, defines the problem, and states the
objectives and scope of the project, setting the stage for the chapters that
follow.

### 1.1 Background

Financial markets are widely believed to react to news. The *efficient-market
hypothesis* argues that prices already incorporate available information,
implying that consistently predicting short-term direction from public news is
hard. Yet a large body of work in *behavioral finance* and *NLP-for-finance*
reports that the **tone and topical mix of news** carry measurable, if small,
predictive signal — particularly for indices and over short horizons.

Most of this literature focuses on English-language sources (e.g., financial
newswires, Twitter/X, earnings calls). Hebrew-language news, and the Israeli
market specifically, are comparatively under-studied. At the same time, modern
LLMs have made it practical to convert unstructured, multi-source headline
streams into clean, structured signals at scale — something that previously
required hand-built lexicons or supervised classifiers per language.

This project sits at that intersection: it uses an LLM to turn a high-volume
Hebrew breaking-news feed into a structured daily sentiment signal, asks
whether that signal helps forecast the TA-125 — and then goes one step
further, operating the resulting model as a **live, self-updating forecasting
service** whose ongoing accuracy is measured against each newly settled
trading day.

### 1.2 Problem Statement

**Can a structured, LLM-derived sentiment signal extracted from Hebrew
breaking-news headlines predict the next-day close-to-close direction of the
TA-125 index, beyond what market data alone provides — and how much signal, if
any, is actually there?**

The challenge has several specific difficulties:

1. **Signal-to-noise.** Daily index direction is notoriously close to random;
   even small, genuine edges are easy to fabricate through data leakage.
2. **Leakage risk.** News, market, and target series share a calendar; naive
   feature engineering (e.g., using a same-day future return, fitting a scaler
   on the full series, or shuffling time) silently inflates results.
3. **Language and source heterogeneity.** Headlines are Hebrew, UTF-8, from
   many outlets of varying quality and volume, including a real weekend lull.
4. **Regime change.** The market environment around `2023-10-07` is a sharp
   structural break that would contaminate any model trained across it.
5. **Research-to-production gap.** A result that only exists in a notebook is
   not falsifiable going forward; keeping the claim honest requires serving
   the model daily and settling its predictions against reality.

### 1.3 Objectives

1. **Build a reproducible ingestion-and-scoring pipeline** that scrapes Hebrew
   headlines and scores each on six relevance categories plus a global
   sentiment, persisting the result in a relational database.
2. **Engineer leakage-safe daily features** combining the news scores with
   market and macro data, with embedding-derived and narrative-based signals.
3. **Train and rigorously hyperparameter-tune a broad model zoo** for next-day
   TA-125 direction, on a strictly chronological, cutoff-bounded split.
4. **Quantify the predictive value honestly** using threshold-free and
   threshold-based metrics, against a Buy&Hold / majority baseline.
5. **Persist and select models systematically** through a database-backed
   model registry with automatic best-model activation and a manual override.
6. **Operate the system live**: a nightly orchestrated pipeline that scores
   the day's news, produces a prediction with the active champion, settles
   yesterday's call against the realized close, and presents everything in an
   interactive dashboard.
7. **Produce reusable artifacts** (a Python package, scripts, notebooks, an
   auto-generated leaderboard, and a deployment runbook) so both the
   experiment and the service are fully reproducible.

### 1.4 Scope and Limitations

**In scope:** Hebrew-headline scraping; LLM scoring into a 7-dimensional
vector; daily feature engineering including embeddings, an embedding-derived
PCA/cluster block, and causal narrative clustering; classification and
forecasting models with HPO; a comparison leaderboard; a model registry with
automatic/manual champion selection; nightly live operation on a two-host
deployment; and a web dashboard.

**Out of scope / limitations:**

- **Target.** The system predicts **close-to-close direction**, not
  overnight-gap or intraday-return magnitude.
- **Cutoff (research track).** All *research* modeling is bounded to
  `≤ 2023-10-07`; the post-cutoff regime is used only as a read-only sanity
  overlay. The *live* track, by design, serves on the full timeline (the
  `FULL` regime) — its performance is reported separately and accumulates
  prospectively, which is the strongest possible guard against hindsight bias.
- **Intraday.** The system is daily-resolution; no tick or minute data.
- **Causality.** The work measures *predictive association*, not economic
  causation.
- **Data quirks.** A non-trivial fraction of "validated" LLM rows are
  all-zero (a known LLM failure mode treated as missing); the corpus mixes
  LLM scoring-model versions across disjoint date ranges (see §3.2, "scoring
  eras").

### 1.5 Methodology

The project follows a staged, gate-driven methodology:

1. **Ingest** Hebrew headlines (backward scrape to ~2015) into `raw_headlines`.
2. **Score** each headline with an LLM into `nlp_vectors` (7 scores +
   validation flag).
3. **Assemble** leakage-safe daily frames: daily-mean scores, per-source score
   pivots, sentiment×relevance interactions, multilingual embedding centroids,
   an embedding-derived PCA/cluster feature block, causal narrative-cluster
   features, and a finance/market block.
4. **Split** chronologically (≈70/15/15) with all transforms fit on train only.
5. **Model & tune** a zoo of classifiers and forecasters with Optuna HPO.
6. **Evaluate** every model on the same sacred out-of-sample window using
   ROC-AUC, F1, accuracy, balanced accuracy, and MCC, plus a backtest overlay.
7. **Compare** all cells in a single auto-generated leaderboard.
8. **Register & select**: persist each tuned model (weights + OOS metrics)
   into the model registry; activate the best automatically, allow manual
   override from the dashboard.
9. **Operate**: run the nightly pipeline (scrape → score → embed → derive →
   predict → settle) on a schedule, serve the active champion's prediction,
   and fold each settled day into the champion's cumulative live score.

### 1.6 Organization of the Project Book

- **Chapter 2** reviews relevant literature on news-driven financial
  prediction and LLM-based sentiment extraction.
- **Chapter 3** details the system architecture, data collection and
  preprocessing, feature engineering, the model registry, the live serving
  layer and dashboard, implementation, and evaluation metrics.
- **Chapter 4** presents the experimental setup, the full results across all
  research tracks and the production registry run, and an analysis and
  discussion of the findings.
- **Chapter 5** concludes and proposes future work.
- **Chapter 6** lists references; **Appendix A** gives the data dictionary,
  schema, and reproduction commands; **Appendix B** summarizes the live
  deployment runbook.

```
 ┌──────────────────┐  headlines  ┌────────────────────┐  7 scores  ┌──────────────┐
 │ mivzakim_scraper │ ──────────▶ │  processing_engine │ ─────────▶ │  PostgreSQL  │
 │  Playwright/FF   │             │  LLM scoring       │ /headline  │ raw_headlines│
 │  mivzakim.net    │             │  (fast / 7-agent)  │            │ nlp_vectors  │
 └──────────────────┘             └────────────────────┘            └──────┬───────┘
        ┌──────────────────────────────────────────────────────────────────┘
        ▼
 ┌────────────────────────────┐  features  ┌──────────────────────────────┐
 │  sentisense/ (features)    │ ─────────▶ │  Model zoo + Optuna HPO      │
 │ scores·embed·PCA·cluster   │            │ trees/LSTM/GRU/TCN/PatchTST/ │
 │ leakage-safe splits        │            │ TFT/N-HiTS/Chronos/TimesFM   │
 └────────────────────────────┘            └──────────────┬───────────────┘
                                                          │ weights + OOS metrics
                                                          ▼
 ┌────────────────────────────┐   serve    ┌──────────────────────────────┐
 │  Live dashboard (FastAPI + │ ◀───────── │  Model registry (Postgres)   │
 │  React SPA): hero, metrics,│  active    │  auto-best + manual override │
 │  EDA, 3-D centroids, sim   │  champion  │  daily predict + settle      │
 └────────────────────────────┘            └──────────────────────────────┘
```
*Figure 1: SentiSense end-to-end pipeline — research stages (top) feeding the
production loop (bottom).*

---

## 2. Literature Review

### 2.1 Overview of Relevant Literature

The project draws on three strands of prior work.

**News sentiment and market prediction.** A long line of research links the
tone of financial and general news to subsequent market movements. Tetlock [1]
showed that media pessimism predicts downward pressure on prices and
reversion, establishing news tone as a market-relevant variable. Bollen et
al. [2] famously linked aggregate mood derived from social media to movements
in the Dow Jones. The consistent theme is that *signal exists but is small and
regime-dependent*, and that careful, leakage-free evaluation is essential —
exactly the posture this project adopts.

**Lexicon vs. model-based sentiment.** Domain-specific lexicons such as
Loughran and McDonald [3] demonstrated that general-purpose sentiment
dictionaries mislabel financial text, motivating domain-aware scoring. Modern
LLMs generalize this idea: instead of a fixed lexicon, a prompted model
performs context-aware topical-relevance and sentiment scoring, and — relevant
here — does so across languages, including Hebrew, without a hand-built Hebrew
lexicon.

**Sequence and foundation models for forecasting.** On the modeling side, the
project surveys the standard time-series toolkit: gradient-boosted trees
(XGBoost [4]) as strong tabular baselines; recurrent networks (LSTM [5], GRU
[6]) and temporal convolutions (TCN [7]) for sequence classification;
transformer forecasters such as the Temporal Fusion Transformer [8] and
PatchTST [9]; deep interpretable forecasters N-BEATS [10] and N-HiTS [11]; and
zero-shot foundation forecasters Chronos [12] and TimesFM [13]. Multilingual
sentence embeddings (multilingual-E5 [14]) provide the Hebrew-aware vector
representations used for the embedding and narrative-clustering features, and
Optuna [15] provides the resumable hyper-parameter search used throughout.

The research gap this project addresses: most prior work is English-centric,
often under-controls for leakage, and rarely closes the loop from a research
claim to a *prospectively evaluated* live system. SentiSense contributes a
**Hebrew-news, LLM-scored, strictly leakage-controlled** evaluation across a
broad model zoo, reports the result honestly rather than selectively — and
then keeps the evaluation running in production, where each new trading day
extends the out-of-sample record.

---

## 3. System Design and Implementation

### 3.1 System Architecture

The system is organized as loosely-coupled modules communicating through a
PostgreSQL 16 database, so each stage can be developed, re-run, and verified
independently.

| Module | Purpose | Entry point |
|---|---|---|
| `mivzakim_scraper/` | Scrape Hebrew headlines (Playwright + Firefox) | `python main.py` |
| `processing_engine/` | LLM scoring (6 relevance + sentiment) | fast pipeline / `process_single_observation` |
| `scripts/` | Data ops: schema, backfill, scoring, retry, standardize, registry training, daily orchestration | `python scripts/<name>.py` |
| `sentisense/` | Features, embeddings, clustering, models, HPO, serving | `python -m sentisense.pipeline` |
| `sentisense/serve/` | Model registry + champion serving | `registry.py` / `champion.py` |
| `ui/` | FastAPI backend + React SPA dashboard | `python -m ui.app` |
| `evaluation/` | Benchmark LLM scoring against a golden dataset | `python -m evaluation.evaluate` |

> **[Figure 2 placeholder: block diagram of the modules above with the
> database at the center; arrows labeled with the table each stage reads or
> writes.]**

**Design principles.**

- **Database as the contract.** All inter-stage data flows through Postgres
  tables (`raw_headlines`, `nlp_vectors`, `headline_embeddings`,
  `daily_embedding_derived`, `embedding_pca_basis`, `model_registry`,
  `model_predictions`, `champion_full_eval`, `narrative_sim*`), decoupling
  scraping, scoring, modeling, serving, and the UI. The dashboard host never
  runs heavy compute — it only reads the database.
- **Single source of truth for constants.** The cutoff date, active model
  name, and score-column contract live in `sentisense/constants.py`, so no
  magic strings leak into feature or model code.
- **Optional, layered dependencies.** Heavy ML/embedding/forecasting libraries
  are `pyproject.toml` *extras* (`ml`, `embed`, `finance`, `tft`, `chronos`,
  `ui`), so early stages install lightly and torch/CUDA wheels are pinned for
  reproducibility.
- **Leakage-safety as an architectural invariant**, enforced at every layer
  (see §3.3).
- **Fail-safe serving.** Every serving path is wrapped so that a missing
  table, an incompatible artifact, or an unreachable auxiliary service
  degrades to a well-defined fallback (pinned champion, cached data, explicit
  "no data" states) — never a broken nightly run or a blank dashboard.

### 3.2 Data Collection and Preprocessing

**Collection.** The scraper drives a headless Firefox via Playwright over
`mivzakim.net`, scraping *backward* in time (`scripts/backfill_history.py`)
from the most recent day toward ~2015, and *forward* daily
(`scripts/daily_scrape_to_db.py`, covering today and yesterday). Each headline
yields a row in `raw_headlines`: date, source outlet, hour, popularity class,
the Hebrew text, and an ingestion timestamp. Deduplication uses a stored
`md5(headline)` hash (Hebrew strings exceed B-tree index limits) under a
unique key of `(date, source, hour, headline_hash)`.

**Scoring.** The processing engine sends each headline to an LLM. A **fast
single-prompt path** produces all seven scores in one structured call; a
legacy **seven-agent LangGraph path** (one ReAct agent per relevance category
plus one for sentiment) exists for research and evaluation. Each result is a
vector of six relevance integers (0–10), one global sentiment integer
(−10…+10), and a `validation_passed` flag, written to `nlp_vectors`. The
corpus contains **~3 million scored headlines**.

**Scoring eras.** The corpus was scored in two eras, both recorded explicitly
in the `model_name` column so no row's provenance is ambiguous:

- **Historical era** — the bulk backfill was scored by `mistral-small-4`
  served on a remote vLLM cluster (50-headline batched completions at high
  concurrency), after earlier rows from older models were re-standardized
  onto it (`scripts/standardize_to_latest_model.py`).
- **Live era** — nightly scoring runs on a **locally hosted Ollama model
  (`gemma4`)** on the GPU node, one headline per structured call. Batch-JSON
  and agentic modes were empirically found unreliable for this model (§4.2.9),
  so the orchestrator selects the scoring flags per backend automatically.
  New headlines are scored **gap-only** (`--unscored-any-model`): a headline
  already covered by any model is never re-scored, so the eras remain
  disjoint by construction.

The dataset builders consume *validated rows from any era*, and every
era-sensitive UI query prefers the active model's row but falls back to any
validated row — so the system remains correct across the era boundary. The
statistical implication of the seam (features scored by different LLMs on
different date ranges) is discussed honestly in §4.5 and §5.

**Quality control and known quirks** (documented in `DATA_HANDOFF.md`):

- **All-zero "validated" rows.** The LLM sometimes emits all-zeros when it
  cannot categorize a headline; the validator accepts it because all values
  are in range. These are treated as missing data.
- **Weekend lull.** Saturday volume is genuinely low (Israeli weekend), not a
  data gap.
- **Encoding / timezone.** All text is UTF-8 Hebrew; event dates/hours are
  Asia/Jerusalem while `created_at` is stored as UTC `TIMESTAMPTZ`.
- **LLM scoring quality gate.** An evaluation harness scores any candidate
  scoring LLM against a hand-labeled golden dataset (MAE, within-1 accuracy,
  Pearson r per category) before it is allowed to write production rows.

### 3.3 Feature Engineering

Leakage-safe feature assembly (`sentisense/features/dataset.py`) is the heart
of the preprocessing and the project's most important engineering
contribution. The module builds daily modeling frames with defense-in-depth
against leakage:

- **Hard cutoff** `≤ 2023-10-07` (research regime) is pushed into the SQL
  (`WHERE rh.date <= :cutoff`) *and* re-applied after the calendar merge.
- **Event date, never ingestion time.** The cutoff and all splits use
  `raw_headlines.date`, never `created_at`.
- **Trading-calendar rollover.** Weekend/holiday news is rolled *forward* to
  the next trading day (Fri/Sat → Sun) via `np.searchsorted(side='left')`;
  market/FX/volatility series are forward-filled.
- **Causal price features.** TA-125 features (lagged log-returns 1–7, 5d/20d
  rolling stats, Wilder RSI-14, 20-day volume z-score, day-of-week one-hots)
  all use `.shift(>=1)`. Cross-asset features (S&P 500, VIX, Brent, USD/ILS,
  VTA-35) are lagged log-returns only.
- **Train-only scaling.** `StandardScaler` (and PCA, scoped by column prefix
  to the embedding block) is fit on the **train slice only**.
- **Honest target.** `Target = (TA125_Price.shift(-1) > TA125_Price)`; the
  trailing row with no next-day price is set to NA and dropped in research
  mode. In *serving* mode (`keep_unlabeled=True`) that same row is retained
  with a `Target = −1` sentinel — the model trains only on real labels and
  **forward-predicts** the sentinel day, so no fabricated label ever enters
  training.
- **Live price extension.** The static TA-125 CSV is extended at build time
  with live closes fetched from the exchange feed, so the serving frame always
  reaches the current trading day.

> **[Figure 4 placeholder: timeline diagram — train/validation/test split, the
> 2023-10-07 cutoff, and the live serving region with the Target=−1 sentinel
> day.]**

**Embedding-derived block.** Each headline is embedded once with a
Hebrew-aware multilingual model (`intfloat/multilingual-e5-base`, 768-d) and
cached in `headline_embeddings`. Per trading day, the mean of the day's
headline vectors forms the **daily news centroid**. From the centroids, a
leak-safe transform basis (StandardScaler → PCA(16) → KMeans(8)) is fit **once
on a training window only** (dates ≤ a recorded `fit_cutoff`) and applied to
every date, yielding 24 features per day: 16 PCA coordinates (`embpca_*`) and
8 distances to the KMeans cluster centers (`embclus_dist_*`), stored in
`daily_embedding_derived`. The fitted basis itself (scaler statistics, PCA
components, and cluster centers) is persisted to `embedding_pca_basis`, which
lets the dashboard project *individual headlines* into exactly the same
16-dimensional space the models consume (§3.5).

**Causal narrative clustering** (`sentisense/cluster/narrative.py`). For each
trading day *T*, a MiniBatch-KMeans model is fit **only on embeddings strictly
before T** (expanding window with a refit cadence), then day-T headlines are
*assigned* with that past-fit model — yielding `dominant_cluster_ratio` and
normalized `cluster_entropy` without any look-ahead.

**Feature views.** Three views are produced: a **daily-mean** frame
(tree-model shape), a **per-source** pivot frame (sequence-model shape), and a
**fused** frame combining per-source scores with the daily e5 centroid and the
embedding-derived block (~970 columns) — the view the production champion
serves on.

### 3.4 Modeling, Hyper-Parameter Optimization, and the Model Registry

**Model zoo.** The forecasting layer evaluates three families under one
leak-safe contract (chronological 70/15/15; tune on the validation slice
only; score once on the sacred last-15% test tail):

- **Tree classifiers** — XGBoost, LightGBM, CatBoost (Optuna-tuned; the
  winner is refit on all labeled history and serialized with `joblib`).
- **Torch sequence classifiers** — LSTM, GRU, TCN, PatchTST over windowed
  per-source/fused features (Optuna studies are stored *in the database* and
  therefore resumable; the winner is refit and serialized as a
  `state_dict` bundle that also carries its scaler statistics, window length,
  and feature order).
- **Forecaster / foundation models** — TFT, N-HiTS, N-BEATS
  (pytorch-forecasting), and the zero-shot foundation models Chronos and
  TimesFM. These carry no persistable artifact — they are registered as
  *re-forecast* entries whose stored parameters (context length, tuned
  decision threshold) suffice to reproduce the forecast live.

**The model registry** (`model_registry` table + `sentisense/serve/registry.py`)
is the production selection mechanism. Each trained candidate is upserted with
its version, family, hyper-parameters, **held-out OOS metrics** (ROC-AUC with
a bootstrap 95% CI, accuracy, MCC, n), serialized artifact, and feature-column
contract. Selection is **auto-best with a sticky manual override**: the
highest-scoring model on the chosen metric is activated automatically, but a
manual activation from the dashboard is never silently overridden. A
rank-normalized **soft-vote ensemble** of the top-K tree models is registered
as its own activatable entry. At most one model is active at a time (enforced
by a partial unique index).

> **[Figure 5 placeholder: registry lifecycle diagram — HPO → OOS evaluation →
> register (weights + metrics) → auto-select / manual override → nightly
> serve.]**

**Champion serving** (`sentisense/serve/champion.py`). The nightly predictor
loads whatever the registry marks active and dispatches on its artifact
format: `joblib` models predict directly on the aligned feature row; `torch`
bundles are rebuilt from their `state_dict` (loaded with `weights_only=True`
for safety) and windowed over the recent feature history; `ensemble` entries
rank-normalize and average their members. A **pinned XGBoost champion**
(versioned JSON config, retrained on all labeled history each night) acts as
the guaranteed fallback: any failure in the registry path logs loudly and
falls back, so the daily prediction never silently breaks.

### 3.5 Live Operation: Orchestration, Serving, and the Dashboard

**Deployment topology.** The system runs on two hosts:

- a **GPU compute node** (NVIDIA RTX 4090) that runs the nightly pipeline —
  scraping, LLM scoring (local Ollama), embedding, derived features, registry
  training, and the champion prediction; and
- a **database/UI host** that runs PostgreSQL 16 and the dashboard (FastAPI +
  built React SPA, managed by a process supervisor).

The two communicate **only through the shared database**: the compute node
writes, the dashboard reads. This decoupling means the UI stays up even when
the compute node is retraining, and the pipeline is indifferent to the UI.

> **[Figure 3 placeholder: two-host topology — GPU node (cron pipeline, LLM,
> registry training) → shared PostgreSQL ← dashboard host (FastAPI/SPA).]**

**Nightly orchestration** (`scripts/daily_live.py`, scheduled via cron after
the TASE close). The orchestrator chains six stages with a lock file (no
double runs), per-stage logging, and a status JSON consumed by the dashboard's
health banner: **scrape** (today + yesterday) → **score** (gap-only; flags
selected automatically per LLM backend) → **embed** (new headlines only) →
**derive** (refresh the embedding-derived block and persist the basis) →
**predict** (the active champion forward-predicts the sentinel day; the
result is upserted into `model_predictions`) → **settle** (yesterday's
prediction is compared with the realized close and its `actual` field is
filled). The orchestrator self-skips non-trading days (Fri/Sat and a
configurable holiday list).

**The dashboard.** A FastAPI backend exposes a read-only JSON API (with
in-process caching) over the shared database; a React SPA renders it. Key
views:

- **Prediction hero** — a large green ▲ UP / red ▼ DOWN card with the current
  day's call, the predicted-class confidence, and the serving model's version.
- **Model performance** — the active champion's metric panel. Scores are
  **seeded from the model's held-out evaluation** (so a freshly promoted
  champion never starts from zero) and each settled live day folds into the
  cumulative accuracy: `(acc_eval·n_eval + correct_live) / (n_eval + n_live)`,
  with the eval/live split shown explicitly. Only the active model's own live
  days count — history from previous champions is never laundered into the
  new one's score.
- **Exploratory data analysis** — headline volume, daily mean sentiment,
  sentiment and relevance distributions, the 6×6 category-correlation
  heatmap, and the validation pass-rate, all computed server-side in SQL.
- **Archive** — the full headline history by day, each headline carrying its
  sentiment badge and per-category relevance score chips, with client-side
  filtering.
- **3-D centroid explorer** — every trading day's news centroid in the shared
  16-d PCA space (axes selectable), with the eight KMeans cluster centers
  drawn as labeled markers; clicking a day opens its **single-day headline
  cloud**, where each headline is projected through the *same persisted
  basis* the models consume, alongside the day centroid. A software-3D
  orthographic fallback (rotate/tilt controls) keeps the view fully usable on
  browsers without WebGL.
- **Simulator** — a narrative-simulation view: per-source **persona votes**
  (each outlet's daily stance derived from its mean scored sentiment,
  compared against the model's call and the realized outcome), plus cached
  agent-based simulation graphs and reports generated off-line by a
  multi-agent narrative engine.
- **Models (operator view)** — the registry leaderboard (version, family,
  OOS ROC-AUC with CI, MCC, accuracy, n) with one-click manual activation;
  hidden from the public navigation.

> **[Figure 6 placeholder: dashboard screenshot — hero + model performance.]**
> **[Figure 7 placeholder: EDA panels screenshot.]**
> **[Figure 8 placeholder: all-days 3-D centroids colored by cluster.]**
> **[Figure 9 placeholder: single-day headline cloud with day centroid and
> cluster centers.]**
> **[Figure 10 placeholder: persona votes vs model call for one day.]**

### 3.6 Implementation Details

**Languages, frameworks, and tooling.** Python 3.12, managed by `uv`.
Persistence uses PostgreSQL 16 via SQLAlchemy 2 + psycopg v3; connection
strings come **only** from the `SENTISENSE_DATABASE_URL` environment variable
and the code fails fast if it is unset (no embedded secrets). Core libraries:
pandas/numpy (features), scikit-learn/XGBoost/LightGBM/CatBoost (tabular),
PyTorch (sequence models), Optuna (HPO, RDB-backed resumable studies),
sentence-transformers (embeddings), pytorch-forecasting + Lightning
(TFT/N-HiTS/N-BEATS), Chronos/TimesFM (foundation forecasters), FastAPI +
uvicorn (API), React + Vite + Plotly (SPA), Playwright (scraping), and
LangGraph (agentic scoring path). Database schema changes ship as idempotent,
numbered SQL migrations (001–007).

**Key implementation decisions and trade-offs.**

- **Notebook → package.** A working but research-grade pipeline lived in
  notebooks. It was extracted into the importable, server-runnable
  `sentisense/` package, hardening the leakage controls in the process (the
  package deliberately does *not* port the notebooks' earlier leaky features
  such as shuffled `StratifiedKFold` or same-day target features).
- **Registry over redeployment.** Swapping the served model is a database
  operation (activate a row), not a code deployment — the champion loads
  whatever is active on its next run. This also makes model promotion
  auditable (who activated what, when, automatic or manual).
- **Serialization safety.** Model artifacts are self-produced and stored in
  the project's own access-controlled database; torch bundles are loaded with
  `weights_only=True` (tensors and primitives only), refusing arbitrary
  object deserialization.
- **Backend-aware scoring.** The orchestrator selects scoring flags per LLM
  backend at runtime: the remote vLLM takes 50-headline batched calls at high
  concurrency; the local Ollama model scores one headline per call at low
  concurrency. An empirical trial (§4.2.9) drove this design.
- **Resumable, cached experimentation.** The comparison driver
  (`scripts/pipeline_compare.py`) writes each finished cell's metrics to
  `leaderboard_cache.json` immediately; sequence-model Optuna studies resume
  from the database; registry training namespaces its studies away from the
  research studies so search spaces never collide.

**Software/hardware.** Development on macOS (CPU); heavy training and the
nightly pipeline on a Linux GPU node (NVIDIA RTX 4090, CUDA 12.3 driver;
torch pinned to CUDA-12.1 wheels, with a CPU fallback index for local work).
PostgreSQL 16 and the dashboard run on a separate Linux host.

### 3.7 Evaluation Metrics

Because next-day direction is near-balanced and accuracy alone is misleading,
the project reports a metric set (`sentisense/models/metrics.py`), all
computed on the **same sacred last-15% out-of-sample window** in research
mode:

- **ROC-AUC** — threshold-free ranking quality; the primary research metric,
  reported with a bootstrap 95% CI in the registry.
- **F1 (macro)** — balances precision/recall across both classes.
- **Accuracy** and **balanced accuracy** — overall and class-balanced hit
  rate. Accuracy is the registry's default *selection* metric for the served
  champion (configurable to ROC-AUC).
- **MCC** — Matthews correlation, robust to class imbalance.

Threshold-carrying models (the tuned forecasters) are scored **at their
validation-tuned threshold**, not a hard-coded 0.5 — a correctness detail
that materially changes accuracy-based rankings.

Three complementary evaluation surfaces exist in production: (a) the
**registry OOS metrics** (held-out test tail, computed once at training
time); (b) the **cumulative live score** (eval-seeded, extended by each
settled prospective day — the strongest evidence, since it cannot be
overfit); and (c) an **in-sample all-days evaluation** (`champion_full_eval`,
the champion fit on all labeled days and scored on those same days) which is
deliberately exposed on the dashboard *as-is*: its near-perfect scores are a
textbook illustration of memorization, and the visible gap between it and the
OOS/live numbers is itself an honest, pedagogical result. A **backtest
overlay** and a **Buy&Hold** benchmark place the statistical metrics in an
economic context.

---

## 4. Results and Analysis

### 4.1 Experimental Setup

Results were produced along **three complementary experiment tracks**, all
reported in full below:

1. **Exploratory notebook tracks** — a sequence of research notebooks
   (`poc.ipynb`, `compare_lstm_features_with_poc.ipynb`,
   `transformer_forecaster.ipynb`, `tuning.ipynb`, `sentisense_analysis.ipynb`,
   `timesfm_explainability.ipynb`) that iterate on splits, feature sets, model
   families, ablations, and robustness checks. These differ deliberately in
   their train/test windows and majority-class baselines, which is itself part
   of the analysis (§4.3).
2. **The unified, hardened package grid** — `scripts/pipeline_compare.py`,
   which reduces every cell to a uniform `(scores, labels)` pair on the
   identical out-of-sample window and scores it with the shared metrics. This
   is the canonical, leakage-hardened cross-model comparison.
3. **The production registry run** — `scripts/train_registry.py`, which
   re-tunes the zoo under the registry's serving contract on the full
   timeline, registers every candidate with its OOS metrics, and activates
   the champion that the live system serves (§4.2.9).

> **Note on reproduction state.** A number of cells in the saved notebooks
> were not executed in the committed copy (e.g. the transformer Optuna "tuned
> leaderboard" and McNemar cells; the `tuning.ipynb` GRU/TCN, multi-seed,
> abstention, and final-report cells; and all of
> `timesfm_explainability.ipynb`). To keep this book honest and reproducible,
> **only metrics that actually rendered in the saved outputs are reported**,
> and each gap is flagged where it occurs.

The unified package grid is a three-axis grid evaluated by
`scripts/pipeline_compare.py`:

- **Model axis** — classifiers (XGBoost, LSTM, GRU, TCN, PatchTST) and
  forecasters (TFT, N-HiTS, N-BEATS, Chronos, TimesFM), plus Buy&Hold.
- **Data-type axis** — `scored` (LLM news scores), `embedded` (768-d e5
  centroid + finance), and `fused` (per-source scores ⊕ centroid). Classifiers
  run on all three; forecasters use scored covariates / univariate only.
- **Regime axis** — `CUT` (≤ 2023-10-07) vs `FULL` (entire timeline).

Each classifier (model × data-type × regime) cell gets its **own resumable
Optuna study**; search spaces are wide (e.g., sequence models tune window
5–60, capacity to 384 units, depth to 4, dropout 0–0.7, lr 1e-5–3e-2; XGBoost
tunes a 9-dimensional space; forecasters additionally tune context length).
HPO selects on a validation slice only; the test tail stays sacred.
Reproducibility is enforced with fixed seeds.

### 4.2 Presentation of Results

The project generated **many** result tables across its experiment tracks.
They are reproduced faithfully below, organized by source, then summarized by
the unified package grid (§4.2.8) and the production registry run (§4.2.9).
Every table predicts the **next-day close-to-close TA-125 direction**.

**Aggregation lineage (how the notebooks relate).** A crucial point for
reading these results is that the notebooks deliberately use *different
news-aggregation strategies*, and the later notebooks expand on the two base
ones:

- **Base A — daily-mean aggregation (`poc.ipynb`).** Each day's headlines are
  collapsed to **per-category means** (`mean_politics … mean_sentiment`, plus
  `std_sentiment`, `pct_negative`, `pct_positive`). This is the compact,
  tree-friendly representation. (The base PoC also added `LastDay_Rise` /
  `LastDay_Pct` features built with a `shift(-1)`, a leaky construction that
  the hardened package deliberately drops.)
- **Base B — per-source wide aggregation (`lstm_forecaster.ipynb`).** Scores
  are **summed per `(date, source)` and pivoted wide** (`<dim>_<source>` ⇒
  320 feature columns), preserving *which outlet produced which signal*, then
  sliced into 30-day windows for the LSTM.
- **Expansions.** `compare_lstm_features_with_poc.ipynb` runs the **PoC tree
  models on Base B's per-source wide features** (directly testing "do
  per-source features beat daily means?"), adding ablations, walk-forward,
  and multi-seed checks. `transformer_forecaster.ipynb` and `tuning.ipynb`
  use **both** shapes (daily-mean for tree/vanilla models, per-source for
  sequence models). The `sentisense/` package generalizes both into
  leakage-safe `mt` (daily-mean) and `ml` (per-source) frames (§3.3).

Each subsection below notes which aggregation it uses.

**Cross-track summary.** Before the detailed tables, Table 1 gives the
one-glance picture: the best configuration in each track, its majority-class
baseline, and whether it beats that baseline / reaches significance. The
recurring theme is that "best" accuracies rarely clear their own baseline,
and none reaches significance.

*Table 1: Best result per experiment track vs its baseline.*

| Track (§) | Aggregation | Best config | Acc | Baseline | Beats base? | ROC-AUC | Significant? |
|---|---|---|---|---|---|---|---|
| PoC (§4.2.1) | daily-mean | XGBoost / LightGBM | 0.5459 | 0.4976 | yes | n/a | no (p≈0.09–0.13) |
| LSTM-features (§4.2.2) | per-source | LGBM "Top sources+Other" | 0.5794 | 0.5675 | +0.012 | 0.5415 | no (p_perm=0.052) |
| LSTM base (§4.2.3) | per-source | LSTM (window 30) | 0.5636 | 0.5773 | **no** | n/a | no (p=0.68) |
| Transformer zoo (§4.2.4) | both | PatchTST_DailyMean | 0.5370 | 0.4931 | yes | 0.5185 | no (p≈0.09) |
| Tuning (§4.2.5) | both | Ensemble / tuned LSTM | 0.4596 | 0.5680 | **no** | no | no |
| Hardened pkg (§4.2.6) | daily-mean | Score-LSTM | 0.5000 | ~0.50 | ~tie | 0.5088 | no (MCC≈0) |
| Unified grid (§4.2.8) | all | GRU [scored] (by ROC-AUC) | 0.5289 | ~0.50 | ~tie | **0.5755** | no |
| Registry (§4.2.9) | fused | PatchTST (active champion) | 0.5780 | ~0.55 | marginal | 0.4795 | CI spans 0.5 |

*(n/a = ROC-AUC not printed numerically in that notebook; "Significant?" =
passes a permutation/binomial test at p<0.05 vs the majority-class baseline.)*

#### 4.2.1 Tree-model proof-of-concept (`poc.ipynb`)

*Aggregation: Base A — daily-mean per category (+ leaky `LastDay` features).*

The earliest experiment established a tree-model baseline. Two evaluation
protocols were run.

**5-fold cross-validation (accuracy):**

| Model | Mean Accuracy | Std | Fold scores |
|---|---|---|---|
| XGBoost | 53.60% | 1.87% | 51.70 / 56.66 / 51.70 / 54.57 / 53.40 |
| LightGBM | 52.40% | 2.49% | 48.04 / 55.35 / 51.70 / 53.00 / 53.93 |
| CatBoost | 53.45% | 3.21% | 47.26 / 56.14 / 53.52 / 55.35 / 54.97 |

*Table 2: PoC tree-model 5-fold cross-validation accuracy.*

**Chronological 80/20 holdout** (train 826 rows 2019-07-17→2022-12-04; test
207 rows 2022-12-05→2023-10-05; test up-rate 49.76%):

| Model | Test Accuracy |
|---|---|
| XGBoost | 54.59% |
| LightGBM | 54.59% |
| CatBoost | 53.62% |

*Table 3: PoC chronological 80/20 holdout accuracy.*

**Significance vs majority-class baseline (49.76%):**

| Model | Acc | p_binom | p_perm | 95% CI |
|---|---|---|---|---|
| XGBoost | 54.59% | 0.0933 | 0.1260 | [48.30%, 61.35%] |
| LightGBM | 54.59% | 0.0933 | 0.1280 | [47.83%, 61.35%] |
| CatBoost | 53.62% | 0.1486 | 0.1800 | [46.86%, 59.92%] |

*Table 4: PoC significance tests vs majority-class baseline.*

XGBoost holdout classification report (207-sample split):¹

| Class | precision | recall | f1 | support |
|---|---|---|---|---|
| 0 (Fall) | 0.56 | 0.45 | 0.50 | 104 |
| 1 (Rise) | 0.54 | 0.64 | 0.58 | 103 |
| accuracy | | | 0.55 | 207 |

*Verdict:* all `p_binom`/`p_perm` are **> 0.05** — no tree model beats the
majority baseline at significance. (McNemar's test was skipped: `statsmodels`
not installed.)

> ¹ `poc.ipynb` contains unresolved git merge-conflict markers; an alternate
> rendered report with a 507-sample support exists. The 207-sample version
> above matches the notebook's stated 80/20 split.

#### 4.2.2 LSTM feature-set vs PoC study (`compare_lstm_features_with_poc.ipynb`)

*Aggregation: Base B — per-source wide (from `lstm_forecaster.ipynb`), fed to
the PoC tree models. This notebook exists specifically to test whether Base
B's per-source features beat Base A's daily means.*

The most extensive notebook compares feature families on the per-source "LSTM
wide" representation, with ablations and robustness checks. Unless noted, the
test window is 2024-03-26→2026-04-28 (504 rows, majority baseline 56.75%).

**Main holdout summary (sorted by accuracy):**

| Experiment | Model | Accuracy | Baseline | Gap | Bal.Acc | ROC-AUC | p_binom | p_perm |
|---|---|---|---|---|---|---|---|---|
| Top sources + Other | LGBM | 0.5794 | 0.5675 | +0.0119 | 0.5230 | 0.5415 | 0.311 | 0.052 |
| Baseline wide | CatBoost | 0.5714 | 0.5675 | +0.0040 | 0.5068 | 0.4760 | 0.447 | 0.234 |
| Top sources + Other | XGBoost | 0.5714 | 0.5675 | +0.0040 | 0.5171 | 0.5359 | 0.447 | 0.130 |
| Baseline wide | XGBoost | 0.5694 | 0.5675 | +0.0020 | 0.5072 | 0.5196 | 0.483 | 0.255 |
| Baseline wide | LGBM | 0.5694 | 0.5675 | +0.0020 | 0.5067 | 0.5453 | 0.483 | 0.278 |
| Top sources + Other | CatBoost | 0.5675 | 0.5675 | 0.0000 | 0.5065 | 0.4963 | 0.519 | 0.315 |

*Table 5: Per-source feature-set holdout comparison (tree models).*

**Feature-group ablation (CatBoost), test up-rate baseline 56.42%:**

| Feature group | N features | Accuracy | Gap | ROC-AUC |
|---|---|---|---|---|
| Basic market only | 6 | 0.5811 | +0.0168 | 0.5415 |
| News + all market features | 344 | 0.5768 | +0.0126 | 0.5074 |
| Market-derived only | 17 | 0.5600 | −0.0042 | 0.5296 |
| News wide only | 321 | 0.5558 | −0.0084 | 0.4841 |

*Table 6: Feature-group ablation (CatBoost) — market-only matches news+market.*

**Walk-forward validation (5 folds, CatBoost):** mean accuracy **0.5967**,
mean baseline 0.5733, mean gap +0.0233, with only **2 / 5** folds beating
baseline (fold accuracies 0.633 / 0.650 / 0.567 / 0.617 / 0.517).

**Multi-seed robustness (CatBoost, 5 seeds):** mean accuracy **0.5714 ±
0.0084** (min 0.560, max 0.581), mean gap +0.0072, **4 / 5** seeds positive;
ROC-AUC ranges 0.507–0.577.

**Feature importance by group (CatBoost):** News 79.2%, Market-derived 17.6%,
Basic market 3.2% of total importance — yet the strongest *individual*
features are price-derived (`TA125_logret_5d_std`, `TA125_logret_lag5`,
`TA125_RSI14`).

*Verdict:* the best configuration (LGBM, "Top sources + Other") reaches
accuracy 0.579 with `p_perm = 0.052` — the closest to significance anywhere —
but its bootstrap 95% CI (0.534–0.623) straddles the baseline. The signal is
**weak and unstable**.

#### 4.2.3 LSTM base forecaster (`lstm_forecaster.ipynb`)

*Aggregation: Base B — per-source wide (320 columns), 30-day windows. This is
the original sequence model the per-source representation was built for.*

The base LSTM is trained on chronological, windowed sequences (window 30,
326 features) with train/val/test = 1,163 / 249 / 250 daily rows
(2019-07-17 → 2026-04-29); test up-rate 57.73%.

**LSTM holdout test:** accuracy **56.36%** (below the 57.73% majority
baseline).

| Class | precision | recall | f1 | support |
|---|---|---|---|---|
| Fall | 0.29 | 0.02 | 0.04 | 93 |
| Rise | 0.57 | 0.96 | 0.72 | 127 |
| accuracy | | | 0.56 | 220 |

*Table 7: LSTM base forecaster holdout classification report.*

**Significance vs baseline (57.73%):** binomial p = 0.6845; permutation
p = 0.8820; bootstrap 95% CI [50.00%, 62.73%].

*Verdict:* the model collapses toward the majority "Rise" class (recall 0.96
vs 0.02), scores **below** baseline, and is statistically indistinguishable
from it. Training accuracy reaches ~0.74 while validation stays ~0.48–0.53 —
clear overfitting on the 320-column per-source representation. (ROC-AUC/MCC
were rendered only inside plot images, so no numeric values are reported
here.)

#### 4.2.4 Transformer model zoo + ablations (`transformer_forecaster.ipynb`)

*Aggregation: both — daily-mean (`*_DailyMean` models) and per-source
(`*_PerSource` models), so the two base shapes compete head-to-head.*

This notebook set an explicit success bar: **≥58% test accuracy with p<0.05
vs majority**. Nine transformer variants were evaluated against tree/linear
baselines (majority-class baseline 49.31%).

**Final leaderboard — transformer vs baselines (best per row):**

| Model | accuracy | balanced_accuracy | f1 | roc_auc | mcc |
|---|---|---|---|---|---|
| ModelB_PatchTST_DailyMean | 0.5370 | 0.5381 | 0.4926 | 0.5185 | 0.0949 |
| CatBoost | 0.5069 | 0.5093 | 0.4940 | 0.5048 | 0.0196 |
| XGBoost | 0.5035 | 0.5020 | 0.4969 | 0.5070 | 0.0040 |
| ModelE_Informer_PerSource | 0.4981 | 0.5000 | 0.3325 | 0.5000 | 0.0000 |
| ModelC_TwoTower_DailyMean | 0.5019 | 0.5000 | 0.3342 | 0.5000 | 0.0000 |
| ModelA_Vanilla_PerSource | 0.4942 | 0.4941 | 0.4940 | 0.4996 | −0.0118 |
| ModelA_Vanilla_DailyMean | 0.4942 | 0.4926 | 0.4018 | 0.5216 | −0.0236 |
| LGBM | 0.4931 | 0.4926 | 0.4922 | 0.4727 | −0.0149 |
| ModelE_Informer_DailyMean | 0.4903 | 0.4889 | 0.4190 | 0.5397 | −0.0309 |
| ModelD_Hierarchical_DailyMean | 0.4903 | 0.4886 | 0.3810 | 0.5339 | −0.0415 |
| ElasticNet | 0.4792 | 0.4798 | 0.4783 | 0.4804 | −0.0405 |
| ModelD_Hierarchical_PerSource | 0.4708 | 0.4711 | 0.4690 | 0.4757 | −0.0583 |
| ModelC_TwoTower_PerSource | 0.4514 | 0.4516 | 0.4487 | 0.4766 | −0.0977 |

*Table 8: Transformer zoo final leaderboard vs tree/linear baselines.*

**Statistical tests (permutation + binomial vs baseline):** no model is
significant — the lowest p-value is PatchTST (`p_binom=0.089`,
`p_perm=0.090`).

**Window-size ablation (PatchTST):** best at window 15–20 (acc ≈ 54–55%,
ROC-AUC up to 0.592); collapses to the majority class at windows 45–60.
**Feature-group ablation:** Market-only 0.5409 > LagReturns-only 0.5292 >
News-only 0.4864.

*Verdict (printed):* "Best transformer: PatchTST 53.81% balacc … Best overall
score < 55% … TA-125 direction at daily frequency may not be predictable from
news sentiment + market features alone." The **≥58% success criterion was not
met**. (The Optuna "tuned leaderboard" and McNemar cells were not executed.)

#### 4.2.5 Sequence-model tuning & robustness (`tuning.ipynb`)

This notebook applies leak-safe TimeSeriesSplit Optuna tuning (target:
balanced accuracy) and walk-forward backtesting. Corpus: 1,898,499 validated
rows, 40 sources.

**Reproducibility sanity (vanilla holdout, baseline 56.80%):** XGBoost
51.06%, LightGBM 52.57%, CatBoost 50.76% — all below baseline.

**Threshold-optimized validation (Youden's J):**

| Model | best_thr | val_balacc | val_acc | val_F1 |
|---|---|---|---|---|
| XGBoost | 0.597 | 0.5363 | 0.5142 | 0.4826 |
| LightGBM | 0.521 | 0.5583 | 0.5628 | 0.5582 |
| CatBoost | 0.525 | 0.5345 | 0.5263 | 0.5243 |

*Table 9: Tuning notebook — threshold-optimized validation (Youden's J).*

**LSTM (Optuna best val balacc 0.5611):** test accuracy at tuned threshold
**0.4553**. **Ensemble (soft-vote):** val balacc 0.5712, **test accuracy
0.4596**. **Walk-forward CatBoost:** mean accuracy **0.5267 ± 0.0814**, mean
baseline 0.5533 (gap −0.0267).

*Verdict:* tuned single models and the ensemble both **underperform the
majority baseline** on the holdout. (GRU/TCN, multi-seed, abstention, and the
final `final_results.csv` cells were not executed in the saved notebook.)

#### 4.2.6 Hardened-package analysis (`sentisense_analysis.ipynb`)

Run directly against the live database (cutoff 2023-10-07). Corpus coverage:
**2,950,339 validated `mistral-small-4` rows** (plus 52,640
`mistral-small:latest`).

**Score-LSTM final holdout (mean ± std over repeats):**

| Metric | mean | std |
|---|---|---|
| accuracy @0.5 | 0.5000 | 0.0058 |
| balanced_accuracy @0.5 | 0.5001 | 0.0057 |
| f1 @0.5 | 0.4990 | 0.0066 |
| roc_auc @0.5 | 0.5088 | 0.0144 |
| mcc @0.5 | 0.0001 | 0.0114 |
| accuracy @tuned | 0.4961 | 0.0136 |
| roc_auc @tuned | 0.5072 | 0.0080 |
| mcc @tuned | 0.0013 | 0.0402 |

*Table 10: Hardened-package score-LSTM final holdout (mean ± std).*

(LSTM Optuna best value 0.538.) *Verdict:* **near-chance** — accuracy ≈ 0.50,
ROC-AUC ≈ 0.51, MCC ≈ 0.00. On the hardened pipeline the score-only LSTM
shows essentially **no directional edge**. (The §2 multi-track, baselines,
and strategy-vs-Buy&Hold cells were not executed; SHAP outputs exist only as
plots.)

#### 4.2.7 Foundation-model explainability (`timesfm_explainability.ipynb`)

This notebook scaffolds zero-shot/covariate-ablation/regime experiments for
Google's TimesFM, but **was not executed in the committed copy** — no numeric
results are available. It is retained as a wired template for future work
(§5).

#### 4.2.8 Unified out-of-sample grid (`leaderboard.md`)

The canonical, leakage-hardened comparison reduces every model × data-type ×
regime cell to the same out-of-sample window and metric set. Sorted by
accuracy descending. Notation: `model [data-type]` for classifiers,
`model [cov=...]` for forecasters.

| model [data-type] | roc_auc | f1 | accuracy |
|---|---|---|---|
| TFT [cov=none] | 0.5391 | 0.4148 | 0.5916 |
| XGBoost [embedded] | 0.5314 | 0.5129 | 0.5890 |
| XGBoost [fused] | 0.5253 | 0.4616 | 0.5759 |
| GRU [fused] | 0.5359 | 0.5238 | 0.5568 |
| PatchTST [fused] | 0.5112 | 0.3679 | 0.5553 |
| Chronos-zeroshot | 0.4266 | 0.3617 | 0.5538 |
| LSTM [embedded] | 0.5128 | 0.5317 | 0.5429 |
| XGBoost [fused] | 0.5396 | 0.5373 | 0.5417 |
| LSTM [fused] | 0.4724 | 0.4427 | 0.5402 |
| Chronos-tuned | 0.4492 | 0.4181 | 0.5381 |
| TFT [cov=scored] | 0.5524 | 0.5033 | 0.5366 |
| XGBoost [embedded] | 0.5217 | 0.5289 | 0.5347 |
| TCN [fused] | 0.5303 | 0.5280 | 0.5318 |
| TCN [scored] | 0.5669 | 0.5281 | 0.5310 |
| TFT [cov=none] | 0.5386 | 0.5212 | 0.5296 |
| GRU [scored] | 0.5755 | 0.4118 | 0.5289 |
| PatchTST [embedded] | 0.4726 | 0.3831 | 0.5283 |
| PatchTST [scored] | 0.4541 | 0.4415 | 0.5208 |
| NHiTS [cov=none] | 0.4808 | 0.4812 | 0.5157 |
| PatchTST [fused] | 0.5040 | 0.5120 | 0.5126 |
| NBEATS | 0.5227 | 0.5080 | 0.5105 |
| TCN [scored] | 0.5422 | 0.4992 | 0.5094 |
| NHiTS [cov=scored] | 0.4830 | 0.5033 | 0.5087 |
| XGBoost [scored] | 0.5338 | 0.5044 | 0.5079 |
| LSTM [scored] | 0.5204 | 0.3918 | 0.5041 |
| XGBoost [scored] | 0.5129 | 0.4997 | 0.5035 |
| PatchTST [scored] | 0.5270 | 0.4321 | 0.5035 |
| TCN [embedded] | 0.4675 | 0.4275 | 0.5022 |
| TFT [cov=scored] | 0.5119 | 0.5002 | 0.5017 |
| NBEATS | 0.5106 | 0.4980 | 0.4983 |
| LSTM [scored] | 0.5125 | 0.4938 | 0.4958 |
| GRU [embedded] | 0.5091 | 0.3414 | 0.4910 |
| NHiTS [cov=none] | 0.4837 | 0.4894 | 0.4895 |
| NHiTS [cov=scored] | 0.4835 | 0.4869 | 0.4869 |
| GRU [embedded] | 0.4642 | 0.4593 | 0.4820 |
| LSTM [fused] | 0.5115 | 0.4797 | 0.4802 |
| TCN [fused] | 0.4552 | 0.4667 | 0.4709 |
| LSTM [embedded] | 0.4715 | 0.3840 | 0.4706 |
| GRU [fused] | 0.4679 | 0.4401 | 0.4669 |
| GRU [scored] | 0.4967 | 0.4221 | 0.4644 |
| PatchTST [embedded] | 0.4552 | 0.4492 | 0.4513 |
| TCN [embedded] | 0.5327 | 0.3269 | 0.4238 |

*Table 11: Unified out-of-sample leaderboard (40+ tuned cells). Coverage: 23
model configurations ran, 21 cached, 2 skipped.*

**Best by ROC-AUC (the headline criterion):** `GRU [scored]` — ROC-AUC =
**0.5755**, F1 = 0.4118.
**Best by accuracy:** `TFT [cov=none]` — accuracy = **0.5916**.

> **[Figure 11 placeholder: scatter of Table 11 — ROC-AUC (x) vs accuracy
> (y), point shape by model family, showing the cloud centered on
> (0.50, ~0.52).]**

#### 4.2.9 Production registry run and the live champion

The final track moves from comparison to *selection*: `train_registry.py`
re-tunes the zoo under the registry's serving contract (fused features, FULL
regime, chronological 70/15/15, per-family Optuna studies) and registers each
candidate with its held-out metrics.

**Registry validation run (tree zoo, low trial budget).** A smoke-budget run
(5 trials/model) validated the end-to-end train → register → select → serve
loop and produced honest, near-chance OOS numbers — consistent with every
research track:

| Model | OOS ROC-AUC | 95% CI | MCC | Accuracy |
|---|---|---|---|---|
| XGBoost | 0.5476 | [0.486, 0.604] | 0.062 | 0.5527 |
| LightGBM | 0.5153 | [0.458, 0.576] | 0.060 | 0.5553 |
| CatBoost | 0.5476 | [0.483, 0.604] | 0.030 | 0.5424 |

*Table 12: Registry validation run — tree zoo OOS metrics (test tail of the
fused/FULL frame). Every ROC-AUC confidence interval spans 0.5.*

**The full-budget run** (100 trials per tree model, 40 per sequence
architecture with 3-seed OOS averaging, plus the foundation-model families)
populated the registry leaderboard that the dashboard's Models panel
displays.

> **[Table/Figure 12 placeholder: export the full registry leaderboard from
> the Models panel (version, family, ROC-AUC + CI, MCC, accuracy, n) once the
> full-budget run's results are final.]**

**The active champion.** Selection by held-out accuracy activated a
**PatchTST** sequence classifier:

| Property | Value |
|---|---|
| Version | `patchtst-20260702-1351` |
| Family | PatchTST (torch sequence classifier, fused features) |
| OOS accuracy | **0.578** (n = 327 held-out days) |
| OOS MCC | 0.087 |
| OOS ROC-AUC | 0.4795 |

*Table 13: Active production champion — held-out evaluation.*

The champion's profile is instructive: its *accuracy* is the zoo's best, but
its *ROC-AUC is below 0.5* — it wins by calibrated class-leaning rather than
by ranking skill, exactly the accuracy/ROC-AUC dissociation the unified grid
already exposed (§4.3). Both metrics are surfaced side-by-side on the
dashboard, and the selection metric is a one-flag choice
(`--select-metric oos_roc_auc | oos_accuracy`), so the trade-off is explicit
rather than hidden.

**Backend trial for the live scoring era.** Before switching nightly scoring
to the locally hosted `gemma4` model, three modes were trialed empirically:
the agentic ReAct path failed (tool-loop recursion), 10-headline batched JSON
failed (unparseable output), and **single-headline structured calls succeeded
20/20** at ≈7.7 headlines/minute — sufficient for the nightly volume
(~1,000 headlines/day). This trial directly produced the backend-aware
scoring design of §3.6.

**Live cumulative record.** From activation onward, each settled trading day
extends the champion's prospective record on the dashboard (eval-seeded
cumulative accuracy, §3.5). This record is the project's strongest ongoing
evidence, since prospective days cannot be overfit.

> **[Figure 12 placeholder: screenshot of the Models panel with the active
> champion highlighted; optionally a second screenshot of the cumulative
> live-accuracy panel after a few weeks of operation.]**

### 4.3 Data Analysis and Interpretation

Reading **across all tracks** of §4.2, several consistent patterns emerge.

1. **Every track converges on near-chance.** Whether trees (§4.2.1–4.2.2,
   §4.2.5, §4.2.9), transformers (§4.2.4), LSTMs (§4.2.3, §4.2.5–4.2.6), or
   foundation models (§4.2.8), ROC-AUC clusters around 0.50 (≈ 0.43–0.59) and
   balanced accuracy hovers near 0.50. The hardened score-LSTM is the
   cleanest statement of this: accuracy 0.500, ROC-AUC 0.509, MCC 0.000
   (§4.2.6); the registry's tree CIs all span 0.5 (§4.2.9).
2. **The majority-class baseline is the real story.** Baselines differ
   sharply by split — 49.31% in the transformer notebook, 49.76% in the PoC,
   56.42–56.80% in the later windows — because the up-rate is
   window-dependent. On the high-up-rate windows, *raw accuracy near 57% does
   not beat "always predict up."* Several models that look respectable on
   accuracy (e.g. tuned LSTM 0.4553, ensemble 0.4596 in §4.2.5) are in fact
   **below** their baseline.
3. **Accuracy, ROC-AUC, and baseline disagree.** The unified grid's
   top-accuracy model (`TFT [cov=none]`, 0.592) has only mediocre ROC-AUC
   (0.539) and low F1 (0.415) — the signature of leaning to the majority
   class. The top-ROC-AUC model (`GRU [scored]`, 0.576) is mid-table on
   accuracy. The production champion repeats the pattern from the other side:
   best-in-zoo accuracy (0.578) with sub-0.5 ROC-AUC (§4.2.9). Reporting a
   single metric would mislead; the multi-metric view is essential.
4. **No data-type or model family dominates.** `scored`, `embedded`, and
   `fused` views land in the same band; foundation models (Chronos zero-shot
   0.427) do not beat trained models; complex transformers do not beat simple
   GRU/TCN/XGBoost. The feature-group ablation is telling: **market-only
   features match or beat news+market**, and **news-only is the weakest**
   (§4.2.2, §4.2.4). Capacity is not the bottleneck; **signal is.**
5. **Significance is essentially never reached.** Across the PoC,
   transformer, and feature-comparison notebooks, every permutation/binomial
   test returns p > 0.05. The single closest call — LGBM "Top sources +
   Other", `p_perm = 0.052` (§4.2.2) — has a bootstrap CI that straddles its
   baseline, and is not robust under walk-forward (2/5 folds positive) or
   multi-seed (mean gap +0.7pp) checks.
6. **In-sample scores are a warning, not a result.** The all-days in-sample
   evaluation (`champion_full_eval`) reaches accuracy ≈ 1.0 — a 600-tree
   XGBoost memorizing 2,586 days of 970 features. Displayed next to the ~0.55
   OOS numbers, it is a built-in demonstration of why leakage-free evaluation
   is non-negotiable in this domain.

### 4.4 Comparison with Existing Approaches

**Internal comparison across tracks.** The notebook tracks (§4.2.1–4.2.5),
the hardened package (§4.2.6, §4.2.8), and the production registry (§4.2.9)
tell a consistent story, but the package and registry numbers are the
trustworthy ones. The exploratory notebooks vary their splits and baselines
and occasionally show small positive gaps on a single favorable window; the
hardened, fixed-window runs flatten these to near-chance. This is exactly the
expected effect of removing window-selection freedom and tightening leakage
controls — the transformer notebook's own ≥58% success criterion is **not
met** anywhere, and its printed verdict already anticipates that "TA-125
direction at daily frequency may not be predictable from news sentiment +
market features alone."

**Comparison with the literature.** The magnitude of the effect is consistent
with the consensus that **news-tone signal for next-day index direction is
small and fragile**, and that much of the apparent predictability reported
elsewhere stems from evaluation leakage or favorable-period selection. The
Buy&Hold benchmark included in the unified grid contextualizes that none of
the directional models offers a dependable economic edge over simply holding
the index. Notably, the feature-importance breakdown (News ≈ 79% of total
importance, §4.2.2) shows the models *do* lean on the news features — they
simply do not convert that into out-of-sample skill.

### 4.5 Discussion of Findings

The honest interpretation is that, **on its own and at daily resolution, the
LLM-scored Hebrew-news stream carries little reliable next-day directional
signal for the TA-125.** This is not a failure of engineering but a
substantive empirical result, and it is credible *because* of the leakage
controls: the pipeline was specifically built to make it hard to fool
oneself.

The project's lasting value is methodological and infrastructural. It
delivers (a) a reproducible, auditable, leakage-safe pipeline spanning
scraping, LLM scoring, feature engineering, and a tuned model zoo; (b) a
uniform, resumable comparison framework; (c) a **production loop** — model
registry, nightly orchestration, settlement, and dashboard — that keeps
extending the out-of-sample record prospectively; and (d) a quantified,
multi-metric baseline against which future richer signals (intraday data,
magnitude targets, alternative LLM scoring schemes, longer horizons) can be
measured.

Limitations include the close-to-close-only target, daily resolution, the
all-zero LLM rows, the mixed scoring-model history — now extended by the
deliberate mistral→gemma **scoring-era seam** in the live track (§3.2), whose
effect on feature comparability is a known, monitored caveat until the
history is re-standardized — and the champion-selection tension between
accuracy and ROC-AUC exposed in §4.2.9. Each is a concrete lever for future
work.

---

## 5. Conclusion and Future Work

**Conclusion.** SentiSense set out to test whether LLM-distilled Hebrew-news
sentiment can predict next-day TA-125 direction — and to do so in a way that
survives contact with production. It produced a complete, leakage-hardened,
reproducible system: scraper, LLM scorer, a ~3M-row scored corpus, daily
feature engineering with embedding-derived and narrative features, a tuned
zoo of 40+ model configurations, a database-backed model registry with
automatic champion selection and manual override, a nightly
scrape-score-predict-settle orchestrator on a two-host deployment, and an
interactive dashboard that presents the prediction, the evidence, and the
data itself. Evaluated rigorously on sacred out-of-sample windows, the best
research models reach only ROC-AUC 0.576 / accuracy 0.592, no result clears
significance, and the deployed champion (PatchTST, OOS accuracy 0.578, CI
spanning chance on ROC-AUC) is served with its limitations displayed rather
than hidden. The contribution is therefore a **trustworthy negative-leaning
baseline plus a live, self-auditing research platform**, not a profitable
predictor.

**Future work.**

1. **Richer targets.** Persist TA-125 OHLC to enable overnight-gap and
   intraday-return (magnitude) targets, which may carry more news signal than
   close-to-close direction.
2. **Longer horizons & event studies.** Test weekly direction and
   event-window responses around high-impact (very negative +
   high-security-relevance) headlines.
3. **Scoring-era standardization.** Re-score the historical corpus under the
   live scoring model (`standardize_to_latest_model.py`) and retrain the zoo
   on a single-era feature space, removing the mistral→gemma seam.
4. **Zero-shot serving.** Extend the champion's dispatch with the
   `reforecast` path so a registered Chronos/TimesFM/TFT winner can be served
   by live re-forecasting, not only evaluated.
5. **Persona analytics.** Backtest the per-source persona votes as
   predictors in their own right ("which outlet is the best forecaster?") and
   consider credibility-weighted aggregation as a feature.
6. **Robustness and monitoring.** Multi-seed registry evaluations as the
   default, drift monitoring on the live feature distributions, and periodic
   automatic re-training gates tied to the cumulative live record.
7. **Explainability.** Execute the TimesFM explainability track (§4.2.7) and
   add SHAP-based attribution for the served champion to the dashboard.

---

## 6. References

[1] P. C. Tetlock, "Giving Content to Investor Sentiment: The Role of Media in
the Stock Market," *Journal of Finance*, vol. 62, no. 3, pp. 1139–1168, 2007.

[2] J. Bollen, H. Mao, and X. Zeng, "Twitter mood predicts the stock market,"
*Journal of Computational Science*, vol. 2, no. 1, pp. 1–8, 2011.

[3] T. Loughran and B. McDonald, "When Is a Liability Not a Liability? Textual
Analysis, Dictionaries, and 10-Ks," *Journal of Finance*, vol. 66, no. 1,
pp. 35–65, 2011.

[4] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in
*Proc. KDD*, 2016, pp. 785–794.

[5] S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural
Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.

[6] K. Cho et al., "Learning Phrase Representations using RNN Encoder–Decoder
for Statistical Machine Translation," in *Proc. EMNLP*, 2014.

[7] S. Bai, J. Z. Kolter, and V. Koltun, "An Empirical Evaluation of Generic
Convolutional and Recurrent Networks for Sequence Modeling," arXiv:1803.01271,
2018.

[8] B. Lim et al., "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting," *International Journal of
Forecasting*, vol. 37, no. 4, pp. 1748–1764, 2021.

[9] Y. Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting
with Transformers (PatchTST)," in *Proc. ICLR*, 2023.

[10] B. N. Oreshkin et al., "N-BEATS: Neural basis expansion analysis for
interpretable time series forecasting," in *Proc. ICLR*, 2020.

[11] C. Challu et al., "N-HiTS: Neural Hierarchical Interpolation for Time
Series Forecasting," in *Proc. AAAI*, 2023.

[12] A. F. Ansari et al., "Chronos: Learning the Language of Time Series,"
*Transactions on Machine Learning Research*, 2024.

[13] A. Das et al., "A decoder-only foundation model for time-series
forecasting (TimesFM)," in *Proc. ICML*, 2024.

[14] L. Wang et al., "Text Embeddings by Weakly-Supervised Contrastive
Pre-training (E5 / multilingual-E5)," arXiv:2212.03533, 2022.

[15] T. Akiba et al., "Optuna: A Next-generation Hyperparameter Optimization
Framework," in *Proc. KDD*, 2019.

---

## 7. Appendix A — Data Dictionary, Schema, and Commands

### A.1 Database schema (PostgreSQL 16)

`raw_headlines` — one row per scraped headline (source of truth):
`id` (PK), `date`, `source`, `hour`, `popularity`, `headline` (Hebrew, UTF-8),
`created_at`, `headline_hash` (`md5(headline)`, stored). Unique on
`(date, source, hour, headline_hash)`.

`nlp_vectors` — one row per `(headline, model)`:
`id` (PK), `headline_id` (FK), `model_name`, six `relevance_*` SMALLINT
(0–10), `global_sentiment` SMALLINT (−10…+10), `validation_passed` BOOLEAN,
`processing_time_seconds`, `errors`, `created_at`. Unique on
`(headline_id, model_name)`.

`headline_embeddings` — one row per `(headline, embed_model)`: 768-d float32
vector stored as raw bytes (`BYTEA`), no vector-extension dependency.

`daily_embedding_derived` — one JSONB row per `(date, embed_model)`: the
leak-safe 16 `embpca_*` + 8 `embclus_dist_*` features, with the recorded
`fit_cutoff`.

`embedding_pca_basis` — the persisted transform basis (scaler mean/scale, PCA
mean/components, KMeans centers) that projects headlines into the same space.

`model_registry` — one row per trained candidate: version (unique), family,
hyper-parameters (JSONB), OOS metrics (ROC-AUC + CI, MCC, accuracy, n),
serialized artifact (`BYTEA`; joblib / torch state-dict / ensemble /
reforecast), feature-column contract, `is_active` (partial-unique: at most
one), `activated_by` (auto | manual), timestamps.

`model_predictions` — the live inference log: `(date, model_version)` unique,
prediction, confidence, `actual` (NULL until settled).

`champion_full_eval` — per-day in-sample evaluation of the champion (see
§3.7).

`narrative_sim`, `narrative_sim_graph`, `narrative_sim_report` — cached
narrative-simulation outputs consumed by the Simulator tab.

### A.2 Score-scale reference

- **Relevance** (six columns): integer 0–10; higher = more relevant to that
  category.
- **Sentiment** (`global_sentiment`): integer −10 (very negative) … +10
  (very positive); 0 = neutral/mixed.
- **`validation_passed`**: TRUE = parseable, in-range LLM output. Always
  filter on TRUE for analysis.

### A.3 Reproduction commands

```bash
# 0 — database (schema auto-initialises from scripts/init_db.sql; migrations 001–007 are idempotent)
docker compose up -d

# 1 — scrape headlines
cd mivzakim_scraper && uv sync && uv run playwright install firefox && uv run python main.py

# 2 — score unscored headlines into nlp_vectors (gap-only, backend-aware)
cd processing_engine && uv sync
uv run python ../scripts/process_headlines.py --fast --unscored-any-model --concurrency 4

# 3 — research pipeline: features → embeddings → models → leaderboard
uv sync --extra ml --extra embed --extra finance          # at repo root
uv run python -m sentisense.pipeline --from features       # leakage-safe, ≤ 2023-10-07

# 4 — full comparison leaderboard (server, run in tmux)
uv sync --extra ml --extra finance --extra embed --extra tft --extra chronos
uv run python scripts/pipeline_compare.py --seq-trials 30 --pf-trials 12 --xgb-trials 60

# 5 — registry training over the full zoo → auto-activate the champion
uv run --extra finance --extra ml --extra tft --extra chronos python scripts/train_registry.py \
    --trials 100 --seq-models lstm,gru,tcn,patchtst --seq-trials 40 --seq-seeds 3 \
    --forecasters chronos,timesfm,tft,nhits,nbeats --select-metric oos_accuracy

# 6 — one nightly cycle by hand (normally run by cron)
uv run --extra finance --extra ml python scripts/daily_live.py

# 7 — dashboard (on the DB/UI host)
cd ui/frontend && npm install && npm run build && cd ../..
uv run --extra ui --extra finance --extra ml python -m ui.app     # serves on :3000
```

### A.4 Repository map

```
mivzakim_scraper/   Playwright scraper for mivzakim.net (Hebrew news)
processing_engine/  LLM scoring pipeline (fast single-prompt + 7-agent LangGraph)
sentisense/         forecasting + serving package
  constants.py        cutoff, model name, score contract
  config.py           modeling/HPO knobs (env-overridable)
  db/                 SQLAlchemy engine (env-only DSN) + migrations 001–007
  ingest/             backfill · score · coverage report
  features/           leak-safe daily dataset assembly (incl. serving mode)
  embed/              multilingual-e5 embeddings · derived PCA/cluster block · basis
  cluster/            causal expanding-window narrative clustering
  models/             sequence datasets, train harness, model zoo, baselines
  hpo/                resumable Optuna HPO + sacred-holdout eval
  serve/              model registry + champion serving (fallback-safe)
  sim/                narrative-simulation client, cache, graph API
  pipeline.py         research orchestrator
ui/                  FastAPI backend (ui/app.py, ui/queries.py) + React SPA (ui/frontend)
evaluation/          LLM-scoring benchmark vs golden dataset
scripts/             init_db.sql · backfill · process/retry/standardize ·
                     pipeline_compare · train_registry · daily_live ·
                     settle_predictions · compute_full_eval · build_embedding_derived
ops/                 crontab template · pm2 process config · startup script
tests/               pytest — cutoff, leakage, calendar rollover, registry serve,
                     projection math, daily orchestration
docs/               RUNBOOK · LIVE_RUNBOOK · MODEL_ZOO · DATA_HANDOFF
*.ipynb             eda · poc · lstm_forecaster · tuning · transformer_forecaster ·
                     sentisense_analysis · timesfm_explainability
```

---

## 8. Appendix B — Live Deployment Runbook (summary)

The full operational document is `docs/LIVE_RUNBOOK.md`; this appendix
summarizes the deployed configuration.

**Hosts.** GPU compute node (pipeline, LLM scoring via local Ollama, registry
training) and a database/UI host (PostgreSQL 16, FastAPI + SPA under a
process supervisor on port 3000). All cross-host traffic is
database-mediated; the only required configuration on each host is
`SENTISENSE_DATABASE_URL` plus the scoring-backend variables.

**Schedule.** Cron on the compute node runs `scripts/daily_live.py` after the
TASE close (the orchestrator itself skips Fri/Sat and listed holidays), and
`scripts/settle_predictions.py` fills in realized outcomes. Registry
re-training (`scripts/train_registry.py`) is run periodically, not nightly —
champion serving is decoupled from training by design.

**Failure modes and their handling.**

| Failure | Behavior |
|---|---|
| Registry table missing / artifact incompatible | champion falls back to the pinned XGBoost config; logged loudly |
| Scoring backend switched (vLLM ↔ Ollama) | orchestrator selects per-backend flags; gap-only scoring prevents re-scoring covered history |
| Simulation service down | Simulator tab disables live runs and serves cached graphs, with an explicit banner |
| No WebGL in the operator's browser | 3-D views render through the software-3D fallback |
| Freshly promoted champion (no live rows) | dashboard seeds metrics from the model's held-out evaluation and labels the eval/live split |

**Data-freshness contract.** Every dashboard panel maps to the pipeline
stage that produces its data (documented per-panel in the runbook); each
degrades to an explicit "no data" state rather than an error when its
producer has not yet run.

---

*Author name, submission date, screenshots for the figure placeholders, and
the final full-budget registry leaderboard export (Table/Figure 12) are to be
completed before submission.*
