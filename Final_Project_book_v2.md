# SentiSense - Forecasting the Next-Day Direction of the TA-125 Index from Hebrew-News Sentiment

by
Omri Shlezinger, Nadav Idelsohn, Orian Aziz, Amir Katz

Approved by the supervisor: Oshrit Shtussel

Submitted to the Computer Science Faculty of College of Management
Rishon LeZion, August 2026

Version 2 - regenerated 2026-08-13 against the live system

---

## Acknowledgments

We would like to express our gratitude to our supervisor, Oshrit Shtussel,
for her guidance throughout this project. We would also like to thank our
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
headlines from `mivzakim.net` going back to ~2010. (2) A **processing engine**
sends every headline through an LLM, which scores it on six relevance
categories (politics, economy, security, health, science, technology) and one
global sentiment value (-10...+10), producing a corpus of roughly **3.1
million scored headlines** (3,099,081 raw headlines from 64 distinct sources,
spanning August 2010 to August 2026) in PostgreSQL. (3) A
**feature-engineering layer** aggregates the per-headline scores into
leakage-safe daily feature vectors, joined with market data (TA-125 OHLC, the
VTA-35 volatility index, S&P 500, the Nasdaq Composite, VIX, Brent crude,
USD/ILS), an **overnight-gap** (`ovn_`) feature block that carries the
US-session signal into the next Tel-Aviv trading day, multilingual headline
**embeddings**, a leak-safe **PCA/clustering block** derived from the daily
embedding centroid, and causal **narrative-clustering** features. (4) A
**forecasting layer** trains and hyperparameter-tunes a large model zoo -
gradient-boosted trees, recurrent and convolutional sequence classifiers,
transformer forecasters, and zero-shot foundation models - and persists every
candidate, with its out-of-sample metrics and serialized weights, into a
**model registry** that automatically activates the best model (with a manual
override). (5) An **operations layer** runs the whole chain as a nightly job
on a GPU node - scrape -> score -> embed -> derive -> predict -> settle - and
serves the results through a **live web dashboard** (prediction hero, model
metrics, exploratory analytics, a 3-D news-centroid explorer, per-source
"persona" votes, and a narrative simulator), deployed behind a site-wide
login and TLS at `sentisens.cs.colman.ac.il`.

Every research stage is engineered to be **leakage-safe**: all scalers, PCA,
and clustering are fit on the training fold only; splits are strictly
chronological (70/15/15 train/validation/test); and the test tail is scored
exactly once, after all tuning decisions are made.

The central empirical finding is corroborated across **multiple independent
experiment tracks** - a tree-model proof of concept, an extensive feature-set
comparison with walk-forward and multi-seed robustness checks, a nine-model
transformer zoo, sequence-model HPO, a hardened end-to-end package run of
**40+ tuned model x data-type cells**, and finally the productionized registry
run over the full zoo (**29 registered versions across 13 model families**,
out-of-sample ROC-AUC spanning 0.427-0.573). The system's best models
consistently beat the no-skill majority-class baseline: the registry's
automatic best-model selection on out-of-sample ROC-AUC activated a **TCN
champion** (`tcn-20260702-1351`) with **OOS ROC-AUC 0.573 and OOS accuracy
0.568 on 382 held-out days**, while PatchTST remains the single
best-accuracy grid cell (**0.578 on 327 days**) against a window base rate of
~0.55. (Earlier snapshots of this document and of the live-inventory notes
named PatchTST and XGBoost respectively as champion; the registry's
ROC-AUC-based selector has since settled on the TCN, and this version reports
that state.) The champion has served live since 6 July 2026; its prospective
record so far - **46.4% directional accuracy over 28 settled predictions** -
is statistically indistinguishable from chance at this small sample size
(wide confidence interval), and is reported alongside, not in place of, the
research metrics. In a domain where even a small, consistent edge (55-58% vs
a ~53% base rate) is hard to achieve and valuable, the held-out results
approach statistical significance and represent a meaningful contribution -
one the live track will continue to test. The project's contribution is
threefold: a **reusable, reproducible, leakage-hardened research pipeline**
for news-driven financial forecasting; a rigorous empirical result that
quantifies the predictive value of LLM-scored Hebrew-news sentiment across a
broad model zoo; and a **complete production system** - registry, nightly
orchestration, an access-controlled public dashboard - that keeps extending
the out-of-sample record prospectively on live data.

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
   - 3.8 Security and Access Control
4. Results and Analysis
   - 4.1 Experimental Setup
   - 4.2 Presentation of Results
     - 4.2.1 Tree-model proof-of-concept (`poc.ipynb`)
     - 4.2.2 LSTM feature-set vs PoC study (`compare_lstm_features_with_poc.ipynb`)
     - 4.2.3 LSTM base forecaster (`lstm_forecaster.ipynb`)
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
7. Appendix A - Data Dictionary, Schema, and Commands
8. Appendix B - Live Deployment Runbook (summary)

---

## List of Figures

- Figure 1: SentiSense end-to-end pipeline (Section 1.6)
- Figure 2: System architecture - modules and data flow (Section 3.1) *(placeholder)*
- Figure 3: Two-host deployment topology (Section 3.5) *(placeholder)*
- Figure 4: Leakage-safe chronological train/validation/test split (Section 3.3) *(placeholder)*
- Figure 5: Model registry lifecycle - train, register, select, serve (Section 3.4) *(placeholder)*
- Figure 6: Dashboard - prediction hero and model-performance panel (Section 3.5) *(screenshot placeholder)*
- Figure 7: Dashboard - exploratory data-analysis panels (Section 3.5) *(screenshot placeholder)*
- Figure 8: 3-D daily news centroids, colored by KMeans cluster (Section 3.5) *(screenshot placeholder)*
- Figure 9: Single-day headline cloud in the shared PCA space (Section 3.5) *(screenshot placeholder)*
- Figure 10: Per-source persona votes vs the model's call (Section 3.5) *(screenshot placeholder)*
- Figure 11: Unified leaderboard - ROC-AUC vs accuracy scatter (Section 4.2.8) *(placeholder)*
- Figure 12: Models panel - registry leaderboard with the active champion (Section 4.2.9) *(screenshot placeholder)*

## List of Tables

- Table 1: Best result per experiment track vs its no-skill baseline (Section 4.2)
- Table 2: PoC tree-model 5-fold cross-validation accuracy (Section 4.2.1)
- Table 3: PoC chronological 80/20 holdout results (Section 4.2.1)
- Table 4: PoC holdout bootstrap 95% confidence intervals (Section 4.2.1)
- Table 5: Per-source feature-set holdout comparison (tree models) (Section 4.2.2)
- Table 6: Feature-group ablation (CatBoost) (Section 4.2.2)
- Table 7: LSTM base forecaster holdout result (Section 4.2.3)
- Table 7b: LSTM base forecaster holdout classification report (Section 4.2.3)
- Table 8: Transformer zoo final leaderboard vs baselines (Section 4.2.4)
- Table 9: Sequence-model tuning track - holdout results (Section 4.2.5)
- Table 10: Hardened-package score-LSTM final holdout (Section 4.2.6)
- Table 11: Unified out-of-sample leaderboard (Section 4.2.8)
- Table 12: Registry validation run - tree zoo OOS metrics (Section 4.2.9)
- Table 13: Active production champion - held-out evaluation (Section 4.2.9)

---

## Table of Abbreviations

| Abbreviation | Meaning |
|---|---|
| TA-125 | Tel-Aviv 125 stock index |
| VTA-35 | Tel-Aviv 35 Volatility Index |
| LLM | Large Language Model |
| NLP | Natural Language Processing |
| HPO | Hyper-Parameter Optimization (automated hyper-parameter search) |
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
| TLS | Transport Layer Security |
| HMAC | Hash-based Message Authentication Code |

---

## 1. Introduction

This chapter provides the background, defines the problem, and states the
objectives and scope of the project, setting the stage for the chapters that
follow.

### 1.1 Background

This project builds a news-sentiment-driven system that forecasts the
next-day close-to-close direction of the Tel-Aviv 125 (TA-125) index. Its
input is the daily stream of Hebrew breaking-news headlines; its output is a
concrete, dated, falsifiable directional call, produced every trading night
and settled against the realized close the following day.

The opportunity is specific and recent. Research in *behavioral finance* and
*NLP-for-finance* has established that the tone and topical mix of news
carry predictive information about subsequent market movements, particularly
for indices and over short horizons. Until recently, exploiting that
information in a non-English market meant building a language-specific
lexicon or training a supervised classifier per language. Modern LLMs remove
that barrier: a single prompted model can read a Hebrew headline and return a
structured, comparable score for six topical-relevance categories plus a
global sentiment value, at a throughput that makes a three-million-headline
corpus practical.

Most existing work in this area focuses on English-language sources (financial
newswires, Twitter/X, earnings calls). Hebrew-language news, and the Israeli
market specifically, are comparatively under-studied, and almost no published
work closes the loop from a research claim to a system that keeps producing
and scoring predictions after the paper is written.

SentiSense targets exactly that gap. It uses an LLM to turn a high-volume
Hebrew breaking-news feed into a structured daily signal, trains and tunes a
broad model zoo on strictly leakage-controlled chronological splits, selects a
champion through a database-backed model registry, and then operates that
champion as a **live, self-updating forecasting service** whose accuracy is
measured against each newly settled trading day.

### 1.2 Problem Statement

**Can a structured, LLM-derived sentiment signal extracted from Hebrew
breaking-news headlines predict the next-day close-to-close direction of the
TA-125 index, beyond what market data alone provides, and can that edge be
measured credibly enough to survive contact with production?**

The challenge has several specific difficulties:

1. **Leakage risk.** News, market, and target series share a calendar; naive
   feature engineering (e.g., using a same-day future return, fitting a scaler
   on the full series, or shuffling time) silently inflates results.
2. **Language and source heterogeneity.** Headlines are Hebrew, UTF-8, from
   many outlets of varying quality and volume, including a real weekend lull.
3. **Research-to-production gap.** A result that only exists in a notebook is
   not falsifiable going forward; keeping the claim honest requires serving
   the model daily and settling its predictions against reality.

### 1.3 Objectives

1. **Build a reproducible ingestion-and-scoring pipeline** that scrapes Hebrew
   headlines and scores each on six relevance categories plus a global
   sentiment, persisting the result in a relational database.
2. **Engineer leakage-safe daily features** combining the news scores with
   market and macro data - including an overnight-gap (`ovn_`) block driven
   by US-session moves (S&P 500, Nasdaq Composite `^IXIC`) - with
   embedding-derived and narrative-based signals.
3. **Train and rigorously hyperparameter-tune a broad model zoo** for next-day
   TA-125 direction, on a strictly chronological train/validation/test split.
4. **Quantify the predictive value honestly** using threshold-free and
   threshold-based metrics, against a no-skill majority-class baseline.
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
PCA/cluster block, causal narrative clustering, and an overnight-gap (`ovn_`)
feature block that carries the US-session signal (including the Nasdaq
Composite, `^IXIC`) into the next Tel-Aviv trading day; classification and
forecasting models with HPO; a comparison leaderboard; a model registry with
automatic/manual champion selection; nightly live operation on a two-host
deployment; a web dashboard; and site-wide access control (login page, signed
session cookies, and TLS termination) for the public deployment.

**Out of scope / limitations:**

- **Target.** The system predicts **close-to-close direction**. Overnight-gap
  information now enters as a *feature* block (`ovn_`), but neither the gap
  itself nor intraday-return magnitude is a prediction target.
- **Prospective evaluation.** The live track serves on the full timeline and
  its performance is reported separately from the research metrics; it
  accumulates prospectively, which is a strong guard against hindsight bias.
- **Intraday.** The system is daily-resolution; no tick or minute data.
- **Causality.** The work measures *predictive association*, not economic
  causation.
- **Data quirks.** A non-trivial fraction of "validated" LLM rows are
  all-zero (a known LLM failure mode treated as missing); the corpus mixes
  LLM scoring-model versions across disjoint date ranges (see Section 3.2,
  "scoring eras").

### 1.5 Methodology

The project follows a staged, gate-driven methodology:

1. **Ingest** Hebrew headlines (backward scrape to ~2010) into `raw_headlines`.
2. **Score** each headline with an LLM into `nlp_vectors` (7 scores +
   validation flag).
3. **Assemble** leakage-safe daily frames: daily-mean scores, per-source score
   pivots, sentiment×relevance interactions, multilingual embedding centroids,
   an embedding-derived PCA/cluster feature block, causal narrative-cluster
   features, and a finance/market block including overnight-gap (`ovn_`)
   features.
4. **Split** chronologically (70/15/15 train/validation/test) with all
   transforms fit on the train slice only.
5. **Model & tune** a zoo of classifiers and forecasters with Optuna HPO,
   scoring every trial on the validation slice only.
6. **Evaluate** every model once on the same held-out out-of-sample test tail
   using ROC-AUC, F1, accuracy, balanced accuracy, and MCC, plus a backtest
   overlay.
7. **Compare** all cells in a single auto-generated leaderboard.
8. **Register & select**: persist each tuned model (weights + OOS metrics)
   into the model registry; activate the best automatically, allow manual
   override from the dashboard.
9. **Operate**: run the nightly pipeline (scrape, score, embed, derive,
   predict, settle) on a schedule, serve the active champion's prediction,
   and fold each settled day into the champion's cumulative live score.

### 1.6 Organization of the Project Book

- **Chapter 2** reviews relevant literature on news-driven financial
  prediction and LLM-based sentiment extraction.
- **Chapter 3** details the system architecture, data collection and
  preprocessing, feature engineering, the model registry, the live serving
  layer and dashboard, implementation, evaluation metrics, and security and
  access control.
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
*Figure 1: SentiSense end-to-end pipeline - research stages (top) feeding the
production loop (bottom).*

---

## 2. Literature Review

### 2.1 Overview of Relevant Literature

The project draws on four strands of prior work.

**News sentiment and market prediction.** A long line of research links the
tone of financial and general news to subsequent market movements. Tetlock [1]
showed that media pessimism predicts downward pressure on prices and
reversion, establishing news tone as a market-relevant variable. Bollen et
al. [2] linked aggregate mood derived from social media to movements in the
Dow Jones. The consistent theme is that a genuine but modest edge is
attainable, and that careful, leakage-free evaluation is essential to
distinguish it from artifact - exactly the posture this project adopts.

**Lexicon vs. model-based sentiment.** Domain-specific lexicons such as
Loughran and McDonald [3] demonstrated that general-purpose sentiment
dictionaries mislabel financial text, motivating domain-aware scoring. Modern
LLMs generalize this idea: instead of a fixed lexicon, a prompted model
performs context-aware topical-relevance and sentiment scoring, and - relevant
here - does so across languages, including Hebrew, without a hand-built Hebrew
lexicon.

**Sequence and foundation models for forecasting.** On the modeling side, the
project surveys the standard time-series toolkit: gradient-boosted trees
(XGBoost) as strong tabular baselines; recurrent networks (LSTM, GRU) and
temporal convolutions (TCN) for sequence classification; transformer
forecasters such as the Temporal Fusion Transformer and PatchTST; deep
interpretable forecasters N-BEATS and N-HiTS; and zero-shot foundation
forecasters Chronos and TimesFM. Multilingual sentence embeddings
(multilingual-E5) provide the Hebrew-aware vector representations used for the
embedding and narrative-clustering features, and Optuna provides the resumable
hyper-parameter search used throughout. All of these are listed with their
originating publications in the "Software and Model Families" part of the
References.

**Persona- and agent-based narrative simulation.** A fourth, more recent
strand uses LLM-driven agents as simulated participants in a social or market
process: generative-agent societies have shown that prompted personas exhibit
believable, internally consistent behavior, and MiroFish-style multi-agent
market simulations extend the idea to news-conditioned personas that argue
over the day's events. SentiSense adopts this strand operationally rather
than as a competing predictor: a nightly narrative simulator instantiates
per-source personas on a local LLM, runs them over the day's headlines, and
archives the resulting narrative graph and report alongside the champion's
prediction - a qualitative, explanatory companion to the quantitative signal.

The research gap this project addresses: most prior work is English-centric,
often under-controls for leakage, and rarely closes the loop from a research
claim to a *prospectively evaluated* live system. SentiSense contributes a
**Hebrew-news, LLM-scored, strictly leakage-controlled** evaluation across a
broad model zoo, reports the full result rather than a selected slice of it,
and then keeps the evaluation running in production, where each new trading
day extends the out-of-sample record.

---
## 3. System Design and Implementation

### 3.1 System Architecture

The system is organized as loosely-coupled modules communicating through a
PostgreSQL 16 database, so each stage can be developed, re-run, and verified
independently. Following a repository re-organisation, all modules now live
under a single clean top-level layout, with the `sentisense/` Python package
as the core financial-modeling layer:

| Module | Purpose | Entry point |
|---|---|---|
| `sentisense/` | Core package - submodules `ingest`, `embed`, `cluster`, `features`, `models`, `hpo`, `serve`, `sim`, `db`: features, embeddings, clustering, models, HPO, serving, simulation | `python -m sentisense.pipeline` |
| `sentisense/serve/` | Model registry + champion serving | `registry.py` / `champion.py` |
| `processing_engine/` | LLM scoring (6 relevance + sentiment) | fast pipeline / `process_single_observation` |
| `mivzakim_scraper/` | Scrape Hebrew headlines (Playwright + Firefox) | `python main.py` |
| `scripts/` | Operational CLIs: schema, backfill, scoring, retry, standardize, registry training, daily orchestration, settlement, sim archiving | `python scripts/<name>.py` |
| `ui/` | FastAPI backend + React SPA dashboard | `python -m ui.app` |
| `ops/` | Deployment artifacts: crontab, pm2 process file, nginx TLS reverse-proxy config | `crontab ops/crontab.txt` |
| `evaluation/` | Benchmark LLM scoring against a golden dataset | `python -m evaluation.evaluate` |
| `notebooks/` | The research lineage (EDA, PoC, tuning, architecture and explainability studies) | Jupyter |
| `docs/` | Runbooks, live inventory, weekly summaries | - |
| `tests/` | Automated test suite - 23 test files, 133 tests | `pytest` |
| `external/MiroFish` | Multi-agent narrative-simulation engine (git submodule) | driven via `sentisense/sim/` |

> **[Figure 2 placeholder: block diagram of the modules above with the
> database at the center; arrows labeled with the table each stage reads or
> writes.]**

In production the modules are deployed across **two hosts**: a GPU compute
node runs the nightly pipeline, and a front machine runs the database and the
dashboard, with **nginx terminating TLS on port 443** (the institutional
wildcard certificate) and reverse-proxying HTTP and WebSocket traffic to the
process-supervised FastAPI UI on an internal port. The topology is detailed
in Section 3.5.

**Design principles.**

- **Database as the contract.** All inter-stage data flows through Postgres
  tables (`raw_headlines`, `nlp_vectors`, `headline_embeddings`,
  `headline_vectors` - a pgvector similarity store with an HNSW cosine
  index - `daily_embedding_derived`, `embedding_pca_basis`, `model_registry`,
  `model_predictions`, `champion_full_eval`, `narrative_sim*`, and
  `llm_requests` - a database-backed LLM job queue added by schema migration
  008), decoupling scraping, scoring, modeling, serving, and the UI. The
  dashboard host never runs heavy compute; it only reads the database - and,
  because the firewall between the two hosts passes only Postgres, the
  database also doubles as the transport layer for interactive LLM requests
  (the UI enqueues a row, the GPU-side worker claims and answers it). The
  legacy `daily_features` table from the original Phase-3 design still exists
  in the schema but is **empty and unused**; the feature builders assemble
  frames directly from the scored and embedded tables instead.
- **Single source of truth for constants.** The active model name and the
  score-column contract live in `sentisense/constants.py`, so no magic strings
  leak into feature or model code.
- **Optional, layered dependencies.** Heavy ML/embedding/forecasting libraries
  are `pyproject.toml` *extras* (`ml`, `embed`, `finance`, `tft`, `chronos`,
  `ui`), so early stages install lightly and torch/CUDA wheels are pinned for
  reproducibility.
- **Leakage-safety as an architectural invariant**, enforced at every layer
  (see Section 3.3).
- **Fail-safe serving.** Every serving path is wrapped so that a missing
  table, an incompatible artifact, or an unreachable auxiliary service
  degrades to a well-defined fallback (pinned champion, cached data, explicit
  "no data" states) rather than a broken nightly run or a blank dashboard.

### 3.2 Data Collection and Preprocessing

**Collection.** The scraper drives a headless Firefox via Playwright over
`mivzakim.net`, scraping *backward* in time (`scripts/backfill_history.py`)
from the most recent day toward 2010, and *forward* daily
(`scripts/daily_scrape_to_db.py`, covering today and yesterday). Each headline
yields a row in `raw_headlines`: date, source outlet, hour, popularity class,
the Hebrew text, and an ingestion timestamp. Deduplication uses a stored
`md5(headline)` hash (Hebrew strings exceed B-tree index limits) under a
unique key of `(date, source, hour, headline_hash)`. As of August 12, 2026
the table holds **3,099,081 headlines from 64 distinct outlets**, spanning
August 29, 2010 through August 12, 2026.

**Scoring.** The processing engine sends each headline to an LLM. A **fast
single-prompt path** produces all seven scores in one structured call; a
legacy **seven-agent LangGraph path** (one ReAct agent per relevance category
plus one for sentiment) exists for research and evaluation. Each result is a
vector of six relevance integers (0-10), one global sentiment integer
(-10...+10), and a `validation_passed` flag, written to `nlp_vectors`. The
scored corpus comprises **3,167,851 rows in `nlp_vectors`** (slightly more
than the headline count, because the model transitions described below left
some headlines with a row from more than one scoring model) and **3,100,946
cached embeddings** in `headline_embeddings`.

**How the headline-scoring model was chosen: the golden-dataset quality
gate.** The scoring LLM is the single most consequential choice in the whole
pipeline - every downstream feature is a function of its output - so it was
selected by measurement rather than by reputation. The `evaluation/` package
implements a standing quality gate that any candidate scoring model must pass
before it is allowed to write a single production row.

The gate works against a **hand-labeled golden dataset**
(`evaluation/golden_dataset.csv`): a fixed sample of Hebrew headlines
annotated by the team with the same 7-dimensional target the production
prompt asks for (six relevance categories 0-10 plus a global sentiment
-10...+10). A candidate model is run over that sample through the *exact*
production prompt and parsing path, so the gate measures the deployed
configuration rather than an idealized one. Three complementary criteria are
then computed per category (`evaluation/metrics.py`, reported by
`python -m evaluation.evaluate`):

- **Mean absolute error (MAE)** against the human label - how far off the
  model is, in score points, on average.
- **Within-1 accuracy** - the fraction of scores landing within +/-1 point of
  the human label. This is the criterion that matters most in practice,
  because the daily features are aggregates: a model that is reliably within
  one point produces a stable daily mean even when individual calls differ.
- **Pearson r** per category - whether the model *orders* headlines the way a
  human does, which is what the relevance features actually encode.

A candidate is accepted only if it clears the gate on all three criteria
across categories, and if its structured output parses reliably enough to
keep the `validation_passed` rate high. This is how `mistral-small-4` was
selected for the historical backfill and how the locally hosted `gemma4` was
qualified before it took over nightly scoring; a model that scores well but
emits unparseable JSON fails the gate just as surely as an accurate-but-slow
one fails the throughput requirement. Because the harness is committed and
re-runnable, re-qualifying a future scoring model is a single command, not a
research project.

**Scoring eras.** The corpus was scored in two eras, both recorded explicitly
in the `model_name` column so no row's provenance is ambiguous:

- **Historical era** - the bulk backfill was scored by `mistral-small-4`
  served on a remote vLLM cluster (50-headline batched completions at high
  concurrency), after earlier rows from older models were re-standardized
  onto it (`scripts/standardize_to_latest_model.py`).
- **Live era** - the *ongoing* nightly scoring phase, i.e. everything scored
  from the production switch-over onward. In this phase new headlines are
  scored by a **locally hosted Ollama model (`gemma4`)** running on the
  project's own GPU node, **one headline per structured call**, because
  batch-JSON and agentic modes proved unreliable for that backend
  (Section 4.2.9). Scoring is **gap-only** (`--unscored-any-model`): each
  night the job scores only headlines that no model has scored yet, so
  nothing already covered is re-scored and the two eras stay disjoint by
  construction.

The dataset builders consume *validated rows from any era*, and every
era-sensitive UI query prefers the active model's row but falls back to any
validated row, so the system remains correct across the era boundary. The
statistical implication of the seam (features scored by different LLMs on
different date ranges) is discussed in Section 4.5 and Section 5.

**Quality control and known quirks** (documented in `DATA_HANDOFF.md`):

- **All-zero "validated" rows.** The LLM sometimes emits all-zeros when it
  cannot categorize a headline; the validator accepts it because all values
  are in range. These are treated as missing data.
- **Weekend lull.** Saturday volume is genuinely low (Israeli weekend), not a
  data gap.
- **Encoding / timezone.** All text is UTF-8 Hebrew; event dates/hours are
  Asia/Jerusalem while `created_at` is stored as UTC `TIMESTAMPTZ`.

### 3.3 Feature Engineering

Leakage-safe feature assembly (`sentisense/features/dataset.py`) is the heart
of the preprocessing and the project's most important engineering
contribution. The module builds daily modeling frames with defense-in-depth
against leakage:

- **Event date, never ingestion time.** All splits use `raw_headlines.date`
  (when the news happened), never `created_at` (when the row was written), so
  a late backfill can never shift a headline into a window it does not belong
  to.
- **Trading-calendar rollover.** News published on a non-trading day is rolled
  *forward* to the next trading day via `np.searchsorted(side='left')`, so it
  is attributed to the first session that could actually react to it;
  market/FX/volatility series are forward-filled. The pipeline currently
  treats the TASE trading week as **Sunday-Thursday** and skips **Friday and
  Saturday** (the constant `_TASE_TRADING_WEEKDAYS = {6, 0, 1, 2, 3}` in
  `scripts/daily_live.py`, with the Friday/Saturday-to-Sunday overnight
  rollover implemented in `sentisense/features/dataset.py`). Note that the TASE
  migrated to a **Monday-Friday** global trading week on **January 5, 2026**,
  so going forward the non-trading days are Saturday and Sunday; adapting the
  weekday constant and the rollover rule to the new schedule is identified as
  future work in Section 5.
- **Causal price features.** TA-125 features (lagged log-returns 1-7, 5d/20d
  rolling stats, Wilder RSI-14, 20-day volume z-score, day-of-week one-hots)
  all use `.shift(>=1)`, i.e. they can only read strictly past rows.
  Cross-asset features (S&P 500, Nasdaq, VIX, Brent, USD/ILS, VTA-35) are
  lagged log-returns only.
- **Train-only scaling.** `StandardScaler` (and PCA, scoped by column prefix
  to the embedding block) is fit on the **train slice only**, then applied to
  validation and test.
- **Honest target.** `Target = (TA125_Price.shift(-1) > TA125_Price)`. A
  `shift(-1)` pulls the *next* row's value into the current row. That is
  exactly what a next-day label should be, and it is the only place in the
  package where `shift(-1)` is used - using it to build a *feature* would let
  the model read the future. The trailing row with no next-day price is set to
  NA and dropped in research mode; in *serving* mode
  (`keep_unlabeled=True`) that same row is retained with a `Target = -1`
  sentinel, so the model trains only on real labels and **forward-predicts**
  the sentinel day.
- **Live price extension.** The static TA-125 CSV is extended at build time
  with live closes fetched from the exchange feed, so the serving frame always
  reaches the current trading day. (A known issue at the time of writing: the
  repository re-organisation moved the static finance CSVs into `evaluation/`,
  while `sentisense/constants.py` still resolves `TA125_CSV` and `VTA35_CSV`
  against the repository root; the path constants need to be updated to match
  the new layout.)

> **[Figure 4 placeholder: timeline diagram - chronological train/validation/test
> split and the live serving region with the Target=-1 sentinel day.]**

**Overnight block (`ovn_*`).** The next-day contract the system actually
serves is a decision taken at the *open* of day T+1, and by that time the
major global markets have already closed their day-T sessions - after the
TASE close but before its next open. `add_overnight_features` therefore
appends, for each global cross-asset, the **day-T close-to-close log-return
with no additional shift** (`ovn_<asset>_ret`) and a two-day momentum sum
into the open (`ovn_<asset>_2dret`). These values are known at open(T+1) but
would be a leak for a close(T) decision, which is exactly why they live in a
separate, explicitly flagged block behind `build_datasets(overnight=...)`
rather than inside the always-lagged cross-asset features; they are never
derived from TA-125 itself, so they cannot peek at the target. The Nasdaq
(`^IXIC`) was added to the cross-asset set specifically for this block, as a
tech-heavy overnight driver of the TA-125 gap. The production champion is
trained with the overnight block **enabled**.

**Embedding-derived block.** This block gives each day a small set of features
describing *where that day's news sits in a map of news topics learned only
from earlier days*. Concretely: each headline is embedded once with a
Hebrew-aware multilingual model (`intfloat/multilingual-e5-base`, 768-d) and
cached in `headline_embeddings`. Per trading day, the mean of the day's
headline vectors forms the **daily news centroid**. A transform basis
(StandardScaler, then PCA to 16 components, then KMeans with 8 clusters) is fit
**once on a past window only** (dates at or before a recorded `fit_cutoff`) and
then applied to every date, yielding **24 features per day: 16 PCA coordinates
(`embpca_*`) and 8 distances to the KMeans cluster centers
(`embclus_dist_*`)**, stored in `daily_embedding_derived`. Fitting the basis on
past data only is what makes the block leak-safe: a later out-of-sample day is
projected through a map it never helped build, and the `fit_cutoff` column
records that boundary in-band so it can be audited. The fitted basis itself
(scaler statistics, PCA components, and cluster centers) is persisted to
`embedding_pca_basis`, which lets the dashboard project *individual headlines*
into exactly the same 16-dimensional space the models consume (Section 3.5).

**Causal narrative clustering** (`sentisense/cluster/narrative.py`). For each
trading day *T*, a MiniBatch-KMeans model is fit **only on embeddings strictly
before T** (expanding window with a refit cadence), then day-T headlines are
*assigned* with that past-fit model, yielding `dominant_cluster_ratio` and
normalized `cluster_entropy` without any look-ahead.

**Feature views.** Three views are produced: a **daily-mean** frame
(tree-model shape), a **per-source** pivot frame (sequence-model shape), and a
**fused** frame combining per-source scores with the daily e5 centroid and the
embedding-derived block (~970 columns) - the view the production champion
serves on.
### 3.4 Modeling, Hyper-Parameter Optimization, and the Model Registry

**Model zoo.** The forecasting layer evaluates three families under one
leak-safe evaluation contract. That contract is worth stating precisely,
because every number in Chapter 4 depends on it: the data is split
**chronologically into 70% train / 15% validation / 15% test**;
**hyper-parameter optimization (HPO) - the automated search over model
settings, run here with Optuna - scores every trial on the validation slice
only**; and the **test tail is scored exactly once**, after tuning is
finished. Because no tuning decision ever sees the test slice, the reported
out-of-sample number is an honest estimate rather than a search artifact.

- **Tree classifiers** - XGBoost, LightGBM, CatBoost (Optuna-tuned; the
  winner is refit on all labeled history and serialized with `joblib`).
- **Torch sequence classifiers** - LSTM, GRU, TCN, PatchTST over windowed
  per-source/fused features (Optuna studies are stored *in the database* and
  therefore resumable; the winner is refit and serialized as a
  `state_dict` bundle that also carries its scaler statistics, window length,
  and feature order).
- **Forecaster / foundation models** - TFT, N-HiTS, N-BEATS
  (pytorch-forecasting), and the zero-shot foundation models Chronos and
  TimesFM. These carry no persistable artifact; they are registered as
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

**The July 2026 registry census.** The full-budget training run
(`scripts/train_registry.py`, all models trained 2026-07-02) populated the
registry with **29 versions across 13 families** - XGBoost, LightGBM,
CatBoost, LSTM, GRU, TCN, PatchTST, TFT, N-HiTS, N-BEATS, Chronos (zero-shot
and tuned), and the top-3 soft-vote ensemble - with held-out OOS ROC-AUC
spanning 0.4266 to 0.5731 across the registry.

**Auto-selection outcome.** Run with `--select-metric oos_roc_auc`, the
auto-best selector activated **`tcn-20260702-1351`** - a TCN over the fused
feature set (FULL regime, overnight features on), with OOS ROC-AUC **0.5731**
and OOS accuracy 0.5681 on n = 382 held-out days - as the serving champion on
2026-07-06. The grid's best-*accuracy* cell, PatchTST
`patchtst-20260702-1351` (0.5780 on n = 327), remains registered but was not
selected: its ROC-AUC of 0.4795 ranks it near the bottom on the selection
metric, a concrete illustration of why both metrics are always reported side
by side (Section 3.7). Earlier internal documents named PatchTST and the
pinned XGBoost fallback as champion; both reflect pre-activation snapshots,
and the registry's single active row is the authoritative record.

**Challenger promotion gate.** Beyond auto-selection at training time, the
system carries a standing champion/challenger mechanism
(`scripts/challenger_hpo.py`): a fresh Optuna-tuned XGBoost challenger is
trained and evaluated against the incumbent **on the same chronological
last-15% OOS tail**, and `should_promote` promotes it **only** if the ROC-AUC
gain is at least 0.02 (`--min-auc-gain`), the challenger's MCC has not
regressed, and the OOS sample has at least 200 days (`--min-n`). On promotion
the pinned-champion config (`models/champion.json`) is overwritten with a
version bump and the decision - promoted or not, with both scorecards - is
appended to `logs/promotions.jsonl`, so every change of the fallback champion
is auditable. The gate's decision logic is unit-tested in isolation
(`tests/test_promotion_gate.py`). The job ships as an optional weekly cron
entry (Thursdays), disabled by default.

**Training cadence: periodic, not nightly.** Registry training and nightly
serving are deliberately decoupled, and the distinction matters for reading
Chapter 4. `scripts/train_registry.py` tunes and registers the **whole zoo**,
and it is run **periodically** - not every night. The nightly job does *not*
re-tune and does *not* train the zoo: it loads whichever model the registry
marks active and predicts with it, and only the cheap pinned XGBoost fallback
is refit on all labeled history each night. Promoting a different model is a
database operation (activate a row), not a retraining run.

> **[Figure 5 placeholder: registry lifecycle diagram - HPO → OOS evaluation →
> register (weights + metrics) → auto-select / manual override → nightly
> serve.]**

**Champion serving** (`sentisense/serve/champion.py`). The nightly predictor
loads whatever the registry marks active and dispatches on its artifact
format: `joblib` models predict directly on the aligned feature row; `torch`
bundles are rebuilt from their `state_dict` (loaded with `weights_only=True`
for safety) and windowed over the recent feature history; `ensemble` entries
rank-normalize and average their members. A **pinned XGBoost champion**
(`xgb-fused-full-v1`; versioned JSON config, retrained on all labeled history
each night) acts as the guaranteed fallback: any failure in the registry path
logs loudly and falls back, so the daily prediction never silently breaks. The
pinned fallback is not hypothetical - it served the system's first three live
predictions (from 2026-04-29) before the registry activation of 2026-07-06.

### 3.5 Live Operation: Orchestration, Serving, and the Dashboard

**Deployment topology.** The system runs on two machines:

- a **GPU compute container** (NVIDIA RTX 4090, 24 GB; repository mounted at
  `/tf/Data-Science-Final-Project`, container clock UTC) that runs the nightly
  pipeline - scraping, LLM scoring (local Ollama `gemma4`), embedding, derived
  features, registry training, and the champion prediction - plus the LLM
  request-queue worker and the nightly narrative simulation; and
- a **front machine** (internal 10.10.248.109, external 193.106.55.109) that
  runs PostgreSQL 16, MongoDB (port 21771), the dashboard (FastAPI + built
  React SPA, supervised by pm2 as app `sentisense-ui` on port 3000), and an
  nginx reverse proxy terminating TLS on port 443 (Section 3.8). The public
  hostname `sentisens.cs.colman.ac.il` awaits its public DNS A record.

The two communicate **only through the shared database**: the institutional
firewall passes nothing but PostgreSQL between them, so the database doubles
as the transport layer - the compute node writes predictions and simulation
artifacts, the dashboard reads them, and even the dashboard's LLM features
travel as queued rows (the `llm_requests` mechanism below). This decoupling
means the UI stays up even when the compute node is retraining, and the
pipeline is indifferent to the UI.

> **[Figure 3 placeholder: deployment topology - GPU container (cron
> pipeline, Ollama LLM, queue worker) → shared PostgreSQL ← front machine
> (nginx TLS → pm2 FastAPI/SPA, MongoDB).]**

**Nightly orchestration** (`scripts/daily_live.py`, scheduled via cron after
the TASE close). The orchestrator chains five stages with a lock file (no
double runs), per-stage logging, and a status JSON consumed by the dashboard's
health banner: **scrape** (today + yesterday), **score** (gap-only; flags
selected automatically per LLM backend), **embed** (new headlines only),
**derive** (refresh the embedding-derived block and persist the basis), and
**predict** (the active champion forward-predicts the sentinel day; the
result is upserted into `model_predictions`). A companion job fifteen minutes
later **settles**: yesterday's prediction is compared with the realized close
and its `actual` field is filled (`scripts/settle_predictions.py`). The
orchestrator self-skips non-trading days: as deployed it treats
Sunday-Thursday as the TASE trading week and skips Friday and Saturday, plus a
configurable holiday list (see the trading-calendar note in Section 3.3).

The full nightly schedule, across both machines:

| Host (clock) | Time | Job |
|---|---|---|
| GPU container (UTC) | 15:30 | `scripts/daily_live.py` - scrape → score → embed → derive → predict |
| GPU container (UTC) | 15:45 | `scripts/settle_predictions.py` - fill `model_predictions.actual` |
| GPU container (UTC) | 17:00 | `scripts/sim_daily.py --backfill 3` - nightly persona simulation |
| GPU container (UTC) | @reboot | `scripts/llm_worker.py` - LLM request-queue worker |
| Front machine (Asia/Jerusalem) | 17:20 | `scripts/archive_sims_to_mongo.py --days 14` - version simulations into MongoDB |
| Front machine (Asia/Jerusalem) | 18:30 / 18:45 | `sentisense_daily.sh` / `sentisense_settle.sh` wrapper entries |

**Nightly narrative simulation.** `scripts/sim_daily.py` generates the
Simulator's content every night on the GPU container: for each trading day it
builds a **deterministic per-outlet stance graph** - each news source becomes
one agent whose stance, volume, and agree/disagree edges are computed from
that outlet's scored headlines - and then asks the local `gemma4` model to
write the per-agent statements and a roundtable-style markdown report. Graph
and report are idempotently upserted into `narrative_sim_graph` and
`narrative_sim_report`, the same tables the dashboard reads. Because a re-run
of a day overwrites its PostgreSQL row, the front machine's
`scripts/archive_sims_to_mongo.py` copies every graph/report row into the
MongoDB collection `sentisense.sim_archive`, keyed by
(`sim_date`, `mode`, `pg_created_at`) - a **versioned history** in which no
simulation run is ever lost to a re-run.

**The LLM request queue.** The dashboard's interactive LLM features cannot
call the GPU box's model directly (the firewall passes only PostgreSQL), so
the database carries the requests: the UI inserts a row into `llm_requests`
(migration 008) with a kind of `ask`, `narrate`, or `simulate`, and
`scripts/llm_worker.py` on the GPU container polls the table every two
seconds, claims jobs with `FOR UPDATE SKIP LOCKED` (safe under concurrent
workers), reclaims any orphaned in-flight rows at startup, answers via the
local Ollama model, and writes the response back for the SPA to poll. This
one queue powers both the analyst panel (grounded question-answering and
day-narration) and on-demand simulation requests.

**The dashboard.** A FastAPI backend exposes a read-only JSON API (with
in-process caching) over the shared database; a React SPA of fourteen
components renders it across three navigation tabs (Dashboard, Archive,
Simulator), a login screen, and a hidden operator view. Key views:

- **Login** - a full-page gate rendered whenever the site-wide password gate
  is enabled and no valid session cookie is present (Section 3.8).
- **Prediction hero** - a large green up / red down card with the current
  day's call, the predicted-class confidence, and the serving model's version.
- **Model performance** - the active champion's metric panel. Scores are
  **seeded from the model's held-out evaluation** (so a freshly promoted
  champion never starts from zero) and each settled live day folds into the
  cumulative accuracy: `(acc_eval*n_eval + correct_live) / (n_eval + n_live)`,
  with the eval/live split shown explicitly. Only the active model's own live
  days count; history from previous champions is never carried into the new
  one's score. The panel itself is a single server-owned JSON document with a
  versioned override chain - active MongoDB version, then file override, then
  live-computed - editable from the operator `PerfVersions` panel
  (Section 3.8).
- **Exploratory data analysis** - headline volume, daily mean sentiment,
  sentiment and relevance distributions, the 6x6 category-correlation
  heatmap, and the validation pass-rate, all computed server-side in SQL.
- **Archive** - the full headline history by day, each headline carrying its
  sentiment badge and per-category relevance score chips, with client-side
  filtering.
- **3-D centroid explorer** (`Centroids3D`) - every trading day's news
  centroid in the shared 16-d PCA space (axes selectable), with the eight
  KMeans cluster centers drawn as labeled markers; clicking a day opens its
  **single-day headline cloud**, where each headline is projected through the
  *same persisted basis* the models consume, alongside the day centroid. A
  software-3D orthographic fallback (rotate/tilt controls) keeps the view
  fully usable on browsers without WebGL.
- **Simulator** - the narrative-simulation view: an **analyst panel**
  (grounded LLM ask/narrate through the request queue), the **persona panel**
  (per-source persona votes - each outlet's daily stance derived from its
  mean scored sentiment - compared against the model's call and the realized
  outcome), the **agent map** of the nightly narrative simulation
  (stance-colored outlets, agree/disagree edges, tappable per-agent
  statements, consensus badges), the roundtable markdown report, and a
  run-new-simulation flow over a websocket that disables itself automatically
  when the simulation engine is unreachable.
- **Models (operator view)** - the registry leaderboard (version, family,
  OOS ROC-AUC with CI, MCC, accuracy, n) with one-click manual activation and
  the embedded `PerfVersions` editor; hidden from the public navigation.

> **[Figure 6 placeholder: dashboard screenshot - hero + model performance.]**
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
uvicorn (API), React + Vite + Plotly (SPA), pymongo (versioned overrides and
simulation archive), Playwright (scraping), and LangGraph (agentic scoring
path). Database schema changes ship as idempotent, numbered SQL migrations
(001-008). The repository is **three independent uv projects** - the root
`sentisense` package, `processing_engine/`, and `mivzakim_scraper/` - each
with its own lockfile and no shared workspace, so the scraper's browser
stack, the LLM-scoring stack, and the ML stack can never entangle their
dependency graphs.

**Key implementation decisions and trade-offs.**

- **Notebook to package.** A working but research-grade pipeline lived in
  notebooks. It was extracted into the importable, server-runnable
  `sentisense/` package, hardening the leakage controls in the process. The
  package deliberately does *not* port the notebooks' earlier leaky
  constructs: shuffled `StratifiedKFold`, same-day target features, and the
  PoC's `LastDay_Rise` / `LastDay_Pct` columns are all absent from the
  hardened code.
- **Registry over redeployment.** Swapping the served model is a database
  operation (activate a row), not a code deployment; the champion loads
  whatever is active on its next run. This also makes model promotion
  auditable (who activated what, when, automatic or manual).
- **Serialization safety.** Model artifacts are self-produced and stored in
  the project's own access-controlled database; torch bundles are loaded with
  `weights_only=True` (tensors and primitives only), refusing arbitrary
  object deserialization.
- **pgvector inside PostgreSQL.** Similarity search over the 768-d headline
  embeddings runs inside the existing PostgreSQL 16 instance via the
  `pgvector` extension (`headline_vectors` table, HNSW cosine index) - no
  separate vector-database service and no new Python dependency, consistent
  with the database-as-the-contract principle of Section 3.1.
- **Backend-aware scoring.** The orchestrator selects scoring flags per LLM
  backend at runtime: the remote vLLM takes 50-headline batched calls at high
  concurrency; the local Ollama model scores one headline per call at low
  concurrency. An empirical trial (Section 4.2.9) drove this design.
- **Resumable, cached experimentation.** The comparison driver
  (`scripts/pipeline_compare.py`) writes each finished cell's metrics to
  `leaderboard_cache.json` immediately; sequence-model Optuna studies resume
  from the database; registry training namespaces its studies away from the
  research studies so search spaces never collide.
- **Supervision without application containers.** The dashboard runs under
  pm2 (autorestart, log capture), nginx fronts it, and cron drives the
  pipeline; the application itself is not containerized. The only container
  is the GPU compute environment, and `scripts/container_startup.sh`
  restores its services idempotently after a container restart.

**Test suite.** The `tests/` directory holds 23 pytest files designed to run
**entirely offline** - no database, no network, no GPU - against small
synthetic frames, so the suite is fast and deterministic anywhere. Coverage
concentrates on the properties that silent bugs would corrupt: leakage safety
(chronological splits, train-only scaling and basis fitting, the
overnight-feature gate, the post-cutoff overlay), the challenger promotion
gate's decision logic, registry/serving round-trips, the streaming centroid
arithmetic, and the horizon-sweep bootstrap mechanics.

**Software/hardware.** Development on macOS (CPU); heavy training and the
nightly pipeline on a Linux GPU node (NVIDIA RTX 4090, 24 GB; CUDA 12.3
driver). Torch is pinned to the CUDA-12.1 wheel index on Linux - the newest
wheel line fully forward-compatible with the node's 12.3 driver, which makes
installs reproducible on the training node - with a CPU fallback index for
local work. PostgreSQL 16, MongoDB, and the dashboard run on the separate
front machine.

### 3.7 Evaluation Metrics

**Accuracy alone is misleading, and this is why the project reports a metric
set rather than a single number.** Next-day index direction is an imbalanced
classification problem whose imbalance shifts with the window: in the
evaluation windows used here the "up" day rate (the *base rate*) ranges from
roughly 49% to 58%. On a window with a 57% up-rate, a model that ignores its
inputs entirely and always predicts "up" scores 57% accuracy. Any accuracy
figure is therefore uninterpretable in isolation - it must be read against
the base rate of the same window, and it must be accompanied by metrics that
a majority-class guesser cannot inflate.

Two reference points make this concrete and are used throughout Chapter 4:

- **The no-skill baseline** is the **majority-class predictor**: it looks at
  the training tail, finds the more common direction, and predicts that
  direction for every test day. Its accuracy equals the base rate of the test
  window (roughly 55% on the current live window). Implemented as
  `MajorityClass` in `sentisense/models/baselines.py`, alongside a
  **Persistence** baseline (predict yesterday's realized direction). A model
  "beats no-skill" only when its accuracy exceeds this base rate. Note that
  the majority-class predictor scores a balanced accuracy of exactly 0.50, an
  ROC-AUC of 0.50, and an MCC of 0.00 by construction, which is precisely why
  those metrics are reported next to accuracy.
- **"Held-out days"** are the days in the **out-of-sample test tail**: the
  final chronological slice of trading days that the model was **never
  trained or tuned on**, scored exactly once. When a result reads
  "OOS accuracy 0.578, n = 327 held-out days", it means 327 trading days the
  model had never seen in any form contributed to that number.

The reported metric set (`sentisense/models/metrics.py`) is computed on the
**same held-out out-of-sample test tail** in research mode:

- **ROC-AUC** - threshold-free ranking quality; the primary research metric,
  reported with a bootstrap 95% CI in the registry.
- **F1 (macro)** - balances precision/recall across both classes.
- **Accuracy** and **balanced accuracy** - overall and class-balanced hit
  rate.
- **MCC** - Matthews correlation, robust to class imbalance.

The registry's *selection* metric is configurable
(`--select-metric oos_accuracy | oos_roc_auc`); the deployed registry was
selected on **OOS ROC-AUC**, which is exactly how the TCN champion out-ranked
the higher-accuracy but poorly-ranked PatchTST cell (Section 3.4).

Threshold-carrying models (the tuned forecasters) are scored **at their
validation-tuned threshold**, not a hard-coded 0.5 - a correctness detail
that materially changes accuracy-based rankings. Where a threshold has to be
chosen from a probability output, the project uses **Youden's J**: for each
candidate threshold, J is the true-positive rate minus the false-positive
rate, and the threshold maximizing J is selected. Youden's J is a
threshold-selection utility, not a reported result.

Three complementary evaluation surfaces exist in production: (a) the
**registry OOS metrics** (held-out test tail, computed once at training
time); (b) the **cumulative live score** (eval-seeded, extended by each
settled prospective day - the strongest evidence, since prospective days
cannot be overfit); and (c) an **in-sample all-days evaluation**
(`champion_full_eval`, the champion fit on all labeled days and scored on
those same days), which is deliberately exposed on the dashboard *as-is*: its
near-perfect scores demonstrate memorization, and the visible gap between it
and the OOS/live numbers is itself an instructive result. Surface (b) has a
concrete mechanical substrate: every night the champion's forward prediction
is upserted into `model_predictions`, and once the next close is known,
`scripts/settle_predictions.py` fills the row's `actual` column - turning the
table into an append-only prospective ledger (29 predictions logged between
2026-04-29 and 2026-08-12, 28 of them settled at the time of writing; Section
4.2.9 reports the resulting live score with the small-sample caveat it
requires). A **backtest overlay** places the statistical metrics in an
economic context.

### 3.8 Security and Access Control

**Posture.** The dashboard fronts a research database on a public network,
so the security goals are deliberately modest and explicit: keep the site
private to the team, keep every secret out of the repository, and keep all
headline data and LLM traffic inside project-controlled machines. Each goal
maps to a concrete mechanism below.

**Site-wide login gate.** The entire dashboard sits behind a login page. The
password lives only in the `SENTISENSE_UI_PASSWORD` environment variable
(never committed; when unset, the gate is off for local development).
`POST /api/login` compares the submitted password in constant time
(`secrets.compare_digest`) and, on success, sets a 30-day `HttpOnly`,
`SameSite=lax` cookie `ss_auth` whose value is an HMAC-SHA256 signature
derived from the password - a **stateless** session that survives server
restarts, with the property that rotating the password invalidates every
outstanding session at once. A FastAPI HTTP middleware rejects (401) every
`/api*` and `/ws*` request that lacks a valid cookie, exempting only
`/api/login` and `/api/auth`; static assets and the SPA shell stay open, and
the shell simply renders the login screen when unauthenticated. Because HTTP
middleware does not cover websocket upgrades, the websocket handler
**re-validates the cookie itself** during the handshake and closes
unauthenticated connections with a dedicated code.

**TLS termination.** nginx 1.29.4 on the front machine terminates TLS 1.2/1.3
on port 443 with the college wildcard certificate `*.cs.colman.ac.il`
(DigiCert/RapidSSL, valid to January 2027), speaks HTTP/2, proxies both HTTP
and websocket upgrades to the loopback dashboard on port 3000, and redirects
port 80 to HTTPS with a permanent 301. The application process itself never
listens on a public TLS port; encryption in transit is the proxy's single
responsibility.

**Secrets live only in the environment.** The pattern established for the
database URL in Section 3.6 applies uniformly: `SENTISENSE_DATABASE_URL`,
`SENTISENSE_UI_PASSWORD`, and `SENTISENSE_MONGO_URL` are read from the
environment at runtime, the code fails fast (or degrades explicitly) when
they are absent, and no credential appears in the repository, the SPA bundle,
or the logs.

**Auditable manual overrides.** The dashboard's model-performance panel is
the one place where an operator can override what the system computes, and
that channel is deliberately **versioned rather than silent**: every edit is
saved as a *new* document in the MongoDB `performance_versions` collection
(document, note, active flag, creation timestamp), activation is an explicit
separate action, deactivation reverts to the live-computed values, and
history is never overwritten. MongoDB is optional by design - a short
connection timeout and warning-only failure mode mean the panel always
renders even with the archive host down.

**Zero-egress narrative-simulation toolchain.** The MiroFish multi-agent
engine behind the Simulator raised two risks: it is AGPL-licensed, and its
upstream defaults call cloud services (a hosted Zep memory store, hosted LLM
APIs). Both are addressed by isolation. For **license isolation**, MiroFish
runs as a separate HTTP service (the `external/MiroFish` submodule, on its
own local port) and `sentisense/sim` is an arm's-length HTTP client - the
AGPL codebase is never imported into, linked with, or vendored inside the
project's own code. For **zero egress**, the hardening specified in
`docs/miro/LOCAL_ONLY.md` replaces every external dependency with a local
one: Zep is self-hosted from source (`scripts/init_zep.sh`), every LLM
endpoint points at the local Ollama server, offline mode is forced for model
hubs, and the upstream Zep client call sites are patched to honor the local
base URL. Enforcement is layered rather than assumed: a preflight check
(`sentisense.sim.preflight.assert_local`) refuses any non-loopback simulation
URL unless explicitly overridden by an environment flag, and
`scripts/verify_local_egress.sh` audits a deployment - confirming the SDK
honors the local `ZEP_API_URL`, scanning the environment files for external
hosts, and optionally watching the live process for outbound connections -
exiting non-zero on any egress risk. No headline data, scores, or prompts
leave the project's machines.

---
## 4. Results and Analysis

### 4.1 Experimental Setup

Results were produced along **three complementary experiment tracks**, all
reported in full below:

1. **Exploratory notebook tracks** - a sequence of research notebooks
   (`poc.ipynb`, `compare_lstm_features_with_poc.ipynb`,
   `transformer_forecaster.ipynb`, `tuning.ipynb`, `sentisense_analysis.ipynb`,
   `timesfm_explainability.ipynb`) that iterate on splits, feature sets, model
   families, ablations, and robustness checks. Their purpose is exploration:
   they deliberately vary their train/test windows, and therefore their
   no-skill baselines, which is itself part of the analysis (Section 4.3).
2. **The unified, hardened package grid** - `scripts/pipeline_compare.py`,
   which reduces every cell to a uniform `(scores, labels)` pair on the
   identical out-of-sample window and scores it with the shared metrics. This
   is the canonical, leakage-hardened cross-model comparison.
3. **The production registry run** - `scripts/train_registry.py`, which
   re-tunes the zoo under the registry's serving contract, registers every
   candidate with its OOS metrics, and activates the champion that the live
   system serves (Section 4.2.9). This track produces the headline result.

**The shared evaluation contract, and why the grid and the registry report
different numbers.** All three tracks obey the same leakage rules -
chronological 70/15/15 split, all transforms fit on the train slice only,
hyper-parameter optimization (HPO) scored on the **validation slice only**,
and the test tail scored exactly once. But the grid and the registry answer
different questions and therefore evaluate under different *contracts*:

- The **unified grid (Section 4.2.8)** is a *comparison* surface. It runs
  every model against every data type, forces all of them onto one shared
  out-of-sample window with one shared metric set, and picks each cell's
  decision threshold by Youden's J. Its output is a like-for-like ranking,
  not a deployable model.
- The **registry run (Section 4.2.9)** is a *selection* surface. It re-tunes
  each family from scratch under the exact contract the live system serves on
  - fused features only, the full available timeline, its own namespaced
  Optuna studies so search spaces never collide with the grid's - and its
  output is a serialized artifact plus the metrics that decide promotion.

Different feature frames, different tuning budgets, and different selection
metrics mean the same architecture legitimately scores differently in the two
tables. This is the reason the grid's PatchTST and the registry's PatchTST are
not the same number: they are answers to different questions, and both are
reported rather than reconciled away.

**How results are tabulated.** Every model-comparison table in this chapter
uses the **same four columns, in the same order**: *Model*, *Accuracy*,
*Baseline*, *ROC-AUC*. *Baseline* is the no-skill majority-class base rate of
that table's own evaluation window (Section 3.7), so accuracy can always be
read against the number it has to beat. Metrics that only some sources
rendered (balanced accuracy, F1, MCC, confidence intervals) are reported in
the prose beneath the relevant table rather than left blank inside it.

> **Note on reproduction state.** A number of cells in the saved notebooks
> were not executed in the committed copy (e.g. the transformer Optuna "tuned
> leaderboard" cells; the `tuning.ipynb` GRU/TCN, multi-seed, abstention, and
> final-report cells; and all of `timesfm_explainability.ipynb`). To keep this
> book reproducible, **only metrics that actually rendered in the saved
> outputs are reported**, and each gap is flagged where it occurs.

The unified package grid is a two-axis grid evaluated by
`scripts/pipeline_compare.py`:

- **Model axis** - classifiers (XGBoost, LSTM, GRU, TCN, PatchTST) and
  forecasters (TFT, N-HiTS, N-BEATS, Chronos, TimesFM).
- **Data-type axis** - `scored` (LLM news scores), `embedded` (768-d e5
  centroid + finance), and `fused` (per-source scores plus centroid).
  Classifiers run on all three; forecasters use scored covariates or
  univariate input only.

Each classifier (model x data-type) cell gets its **own resumable Optuna
study**; search spaces are wide (e.g., sequence models tune window 5-60,
capacity to 384 units, depth to 4, dropout 0-0.7, lr 1e-5-3e-2; XGBoost tunes
a 9-dimensional space; forecasters additionally tune context length).
Reproducibility is enforced with fixed seeds.

### 4.2 Presentation of Results

The results below are organized by source: the exploratory notebook tracks
first, then the unified package grid (Section 4.2.8), then the production
registry run and the live champion (Section 4.2.9), which is where the
system's headline number appears. Every table predicts the **next-day
close-to-close TA-125 direction**, and every table uses the shared column set
described in Section 4.1.

**Aggregation lineage (how the notebooks relate).** The notebooks deliberately
use *different news-aggregation strategies*, and the later ones expand on two
base representations:

- **Base A - daily-mean aggregation (`poc.ipynb`).** Each day's headlines are
  collapsed to **per-category means** (`mean_politics` through
  `mean_sentiment`, plus `std_sentiment`, `pct_negative`, `pct_positive`).
  This is the compact, tree-friendly representation. The PoC additionally
  carried two `LastDay_*` features that were not leak-safe; the hardened
  `sentisense/` package drops them entirely (Section 3.3), which is one reason
  PoC numbers are not directly comparable to later tracks.
- **Base B - per-source wide aggregation (`lstm_forecaster.ipynb`).** Scores
  are **summed per `(date, source)` and pivoted wide** (`<dim>_<source>`,
  giving 320 feature columns), preserving *which outlet produced which
  signal*, then sliced into 30-day windows for the LSTM.
- **Expansions.** `compare_lstm_features_with_poc.ipynb` runs the **PoC tree
  models on Base B's per-source wide features**, directly testing whether
  per-source features beat daily means, and adds ablations, walk-forward, and
  multi-seed checks. `transformer_forecaster.ipynb` and `tuning.ipynb` use
  **both** shapes (daily-mean for tree/vanilla models, per-source for sequence
  models). The `sentisense/` package generalizes both into leakage-safe `mt`
  (daily-mean) and `ml` (per-source) frames (Section 3.3).

Each subsection below states its purpose in one sentence and notes which
aggregation it uses.

**Cross-track summary.** Table 1 gives the one-glance picture: the best
configuration in each track and how it compares to the no-skill
majority-class baseline. The baseline is defined identically everywhere - the
majority-class predictor's accuracy - but its *value* differs by row because
the tracks evaluate on different windows and the up-day base rate is
window-dependent (it ranges from 0.4931 to 0.5773 across these windows). Each
row's accuracy must therefore be read against the baseline on its own row,
never against another row's. The exploratory rows are early or narrow
experiments; the rows that carry the project's conclusions are the unified
grid and, above all, the production registry (Section 4.2.9), which
contributes both its best-accuracy cell and the activated champion.

*Table 1: Best result per experiment track vs its no-skill baseline. Baseline
is the majority-class base rate of that track's own evaluation window; the
values differ across rows because the windows differ, so accuracy is only
meaningful relative to the baseline on the same row.*

| Model | Accuracy | Baseline | ROC-AUC |
|---|---|---|---|
| XGBoost / LightGBM - PoC, daily-mean (Section 4.2.1) | 0.5459 | 0.4976 | n/a |
| LGBM "Top sources + Other" - per-source (Section 4.2.2) | 0.5794 | 0.5675 | 0.5415 |
| LSTM window 30 - per-source (Section 4.2.3) | 0.5636 | 0.5773 | n/a |
| PatchTST_DailyMean - transformer zoo (Section 4.2.4) | 0.5370 | 0.4931 | 0.5185 |
| Ensemble (soft-vote) - tuning track (Section 4.2.5) | 0.4596 | 0.5680 | n/a |
| Score-LSTM - hardened package (Section 4.2.6) | 0.5000 | ~0.50 | 0.5088 |
| GRU [scored] - unified grid, best ROC-AUC (Section 4.2.8) | 0.5289 | ~0.50 | **0.5755** |
| TFT [cov=none] - unified grid, best accuracy (Section 4.2.8) | **0.5916** | ~0.50 | 0.5391 |
| PatchTST - registry best accuracy (Section 4.2.9) | **0.5780** | ~0.55 | 0.4795 |
| **TCN - production champion (Section 4.2.9)** | 0.5681 | ~0.55 | **0.5731** |

*(n/a = ROC-AUC was not printed numerically in that notebook's saved output.)*

The rows that matter for the project's claim are the last three blocks. On
the unified grid, the best model reaches **accuracy 0.5916** and the best
ranker reaches **ROC-AUC 0.5755**, both above the no-skill line. Under the
production contract, the registry's best-accuracy cell (PatchTST) reaches
**accuracy 0.578 on 327 held-out days against a ~0.55 base rate**, while the
registry's automatic selector - which ranks candidates by out-of-sample
ROC-AUC - activated the **TCN** champion at **ROC-AUC 0.5731 with accuracy
0.5681 on 382 held-out days**. Both registry cells hold an edge of roughly
1-3 percentage points over always predicting the majority direction,
sustained on days the models never saw; Section 4.2.9 explains why the
champion slot went to the ranker rather than the accuracy leader.

#### 4.2.1 Tree-model proof-of-concept (`poc.ipynb`)

*Purpose: establish whether daily-mean news features carry any tradable
directional information at all, using standard gradient-boosted trees.*

*Aggregation: Base A - daily-mean per category.*

The earliest experiment established a tree-model reference point. Two
evaluation protocols were run, and they report different numbers for the same
models. **This is a difference of evaluation contract, not of model quality,
and it is worth stating explicitly because the same pattern recurs later in
the chapter:**

- The **5-fold cross-validation** figures below average five folds drawn from
  across the whole PoC period. Each fold has its own class balance, and folds
  are averaged, so the result describes *typical* performance over a mixed
  set of market conditions.
- The **chronological 80/20 holdout** figure is a single, strictly forward
  test on one specific window (test up-rate 49.76%).

The holdout number is therefore the one comparable to the rest of this
chapter, and the cross-validation number is reported for completeness rather
than as a competing claim. Both are also inflated relative to the hardened
tracks, because the PoC frame still carried the two `LastDay_*` features that
the hardened package later removed.

**5-fold cross-validation (accuracy):**

| Model | Mean Accuracy | Std | Fold scores |
|---|---|---|---|
| XGBoost | 53.60% | 1.87% | 51.70 / 56.66 / 51.70 / 54.57 / 53.40 |
| LightGBM | 52.40% | 2.49% | 48.04 / 55.35 / 51.70 / 53.00 / 53.93 |
| CatBoost | 53.45% | 3.21% | 47.26 / 56.14 / 53.52 / 55.35 / 54.97 |

*Table 2: PoC tree-model 5-fold cross-validation accuracy.*

**Chronological 80/20 holdout** (train 826 rows 2019-07-17 to 2022-12-04; test
207 rows 2022-12-05 to 2023-10-05; test up-rate 49.76%):

| Model | Accuracy | Baseline | ROC-AUC |
|---|---|---|---|
| XGBoost | 0.5459 | 0.4976 | n/a |
| LightGBM | 0.5459 | 0.4976 | n/a |
| CatBoost | 0.5362 | 0.4976 | n/a |

*Table 3: PoC chronological 80/20 holdout results. ROC-AUC was not printed
numerically in this notebook's saved output.*

All three trees clear the 0.4976 no-skill baseline, XGBoost and LightGBM by
about 4.8 percentage points. Bootstrap 95% confidence intervals on the
holdout accuracy are wide, as expected on a 207-day window:

| Model | Accuracy | 95% CI |
|---|---|---|
| XGBoost | 0.5459 | [0.4830, 0.6135] |
| LightGBM | 0.5459 | [0.4783, 0.6135] |
| CatBoost | 0.5362 | [0.4686, 0.5992] |

*Table 4: PoC holdout bootstrap 95% confidence intervals.*

XGBoost holdout classification report (207-sample split):¹

| Class | precision | recall | f1 | support |
|---|---|---|---|---|
| 0 (Fall) | 0.56 | 0.45 | 0.50 | 104 |
| 1 (Rise) | 0.54 | 0.64 | 0.58 | 103 |
| accuracy | | | 0.55 | 207 |

*Reading:* the proof of concept did what a proof of concept should do - it
showed the trees clearing the no-skill line on a forward window, on a small
sample and with a feature frame that had not yet been hardened. That was
enough to justify building the leakage-safe pipeline; it is not itself the
project's evidence.

> ¹ `poc.ipynb` contains unresolved git merge-conflict markers; an alternate
> rendered report with a 507-sample support exists. The 207-sample version
> above matches the notebook's stated 80/20 split.

#### 4.2.2 LSTM feature-set vs PoC study (`compare_lstm_features_with_poc.ipynb`)

*Purpose: test whether keeping track of which outlet published which headline
(per-source features) predicts better than collapsing the day to category
means.*

*Aggregation: Base B - per-source wide (from `lstm_forecaster.ipynb`), fed to
the PoC tree models.*

This is the most extensive notebook: it compares feature families on the
per-source "LSTM wide" representation, with ablations and robustness checks.
Unless noted, the test window is 2024-03-26 to 2026-04-28 (504 rows, no-skill
baseline 0.5675).

**Main holdout summary (sorted by accuracy):**

| Model | Accuracy | Baseline | ROC-AUC |
|---|---|---|---|
| LGBM - Top sources + Other | 0.5794 | 0.5675 | 0.5415 |
| CatBoost - Baseline wide | 0.5714 | 0.5675 | 0.4760 |
| XGBoost - Top sources + Other | 0.5714 | 0.5675 | 0.5359 |
| XGBoost - Baseline wide | 0.5694 | 0.5675 | 0.5196 |
| LGBM - Baseline wide | 0.5694 | 0.5675 | 0.5453 |
| CatBoost - Top sources + Other | 0.5675 | 0.5675 | 0.4963 |

*Table 5: Per-source feature-set holdout comparison (tree models). Balanced
accuracy ranges 0.5065 to 0.5230 across these rows.*

**Feature-group ablation (CatBoost), no-skill baseline 0.5642:**

| Model | Accuracy | Baseline | ROC-AUC |
|---|---|---|---|
| Basic market only (6 features) | 0.5811 | 0.5642 | 0.5415 |
| News + all market features (344 features) | 0.5768 | 0.5642 | 0.5074 |
| Market-derived only (17 features) | 0.5600 | 0.5642 | 0.5296 |
| News wide only (321 features) | 0.5558 | 0.5642 | 0.4841 |

*Table 6: Feature-group ablation (CatBoost). A compact market-feature set
performs comparably to the full news-plus-market frame on this window.*

**Walk-forward validation (5 folds, CatBoost):** mean accuracy **0.5967** vs
mean baseline 0.5733, a mean gap of +0.0233 (fold accuracies 0.633 / 0.650 /
0.567 / 0.617 / 0.517).

**Multi-seed robustness (CatBoost, 5 seeds):** mean accuracy **0.5714 +/-
0.0084** (min 0.560, max 0.581), mean gap +0.0072, with **4 of 5** seeds above
baseline; ROC-AUC ranges 0.507-0.577.

*Reading:* the best configuration (LGBM, "Top sources + Other") reaches
accuracy 0.5794 against a 0.5675 baseline, and the per-source representation
holds its edge across seeds. On this window the per-source features do not
give a decisive advantage over a compact market-feature set, which is a useful
negative finding about the *representation*, not about the system.

#### 4.2.3 LSTM base forecaster (`lstm_forecaster.ipynb`)

*Purpose: establish how a recurrent sequence model behaves on the raw
320-column per-source representation it was designed for.*

*Aggregation: Base B - per-source wide (320 columns), 30-day windows.*

The base LSTM is trained on chronological, windowed sequences (window 30,
326 features) with train/validation/test = 1,163 / 249 / 250 daily rows
(2019-07-17 to 2026-04-29); the test window's no-skill baseline is 0.5773.

| Model | Accuracy | Baseline | ROC-AUC |
|---|---|---|---|
| LSTM (window 30) | 0.5636 | 0.5773 | n/a |

*Table 7: LSTM base forecaster holdout result. ROC-AUC was rendered only
inside plot images, so no numeric value is available.*

| Class | precision | recall | f1 | support |
|---|---|---|---|---|
| Fall | 0.29 | 0.02 | 0.04 | 93 |
| Rise | 0.57 | 0.96 | 0.72 | 127 |
| accuracy | | | 0.56 | 220 |

*Table 7b: LSTM base forecaster holdout classification report.*

Bootstrap 95% CI on accuracy: [0.5000, 0.6273].

*Reading:* on this window the model leans heavily to the majority "Rise" class
(recall 0.96 versus 0.02) and lands just under its baseline. Training accuracy
reaches ~0.74 while validation stays around 0.48-0.53, which identifies the
cause: a 320-column per-source frame gives an unregularized LSTM too much
capacity for the number of trading days available. This result is what
motivated the dimensionality reduction used later - the fused frame's PCA
block - and the far tighter HPO contract of the registry track.

#### 4.2.4 Transformer model zoo + ablations (`transformer_forecaster.ipynb`)

*Purpose: find out whether attention-based architectures extract more from the
same features than trees or recurrent models do.*

*Aggregation: both - daily-mean (`*_DailyMean` models) and per-source
(`*_PerSource` models), so the two base shapes compete head-to-head.*

Nine transformer variants were evaluated against tree and linear reference
models on a window whose no-skill baseline is 0.4931.

**Final leaderboard (best per row):**

| Model | Accuracy | Baseline | ROC-AUC |
|---|---|---|---|
| ModelB_PatchTST_DailyMean | 0.5370 | 0.4931 | 0.5185 |
| CatBoost | 0.5069 | 0.4931 | 0.5048 |
| XGBoost | 0.5035 | 0.4931 | 0.5070 |
| ModelC_TwoTower_DailyMean | 0.5019 | 0.4931 | 0.5000 |
| ModelE_Informer_PerSource | 0.4981 | 0.4931 | 0.5000 |
| ModelA_Vanilla_PerSource | 0.4942 | 0.4931 | 0.4996 |
| ModelA_Vanilla_DailyMean | 0.4942 | 0.4931 | 0.5216 |
| LGBM | 0.4931 | 0.4931 | 0.4727 |
| ModelE_Informer_DailyMean | 0.4903 | 0.4931 | 0.5397 |
| ModelD_Hierarchical_DailyMean | 0.4903 | 0.4931 | 0.5339 |
| ElasticNet | 0.4792 | 0.4931 | 0.4804 |
| ModelD_Hierarchical_PerSource | 0.4708 | 0.4931 | 0.4757 |
| ModelC_TwoTower_PerSource | 0.4514 | 0.4931 | 0.4766 |

*Table 8: Transformer zoo final leaderboard vs tree/linear reference models.
Balanced accuracy and MCC track accuracy closely here; PatchTST leads on both
(balanced accuracy 0.5381, MCC 0.0949).*

**Window-size ablation (PatchTST):** best at window 15-20 (accuracy around
0.54-0.55, ROC-AUC up to 0.592); the model collapses to the majority class at
windows 45-60.
**Feature-group ablation:** Market-only 0.5409, LagReturns-only 0.5292,
News-only 0.4864.

*Reading:* PatchTST is the clear winner of this track, beating the baseline by
4.4 percentage points and leading every reference model on accuracy, balanced
accuracy, and MCC. The window ablation is the actionable finding: the
architecture needs a short context (15-20 days) and degrades badly with a long
one. Both results carried directly into the production track, where a tuned
PatchTST on the fused frame became the registry's best-accuracy cell -
though, as Section 4.2.9 details, not its champion. (The Optuna "tuned
leaderboard" cells were not executed in the saved notebook.)

#### 4.2.5 Sequence-model tuning & robustness (`tuning.ipynb`)

*Purpose: stress-test the tuning procedure itself - does leak-safe
TimeSeriesSplit Optuna tuning on daily-mean features transfer to a forward
holdout?*

This notebook applies leak-safe TimeSeriesSplit Optuna tuning (target:
balanced accuracy) and walk-forward backtesting. Corpus: 1,898,499 validated
rows, 40 sources. The holdout window's no-skill baseline is 0.5680.

| Model | Accuracy | Baseline | ROC-AUC |
|---|---|---|---|
| LightGBM (vanilla holdout) | 0.5257 | 0.5680 | n/a |
| XGBoost (vanilla holdout) | 0.5106 | 0.5680 | n/a |
| CatBoost (vanilla holdout) | 0.5076 | 0.5680 | n/a |
| LSTM (Optuna, tuned threshold) | 0.4553 | 0.5680 | n/a |
| Ensemble (soft-vote, tuned threshold) | 0.4596 | 0.5680 | n/a |

*Table 9: Sequence-model tuning track - holdout results. ROC-AUC was not
printed numerically in this notebook's saved output.*

Threshold selection on the validation slice (by Youden's J) produced
thresholds of 0.597 (XGBoost), 0.521 (LightGBM), and 0.525 (CatBoost), with
validation balanced accuracies of 0.5363, 0.5583, and 0.5345 respectively; the
tuned LSTM reached validation balanced accuracy 0.5611 and the soft-vote
ensemble 0.5712. Walk-forward CatBoost gave mean accuracy 0.5267 +/- 0.0814
against a mean baseline of 0.5533.

*Reading:* this track is the chapter's cleanest example of validation-to-test
transfer failure. Every configuration looks reasonable on the validation slice
and then lands below the baseline on the holdout, on a high-base-rate window
that is unforgiving to any model that does not lean to the majority class.
The lesson - that a threshold and a hyper-parameter set chosen on one slice
need re-checking against the base rate of the slice they will be scored on -
is why the registry track re-tunes under its own serving contract rather than
importing settings from here. (GRU/TCN, multi-seed, abstention, and the final
`final_results.csv` cells were not executed in the saved notebook.)

#### 4.2.6 Hardened-package analysis (`sentisense_analysis.ipynb`)

*Purpose: isolate the contribution of the LLM news scores alone, by running a
sequence model on the `scored` frame with every leakage control enabled and no
market features to lean on.*

Run directly against the live database. Corpus coverage: **2,950,339 validated
`mistral-small-4` rows** (plus 52,640 `mistral-small:latest`).

| Model | Accuracy | Baseline | ROC-AUC |
|---|---|---|---|
| Score-LSTM (threshold 0.5) | 0.5000 | ~0.50 | 0.5088 |
| Score-LSTM (tuned threshold) | 0.4961 | ~0.50 | 0.5072 |

*Table 10: Hardened-package score-LSTM final holdout, averaged over repeats.
Standard deviations across repeats: accuracy 0.0058, ROC-AUC 0.0144, MCC
0.0114 at threshold 0.5. Balanced accuracy 0.5001, F1 0.4990, MCC 0.0001.*

*Reading:* this is an ablation, and its value is in what it isolates. Stripped
of market context and run under the full hardened contract, the news scores on
their own do not carry next-day directional information for a single LSTM
(LSTM Optuna best value 0.538). Read together with the feature-group
ablations in Sections 4.2.2 and 4.2.4, it locates where the system's edge
actually comes from: the *combination* of news features with market context in
the fused frame, which is exactly the frame the production champion serves on.
(SHAP outputs exist only as plots in the saved notebook.)

#### 4.2.7 Foundation-model explainability (`timesfm_explainability.ipynb`)

This notebook scaffolds zero-shot and covariate-ablation experiments for
Google's TimesFM, but **was not executed in the committed copy**, so no
numeric results are available. It is retained as a wired template for future
work (Section 5).

#### 4.2.8 Unified out-of-sample grid (`leaderboard.md`)

*Purpose: rank every model against every data type on one shared
out-of-sample window, so architectures can be compared like for like.*

This is the *comparison* contract described in Section 4.1: each cell is
reduced to the same `(scores, labels)` pair on the identical window, scored
with the same metric set, with its decision threshold chosen by Youden's J.
The no-skill baseline on this shared window is approximately 0.50. Sorted by
accuracy descending. Notation: `model [data-type]` for classifiers,
`model [cov=...]` for forecasters. Where a model appears twice, the two rows
are distinct tuned cells that survived the cache.

| Model | Accuracy | Baseline | ROC-AUC |
|---|---|---|---|
| TFT [cov=none] | 0.5916 | ~0.50 | 0.5391 |
| XGBoost [embedded] | 0.5890 | ~0.50 | 0.5314 |
| XGBoost [fused] | 0.5759 | ~0.50 | 0.5253 |
| GRU [fused] | 0.5568 | ~0.50 | 0.5359 |
| PatchTST [fused] | 0.5553 | ~0.50 | 0.5112 |
| Chronos-zeroshot | 0.5538 | ~0.50 | 0.4266 |
| LSTM [embedded] | 0.5429 | ~0.50 | 0.5128 |
| XGBoost [fused] | 0.5417 | ~0.50 | 0.5396 |
| LSTM [fused] | 0.5402 | ~0.50 | 0.4724 |
| Chronos-tuned | 0.5381 | ~0.50 | 0.4492 |
| TFT [cov=scored] | 0.5366 | ~0.50 | 0.5524 |
| XGBoost [embedded] | 0.5347 | ~0.50 | 0.5217 |
| TCN [fused] | 0.5318 | ~0.50 | 0.5303 |
| TCN [scored] | 0.5310 | ~0.50 | 0.5669 |
| TFT [cov=none] | 0.5296 | ~0.50 | 0.5386 |
| GRU [scored] | 0.5289 | ~0.50 | 0.5755 |
| PatchTST [embedded] | 0.5283 | ~0.50 | 0.4726 |
| PatchTST [scored] | 0.5208 | ~0.50 | 0.4541 |
| NHiTS [cov=none] | 0.5157 | ~0.50 | 0.4808 |
| PatchTST [fused] | 0.5126 | ~0.50 | 0.5040 |
| NBEATS | 0.5105 | ~0.50 | 0.5227 |
| TCN [scored] | 0.5094 | ~0.50 | 0.5422 |
| NHiTS [cov=scored] | 0.5087 | ~0.50 | 0.4830 |
| XGBoost [scored] | 0.5079 | ~0.50 | 0.5338 |
| LSTM [scored] | 0.5041 | ~0.50 | 0.5204 |
| XGBoost [scored] | 0.5035 | ~0.50 | 0.5129 |
| PatchTST [scored] | 0.5035 | ~0.50 | 0.5270 |
| TCN [embedded] | 0.5022 | ~0.50 | 0.4675 |
| TFT [cov=scored] | 0.5017 | ~0.50 | 0.5119 |
| NBEATS | 0.4983 | ~0.50 | 0.5106 |
| LSTM [scored] | 0.4958 | ~0.50 | 0.5125 |
| GRU [embedded] | 0.4910 | ~0.50 | 0.5091 |
| NHiTS [cov=none] | 0.4895 | ~0.50 | 0.4837 |
| NHiTS [cov=scored] | 0.4869 | ~0.50 | 0.4835 |
| GRU [embedded] | 0.4820 | ~0.50 | 0.4642 |
| LSTM [fused] | 0.4802 | ~0.50 | 0.5115 |
| TCN [fused] | 0.4709 | ~0.50 | 0.4552 |
| LSTM [embedded] | 0.4706 | ~0.50 | 0.4715 |
| GRU [fused] | 0.4669 | ~0.50 | 0.4679 |
| GRU [scored] | 0.4644 | ~0.50 | 0.4967 |
| PatchTST [embedded] | 0.4513 | ~0.50 | 0.4552 |
| TCN [embedded] | 0.4238 | ~0.50 | 0.5327 |

*Table 11: Unified out-of-sample leaderboard (40+ tuned cells) against a
~0.50 no-skill baseline. Coverage: 23 model configurations ran, 21 cached,
2 skipped. F1 for each cell is available in the generated `leaderboard.md`.*

**Best by accuracy:** `TFT [cov=none]` at **0.5916**, roughly nine points
above the no-skill line on this window.
**Best by ROC-AUC:** `GRU [scored]` at **0.5755** - the strongest *ranker* in
the zoo, meaning it orders up-days above down-days better than any other cell.

Around a third of the grid's cells clear the no-skill line, and the top of the
table does so by a wide margin. The grid's job, though, is ranking rather than
deployment: no single cell here is tuned under the serving contract, which is
what the next section does.

> **[Figure 11 placeholder: scatter of Table 11 - ROC-AUC (x) vs accuracy
> (y), point shape by model family, showing the cloud centered on
> (0.50, ~0.52).]**

#### 4.2.9 Production registry run and the live champion

*Purpose: select and deploy one model under the exact contract the live system
serves on. This is the track that produces the project's headline result.*

`train_registry.py` re-tunes the zoo under the registry's serving contract -
fused features, the full available timeline, chronological 70/15/15,
per-family Optuna studies in registry-namespaced storage - and registers each
candidate with its held-out metrics. As explained in Section 4.1, this is a
different evaluation contract from the unified grid (all data types, Youden
thresholds, comparison-only), which is why the same architecture scores
differently in Table 11 and Table 13. Both numbers are real; they measure
different things.

**Registry validation run (tree zoo, low trial budget).** A smoke-budget run
(5 trials per model) validated the end-to-end train, register, select, and
serve loop:

| Model | Accuracy | Baseline | ROC-AUC |
|---|---|---|---|
| LightGBM | 0.5553 | ~0.55 | 0.5153 |
| XGBoost | 0.5527 | ~0.55 | 0.5476 |
| CatBoost | 0.5424 | ~0.55 | 0.5476 |

*Table 12: Registry validation run - tree zoo OOS metrics on the test tail of
the fused frame. ROC-AUC 95% confidence intervals: XGBoost [0.486, 0.604],
LightGBM [0.458, 0.576], CatBoost [0.483, 0.604]; MCC 0.062, 0.060, and 0.030
respectively. At five trials per model this run was a plumbing check, not a
tuning result.*

**The full-budget run** (100 trials per tree model, 40 per sequence
architecture with 3-seed OOS averaging, plus the foundation-model families)
was executed on 2026-07-02 and populated the registry with **29 registered
model versions** spanning thirteen families: XGBoost, LightGBM, CatBoost,
LSTM, GRU, TCN, PatchTST, TFT, N-HiTS, N-BEATS, Chronos (zero-shot), Chronos
(tuned), and a top-3 soft-vote ensemble. Out-of-sample ROC-AUC across the 29
versions ranges from 0.4266 to 0.5731. The search was backed by the
registry's in-database Optuna storage, which by the end of the run held 58
studies totalling roughly 2,073 trials. This is the leaderboard the
dashboard's Models panel displays, and it produced the champion below.

> **[Placeholder: export the full 29-version registry
> leaderboard from the Models panel (version, family, ROC-AUC + CI, MCC,
> accuracy, n).]**

**The active champion - the system's headline result.** The registry's
automatic selector ranks candidates by **out-of-sample ROC-AUC**
(`--select-metric oos_roc_auc`, the default) and activated a **TCN** sequence
classifier on the fused frame, trained under the FULL regime with the
overnight target enabled:

| Model | Accuracy | Baseline | ROC-AUC |
|---|---|---|---|
| **TCN (`tcn-20260702-1351`), active champion** | 0.5681 | ~0.55 | **0.5731** |
| PatchTST (`patchtst-20260702-1351`), registry best accuracy | **0.5780** | ~0.55 | 0.4795 |

*Table 13: Active production champion and the registry's best-accuracy cell -
held-out evaluation. The TCN is evaluated on n = 382 held-out days, the
PatchTST on n = 327; in both cases the days were never used for training or
tuning. Family: torch sequence classifiers, fused features.*

The selection deserves a plain statement, because the two rows of Table 13
pull in opposite directions. The PatchTST cell has the registry's best
held-out *accuracy* (0.5780) but a *ROC-AUC below 0.5* (0.4795): it makes
well-placed directional calls at its tuned threshold while ordering its
probabilities worse than chance. The TCN is the registry's best *ranker*
(0.5731) and gives up only about one accuracy point (0.5681). Because the
serving path consumes the model's probability - the dashboard displays
calibrated confidence, and the settlement record is threshold-sensitive - the
registry's default selection metric is ranking quality, and the auto-best
selector therefore promoted the TCN. The alternative remains a one-flag
choice (`--select-metric oos_accuracy`), and both metrics are shown side by
side on the dashboard, so the trade-off is explicit rather than hidden. (An
earlier draft of this book, and the operational inventory document, named the
PatchTST cell and the initial XGBoost champion respectively as the serving
model; both described earlier or provisional states of the registry, and the
serving champion since 2026-07-06 is the TCN above.)

**This is the number the system is judged on: ROC-AUC 0.5731 with 56.8%
directional accuracy on 382 held-out days, against a no-skill majority-class
baseline of about 55% and a longer-run base rate near 53%.** The champion
beats the no-skill predictor by roughly 1-2 accuracy points while being the
best-ordered ranker in the zoo, and holds that margin over more than a year
of trading days it never saw. An edge of this size is not dramatic, and it is
not meant to be: in daily index-direction forecasting a consistent 55-58%
against a ~53% base rate is a genuinely difficult result to obtain and a
valuable one to hold, which is why the margin is reported precisely rather
than rounded up.

**Backend trial for the live scoring era.** Before switching nightly scoring
to the locally hosted `gemma4` model, three modes were trialed: the agentic
ReAct path failed (tool-loop recursion), 10-headline batched JSON failed
(unparseable output), and **single-headline structured calls succeeded 20/20**
at about 7.7 headlines per minute - sufficient for the nightly volume of
roughly 1,000 headlines per day. This trial directly produced the
backend-aware scoring design of Section 3.6.

**Live prospective record.** From activation onward, each settled trading day
extends the champion's prospective record on the dashboard (eval-seeded
cumulative accuracy, Section 3.5). Between 2026-04-29 and 2026-08-12 the
production loop issued **29 next-day predictions**: three by the initial
champion `xgb-fused-full-v1` (activated 2026-04-29, before the full-budget
run existed) and 26 by the TCN champion after the 2026-07-06 switchover. Of
these, **28 have settled, with 13 correct - a live directional accuracy of
46.4%**, below both the champion's 56.8% OOS accuracy and the ~55% majority
baseline. The exact binomial 95% confidence interval on 13/28 spans **0.275
to 0.661**: the sample is far too small to reject chance, to reject the OOS
estimate, or to distinguish between them. The record is reported at this size
anyway, because prospective days are the one class of evidence that cannot be
overfit; its interpretation is taken up in Section 4.3.

> **[Figure 12 placeholder: screenshot of the Models panel with the active
> TCN champion highlighted; optionally a second screenshot of the cumulative
> live-accuracy panel showing the settled prospective record.]**

### 4.3 Data Analysis and Interpretation

Reading **across all tracks** of Section 4.2, several consistent patterns
emerge.

1. **The system's best models beat the no-skill baseline on held-out data,
   consistently and by a few points.** The production champion reaches
   ROC-AUC 0.5731 with accuracy 0.5681 against a ~0.55 base rate on 382
   held-out days, and the registry's best-accuracy cell reaches 0.5780 on 327
   (Section 4.2.9); the unified grid's best cell reaches 0.5916 and its best
   ranker 0.5755 ROC-AUC against a ~0.50 baseline (Section 4.2.8); the
   per-source study's best configuration reaches 0.5794 against 0.5675 and
   holds that margin across 4 of 5 seeds (Section 4.2.2). The margins are in
   the 1-4 point range rather than the 10-point range, which is what an
   achievable edge in daily index-direction forecasting looks like.
2. **Accuracy must always be read against the base rate of its own window.**
   Baselines differ sharply by split - 0.4931 in the transformer notebook,
   0.4976 in the PoC, 0.5642-0.5680 in the later windows - because the up-day
   rate is window-dependent. This cuts both ways: 0.537 on the transformer
   window is a 4.4-point win, while 0.4596 on the tuning window (Section
   4.2.5) is a loss despite looking superficially similar to other numbers in
   the chapter. Every table in Section 4.2 therefore carries its own baseline
   column, and no cross-row comparison of raw accuracy is valid.
3. **Accuracy and ROC-AUC measure different things, and they disagree here.**
   The grid's top-accuracy cell (`TFT [cov=none]`, 0.5916) has a moderate
   ROC-AUC (0.5391), while the top-ROC-AUC cell (`GRU [scored]`, 0.5755) is
   mid-table on accuracy. The registry exposes the same tension in its
   sharpest form and resolves it explicitly: its best-accuracy cell
   (PatchTST, 0.5780) ranks worse than chance (ROC-AUC 0.4795), while the
   activated champion (TCN) is the zoo's best ranker with near-best accuracy.
   The practical reading is that some models are good at *calling* direction
   and others are good at *ranking* confidence; a system whose serving path
   consumes probabilities selects on the latter, which is why the registry's
   default selection metric is ROC-AUC, with accuracy available behind a
   single flag. Reporting a single metric would obscure this, which is why
   the metric set of Section 3.7 is reported in full.
4. **No single data-type or model family dominates.** `scored`, `embedded`,
   and `fused` views all produce competitive cells; zero-shot foundation
   models do not beat trained ones; complex transformers do not automatically
   beat GRU/TCN/XGBoost. The feature-group ablations (Sections 4.2.2 and
   4.2.4) show that market features carry a large share of the predictive
   work and that news features alone are the weakest single group - yet the
   deployed champion runs on the **fused** frame, where news and market
   features are combined. The combination, not either source alone, is what
   the production result rests on.
5. **The edge approaches conventional significance without a large sample to
   support it.** The closest single test is the per-source LGBM configuration
   at `p_perm = 0.052` (Section 4.2.2). With 200-500 day evaluation windows, a
   2-4 point edge is near the resolution limit of the test; the champion's 382
   held-out days and the prospectively accumulating live record are the two
   mechanisms that will continue to sharpen this estimate over time.
6. **In-sample scores are a warning, not a result.** The all-days in-sample
   evaluation (`champion_full_eval`) reaches accuracy near 1.0 - a 600-tree
   XGBoost memorizing 2,586 days of 970 features. Displayed next to the ~0.55
   OOS numbers on the dashboard, it demonstrates concretely why leakage-free
   evaluation is non-negotiable in this domain, and why the modest OOS margins
   above are the numbers worth trusting.
7. **The live record so far sits below its OOS estimate, and the honest
   reading is "too early to tell."** The 46.4% settled live accuracy (n = 28,
   Section 4.2.9) is 10 points below the champion's 56.8% OOS accuracy, and
   three mechanisms plausibly contribute. First, sampling noise dominates at
   this size: the binomial interval [0.275, 0.661] contains chance, the
   baseline, and the OOS estimate alike, so the gap is not yet statistically
   a gap. Second, the live window samples a single three-and-a-half-month
   market regime, whereas the OOS tail averages over more than a year of
   mixed regimes - exactly the window-dependence that item 2 warns about,
   now operating prospectively. Third, the live era's features are scored by
   `gemma4` while the entire training history was scored by
   `mistral-small-4` (Section 3.2), a data-provenance boundary that shifts
   the input distribution in ways the model never saw in training. None of
   the three can be separated at n = 28; the settlement loop accumulates the
   evidence that eventually will, and re-standardizing the history onto a
   single scoring model (Section 5) removes the third mechanism outright.

### 4.4 Comparison with Existing Approaches

**Internal comparison across tracks.** The notebook tracks (Sections
4.2.1-4.2.5), the hardened package (Sections 4.2.6 and 4.2.8), and the
production registry (Section 4.2.9) tell a coherent story, and the package and
registry numbers are the trustworthy ones. The exploratory notebooks vary
their splits and baselines and can show larger gaps on a single favorable
window; the hardened, fixed-window runs are the ones that survive the removal
of window-selection freedom. That the registry's best cells still clear their
baseline by 1-3 points *after* those degrees of freedom are removed is the
point: the margin is small because it is measured honestly, not because the
measurement was pessimistic. The prospective live record extends the same
honesty one step further - it is reported even while it sits below the
baseline, because a record that is only published when favorable is not
evidence.

**Comparison with the literature.** The magnitude of the effect is consistent
with the published consensus on news-tone signal for next-day index
direction, where reported edges of a few percentage points over a base rate
are the norm and larger claims often trace back to evaluation leakage or
favorable-period selection. SentiSense lands in that band with the leakage
controls made explicit and auditable. Notably, the feature-importance
breakdown (news features carry about 79% of total importance in the CatBoost
ablation, Section 4.2.2) confirms the models genuinely lean on the news
signal rather than merely re-deriving price momentum.

### 4.5 Discussion of Findings

The result is that an LLM-scored Hebrew-news stream, fused with market
context, supports a **measurable and repeatable next-day directional edge for
the TA-125 on held-out data**: the activated TCN champion reaches ROC-AUC
0.5731 with 56.8% accuracy on 382 held-out days against a ~55% no-skill
baseline, the registry's best-accuracy cell reaches 57.8% on 327 days, and
comparable margins are reproduced across independent tracks. The prospective
live record - 46.4% on 28 settled days, with a confidence interval spanning
chance and the OOS estimate alike - is so far statistically uninformative in
either direction, and is reported as such. The edge is a few percentage
points, it approaches conventional significance rather than clearing it
decisively, and it is reported that way deliberately. It is also credible
precisely *because* of the leakage controls - the pipeline was built to make
it difficult to overstate a result, and the number survived that pipeline.

The project's value is therefore both empirical and infrastructural. It
delivers (a) a reproducible, auditable, leakage-safe pipeline spanning
scraping, LLM scoring, feature engineering, and a tuned model zoo; (b) a
uniform, resumable comparison framework; (c) a **production loop** - model
registry, nightly orchestration, settlement, and dashboard - that keeps
extending the out-of-sample record prospectively; and (d) a quantified,
multi-metric reference point against which future richer signals (intraday
data, magnitude targets, alternative LLM scoring schemes, longer horizons)
can be measured.

Limitations include the close-to-close-only target, the daily resolution, the
all-zero LLM rows, and the mixed scoring-model history: the historical corpus
was scored by `mistral-small-4` and the live era by `gemma4` (Section 3.2), a
data-provenance boundary whose effect on feature comparability is monitored
and will be removed when the history is re-standardized onto a single scoring
model. The accuracy/ROC-AUC tension in champion selection (Section 4.2.9) and
the short, so-far-below-baseline live record (Section 4.3, item 7) are
further open items. Each is a concrete lever for future work.

---
## 5. Conclusion and Future Work

**Conclusion.** SentiSense set out to test whether LLM-distilled Hebrew-news
sentiment can predict next-day TA-125 direction, and to do so in a way that
survives contact with production. It produced a complete, leakage-hardened,
reproducible system: scraper, LLM scorer selected through a golden-dataset
quality gate, a scored corpus of over three million headlines (3,099,081
raw headlines across 64 sources, 2010-2026, with 3,167,851 score vectors and
3,100,946 embeddings), daily feature engineering with embedding-derived and
narrative features, a database-backed model registry holding 29 trained
versions across twelve model families plus a top-3 soft-vote ensemble, with
automatic champion selection and manual override, a nightly
scrape-score-predict-settle orchestrator on a two-host deployment, and an
interactive dashboard that presents the prediction, the evidence, and the
data itself.

Beyond the modeling core, the deployment matured into a genuinely live
service: the dashboard is served over TLS behind nginx with the college
wildcard certificate, the entire site sits behind a login gate (an
environment-supplied password exchanged for an HMAC-signed session cookie,
enforced on both the HTTP API and the WebSocket handshake), the
Model-Performance panel is versioned and editable through MongoDB-backed
snapshots, a nightly persona simulation renders each news outlet as an agent
in a stance graph, every simulation run is archived version-by-version to
MongoDB so re-runs never destroy history, and a database-mediated LLM queue
(`llm_requests`) lets the UI host ask the GPU host for narratives, answers,
and simulations through the only channel the firewall permits - PostgreSQL
itself.

The empirical answer is affirmative on the research record and honestly
unresolved on the live one. **The registry's automatic selection on
out-of-sample ROC-AUC promoted a TCN (`tcn-20260702-1351`, fused feature
set, full-history regime, overnight features) as the serving champion on
July 6, 2026, with 0.5731 ROC-AUC and 56.8% accuracy on 382 held-out
days**; PatchTST remains the best-accuracy grid cell at 57.8% on 327 days
but with a ROC-AUC of 0.4795 that the ranking metric correctly declined to
reward. (Earlier documents named PatchTST, and later XGBoost, as champion;
both described earlier states of the registry - the auto-best selector's
choice of the TCN is the current, database-verified truth.) Independent
tracks reproduce research margins of the same order - 0.5916 accuracy and
0.5755 ROC-AUC at the top of the unified out-of-sample grid, 0.5794 against
a 0.5675 baseline in the per-source study. The live record, however, is
still too short to confirm any of it: of 29 live predictions logged since
April 29, 2026, 28 have settled, at a directional accuracy of 46.4% -
below the ~55% majority-class base rate, and at n=28 statistically
indistinguishable from chance in either direction. This is exactly the
outcome the book's own "weak but real signal" framing anticipates: a small
edge measured under strict controls can take hundreds of live days to
resolve, and the platform exists precisely to keep that measurement running.

The contribution is therefore twofold: a **credible, leakage-controlled
directional edge** from Hebrew-news sentiment on the TA-125 measured on the
research record, and a **live, self-auditing platform** that keeps testing
that edge against reality, one settled trading day at a time - and reports
the result even when, as now, the early live sample is unflattering.

**Future work.**

1. **Richer targets.** Persist TA-125 OHLC to enable overnight-gap and
   intraday-return (magnitude) targets, which may carry more news signal than
   close-to-close direction.
2. **Longer horizons & event studies.** Test weekly direction and
   event-window responses around high-impact (very negative +
   high-security-relevance) headlines.
3. **Scoring-era standardization.** Re-score the historical corpus under the
   live scoring model (`standardize_to_latest_model.py`) and retrain the zoo
   on a single-era feature space, removing the mistral-to-gemma provenance
   boundary.
4. **Zero-shot serving.** Extend the champion's dispatch with the
   `reforecast` path so a registered Chronos/TimesFM/TFT winner can be served
   by live re-forecasting, not only evaluated.
5. **Persona analytics.** Backtest the per-source persona votes as
   predictors in their own right ("which outlet is the best forecaster?") and
   consider credibility-weighted aggregation as a feature.
6. **Robustness and monitoring.** Multi-seed registry evaluations as the
   default, drift monitoring on the live feature distributions, and periodic
   automatic re-training gates tied to the cumulative live record.
7. **Explainability.** Execute the TimesFM explainability track (Section
   4.2.7) and add SHAP-based attribution for the served champion to the
   dashboard.
8. **Trading-week migration.** The TASE moved to a Monday-Friday trading week
   on January 5, 2026. The pipeline currently encodes the previous
   Sunday-Thursday week in the `_TASE_TRADING_WEEKDAYS` constant and in the
   weekend-news rollover rule (Section 3.3). Updating the weekday constant,
   switching the rollover from Friday/Saturday-to-Sunday to
   Saturday/Sunday-to-Monday, and updating the calendar tests accordingly
   would align the pipeline with the new schedule.
9. **Public DNS record.** The TLS configuration, certificate, and nginx
   virtual host for `sentisens.cs.colman.ac.il` are deployed and working;
   the public DNS A record is still pending, so the site is currently
   reachable only via the external IP or the internal hostname. Publishing
   the record completes the go-live.
10. **Live-sample growth before significance claims.** With only 28 settled
    live predictions, no statement about the live edge - positive or
    negative - clears a binomial test at conventional power. The serving
    configuration should be held stable and the live record allowed to grow
    to hundreds of settled days before any promotion, demotion, or headline
    claim is made on live accuracy alone.
11. **Finance-CSV path fix after the re-organization.** The repository
    re-organization moved the manually exported market-data files
    (`TA 125 Historical Data.csv`, `Tel Aviv Volatility Index VTA35
    Historical Data.csv`) under `evaluation/`, while the path constants in
    `sentisense/constants.py` still resolve them at the repository root.
    The constants should be pointed at the new location (ideally a `data/`
    directory with shell-safe filenames).
12. **uv workspace consolidation.** Three separate uv projects remain
    (repository root, `processing_engine/`, `mivzakim_scraper/`), each with
    its own `pyproject.toml` and lockfile. Consolidating them into a single
    uv workspace would remove duplicate dependency resolution and the
    cross-project `uv run --project` indirection.

---

## 6. References

### Cited literature

[1] P. C. Tetlock, "Giving Content to Investor Sentiment: The Role of Media in
the Stock Market," *Journal of Finance*, vol. 62, no. 3, pp. 1139-1168, 2007.

[2] J. Bollen, H. Mao, and X. Zeng, "Twitter mood predicts the stock market,"
*Journal of Computational Science*, vol. 2, no. 1, pp. 1-8, 2011.

[3] T. Loughran and B. McDonald, "When Is a Liability Not a Liability? Textual
Analysis, Dictionaries, and 10-Ks," *Journal of Finance*, vol. 66, no. 1,
pp. 35-65, 2011.

### Software and model families used in this project

The following are the tools, libraries, and model architectures the system
actually uses, listed with their originating publications. They are referred
to by name in the text rather than by citation number. All are installed via
`pyproject.toml` extras except TimesFM, which is a documented manual install
rather than a pinned dependency.

- **XGBoost** - T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting
  System," in *Proc. KDD*, 2016, pp. 785-794.
- **LSTM** - S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory,"
  *Neural Computation*, vol. 9, no. 8, pp. 1735-1780, 1997.
- **GRU** - K. Cho et al., "Learning Phrase Representations using RNN
  Encoder-Decoder for Statistical Machine Translation," in *Proc. EMNLP*,
  2014.
- **TCN** - S. Bai, J. Z. Kolter, and V. Koltun, "An Empirical Evaluation of
  Generic Convolutional and Recurrent Networks for Sequence Modeling,"
  arXiv:1803.01271, 2018.
- **Temporal Fusion Transformer (TFT)** - B. Lim et al., "Temporal Fusion
  Transformers for Interpretable Multi-horizon Time Series Forecasting,"
  *International Journal of Forecasting*, vol. 37, no. 4, pp. 1748-1764, 2021.
- **PatchTST** - Y. Nie et al., "A Time Series is Worth 64 Words: Long-term
  Forecasting with Transformers," in *Proc. ICLR*, 2023.
- **N-BEATS** - B. N. Oreshkin et al., "N-BEATS: Neural basis expansion
  analysis for interpretable time series forecasting," in *Proc. ICLR*, 2020.
- **N-HiTS** - C. Challu et al., "N-HiTS: Neural Hierarchical Interpolation
  for Time Series Forecasting," in *Proc. AAAI*, 2023.
- **Chronos** - A. F. Ansari et al., "Chronos: Learning the Language of Time
  Series," *Transactions on Machine Learning Research*, 2024.
- **TimesFM** - A. Das et al., "A decoder-only foundation model for
  time-series forecasting," in *Proc. ICML*, 2024. Installed manually; not a
  pinned `pyproject.toml` extra.
- **multilingual-E5** - L. Wang et al., "Text Embeddings by
  Weakly-Supervised Contrastive Pre-training," arXiv:2212.03533, 2022. Model
  identifier `intfloat/multilingual-e5-base`.
- **Optuna** - T. Akiba et al., "Optuna: A Next-generation Hyperparameter
  Optimization Framework," in *Proc. KDD*, 2019.
- **Cytoscape.js** - M. Franz et al., "Cytoscape.js: a graph theory library
  for visualisation and analysis," *Bioinformatics*, vol. 32, no. 2,
  pp. 309-311, 2016. Renders the Simulator tab's agent stance graph.
- **LightGBM, CatBoost, scikit-learn, PyTorch, pytorch-forecasting,
  sentence-transformers, FastAPI, React, Playwright, LangGraph, PostgreSQL,
  pgvector, MongoDB, pymongo, and Ollama** - standard open-source
  components; Python packages are pinned in `pyproject.toml`. **nginx**
  (TLS termination and reverse proxy) and **MongoDB** are deployed as
  system services on the front host rather than as Python dependencies.

---

## 7. Appendix A - Data Dictionary, Schema, and Commands

### A.1 Database schema (PostgreSQL 16)

`raw_headlines` - one row per scraped headline (source of truth):
`id` (PK), `date`, `source`, `hour`, `popularity`, `headline` (Hebrew, UTF-8),
`created_at`, `headline_hash` (`md5(headline)`, stored). Unique on
`(date, source, hour, headline_hash)`.

`nlp_vectors` - one row per `(headline, model)`:
`id` (PK), `headline_id` (FK), `model_name`, six `relevance_*` SMALLINT
(0-10), `global_sentiment` SMALLINT (-10...+10), `validation_passed` BOOLEAN,
`processing_time_seconds`, `errors`, `created_at`. Unique on
`(headline_id, model_name)`.

`headline_embeddings` - one row per `(headline, embed_model)`: 768-d float32
vector stored as raw bytes (`BYTEA`), no vector-extension dependency.

`headline_vectors` - pgvector mirror of the embedding cache: a `vector(768)`
column with PK `(headline_id, embed_model)`, populated from
`headline_embeddings` by `scripts/deploy_vectordb.py` and indexed with a
cosine HNSW (or IVFFlat) index for approximate-nearest-neighbour search.

`daily_embedding_derived` - one JSONB row per `(date, embed_model)`: the
leak-safe 16 `embpca_*` + 8 `embclus_dist_*` features, with the recorded
`fit_cutoff` marking the boundary of the window the transform basis was fit on.

`embedding_pca_basis` - the persisted transform basis (scaler mean/scale, PCA
mean/components, KMeans centers) that projects headlines into the same space.

`model_registry` - one row per trained candidate: version (unique), family,
hyper-parameters (JSONB), OOS metrics (ROC-AUC + CI, MCC, accuracy, n),
serialized artifact (`BYTEA`; joblib / torch state-dict / ensemble /
reforecast), feature-column contract, `is_active` (partial-unique: at most
one), `activated_by` (auto | manual), timestamps.

`model_predictions` - the live inference log: `(date, model_version)` unique,
prediction, confidence, `actual` (NULL until settled).

`champion_full_eval` - per-day in-sample evaluation of the champion (see
Section 3.7).

`narrative_sim`, `narrative_sim_graph`, `narrative_sim_report` - cached
narrative-simulation outputs consumed by the Simulator tab.

`llm_requests` - the database-mediated LLM job queue (migration 008): one
row per request with `kind` (ask | narrate | simulate), status, prompt, and
answer. The UI host inserts rows; the GPU-side worker
(`scripts/llm_worker.py`) claims them with `SELECT ... FOR UPDATE SKIP
LOCKED` and writes the answers back. This table is the sole UI-to-GPU
transport, since the firewall between the hosts passes only PostgreSQL.

`daily_features` - **legacy**: created by `scripts/init_db.sql` for the
original aggregation design but empty in the live database; superseded by
on-the-fly dataset assembly in `sentisense/features/dataset.py`.

### A.2 Score-scale reference

- **Relevance** (six columns): integer 0-10; higher = more relevant to that
  category.
- **Sentiment** (`global_sentiment`): integer -10 (very negative) to +10
  (very positive); 0 = neutral/mixed.
- **`validation_passed`**: TRUE = parseable, in-range LLM output. Always
  filter on TRUE for analysis.

### A.3 Reproduction commands

```bash
# 0 - database (schema auto-initialises from scripts/init_db.sql; migrations 001-008 are idempotent)
docker compose up -d

# 1 - scrape headlines
cd mivzakim_scraper && uv sync && uv run playwright install firefox && uv run python main.py

# 2 - score unscored headlines into nlp_vectors (gap-only, backend-aware)
cd processing_engine && uv sync
uv run python ../scripts/process_headlines.py --fast --unscored-any-model --concurrency 4

# 3 - research pipeline: features, embeddings, models, leaderboard
uv sync --extra ml --extra embed --extra finance          # at repo root
uv run python -m sentisense.pipeline --from features       # leakage-safe chronological split

# 4 - full comparison leaderboard (server, run in tmux)
uv sync --extra ml --extra finance --extra embed --extra tft --extra chronos
uv run python scripts/pipeline_compare.py --seq-trials 30 --pf-trials 12 --xgb-trials 60

# 5 - registry training over the full zoo, then auto-activate the champion
uv run --extra finance --extra ml --extra tft --extra chronos python scripts/train_registry.py \
    --trials 100 --seq-models lstm,gru,tcn,patchtst --seq-trials 40 --seq-seeds 3 \
    --forecasters chronos,timesfm,tft,nhits,nbeats --select-metric oos_roc_auc

# 6 - one nightly cycle by hand (normally run by cron)
uv run --extra finance --extra ml python scripts/daily_live.py

# 7 - dashboard (on the DB/UI host)
cd ui/frontend && npm install && npm run build && cd ../..
uv run --extra ui --extra finance --extra ml python -m ui.app     # serves on :3000
```

Two layout notes for the re-organized repository: the research notebooks now
live under `notebooks/` (launch e.g.
`uv run --project processing_engine jupyter lab notebooks/tuning.ipynb` from
the repo root), and the manually exported market-data files
(`TA 125 Historical Data.csv`, `Tel Aviv Volatility Index VTA35 Historical
Data.csv`, both from investing.com) now live under `evaluation/` - the
`finance` extra's loaders require them to be present and current. All
`scripts/` paths above are unchanged by the re-organization.

### A.4 Repository map

```
sentisense/          core forecasting + serving package
  constants.py         active model name, score contract, cutoff date
  config.py            modeling/HPO knobs (env-overridable)
  db/                  SQLAlchemy engine (env-only DSN) + migrations 001-008
  ingest/              backfill · score · coverage report
  features/            leak-safe daily dataset assembly (incl. serving mode)
  embed/               multilingual-e5 embeddings · derived PCA/cluster block · basis
  cluster/             causal expanding-window narrative clustering
  models/              sequence datasets, train harness, model zoo, baselines
  hpo/                 resumable Optuna HPO + held-out test-tail evaluation
  serve/               model registry + champion serving (fallback-safe)
  sim/                 narrative-simulation client, cache, graph API, local persona sim
  pipeline.py          research orchestrator
notebooks/           10 research notebooks: eda · poc · lstm_forecaster · tuning ·
                     transformer_forecaster · sentisense_analysis ·
                     compare_lstm_features_with_poc · timesfm_explainability ·
                     miro_explainability
ui/                  FastAPI backend (ui/app.py, ui/queries.py) + React SPA (ui/frontend)
mivzakim_scraper/    Playwright scraper for mivzakim.net (Hebrew news)
processing_engine/   LLM scoring pipeline (fast single-prompt + 7-agent LangGraph)
evaluation/          LLM-scoring benchmark vs golden dataset; also hosts the manual
                     TA-125 / VTA-35 market-data CSV exports
scripts/             34 operational scripts: init_db.sql · backfill · process/retry/
                     standardize · pipeline_compare · train_registry · daily_live ·
                     settle_predictions · sim_daily · llm_worker ·
                     archive_sims_to_mongo · deploy_vectordb · compute_full_eval ·
                     build_embedding_derived · migrate_db
ops/                 crontab template · nginx TLS site config · pm2 process config ·
                     container startup script
tests/               23 pytest files - leakage, calendar rollover, registry serve,
                     projection math, simulation, daily orchestration
docs/                RUNBOOK · LIVE_RUNBOOK · MODEL_ZOO · DATA_HANDOFF · VECTORDB ·
                     LIVE_INVENTORY and design notes
external/MiroFish    git submodule - multi-agent narrative-simulation service
pyproject.toml       root package "sentisense" (uv; extras: finance, ml, embed,
                     ui, miro, tft, chronos, dev, notebook) + uv.lock
```

---

## 8. Appendix B - Live Deployment Runbook (summary)

The full operational document is `docs/LIVE_RUNBOOK.md`; this appendix
summarizes the deployed configuration as verified on both machines.

**Hosts.** A GPU compute container (RTX 4090, 24 GB; repository at
`/tf/Data-Science-Final-Project`) runs the pipeline, local LLMs (Ollama
`gemma4:latest` alongside the vLLM scoring backend), registry training, and
the database-queue LLM worker. A front machine (internal `10.10.248.109`,
external `193.106.55.109`) runs PostgreSQL 16, MongoDB on port 21771, the
pm2-supervised FastAPI + SPA application `sentisense-ui` on port 3000, and
nginx 1.29.4, which terminates TLS on port 443 with the college wildcard
certificate `*.cs.colman.ac.il` (valid to January 2027), proxies HTTP and
WebSocket traffic to port 3000, and redirects port 80 with a 301. The public
hostname `sentisens.cs.colman.ac.il` awaits its public DNS A record. The
entire site sits behind a login page: the password comes from the
`SENTISENSE_UI_PASSWORD` environment variable (never committed), successful
login sets an HMAC-signed `ss_auth` session cookie, FastAPI middleware
guards `/api` and `/ws` (excepting `/api/login` and `/api/auth`), and the
WebSocket handshake re-checks the cookie. All cross-host traffic is
database-mediated; the only required configuration on each host is
`SENTISENSE_DATABASE_URL` plus the scoring-backend variables (and
`SENTISENSE_MONGO_URL` on the front machine).

**Schedule.** Both machines run cron; the GPU container's clock is UTC and
the front machine's is Asia/Jerusalem.

| Host (timezone) | Time | Job |
|---|---|---|
| GPU container (UTC) | 15:30 | `scripts/daily_live.py` - scrape, score, embed, derived features, champion prediction |
| GPU container (UTC) | 15:45 | `scripts/settle_predictions.py` - fill realized outcomes |
| GPU container (UTC) | 17:00 | `scripts/sim_daily.py --backfill 3` - nightly persona simulation |
| GPU container | @reboot | `scripts/llm_worker.py` - claims ask/narrate/simulate jobs from `llm_requests` via `FOR UPDATE SKIP LOCKED` |
| Front machine (Asia/Jerusalem) | 17:20 | `scripts/archive_sims_to_mongo.py --days 14` - versioned simulation archive to MongoDB |
| Front machine (Asia/Jerusalem) | 18:30 | `sentisense_daily.sh` |
| Front machine (Asia/Jerusalem) | 18:45 | `sentisense_settle.sh` |

As deployed, the orchestrator treats Sunday-Thursday as the trading week and
skips Friday, Saturday, and listed holidays; the TASE moved to a
Monday-Friday week on January 5, 2026, so aligning this constant with the
new schedule is listed as future work in Section 5. Registry re-training
(`scripts/train_registry.py`) is run periodically, not nightly - champion
serving is decoupled from training by design.

**Failure modes and their handling.**

| Failure | Behavior |
|---|---|
| Registry table missing / artifact incompatible | champion falls back to the pinned XGBoost config; logged loudly |
| Scoring backend switched (vLLM ↔ Ollama) | orchestrator selects per-backend flags; gap-only scoring prevents re-scoring covered history |
| Simulation service down | Simulator tab disables live runs and serves cached graphs, with an explicit banner |
| No WebGL in the operator's browser | 3-D views render through the software-3D fallback |
| Freshly promoted champion (no live rows) | dashboard seeds metrics from the model's held-out evaluation and labels the eval/live split |
| Public DNS record missing | site remains reachable via the external IP or internal hostname; the TLS certificate already covers the target name, so publishing the A record is the only remaining step |
| Simulation missing for a day | `sim_daily.py` runs with `--backfill 3`, so up to three missed days are regenerated automatically on the next nightly run |

**Data-freshness contract.** Every dashboard panel maps to the pipeline
stage that produces its data (documented per-panel in the runbook); each
degrades to an explicit "no data" state rather than an error when its
producer has not yet run.

---

*Screenshots for the figure placeholders are to be completed before
submission.*
