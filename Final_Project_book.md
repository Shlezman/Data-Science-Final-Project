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

The system has four stages. (1) A **scraper** collects Hebrew breaking-news
headlines from `mivzakim.net` going back to ~2015. (2) A **processing engine**
sends every headline through an LLM, which scores it on six relevance
categories (politics, economy, security, health, science, technology) and one
global sentiment value (−10…+10). This produced a corpus of roughly **1.9
million scored headlines** stored in PostgreSQL. (3) A **feature-engineering
layer** aggregates these per-headline scores into leakage-safe daily feature
vectors, joined with market data (TA-125 OHLC, the VTA-35 volatility index,
S&P 500, VIX, Brent crude, and the USD/ILS exchange rate), and optionally with
multilingual headline **embeddings** and causal **narrative-clustering**
features. (4) A **forecasting layer** trains and hyperparameter-tunes a large
model zoo — gradient-boosted trees, recurrent and convolutional sequence
classifiers, transformer forecasters, and zero-shot foundation models — to
predict next-day TA-125 direction.

Every stage is engineered to be **leakage-safe**: a hard data cutoff of
`2023-10-07` (the regime break preceding a major market shock) is enforced in
SQL and re-applied after feature assembly; all scalers, PCA, and clustering are
fit on the training fold only; and splits are strictly chronological.

The central empirical finding is sobering and honest, and it is corroborated
across **multiple independent experiment tracks** — a tree-model proof of
concept, an extensive feature-set comparison with walk-forward and multi-seed
robustness checks, a nine-model transformer zoo, sequence-model HPO, and a
hardened end-to-end package run of **40+ tuned model × data-type × regime
cells**. In every track, out-of-sample performance hovers close to the no-skill
baseline. On the unified grid the best model by ROC-AUC reaches only **0.576**
and the best by accuracy **0.592**; on the hardened pipeline the score-only LSTM
sits at accuracy 0.500 / ROC-AUC 0.509 / MCC 0.000; and no model anywhere meets
the transformer track's pre-registered ≥58% success criterion or clears a
permutation/binomial significance test. The project's contribution is therefore
twofold: a **reusable, reproducible, leakage-hardened pipeline** for news-driven
financial forecasting, and a rigorous, negative-leaning result that quantifies
how little next-day directional signal the LLM-scored Hebrew-news stream carries
on its own.

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
   - 3.3 Implementation Details
   - 3.4 Evaluation Metrics
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
   - 4.3 Data Analysis and Interpretation
   - 4.4 Comparison with Existing Approaches
   - 4.5 Discussion of Findings
5. Conclusion and Future Work
6. References
7. Appendix A — Data Dictionary, Schema, and Commands

---

## List of Figures

- Figure 1: SentiSense end-to-end pipeline (§1.6)

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

---

## Table of Abbreviations

| Abbreviation | Meaning |
|---|---|
| TA-125 | Tel-Aviv 125 stock index |
| VTA-35 | Tel-Aviv 35 Volatility Index |
| LLM | Large Language Model |
| NLP | Natural Language Processing |
| HPO | Hyper-Parameter Optimization |
| ROC-AUC | Area Under the Receiver Operating Characteristic Curve |
| F1 | F1 score (harmonic mean of precision and recall) |
| MCC | Matthews Correlation Coefficient |
| LSTM | Long Short-Term Memory network |
| GRU | Gated Recurrent Unit |
| TCN | Temporal Convolutional Network |
| TFT | Temporal Fusion Transformer |
| PCA | Principal Component Analysis |
| OHLC | Open / High / Low / Close (price data) |
| FX | Foreign Exchange rate |
| DSN | Database Source Name (connection string) |

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
Hebrew breaking-news feed into a structured daily sentiment signal, and asks
whether that signal helps forecast the TA-125.

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

### 1.3 Objectives

1. **Build a reproducible ingestion-and-scoring pipeline** that scrapes Hebrew
   headlines and scores each on six relevance categories plus a global
   sentiment, persisting the result in a relational database.
2. **Engineer leakage-safe daily features** combining the news scores with
   market and macro data, with optional embedding- and narrative-based signals.
3. **Train and rigorously hyperparameter-tune a broad model zoo** for next-day
   TA-125 direction, on a strictly chronological, cutoff-bounded split.
4. **Quantify the predictive value honestly** using threshold-free and
   threshold-based metrics, against a Buy&Hold / majority baseline.
5. **Produce reusable artifacts** (a Python package, scripts, notebooks, and an
   auto-generated leaderboard) so the experiment is fully reproducible.

### 1.4 Scope and Limitations

**In scope:** Hebrew-headline scraping; LLM scoring into a 7-dimensional vector;
daily feature engineering; embeddings and causal narrative clustering;
classification and forecasting models with HPO; a comparison leaderboard.

**Out of scope / limitations:**

- **Target.** The database stores only a boolean next-day direction; continuous
  TA-125 OHLC exists only in a CSV. The project therefore predicts
  **close-to-close direction**, not overnight-gap or intraday-return magnitude.
- **Cutoff.** All modeling is bounded to `≤ 2023-10-07`; the post-cutoff regime
  is used only as a read-only sanity overlay, never for training.
- **Intraday.** The system is daily-resolution; no tick or minute data.
- **Causality.** The work measures *predictive association*, not economic
  causation.
- **Data quirks.** A non-trivial fraction of "validated" LLM rows are all-zero
  (a known LLM failure mode treated as missing); the corpus mixes LLM model
  versions across disjoint date ranges.

### 1.5 Methodology

The project follows a staged, gate-driven methodology:

1. **Ingest** Hebrew headlines (backward scrape to ~2015) into `raw_headlines`.
2. **Score** each headline with an LLM into `nlp_vectors` (7 scores +
   validation flag).
3. **Assemble** leakage-safe daily frames: daily-mean scores, per-source score
   pivots, sentiment×relevance interactions, multilingual embedding centroids,
   causal narrative-cluster features, and a finance/market block.
4. **Split** chronologically (≈70/15/15) with all transforms fit on train only.
5. **Model & tune** a zoo of classifiers and forecasters with Optuna HPO.
6. **Evaluate** every model on the same sacred out-of-sample window using
   ROC-AUC, F1, accuracy, balanced accuracy, and MCC, plus a backtest overlay.
7. **Compare** all cells in a single auto-generated leaderboard.

### 1.6 Organization of the Project Book

- **Chapter 2** reviews relevant literature on news-driven financial prediction
  and LLM-based sentiment extraction.
- **Chapter 3** details the system architecture, data collection and
  preprocessing, implementation, and evaluation metrics.
- **Chapter 4** presents the experimental setup, the full results leaderboard,
  and an analysis and discussion of the findings.
- **Chapter 5** concludes and proposes future work.
- **Chapter 6** lists references; **Appendix A** gives the data dictionary,
  schema, and reproduction commands.

```
 ┌──────────────────┐  headlines  ┌────────────────────┐  7 scores  ┌─────────────┐
 │ mivzakim_scraper │ ──────────▶ │  processing_engine │ ─────────▶ │ PostgreSQL  │
 │  Playwright/FF   │             │  LLM scoring       │/headline   │ raw_headlines│
 │  mivzakim.net    │             │  (fast / 7-agent)  │            │ nlp_vectors  │
 └──────────────────┘             └────────────────────┘            └──────┬──────┘
        ┌──────────────────────────────────────────────────────────────── ┘
        ▼
 ┌────────────────────────────┐  features  ┌─────────────────────────────┐
 │  sentisense/ (Phase 2&3)   │ ─────────▶ │  Forecasting model zoo       │
 │ features·embed·cluster·HPO │            │ trees/LSTM/GRU/TCN/PatchTST/ │
 │ leakage-safe, ≤ 2023-10-07 │            │ TFT/N-HiTS/Chronos → TA-125  │
 └────────────────────────────┘            └─────────────────────────────┘
```
*Figure 1: SentiSense end-to-end pipeline.*

---

## 2. Literature Review

### 2.1 Overview of Relevant Literature

The project draws on three strands of prior work.

**News sentiment and market prediction.** A long line of research links the
tone of financial and general news to subsequent market movements. Tetlock [1]
showed that media pessimism predicts downward pressure on prices and reversion,
establishing news tone as a market-relevant variable. Bollen et al. [2] famously
linked aggregate mood derived from social media to movements in the Dow Jones.
The consistent theme is that *signal exists but is small and regime-dependent*,
and that careful, leakage-free evaluation is essential — exactly the posture
this project adopts.

**Lexicon vs. model-based sentiment.** Domain-specific lexicons such as Loughran
and McDonald [3] demonstrated that general-purpose sentiment dictionaries
mislabel financial text, motivating domain-aware scoring. Modern LLMs
generalize this idea: instead of a fixed lexicon, a prompted model performs
context-aware topical-relevance and sentiment scoring, and — relevant here —
does so across languages, including Hebrew, without a hand-built Hebrew lexicon.

**Sequence and foundation models for forecasting.** On the modeling side, the
project surveys the standard time-series toolkit: gradient-boosted trees
(XGBoost [4]) as strong tabular baselines; recurrent networks (LSTM [5], GRU
[6]) and temporal convolutions (TCN [7]) for sequence classification;
transformer forecasters such as the Temporal Fusion Transformer [8] and PatchTST
[9]; deep interpretable forecasters N-BEATS [10] and N-HiTS [11]; and zero-shot
foundation forecasters Chronos [12] and TimesFM [13]. Multilingual sentence
embeddings (multilingual-E5 [14]) provide the Hebrew-aware vector
representations used for the embedding and narrative-clustering features.

The research gap this project addresses: most prior work is English-centric and
often under-controls for leakage. SentiSense contributes a **Hebrew-news,
LLM-scored, strictly leakage-controlled** evaluation across a broad model zoo,
and reports the result honestly rather than selectively.

---

## 3. System Design and Implementation

### 3.1 System Architecture

The system is organized as four loosely-coupled modules communicating through a
PostgreSQL 16 database, so each stage can be developed, re-run, and verified
independently.

| Module | Purpose | Entry point |
|---|---|---|
| `mivzakim_scraper/` | Scrape Hebrew headlines (Playwright + Firefox) | `python main.py` |
| `processing_engine/` | LLM scoring (6 relevance + sentiment) | `process_single_observation` / fast pipeline |
| `scripts/` | Data ops: schema, backfill, scoring, retry, standardize | `python scripts/<name>.py` |
| `sentisense/` | Features, embeddings, clustering, models, HPO, orchestration | `python -m sentisense.pipeline` |
| `evaluation/` | Benchmark LLM scoring against a golden dataset | `python -m evaluation.evaluate` |

**Design principles.**

- **Database as the contract.** All inter-stage data flows through four tables
  (`raw_headlines`, `nlp_vectors`, `daily_features`, `model_predictions`),
  decoupling scraping, scoring, and modeling.
- **Single source of truth for constants.** The cutoff date, active model name,
  and score-column contract live in `sentisense/constants.py`, so no magic
  strings leak into feature or model code.
- **Optional, layered dependencies.** Heavy ML/embedding/forecasting libraries
  are `pyproject.toml` *extras* (`ml`, `embed`, `finance`, `tft`, `chronos`),
  so early stages install lightly and torch/CUDA wheels are pinned for
  reproducibility.
- **Leakage-safety as an architectural invariant**, enforced at every layer
  (see §3.2).

### 3.2 Data Collection and Preprocessing

**Collection.** The scraper drives a headless Firefox via Playwright over
`mivzakim.net`, scraping *backward* in time (`scripts/backfill_history.py`)
from the most recent day toward ~2015. Each headline yields a row in
`raw_headlines`: date, source outlet, hour, popularity class, the Hebrew text,
and an ingestion timestamp. Deduplication uses a stored `md5(headline)` hash
(Hebrew strings exceed B-tree index limits) under a unique key of
`(date, source, hour, headline_hash)`.

**Scoring.** The processing engine sends each headline to an LLM. A **fast
single-prompt path** produces all seven scores in one structured call (used for
the bulk backfill on a vLLM `mistral-small-4` server); a legacy **seven-agent
LangGraph path** exists for local Ollama. Each result is a vector of six
relevance integers (0–10), one global sentiment integer (−10…+10), and a
`validation_passed` flag, written to `nlp_vectors`. The corpus contains
**~1.9M scored headlines**.

**Quality control and known quirks** (documented in `DATA_HANDOFF.md`):

- **All-zero "validated" rows.** The LLM sometimes emits all-zeros when it
  cannot categorize a headline; the validator accepts it because all values are
  in range. These are treated as missing data.
- **Mixed model versions.** Earlier rows scored by `mistral-large-2` /
  `mistral-small3.2` were standardized onto `mistral-small-4`; analytical
  queries pin the active model to avoid double-counting.
- **Weekend lull.** Saturday volume is genuinely low (Israeli weekend), not a
  data gap.
- **Encoding / timezone.** All text is UTF-8 Hebrew; event dates/hours are
  Asia/Jerusalem while `created_at` is stored as UTC `TIMESTAMPTZ`.

**Leakage-safe feature assembly** (`sentisense/features/dataset.py`). This is
the heart of the preprocessing and the project's most important engineering
contribution. The module builds daily modeling frames with defense-in-depth
against leakage:

- **Hard cutoff** `≤ 2023-10-07` is pushed into the SQL (`WHERE rh.date <=
  :cutoff`) *and* re-applied after the calendar merge.
- **Event date, never ingestion time.** The cutoff and all splits use
  `raw_headlines.date`, never `created_at`.
- **Trading-calendar rollover.** Weekend/holiday news is rolled *forward* to
  the next trading day (Fri/Sat → Sun) via `np.searchsorted(side='left')`;
  market/FX/volatility series are forward-filled.
- **Causal price features.** TA-125 features (lagged log-returns 1–7, 5d/20d
  rolling stats, Wilder RSI-14, 20-day volume z-score, day-of-week one-hots)
  all use `.shift(>=1)`. Cross-asset features (S&P 500, VIX, Brent, USD/ILS,
  VTA-35) are lagged log-returns only.
- **Train-only scaling.** `StandardScaler` (and optional PCA, scoped by column
  prefix to the embedding block) is fit on the **train slice only**. The
  notebook's earlier full-frame `MinMaxScaler` on VTA-35 — a leak — was replaced
  by a leak-free zero-fill plus a `VTA35_missing` indicator.
- **Honest target.** `Target = (TA125_Price.shift(-1) > TA125_Price)`; the
  trailing row with no next-day price is explicitly set to NA and dropped, so no
  fabricated label is ever produced.

Three feature "views" are produced: a **daily-mean** frame (tree-model shape), a
**per-source** pivot frame (sequence-model shape), and — when embeddings are
cached — an **embedding-centroid** frame and a **fused** frame combining
per-source scores with the daily e5 centroid.

### 3.3 Implementation Details

**Languages, frameworks, and tooling.** Python 3.12, managed by `uv`.
Persistence uses PostgreSQL 16 via SQLAlchemy 2 + psycopg v3; connection
strings come **only** from the `SENTISENSE_DATABASE_URL` environment variable
and the code fails fast if it is unset (no embedded secrets). Core libraries:
pandas/numpy (features), scikit-learn/XGBoost/LightGBM/CatBoost (tabular),
PyTorch (sequence models), Optuna (HPO), sentence-transformers (embeddings),
pytorch-forecasting + Lightning (TFT/N-HiTS/N-BEATS), and Chronos/TimesFM
(foundation forecasters).

**Key implementation decisions and trade-offs.**

- **Notebook → package.** A working but research-grade pipeline lived in
  notebooks (`transformer_forecaster.ipynb`, `lstm_forecaster.ipynb`,
  `tuning.ipynb`). It was extracted into the importable, server-runnable
  `sentisense/` package, hardening the leakage controls in the process (the
  package deliberately does *not* port the notebooks' earlier leaky features
  such as shuffled `StratifiedKFold` or same-day target features).
- **Embeddings.** A Hebrew-aware multilingual model
  (`intfloat/multilingual-e5-base`) is used rather than an English-centric
  default; embeddings are cached so downstream stages never recompute them.
- **Causal narrative clustering** (`sentisense/cluster/narrative.py`). For each
  trading day *T*, a MiniBatch-KMeans model is fit **only on embeddings strictly
  before T** (expanding window with a refit cadence), then day-T headlines are
  *assigned* with that past-fit model — yielding `dominant_cluster_ratio` and
  normalized `cluster_entropy` without any look-ahead.
- **Resumable, cached experimentation.** The comparison driver
  (`scripts/pipeline_compare.py`) writes each finished cell's metrics to
  `leaderboard_cache.json` immediately, so adding or fixing one model costs only
  that cell, not the whole grid.

**Software/hardware.** Development on macOS (CPU); heavy training on a Linux
server with an NVIDIA RTX 4090 (CUDA 12.3 driver). Torch is pinned to CUDA-12.1
wheels for that driver, with a CPU fallback index for local work. PostgreSQL
runs via `docker-compose`.

### 3.4 Evaluation Metrics

Because next-day direction is near-balanced and accuracy alone is misleading,
the project reports a metric set (`sentisense/models/metrics.py`), all computed
on the **same sacred last-15% out-of-sample window**:

- **ROC-AUC** — threshold-free ranking quality; the primary headline metric.
- **F1 (macro)** — balances precision/recall across both classes.
- **Accuracy** and **balanced accuracy** — overall and class-balanced hit rate.
- **MCC** — Matthews correlation, robust to class imbalance.

A **backtest overlay** and a **Buy&Hold** benchmark place the statistical
metrics in an economic context, and a read-only post-cutoff "buy-only" overlay
sanity-checks behavior on the held-out future regime (never used for training).

---

## 4. Results and Analysis

### 4.1 Experimental Setup

Results were produced along **two complementary experiment tracks**, both
reported in full below:

1. **Exploratory notebook tracks** — a sequence of research notebooks
   (`poc.ipynb`, `compare_lstm_features_with_poc.ipynb`,
   `transformer_forecaster.ipynb`, `tuning.ipynb`, `sentisense_analysis.ipynb`,
   `timesfm_explainability.ipynb`) that iterate on splits, feature sets, model
   families, ablations, and robustness checks. These differ deliberately in
   their train/test windows and majority-class baselines, which is itself part
   of the analysis (§4.3).
2. **The unified, hardened package grid** — `scripts/pipeline_compare.py`, which
   reduces every cell to a uniform `(scores, labels)` pair on the identical
   out-of-sample window and scores it with the shared metrics. This is the
   canonical, leakage-hardened cross-model comparison.

> **Note on reproduction state.** A number of cells in the saved notebooks were
> not executed in the committed copy (e.g. the transformer Optuna "tuned
> leaderboard" and McNemar cells; the `tuning.ipynb` GRU/TCN, multi-seed,
> abstention, and final-report cells; and all of `timesfm_explainability.ipynb`).
> To keep this book honest and reproducible, **only metrics that actually
> rendered in the saved outputs are reported**, and each gap is flagged where it
> occurs.

The unified package grid is a three-axis grid evaluated by
`scripts/pipeline_compare.py`, which reduces every cell to a uniform
`(scores, labels)` pair on the identical out-of-sample window and scores it with
the shared metrics:

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

The project generated **many** result tables across its experiment tracks. They
are reproduced faithfully below, organized by source, and then summarized by the
unified package grid (§4.2.8). Every table predicts the **next-day
close-to-close TA-125 direction**.

**Aggregation lineage (how the notebooks relate).** A crucial point for reading
these results is that the notebooks deliberately use *different news-aggregation
strategies*, and the later notebooks expand on the two base ones:

- **Base A — daily-mean aggregation (`poc.ipynb`).** Each day's headlines are
  collapsed to **per-category means** (`mean_politics … mean_sentiment`, plus
  `std_sentiment`, `pct_negative`, `pct_positive`). This is the compact,
  tree-friendly representation. (The base PoC also added `LastDay_Rise` /
  `LastDay_Pct` features built with a `shift(-1)`, a leaky construction that the
  hardened package deliberately drops.)
- **Base B — per-source wide aggregation (`lstm_forecaster.ipynb`).** Scores are
  **summed per `(date, source)` and pivoted wide** (`<dim>_<source>` ⇒ 320
  feature columns), preserving *which outlet produced which signal*, then sliced
  into 30-day windows for the LSTM.
- **Expansions.** `compare_lstm_features_with_poc.ipynb` runs the **PoC tree
  models on Base B's per-source wide features** (directly testing "do per-source
  features beat daily means?"), adding ablations, walk-forward, and multi-seed
  checks. `transformer_forecaster.ipynb` and `tuning.ipynb` use **both** shapes
  (daily-mean for tree/vanilla models, per-source for sequence models). The
  `sentisense/` package generalizes both into leakage-safe `mt` (daily-mean) and
  `ml` (per-source) frames (§3.2).

Each subsection below notes which aggregation it uses.

**Cross-track summary.** Before the detailed tables, Table 1 gives the
one-glance picture: the best configuration in each track, its majority-class
baseline, and whether it beats that baseline / reaches significance. The
recurring theme is that "best" accuracies rarely clear their own baseline, and
none reaches significance.

*Table 1: Best result per experiment track vs its baseline.*

| Track (§) | Aggregation | Best config | Acc | Baseline | Beats base? | ROC-AUC | Significant? |
|---|---|---|---|---|---|---|---|
| PoC (§4.2.1) | daily-mean | XGBoost / LightGBM | 0.5459 | 0.4976 | yes | n/a | no (p≈0.09–0.13) |
| LSTM-features (§4.2.2) | per-source | LGBM "Top sources+Other" | 0.5794 | 0.5675 | +0.012 | 0.5415 | no (p_perm=0.052) |
| LSTM base (§4.2.3) | per-source | LSTM (window 30) | 0.5636 | 0.5773 | **no** | n/a | no (p=0.68) |
| Transformer zoo (§4.2.4) | both | PatchTST_DailyMean | 0.5370 | 0.4931 | yes | 0.5185 | no (p≈0.09) |
| Tuning (§4.2.5) | both | Ensemble / tuned LSTM | 0.4596 | 0.5680 | **no** | n/a | no |
| Hardened pkg (§4.2.6) | daily-mean | Score-LSTM | 0.5000 | ~0.50 | ~tie | 0.5088 | no (MCC≈0) |
| Unified grid (§4.2.8) | all | GRU [scored] (by ROC-AUC) | 0.5289 | ~0.50 | ~tie | **0.5755** | no |

*(n/a = ROC-AUC not printed numerically in that notebook; "Significant?" = passes
a permutation/binomial test at p<0.05 vs the majority-class baseline.)*

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

**Chronological 80/20 holdout** (train 826 rows 2019-07-17→2022-12-04; test 207
rows 2022-12-05→2023-10-05; test up-rate 49.76%):

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
> rendered report with a 507-sample support exists. The 207-sample version above
> matches the notebook's stated 80/20 split.

#### 4.2.2 LSTM feature-set vs PoC study (`compare_lstm_features_with_poc.ipynb`)

*Aggregation: Base B — per-source wide (from `lstm_forecaster.ipynb`), fed to
the PoC tree models. This notebook exists specifically to test whether Base B's
per-source features beat Base A's daily means.*

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

**Walk-forward validation (5 folds, CatBoost):** mean accuracy **0.5967**, mean
baseline 0.5733, mean gap +0.0233, with only **2 / 5** folds beating baseline
(fold accuracies 0.633 / 0.650 / 0.567 / 0.617 / 0.517).

**Multi-seed robustness (CatBoost, 5 seeds):** mean accuracy **0.5714 ±
0.0084** (min 0.560, max 0.581), mean gap +0.0072, **4 / 5** seeds positive;
ROC-AUC ranges 0.507–0.577.

**Feature importance by group (CatBoost):** News 79.2%, Market-derived 17.6%,
Basic market 3.2% of total importance — yet the strongest *individual* features
are price-derived (`TA125_logret_5d_std`, `TA125_logret_lag5`, `TA125_RSI14`).

*Verdict:* the best configuration (LGBM, "Top sources + Other") reaches accuracy
0.579 with `p_perm = 0.052` — the closest to significance anywhere — but its
bootstrap 95% CI (0.534–0.623) straddles the baseline. The signal is **weak and
unstable**.

#### 4.2.3 LSTM base forecaster (`lstm_forecaster.ipynb`)

*Aggregation: Base B — per-source wide (320 columns), 30-day windows. This is
the original sequence model the per-source representation was built for.*

The base LSTM is trained on chronological, windowed sequences (window 30,
326 features) with train/val/test = 1,163 / 249 / 250 daily rows
(2019-07-17 → 2026-04-29); test up-rate 57.73%.

**LSTM holdout test:** accuracy **56.36%** (below the 57.73% majority baseline).

| Class | precision | recall | f1 | support |
|---|---|---|---|---|
| Fall | 0.29 | 0.02 | 0.04 | 93 |
| Rise | 0.57 | 0.96 | 0.72 | 127 |
| accuracy | | | 0.56 | 220 |

*Table 7: LSTM base forecaster holdout classification report.*

**Significance vs baseline (57.73%):** binomial p = 0.6845; permutation
p = 0.8820; bootstrap 95% CI [50.00%, 62.73%].

*Verdict:* the model collapses toward the majority "Rise" class (recall 0.96 vs
0.02), scores **below** baseline, and is statistically indistinguishable from it.
Training accuracy reaches ~0.74 while validation stays ~0.48–0.53 — clear
overfitting on the 320-column per-source representation. (ROC-AUC/MCC were
rendered only inside plot images, so no numeric values are reported here.)

#### 4.2.4 Transformer model zoo + ablations (`transformer_forecaster.ipynb`)

*Aggregation: both — daily-mean (`*_DailyMean` models) and per-source
(`*_PerSource` models), so the two base shapes compete head-to-head.*

This notebook set an explicit success bar: **≥58% test accuracy with p<0.05 vs
majority**. Nine transformer variants were evaluated against tree/linear
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
significant — the lowest p-value is PatchTST (`p_binom=0.089`, `p_perm=0.090`).

**Window-size ablation (PatchTST):** best at window 15–20 (acc ≈ 54–55%,
ROC-AUC up to 0.592); collapses to the majority class at windows 45–60.
**Feature-group ablation:** Market-only 0.5409 > LagReturns-only 0.5292 >
News-only 0.4864.

*Verdict (printed):* "Best transformer: PatchTST 53.81% balacc … Best overall
score < 55% … TA-125 direction at daily frequency may not be predictable from
news sentiment + market features alone." The **≥58% success criterion was not
met**. (The Optuna "tuned leaderboard" and McNemar cells were not executed.)

#### 4.2.5 Sequence-model tuning & robustness (`tuning.ipynb`)

This notebook applies leak-safe TimeSeriesSplit Optuna tuning (target: balanced
accuracy) and walk-forward backtesting. Corpus: 1,898,499 validated rows,
40 sources.

**Reproducibility sanity (vanilla holdout, baseline 56.80%):** XGBoost 51.06%,
LightGBM 52.57%, CatBoost 50.76% — all below baseline.

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

*Verdict:* tuned single models and the ensemble both **underperform the majority
baseline** on the holdout. (GRU/TCN, multi-seed, abstention, and the final
`final_results.csv` cells were not executed in the saved notebook.)

#### 4.2.6 Hardened-package analysis (`sentisense_analysis.ipynb`)

Run directly against the live database (cutoff 2023-10-07). Corpus coverage:
**2,950,339 validated `mistral-small-4` rows** (plus 52,640 `mistral-small:latest`).

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
ROC-AUC ≈ 0.51, MCC ≈ 0.00. On the hardened pipeline the score-only LSTM shows
essentially **no directional edge**. (The §2 multi-track, baselines, and
strategy-vs-Buy&Hold cells were not executed; SHAP outputs exist only as plots.)

#### 4.2.7 Foundation-model explainability (`timesfm_explainability.ipynb`)

This notebook scaffolds zero-shot/covariate-ablation/regime experiments for
Google's TimesFM, but **was not executed in the committed copy** — no numeric
results are available. It is retained as a wired template for future work
(§5).

#### 4.2.8 Unified out-of-sample grid (`leaderboard.md`)

The canonical, leakage-hardened comparison reduces every model × data-type ×
regime cell to the same out-of-sample window and metric set. Sorted by accuracy
descending. Notation: `model [data-type]` for classifiers, `model [cov=...]`
for forecasters.

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

### 4.3 Data Analysis and Interpretation

Reading **across all six tracks** of §4.2, several consistent patterns emerge.

1. **Every track converges on near-chance.** Whether trees (§4.2.1–4.2.2,
   §4.2.5), transformers (§4.2.4), LSTMs (§4.2.3, §4.2.5–4.2.6), or foundation
   models (§4.2.8), ROC-AUC clusters around 0.50 (≈ 0.43–0.59) and balanced
   accuracy hovers near 0.50. The hardened score-LSTM is the cleanest statement
   of this: accuracy 0.500, ROC-AUC 0.509, MCC 0.000 (§4.2.6).
2. **The majority-class baseline is the real story.** Baselines differ sharply
   by split — 49.31% in the transformer notebook, 49.76% in the PoC, 56.42–
   56.80% in the later windows — because the up-rate is window-dependent. On the
   high-up-rate windows, *raw accuracy near 57% does not beat "always predict
   up."* Several models that look respectable on accuracy (e.g. tuned LSTM
   0.4553, ensemble 0.4596 in §4.2.5) are in fact **below** their baseline.
3. **Accuracy, ROC-AUC, and baseline disagree.** The unified grid's top-accuracy
   model (`TFT [cov=none]`, 0.592) has only mediocre ROC-AUC (0.539) and low F1
   (0.415) — the signature of leaning to the majority class. The top-ROC-AUC
   model (`GRU [scored]`, 0.576) is mid-table on accuracy. Reporting a single
   metric would mislead; the multi-metric view is essential.
4. **No data-type or model family dominates.** `scored`, `embedded`, and `fused`
   views land in the same band; foundation models (Chronos zero-shot 0.427) do
   not beat trained models; complex transformers do not beat simple
   GRU/TCN/XGBoost. The feature-group ablation is telling: **market-only features
   match or beat news+market**, and **news-only is the weakest** (§4.2.2, §4.2.4).
   Capacity is not the bottleneck; **signal is.**
5. **Significance is essentially never reached.** Across the PoC, transformer,
   and feature-comparison notebooks, every permutation/binomial test returns
   p > 0.05. The single closest call — LGBM "Top sources + Other", `p_perm =
   0.052` (§4.2.2) — has a bootstrap CI that straddles its baseline, and is not
   robust under walk-forward (2/5 folds positive) or multi-seed (mean gap
   +0.7pp) checks.

### 4.4 Comparison with Existing Approaches

**Internal comparison across tracks.** The notebook tracks (§4.2.1–4.2.5) and
the hardened package (§4.2.6, §4.2.8) tell a consistent story, but the package
numbers are the trustworthy ones. The exploratory notebooks vary their splits
and baselines and occasionally show small positive gaps on a single favorable
window; the hardened, fixed-window package run flattens these to near-chance.
This is exactly the expected effect of removing window-selection freedom and
tightening leakage controls — the transformer notebook's own ≥58% success
criterion is **not met** anywhere, and its printed verdict already anticipates
that "TA-125 direction at daily frequency may not be predictable from news
sentiment + market features alone."

**Comparison with the literature.** The magnitude of the effect is consistent
with the consensus that **news-tone signal for next-day index direction is small
and fragile**, and that much of the apparent predictability reported elsewhere
stems from evaluation leakage or favorable-period selection. The Buy&Hold
benchmark included in the unified grid contextualizes that none of the
directional models offers a dependable economic edge over simply holding the
index. Notably, the feature-importance breakdown (News ≈ 79% of total
importance, §4.2.2) shows the models *do* lean on the news features — they
simply do not convert that into out-of-sample skill.

### 4.5 Discussion of Findings

The honest interpretation is that, **on its own and at daily resolution, the
LLM-scored Hebrew-news stream carries little reliable next-day directional
signal for the TA-125.** This is not a failure of engineering but a substantive
empirical result, and it is credible *because* of the leakage controls: the
pipeline was specifically built to make it hard to fool oneself.

The project's lasting value is methodological. It delivers (a) a reproducible,
auditable, leakage-safe pipeline spanning scraping, LLM scoring, feature
engineering, and a tuned model zoo; (b) a uniform, resumable comparison
framework; and (c) a quantified, multi-metric baseline against which future
richer signals (intraday data, magnitude targets, alternative LLM scoring
schemes, longer horizons) can be measured. Limitations include the
close-to-close-only target, daily resolution, the all-zero LLM rows, and the
mixed scoring-model history — each a concrete lever for future work.

---

## 5. Conclusion and Future Work

**Conclusion.** SentiSense set out to test whether LLM-distilled Hebrew-news
sentiment can predict next-day TA-125 direction. It produced a complete,
leakage-hardened, reproducible system — scraper, LLM scorer, ~1.9M-row scored
corpus, daily feature engineering, embeddings, causal narrative clustering, and
a tuned zoo of 40+ model configurations — and evaluated them rigorously on a
sacred out-of-sample window bounded by a hard `2023-10-07` cutoff. The result is
a small, fragile edge at best (top ROC-AUC 0.576, top accuracy 0.592), broadly
indistinguishable from chance. The contribution is therefore a **trustworthy
negative-leaning baseline plus a reusable research platform**, not a profitable
predictor.

**Future work.**

1. **Richer targets.** Persist TA-125 OHLC to enable overnight-gap and
   intraday-return (magnitude) targets, which may carry more news signal than
   close-to-close direction.
2. **Longer horizons & event studies.** Test weekly direction and event-window
   responses around high-impact (very negative + high-security-relevance)
   headlines.
3. **Better signal extraction.** Per-source credibility weighting, headline
   embeddings fed as PCA-reduced covariates to forecasters, and improved LLM
   prompts that reduce the all-zero failure mode.
4. **Robustness.** Multi-seed ablations (≥3 seeds, mean±std), walk-forward
   evaluation across multiple windows, and statistical significance testing of
   the small observed edges.
5. **Operationalization.** Wire the empty `daily_features` / `model_predictions`
   tables into a scheduled daily prediction job with a monitoring dashboard
   (the planned fifth module).

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
Multi-horizon Time Series Forecasting," *International Journal of Forecasting*,
vol. 37, no. 4, pp. 1748–1764, 2021.

[9] Y. Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with
Transformers (PatchTST)," in *Proc. ICLR*, 2023.

[10] B. N. Oreshkin et al., "N-BEATS: Neural basis expansion analysis for
interpretable time series forecasting," in *Proc. ICLR*, 2020.

[11] C. Challu et al., "N-HiTS: Neural Hierarchical Interpolation for Time
Series Forecasting," in *Proc. AAAI*, 2023.

[12] A. F. Ansari et al., "Chronos: Learning the Language of Time Series,"
*Transactions on Machine Learning Research*, 2024.

[13] A. Das et al., "A decoder-only foundation model for time-series forecasting
(TimesFM)," in *Proc. ICML*, 2024.

[14] L. Wang et al., "Text Embeddings by Weakly-Supervised Contrastive
Pre-training (E5 / multilingual-E5)," arXiv:2212.03533, 2022.

---

## 7. Appendix A — Data Dictionary, Schema, and Commands

### A.1 Database schema (PostgreSQL 16)

`raw_headlines` — one row per scraped headline (source of truth):
`id` (PK), `date`, `source`, `hour`, `popularity`, `headline` (Hebrew, UTF-8),
`created_at`, `headline_hash` (`md5(headline)`, stored). Unique on
`(date, source, hour, headline_hash)`.

`nlp_vectors` — one row per `(headline, model)`:
`id` (PK), `headline_id` (FK), `model_name`, six `relevance_*` SMALLINT (0–10),
`global_sentiment` SMALLINT (−10…+10), `validation_passed` BOOLEAN,
`processing_time_seconds`, `errors` TEXT[], `created_at`. Unique on
`(headline_id, model_name)`.

`daily_features` — per-day aggregate vector (schema exists; first written by the
feature layer). `model_predictions` — inference log (schema exists).

### A.2 Score-scale reference

- **Relevance** (six columns): integer 0–10; higher = more relevant to that
  category.
- **Sentiment** (`global_sentiment`): integer −10 (very negative) … +10 (very
  positive); 0 = neutral/mixed.
- **`validation_passed`**: TRUE = parseable, in-range LLM output. Always filter
  on TRUE (and on the active `model_name`) for analysis.

### A.3 Reproduction commands

```bash
# 0 — database (schema auto-initialises from scripts/init_db.sql)
docker compose up -d

# 1 — scrape headlines
cd mivzakim_scraper && uv sync && uv run playwright install firefox && uv run python main.py

# 2 — score unscored headlines into nlp_vectors
cd processing_engine && uv sync
uv run python ../scripts/process_headlines.py --fast --headlines-per-call 50 --concurrency 50

# 3 — forecast (Phase 2&3): features → embeddings → models → leaderboard
uv sync --extra ml --extra embed --extra finance        # at repo root
uv run python -m sentisense.pipeline --from features     # leakage-safe, ≤ 2023-10-07

# 4 — full comparison leaderboard (server, run in tmux)
uv sync --extra ml --extra finance --extra embed --extra tft --extra chronos
uv run python scripts/pipeline_compare.py --seq-trials 30 --pf-trials 12 --xgb-trials 60
```

### A.4 Repository map

```
mivzakim_scraper/   Playwright scraper for mivzakim.net (Hebrew news)
processing_engine/  LLM scoring pipeline (fast single-prompt + 7-agent LangGraph)
sentisense/         Phase 2&3 forecasting package
  constants.py        cutoff, model name, score contract
  config.py           modeling/HPO knobs (env-overridable)
  db/                 SQLAlchemy engine (env-only DSN)
  ingest/             backfill · score · coverage report
  features/           leak-safe daily dataset assembly
  embed/              multilingual-e5 headline embeddings + cache
  cluster/            causal expanding-window narrative clustering
  models/             sequence datasets, train harness, model zoo, baselines
  hpo/                resumable Optuna HPO + sacred-holdout eval
  pipeline.py         end-to-end orchestrator
evaluation/         LLM-scoring benchmark vs golden dataset
scripts/            init_db.sql · backfill · process/retry/standardize · compare
tests/              pytest — cutoff, leakage, calendar rollover, connection
docs/               RUNBOOK · MODEL_ZOO · VECTORDB · sentisense-understanding
*.ipynb             eda · poc · lstm_forecaster · tuning · transformer_forecaster
```

*End of first version. Author name, date, and any institution-specific front
matter are placeholders to be completed before submission.*
