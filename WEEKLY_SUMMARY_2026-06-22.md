# SentiSense — Weekly Progress Summary

**Window:** 2026-06-15 → 2026-06-22 (last 7 days, `main` branch)
**Author:** Omri Shlezman
**For:** Mentor review

---

## TL;DR

In the last week the project moved from a single hand-tuned LSTM proof-of-concept to a
**full, leakage-safe, reproducible benchmarking platform** that compares 11 model families
across 3 feature representations and 2 market regimes on an identical out-of-sample window.

- **44 commits**, **4 PRs merged** (#18, #23, #24, #26), **~20,300 lines added** across 40 files.
- A single command (`pipeline_compare.py`) now produces a 46-cell leaderboard with a
  per-cell coverage report (no silent skips) and a resumable cache.
- A **pgvector** embedding store was deployed on the GPU server (no Docker) and filled from
  the embedding cache.
- **Headline scientific finding:** next-day TA-125 direction is **statistically
  indistinguishable from chance** on held-out data (best OOS ROC-AUC ≈ 0.58, most cells
  0.47–0.57, MCC ≈ 0). This is consistent with the weak-form Efficient Market Hypothesis and
  is itself a defensible, rigorously-established result — not a failure.
- **Two PRs in review** (land next): **#25** accuracy-boost (overnight features + confidence
  intervals + ensemble) and **#22** MiroFish multi-agent narrative simulation.

---

## What shipped (by PR)

| PR | Title | Size | Theme |
|----|-------|------|-------|
| [#18](https://github.com/Shlezman/Data-Science-Final-Project/pull/18) | Phase 2&3 — leakage-safe financial modeling + LSTM HPO pipeline | +8,571 / −230 | Core modeling foundation |
| [#23](https://github.com/Shlezman/Data-Science-Final-Project/pull/23) | Enrich leaderboard with TFT, Chronos, PatchTST, TCN, GRU (+ HPO) | +1,310 / −80 | Model zoo |
| [#24](https://github.com/Shlezman/Data-Science-Final-Project/pull/24) | pgvector embedding store (deploy + fill, no Docker) | +322 / −0 | Vector DB |
| [#26](https://github.com/Shlezman/Data-Science-Final-Project/pull/26) | Leaderboard refresh | +64 / −0 | Results |

### 1. Leakage-safe modeling foundation (PR #18)
The biggest change of the week. Established the rules and plumbing that make every later
result trustworthy:
- **Next-day target** `close(T+1) > close(T)` with a strict no-look-ahead feature contract;
  all news from a non-trading day (Fri/Sat) is rolled into the next Sunday's row so weekend
  headlines aren't lost but never leak forward.
- **Regime split** at the Oct-7-2023 structural break: `CUT` (≤ 2023-10-07) vs `FULL`
  (entire timeline), so we can tell genuine signal from post-war trend-riding.
- **Three feature representations**, all built leak-safe and cutoff-aware:
  - `scored` — daily LLM news scores (mean + per-source).
  - `embedded` — daily e5 embedding centroid (768-d) + finance block.
  - `fused` — per-source scores ⊕ embedding centroid.
- **Calibration guard** (isotonic only when it helps, with a class-collapse check) and a
  **post-cutoff buy-overlay** scorecard for the trend regime.
- New torch-free `metrics` module so the leaderboard runs without importing torch.

### 2. Model zoo + HPO (PR #23)
Went from 1 model to a directly-comparable fleet, each hyperparameter-tuned:
- **Classifiers:** XGBoost (Optuna), LSTM, **GRU**, **TCN**, **PatchTST**.
- **Forecasters** (forecast → direction bridge): **TFT**, **N-HiTS**, **N-BEATS**
  (pytorch-forecasting), **Chronos** (Amazon zero-shot foundation model), **TimesFM**
  (Google). Plus a **Buy&Hold** baseline in every regime.
- Generic Optuna sequence-HPO harness (`optuna_seq`) so every torch model is tuned on the
  same footing; foundation forecasters sweep context length + decision threshold.
- Dependency hardening that cost real debugging time and is worth noting: **torch pinned to
  CUDA-12.1 wheels** to match the server's 12.3 driver (the default 12.9 wheel silently
  crashed every torch model), and **transformers pinned < 4.49** for Chronos compatibility.

### 3. pgvector embedding store (PR #24)
- `scripts/deploy_vectordb.py` deploys pgvector into the native PostgreSQL 14 (no Docker, no
  superuser assumptions) and fills it from the embedding cache.
- Index build hardened for the real machine: single-threaded build (avoids `/dev/shm`
  disk-full), `ivfflat` with `lists = sqrt(n)` for >1M rows, bounded maintenance memory.

### 4. Infrastructure that makes the above usable
- **`pipeline_compare.py`** — one command → the whole grid → `leaderboard.md`.
- **Resumable per-cell cache** — each finished cell is persisted the instant it completes; a
  re-run or crash-resume reuses done cells and only computes new/changed ones.
- **Coverage report** — `leaderboard.md` ends with every cell marked ran / skipped + reason,
  so a missing model is never silent.
- **`gap_fill.py`** — orchestrates scrape → score → embed → finance to extend the dataset.
- Docs: `docs/MODEL_ZOO.md`, `docs/VECTORDB.md`. Tests: `test_model_zoo`, `test_vectordb`,
  `test_postcutoff_overlay`, `test_timesfm_mapping`.

---

## Results — the leaderboard (out-of-sample, last-15% window)

The full 46-cell table lives in [`leaderboard.md`](leaderboard.md). The honest reading:

**Direction is hard.** Best out-of-sample ROC-AUC across every model/feature/regime
combination was **GRU [scored/CUT] = 0.576** (n=242 — a small, noisy window). The vast
majority of cells sit in **0.47–0.57**, and **MCC hovers around 0** everywhere. No model
clears chance by a margin that survives the sample size.

**High Sharpe in FULL is trend, not skill.** In the `FULL` regime several models show
eye-catching Sharpe ratios (2.5–3.0) and large cumulative returns — but **Buy&Hold in the
same regime scores Sharpe 2.82**. Those models are simply staying long through the
post-2023 rally, not timing direction. ROC-AUC there is still ≈ 0.5.

| Reading | Evidence |
|---|---|
| Best discriminative cell | GRU scored/CUT, ROC-AUC 0.576 (n=242) |
| Typical cell | ROC-AUC 0.47–0.57, MCC ≈ 0 |
| "Profitable" FULL cells | Sharpe ≈ Buy&Hold → riding the rally, not skill |
| Verdict | Next-day direction ≈ coin flip on held-out data |

This is **consistent with the weak-form Efficient Market Hypothesis**: daily index direction
is not reliably predictable from public news sentiment. Establishing that *rigorously* —
leak-safe, regime-aware, multi-model, with an honest baseline — is the scientific
contribution.

---

## Upcoming — two open PRs in review

Two pull requests are open and will land next.

### PR #25 — accuracy boost (`feat/accuracy-boost`, +660 / −87)

An experiment to squeeze out any remaining signal and, more importantly, to **quantify the
uncertainty** so "≈ chance" becomes a defensible statistical claim rather than a hunch:

- **Overnight global features** — day-T close returns of S&P 500, Nasdaq, VIX, Brent, USD-ILS.
  These are known before the next TA-125 open, so they are leak-safe for an *open(T+1)*
  decision contract (one step ahead of the close-safe baseline). This is the most plausible
  source of a real edge.
- **Bootstrap ROC-AUC confidence intervals** on every cell — the arbiter for "is this cell
  distinguishable from 0.5?".
- **Soft-vote ensemble** (rank-normalized) and **abstention analysis** (accuracy vs coverage
  when acting only on the most-confident predictions).
- **Engineering robustness:** a streaming daily-embedding-centroid loader that fixed a
  full-history out-of-memory crash (the old path materialized ~3M × 768 vectors ≈ 9 GB at
  once; the new one accumulates per-date sums over DB chunks, peak RAM ~0.3 GB, with a test
  proving the result is identical).

**Status:** the overnight grid is queued to run on the GPU server; results + the CI verdict
will follow.

### PR #22 — MiroFish narrative simulation (`feat/miro-simulation`, +3,413 / −61)

A new **narrative-simulation layer**. MiroFish is a multi-agent social-simulation engine
(CAMEL-AI OASIS + GraphRAG + agent memory, vendored as a submodule at `external/MiroFish`).
It does *not* predict prices — it simulates how a crowd's narrative around the day's news
evolves, and we mine that for signal and explainability. Three uses:

- **A — Causal sim-feature.** Each trading day is seeded strictly on news ≤ T (leak-safe),
  the agents are interviewed, and their answers are reduced to a deterministic numeric vote
  (direction score, confidence, disagreement, magnitude). These `sim_*` columns feed the
  existing XGBoost / LSTM / TimesFM forecasters via a **with-sim vs without-sim ablation**
  (`pipeline_compare --with-sim`).
- **B — Event-study explainability.** Rich per-agent reports + the agent interaction graph
  on high-impact dates (war onset, rate decisions) — stored for the report.
- **C — Live forward gauge.** One sim/day going forward, with the agent graph persisted for
  the future dashboard UI.

Two complementary **simulation modes** run per day (each cached separately):

| Mode | Seed | Answers |
|------|------|---------|
| `source` | one section per news outlet, perspective-aware | "What does each channel's crowd think, and where do they diverge?" |
| `flat` | whole-day news pooled, source stripped | "What does the day's news say as one aggregate narrative?" |

Their divergence (`sim_src_flat_gap`, `sim_src_flat_agree`) is itself a feature — it flags
days where channel framing matters.

**Engineering notes:**
- **Local-only, zero external egress** — LLM is Gemma-4 on the 4090 (Ollama/vLLM,
  OpenAI-compatible), memory/graph via a **self-hosted Zep** (not Zep Cloud). Honors the org
  data-handling policy; verified by `scripts/verify_local_egress.sh`.
- SentiSense talks to MiroFish over **HTTP only** (`sentisense/sim/` client orchestrates the
  9-step build→simulate→report pipeline with polling) — it never imports the AGPL engine
  in-process.
- New tables (`narrative_sim`, `narrative_sim_graph`, `narrative_sim_report`) + migrations,
  plus 4 test modules.

**Honest north star (same as the rest of the project):** the sim-feature's lift is something
to be **measured, not assumed**. Even at zero lift, the explainability reports + agent-graph
assets are valuable on their own.

---

## Next steps

1. **Finish the overnight run** and report the confidence-interval verdict: does *any* cell's
   ROC-AUC lower bound clear 0.5? (Expected: no — which closes the direction question
   cleanly.)
2. **Pivot the target where signal is more plausible.** Daily *direction* is near-chance, but
   *volatility/magnitude* is far more predictable in finance. Candidate next experiments:
   - predict next-day **absolute return / realized volatility** (regression) instead of sign;
   - **multi-day horizon** (5-day direction) where sentiment may aggregate into signal;
   - **event-conditioned** prediction (only days with a high-relevance news spike).
3. **Merge the two open PRs** (#25 accuracy-boost, #22 MiroFish) and run the **with-sim
   ablation** — does the narrative-simulation feature move any cell's ROC-AUC beyond its
   confidence interval? (Measured, not assumed.)
4. **Write up the negative result** for the report with the leaderboard + CIs as evidence —
   a clean, well-instrumented "markets are efficient at the daily horizon" finding, with the
   MiroFish explainability reports as the qualitative companion.

---

*Generated from `main` git history (44 commits, 2026-06-15 → 2026-06-22).*
