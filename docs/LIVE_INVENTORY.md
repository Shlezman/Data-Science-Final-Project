# SentiSense go-live — Phase 0 inventory + architecture

Branch `feat/go-live` (off `origin/main`, with `feat/miro-simulation` merged in). Everything
below is **reused, not rewritten**. Decisions: integrate mirofish into this branch · everything
runs inside the `/tf` container (UI on `:3000` exposed to host, cron in-container) · UI =
FastAPI + Vite/React SPA.

## Reusable entrypoints (what each workstream calls)

### Daily pipeline stages (chain in this order)
| # | Stage | Entrypoint | Writes | Idempotent on |
|---|-------|-----------|--------|---------------|
| 1 | Scrape | `scripts/daily_scrape_to_db.py` (`--days 2`) | `raw_headlines` | `(date,source,hour,headline_hash)` ON CONFLICT |
| 2 | Score | `scripts/process_headlines.py` (`--fast`) | `nlp_vectors` | `(headline_id,model_name)` — only unscored |
| 3 | Embed | `python -m sentisense.embed.embeddings` (`--scope all`) | `headline_embeddings` | `(headline_id,embed_model)` — only un-embedded |
| 4 | Derived | `scripts/build_embedding_derived.py` | `daily_embedding_derived` | upsert `(date,embed_model)` |
| 5 | Features | `sentisense.features.build_fused_dataset(cutoff, overnight=True)` | in-memory frame | leak-safe cutoff |
| 6 | Predict | **NEW** `sentisense/serve/champion.py` (see below) | `model_predictions` | `(date,model_version)` |
| 7 | Settle | **NEW** backfill `model_predictions.actual` once T+1 close is known | `model_predictions.actual` | idempotent UPDATE |

Env: `SENTISENSE_DATABASE_URL`, `SENTISENSE_OPENAI_*`/`SENTISENSE_LLM_BACKEND` (scoring),
`SENTISENSE_EMBED_*`. uv extras: scrape/score = base; embed = `embed`; features/predict =
`finance` + `ml`.

### Champion model (financial direction — corrected)
The relevant metrics live in **`sentisense/models/metrics.py`** (`direction_metrics` →
roc_auc/accuracy/f1/mcc, `metrics_at`) and the leaderboard in `scripts/pipeline_compare.py` —
NOT `evaluation/` (that's the golden-dataset LLM-scoring eval). The financial verdict is
**chance** (best FULL-regime OOS ROC-AUC ≈ 0.53–0.56, CIs span 0.5). The served **CHAMPION**
is therefore chosen for robustness + cheap daily retrain, not skill:

> **Champion = XGBoost on the `fused` dataset, FULL regime, overnight features on.**
> Tree model, GPU (`device=cuda`), trains in seconds on all history, consumes every feature
> family (scores + per-source + centroid + derived PCA/cluster + finance + overnight). Pinned
> as a versioned artifact + config; reuses `sentisense/models/xgb_hpo.py`.

### Mirofish simulator (drives WS3-C)
| Need | Symbol | Notes |
|------|--------|-------|
| Run a sim for a date | `sentisense.sim.runner.run_day(client, engine, date, mode=…, seed_idx=…)` | writes `narrative_sim` + `_graph` + `_report`; `None` if cached |
| Daily one-shot | `scripts/miro_daily.py` | finds latest trading day w/ news, idempotent |
| Modes | `config.SIM_MODES` = `source` (per-outlet) · `flat` (pooled) | env `SENTISENSE_MIRO_MODES` |
| Agent graph for UI | `sentisense.sim.graph_api.graph_for_date(date, mode=…)` / `latest_graph(mode=…)` | returns `{nodes:[{id,type,label,attrs}], edges:[{src,dst,type,weight}], meta}` |
| Report | `graph_api.report_for_date(date, mode=…)` | markdown + sections |
| Live stream | **none** | poll-based; only final results. UI streams *progress*, not true steps |

MiroFish itself = `external/MiroFish` (AGPL submodule), run as a **separate HTTP service**
(default `http://localhost:5001`); `sentisense/sim` is an HTTP client (only new dep: `requests`,
via `--extra miro`). It never imports MiroFish in-process.

## Data the UI reads (all in Postgres)
- **Dashboard accuracy + confusion matrix** ← `model_predictions(date, model_version,
  prediction, confidence, actual)`. Confusion matrix = `prediction` vs `actual` (NULL until
  settled). Compute with the existing financial metrics, not the golden-dataset module.
- **Live last-day headlines** ← `raw_headlines` ⟕ `nlp_vectors` (filter `model_name`,
  `validation_passed`, exclude all-zero rows), `date >= today-1`.
- **Archive** ← same join, `WHERE date = :d` paginated.
- **Simulator** ← `narrative_sim*` via `graph_api`.

## Trading-calendar guard (reuse)
TA-125 CSV (repo root) is the TASE calendar; `sentisense.features.dataset._finance_base()` →
`(base, trading_days, price_full)`. "Run today?" = today (in `Asia/Jerusalem`) ∈ `trading_days`.
Gap: no explicit holiday list — the CSV is de-facto truth; run ≥6 h after close.

## Conventions to follow
loguru → stderr + `logs/<script>_{time}.log`; `--dry-run` everywhere; exit 0 ok / 1 fail;
`Asia/Jerusalem` for "today"; `ON CONFLICT DO NOTHING`/upsert for idempotency; secrets via env.

## New code this branch adds (no rewrites)
- `sentisense/serve/champion.py` — pinned champion: train-on-history → forward-predict T+1 →
  write `model_predictions`. (Needs a builder tweak to retain the latest *unlabeled* day.)
- `scripts/daily_live.py` — orchestrator chaining stages 1–7 (lock, calendar, status, logs).
- `scripts/settle_predictions.py` — backfill `actual` from realized TA-125.
- `scripts/challenger_hpo.py` — optional gated challenger (WS2).
- `ui/` — FastAPI app (REST + ws) serving the Vite/React SPA on `:3000`.
- ops: cron entries, pm2/supervisor unit, runbook.
