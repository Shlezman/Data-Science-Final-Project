# MiroFish → SentiSense integration plan

Branch: **`feat/miro-simulation`** (off `main`). MiroFish vendored as a submodule at
`external/MiroFish` (AGPL-3.0; this project is open-source + runs on a personal remote
box, not org compute → import/use is fine).

## Vision
MiroFish is a **multi-agent social-simulation** engine (CAMEL-AI OASIS + GraphRAG +
agent memory). It does NOT predict prices — it simulates how a crowd's narrative
evolves. We use it as SentiSense's **narrative-simulation layer**:

- **A. Causal sim-feature** → a daily "expected market reaction" vector fed to the
  existing XGBoost / LSTM / TimesFM forecasters (with-sim vs without-sim ablation).
- **B. Event-study explainability** → rich agent reports + the agent graph on high-impact
  dates (war onset, rate decisions).
- **C. Live forward gauge** → one sim/day going forward + the **agent graph persisted for
  the future UI**.

Honest north star: next-day TA-125 direction is ~chance (EMH). Success = a *causal*
sim-feature whose lift is **measured, not assumed**, plus the explainability + agent-graph
assets (valuable regardless).

## Confirmed MiroFish API (read from source — Flask, AGPL)
Blueprints (`backend/app/__init__.py`): `/api/graph`, `/api/simulation`, `/api/report`.
Pipeline (stateful, multi-call, async + poll):
1. create project (seed text) → `project_id`  (under `/api/graph/project*`)
2. `POST /api/graph/build {project_id}` → `graph_id` (poll `GET /api/graph/task/<task_id>`)
3. `POST /api/simulation/create {project_id, graph_id, enable_twitter, enable_reddit}` → `simulation_id`
4. `POST /api/simulation/prepare {simulation_id}` (poll `/prepare/status`) — personas/config
5. `POST /api/simulation/start {simulation_id}` (poll `GET /api/simulation/<id>/run-status`)
6. `POST /api/report/generate {simulation_id}` (poll `/generate/status`) → `report_id`
7. `GET /api/report/<id>/sections` → `[{filename, section_index, content(md)}]` — explainability
8. `GET /api/graph/data/<graph_id>` → `{data: <nodes/edges>}` — **UI agent graph**
9. `POST /api/simulation/interview` / `/interview/all` → per-agent answers → **structured
   numeric vote** (the deterministic feature; avoids NLP-parsing the markdown report)

LLM: OpenAI-SDK (`app/utils/llm_client.py`, `.env LLM_BASE_URL/MODEL_NAME`). Memory/graph:
Zep (`zep-cloud`), defaults to Zep Cloud.

## Config (Gemma-4 + self-hosted Zep, all local)
MiroFish `.env` (on the remote box):
- `LLM_BASE_URL=http://localhost:11434/v1`, `LLM_MODEL_NAME=gemma-4`, `LLM_API_KEY=local` (Gemma-4 on the 4090 via Ollama/vLLM, OpenAI-compat).
- `ZEP_API_KEY` → self-hosted Zep (see `scripts/init_zep.sh`), not app.getzep.com.
- No external egress. (Org data-handling honored.)

## Data model — `sentisense/db/migrations/002_narrative_sim.sql`
- `narrative_sim` (date, seed_hash, llm_model, seed_idx) → dir_score, confidence,
  disagreement, magnitude, dominant_narrative, n_agents, n_steps, sim_id, created_at
- `narrative_sim_graph` (sim_run_id, date, graph_id, nodes JSONB, edges JSONB, meta JSONB) — UI
- `narrative_sim_report` (sim_run_id, date, report_id, sections JSONB, report_md TEXT) — explainability

## SentiSense code (HTTP client; never imports MiroFish in-process)
`sentisense/sim/`: `config.py`, `miro_client.py` (orchestrates the 9-step pipeline + polling),
`extract.py` (interview votes → numeric feature; sections → explainability), `graph.py`
(normalize `/graph/data` → `{nodes,edges,meta}`), `runner.py` (per-day **causal** seed ≤T,
multi-seed, idempotent cache).
`scripts/run_miro_window.py` (A, windowed), `scripts/miro_event_study.py` (B),
daily-pipeline hook (C), `scripts/init_zep.sh` (self-host Zep).
`pipeline_compare.py` `--with-sim` ablation arm. `miro_explainability.ipynb`.

## Invariants (carried over)
1. **Leakage:** day-T seed = headlines strictly ≤ T (bounded lookback); question asks T+1;
   cache keyed by `seed_hash`. Leak-safety unit test.
2. **Reproducibility:** pin MiroFish commit + Gemma-4 tag + agent config + seeds; sim is
   stochastic → multi-seed, report mean±std.
3. **Data in-house:** Gemma local, self-hosted Zep, zero external egress.
4. **Out-of-sample only** for the ablation; honest reporting.
5. **Cost reality:** each day-sim = graph-build + multi-agent sim + report = expensive →
   mode A is a **windowed** ablation (last N days, cost-budgeted) + forward-live (C);
   **NOT** a 13-year backfill. Serialize sims vs the modeling GPU jobs.

## Phases
- **P0 — confirm API / report / graph / cost** ✅ (done: endpoints mapped; cost probe = run one day-sim on the 4090 to size the window — server-side).
- **P1 — infra:** branch, submodule, `sentisense/sim/` skeleton, migration, `init_zep.sh`, config, security review. ◀ in progress
- **P2 — client + extractor + graph** (offline fixtures + leak-safety test).
- **P3 — causal runner + cache** (`run_miro_window.py`).
- **P4 — mode A feature + `--with-sim` ablation** (windowed, OOS).
- **P5 — mode B event study** (notebook).
- **P6 — mode C live + UI graph contract** (daily hook + documented graph JSON schema).
- **P7 — validate + adversarial review + docs.**
