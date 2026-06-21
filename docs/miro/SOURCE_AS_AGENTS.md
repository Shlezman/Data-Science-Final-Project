# Source-as-agent: handling multi-channel perspective

## How a day-sim actually works (no summarization)
For decision day T the pipeline is **raw headlines → knowledge graph → agents → vote**, with
no "summarize then feed" step:

1. **Seed** (`runner.build_sim_seed`): headlines in `(T−lookback, T]`, balanced and grouped
   by source, rendered as text. Strictly ≤ T (leak-safe; `seed_hash` covers only ≤T content).
2. **Graph** (MiroFish `graph_builder`): the seed is chunked and sent to Zep/GraphRAG, which
   uses the LLM to **extract entities + relations** into a knowledge graph (extraction, not
   a summary).
3. **Agents** (`oasis_profile_generator`): MiroFish generates **one OASIS agent per graph
   entity** (people / orgs / topics found in the news), partly randomized.
4. **Sim + interview**: agents interact over rounds; `interview/all` polls each → votes →
   `extract.votes_to_features` → numeric `dir_score` / `confidence` / `disagreement`.

The qualitative **report** is produced at the end for humans — it is *not* fed to the agents.

## The problem this addresses
Headlines come from many outlets with different agendas. Naively concatenated, the day's
news became one homogeneous corpus: outlets were flattened into a single entity graph, a
viral story repeated by many channels was over-counted, and a prolific outlet's volume
dominated the seed. Channel perspective was invisible to the sim.

## A1 — what we changed (SentiSense side, no MiroFish fork)
Config (`sentisense/sim/config.py`, all env-overridable):
- `SOURCE_AS_AGENTS=true` — render the seed as one **section per source** with a preamble
  telling GraphRAG to treat each outlet as a distinct voice, so outlets surface as entities
  (→ agents).
- `SEED_PER_SOURCE_CAP=40` — newest-N **per source** before the overall `SEED_TOTAL_CAP=250`,
  so a high-volume channel can't drown a sparse one (the volume-skew fix). `SEED_FETCH_CAP`
  bounds the DB read.
- `SIM_ENTITY_TYPES` (optional, e.g. `"Organization,Source"`) — threaded into `/prepare` to
  scope entity→profile generation toward source/org entities.

Leak-safety unchanged: balancing/grouping operate only on the ≤T window; `seed_hash` still a
pure function of ≤T text.

## A3 — verify on the box, then decide
MiroFish's `/prepare` API exposes **no custom-persona hook** — profiles are auto-generated
from graph entities. A1 makes outlets *likely* to become entities but cannot guarantee a 1:1
channel↔agent mapping. So `run_day` logs a **coverage** line each sim:

```
source→graph coverage 7/9 (78%) — missing ['SomeOutlet', ...]
```
(`extract.source_agent_coverage` matches source names against graph node ids/labels.)

- **High coverage** (most sources became nodes) → A1 is sufficient; done.
- **Low coverage** → apply the A2 fallback below.

## A2 — fallback patch (apply only if A3 coverage is low)
Inject one explicit agent profile per source instead of relying on entity extraction. Exact
seam (from the source audit):
- `backend/app/api/simulation.py` `/prepare` (`prepare_simulation`, ~line 359): accept a new
  `custom_profiles` list in the request payload.
- `backend/app/services/oasis_profile_generator.py`: when `custom_profiles` is supplied, emit
  them verbatim to `reddit_profiles.json` / `twitter_profiles.csv` instead of generating from
  graph entities.
- SentiSense side: pass `custom_profiles` (one persona per source, agenda described from the
  source's headlines) through `MiroClient.prepare`.

This will be delivered as `scripts/patch_mirofish_source_agents.py` (same idempotent +`.bak`
+`--revert` shape as `patch_mirofish_zep_local.py`) **on demand** — deferred until on-box
coverage confirms it's needed, per the hybrid plan.
