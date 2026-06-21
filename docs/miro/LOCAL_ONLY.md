# Running MiroFish fully local (zero external egress)

SentiSense talks to MiroFish over loopback HTTP only. MiroFish *itself*, out of the box,
egresses to **three** external clouds. This doc lists every one (from an adversarial source
audit of the vendored `external/MiroFish`) and the exact setting that pins it local.

> **TL;DR** — copy `scripts/mirofish.env.local-only.example` → `external/MiroFish/backend/.env`,
> fill the two `<host>` values, run `scripts/init_zep.sh`, then `bash scripts/verify_local_egress.sh`.
> Our own layer additionally refuses a non-loopback MiroFish URL (`assert_local`) and defaults
> the Twitter/Reddit OASIS platforms **off**.

## The three egress classes

| # | Component | Default destination | Confirmed | Fix |
|---|-----------|---------------------|-----------|-----|
| 1 | **LLM** — every agent "thought", report, persona, sim-config (`llm_client.py`, `oasis_profile_generator.py`, `simulation_config_generator.py`, OASIS subprocess scripts) | `https://api.openai.com/v1` (code default) / `dashscope.aliyuncs.com` (shipped `.env.example`) | ✅ env override works, no code change | `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL_NAME` + `OPENAI_API_BASE_URL`/`OPENAI_BASE_URL`; for parallel runs also `LLM_BOOST_BASE_URL`. **Never leave blank** — blank falls back to OpenAI. |
| 2 | **Zep Cloud** — knowledge-graph build/search/memory (5 `Zep(api_key=…)` sites: `graph_builder.py:51`, `zep_tools.py:430`, `zep_entity_reader.py:86`, `zep_graph_memory_updater.py:246`, `oasis_profile_generator.py:208`) | `https://api.getzep.com/api/v2` | ⚠️ **contested** — see below | `ZEP_API_URL` env **if** the SDK honors it, else `scripts/patch_mirofish_zep_local.py` |
| 3 | **HuggingFace** — Twitter recsys model auto-download (TwHIN-BERT + MiniLM) via `camel-oasis` | `huggingface.co` / `cdn-lfs` | ✅ Twitter path only; **Reddit path does not egress** | `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` + populated `HF_HOME`, **or** leave Twitter off (our default) |

Not runtime egress: a CI workflow pushes an image to `ghcr.io` (`.github/workflows/docker-image.yml`) — build-time only, never runs during a sim. No live Twitter/Reddit API calls exist (agents are LLM-simulated; the only key the social sim needs is the *LLM* key, not a Twitter/Reddit key).

## The Zep contradiction (item 2) — resolve it on the box

The audit's verifiers split on whether `zep-cloud==3.13.0`'s `Zep.__init__` reads `ZEP_API_URL`:
some matched it verbatim against the v3.13.0 tag, others found the env hook only on a newer
branch and concluded a code change is required. **Do not guess — measure:**

```bash
bash scripts/verify_local_egress.sh        # step 1 introspects the installed SDK
```
- **"SDK reads ZEP_API_URL"** → just set `ZEP_API_URL=http://localhost:8000` (no `/api/v2` suffix; the SDK appends it). Done.
- **"SDK does NOT read ZEP_API_URL"** → `python scripts/patch_mirofish_zep_local.py` injects an explicit `base_url` (built from `ZEP_API_URL`) at all 5 call sites. The SDK *always* honors an explicit `base_url`. Idempotent, writes `.bak`, `--revert` to undo.

Either way keep `ZEP_API_KEY` set to any non-empty value — `Config.validate()` hard-requires it and each constructor rejects an empty key.

## Verify

```bash
# static: SDK env-hook check + scan MiroFish/.env for any external host in egress keys
bash scripts/verify_local_egress.sh
# live: watch the running backend's sockets (loopback only is the pass condition)
bash scripts/verify_local_egress.sh <mirofish_backend_pid>
# or, ad-hoc, while a sim runs:
lsof -nP -iTCP -sTCP:ESTABLISHED | grep -E 'getzep|api\.openai|dashscope|huggingface'
```
A clean run shows only `127.0.0.1` / `::1` connections and exits 0.

## What SentiSense enforces on its side
- `sentisense.sim.preflight.assert_local` — `MiroClient` refuses a non-loopback `SENTISENSE_MIRO_URL` unless `SENTISENSE_MIRO_ALLOW_REMOTE=1`.
- `ENABLE_TWITTER` / `ENABLE_REDDIT` default **off** (`config.py`) — avoids the HF download (item 3) and the extra Twitter LLM subprocess. Opt in only after the stack is verified local.
