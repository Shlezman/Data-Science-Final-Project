# SentiSense Processing Engine

A multi-agent AI pipeline for scoring Hebrew news headlines on **topic relevance** and **sentiment tone**, using LangGraph-orchestrated ReAct agents backed by a local Ollama LLM.

---

## Pipeline Architecture

```
                        ┌─────────────────────────────────┐
         dict input ──► │           INGESTION              │
                        │  Validates via ObservationInput  │
                        │  (Pydantic) — cleans & seeds     │
                        │  the PipelineState envelope      │
                        └────────────────┬────────────────┘
                                         │
                          Fan-out (7 parallel async branches)
            ┌──────────┬───────────┬─────┴──────┬───────────┬────────────┬─────────────┐
            │          │           │            │           │            │             │
    ┌───────▼──┐ ┌─────▼─────┐ ┌──▼──────┐ ┌──▼──────┐ ┌──▼──────┐ ┌──▼──────┐ ┌───▼──────┐
    │ Politics │ │  Economy  │ │Security │ │ Health  │ │ Science │ │  Tech   │ │Sentiment │
    │  &  Gov  │ │ &Finance  │ │&Military│ │&Medicine│ │&Climate │ │         │ │  (tone)  │
    │  0 – 10  │ │  0 – 10   │ │  0–10   │ │  0–10   │ │  0–10   │ │  0–10   │ │ -10..+10 │
    └───────┬──┘ └─────┬─────┘ └──┬──────┘ └──┬──────┘ └──┬──────┘ └──┬──────┘ └───┬──────┘
            │          │          │            │           │            │             │
            └──────────┴──────────┴─────┬──────┴───────────┴────────────┴─────────────┘
                                         │  Fan-in (all 7 must complete)
                        ┌────────────────▼────────────────┐
                        │           VALIDATION             │
                        │  Range-checks all 7 scores,      │
                        │  sets validation_passed flag     │
                        └────────────────┬────────────────┘
                                         │
                        ┌────────────────▼────────────────┐
                        │          AGGREGATION             │
                        │  Flattens state into 7-column    │
                        │  output dict (CSV / DB ready)    │
                        └────────────────┬────────────────┘
                                         │
                                    dict output
```

---

## How Each Stage Works

### 1. Ingestion

Receives a raw dict from the caller, validates it against the `ObservationInput` Pydantic model, and seeds the shared `PipelineState` TypedDict. If validation fails (e.g. empty headline), an error is recorded and `validation_passed` is set to `False` immediately.

### 2. The 7 ReAct Agents (parallel fan-out)

All 7 agents run **concurrently** in separate async branches. Each is built with `langgraph.prebuilt.create_react_agent` and shares a single `ChatOllama` instance.

**What a ReAct agent does, step by step:**

1. **Receives** a `HumanMessage` containing the Hebrew headline.
2. **Calls tools** — the agent autonomously decides which tools to invoke. Tools are local, deterministic, and have no network calls:
   - *Shared (all agents):* `clean_hebrew_text`, `transliterate_hebrew`, `count_headline_words`, `detect_urgency_signals`, `extract_numbers_and_percentages`
   - *Category-specific:* domain keyword scanners (e.g. politics gets party/government term detectors; economy gets finance/market detectors)
   - *Sentiment-specific:* positive/negative word detectors, conflict and achievement language scanners
3. **Reasons** over the tool results using chain-of-thought, producing a structured `RelevancyOutput` or `SentimentOutput` Pydantic object.
4. **Writes** an `AgentResult` (`score` + `chain_of_thought`) into its dedicated slot in `PipelineState`.

The system prompts enforce a strict tool-first discipline: agents must call the domain scanners before scoring, so scores are grounded in curated Hebrew keyword lexicons rather than free-form LLM guesses.

If an agent fails, **tenacity** retries it (exponential backoff, up to 3 attempts). If all retries fail, a fallback `score=0` is written and the error is appended to the `errors` list — the pipeline never raises.

### 3. Validation

A post-fan-in quality gate that checks every score is within its legal range (`0–10` for relevancy, `-10..+10` for sentiment) and that no agent reported an error. Sets `validation_passed = True` only when everything is clean.

### 4. Aggregation

Flattens the scattered `PipelineState` keys into a single flat dict with exactly 7 named columns, ready for Postgres insertion or CSV export:

| Column | Type | Description |
|---|---|---|
| `relevance_category_1` | int 0–10 | Politics & Government |
| `relevance_category_2` | int 0–10 | Economy & Finance |
| `relevance_category_3` | int 0–10 | Security & Military |
| `relevance_category_4` | int 0–10 | Health & Medicine |
| `relevance_category_5` | int 0–10 | Science & Climate |
| `relevance_category_6` | int 0–10 | Technology |
| `global_sentiment` | int -10..+10 | Text tone (0 = neutral) |

Plus pass-through metadata: `date`, `source`, `hour`, `popularity`, `headline`, `validation_passed`, `errors`, `processing_time_seconds`.

---

## Scoring Rubric

**Relevance (0–10):**
- `0` — completely unrelated to the category
- `1–3` — tangential mention only
- `4–6` — moderate overlap
- `7–9` — strongly related
- `10` — quintessential example of the category

**Sentiment (-10 to +10):**
Reflects the **tone of the text**, not a financial prediction.
- `-10` — catastrophic / devastating language
- `0` — neutral / purely factual
- `+10` — celebratory / triumphant language
Most headlines fall between -5 and +5; scores beyond ±7 require overwhelming evidence.

---

## Configuration

All knobs are overridable via `SENTISENSE_*` environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `SENTISENSE_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `SENTISENSE_OLLAMA_MODEL` | `qwen2.5:14b` | Model to use |
| `SENTISENSE_OLLAMA_TEMPERATURE` | `0.1` | LLM temperature |
| `SENTISENSE_OLLAMA_TIMEOUT` | `120` | Request timeout (s) |
| `SENTISENSE_OLLAMA_NUM_CTX` | `8192` | Context window tokens |
| `SENTISENSE_AGENT_RECURSION_LIMIT` | `10` | Max ReAct loop steps |
| `SENTISENSE_RETRY_MAX_ATTEMPTS` | `3` | Retries per agent |
| `SENTISENSE_RETRY_WAIT_MIN` | `2` | Min backoff (s) |
| `SENTISENSE_RETRY_WAIT_MAX` | `10` | Max backoff (s) |
| `SENTISENSE_LOG_LEVEL` | `DEBUG` | Loguru log level |

---

## Usage

**Install:**
```bash
pip install -e .
```

**Smoke test (requires Ollama running):**
```bash
python -m processing_engine
```

**Programmatic API:**
```python
import asyncio
from processing_engine import process_single_observation

result = asyncio.run(process_single_observation({
    "date": "2025-01-15",
    "source": "כאן חדשות",
    "hour": "14:30",
    "popularity": "important",
    "headline": "בנק ישראל הכריז על העלאת הריבית ב-0.25% לאחר עלייה באינפלציה",
}))
# result["relevance_category_2"]  → Economy score, e.g. 9
# result["global_sentiment"]      → Tone score, e.g. -2
```

The graph is **lazy-compiled** on first call and cached as a module-level singleton for subsequent calls.

---

## Evaluation

See [`evaluation/`](evaluation/) for the harness that compares model outputs against a manually-labelled golden dataset and produces per-model metrics (MAE, Within-1 Accuracy, Pearson r).
