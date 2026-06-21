"""MiroFish service config (env-overridable). The MiroFish backend runs as a SEPARATE
local service (Flask); SentiSense only talks to it over HTTP — it never imports MiroFish
code in-process. The LLM (Gemma-4) + Zep are configured inside MiroFish's own .env on the
box; here we only need where the service lives + polling/agent knobs.
"""

from __future__ import annotations

import os

MIRO_URL = os.getenv("SENTISENSE_MIRO_URL", "http://localhost:5001").rstrip("/")
POLL_SECONDS = int(os.getenv("SENTISENSE_MIRO_POLL", "5"))
TIMEOUT_SECONDS = int(os.getenv("SENTISENSE_MIRO_TIMEOUT", "3600"))   # per async stage

# Recorded with each cached run (reproducibility); must match MiroFish's .env LLM_MODEL_NAME.
LLM_MODEL = os.getenv("SENTISENSE_MIRO_LLM", "gemma-4")

ENABLE_TWITTER = os.getenv("SENTISENSE_MIRO_TWITTER", "true").lower() in ("1", "true", "yes")
ENABLE_REDDIT = os.getenv("SENTISENSE_MIRO_REDDIT", "true").lower() in ("1", "true", "yes")

# Lookback window for the per-day causal seed (days of headlines ≤ T fed as seed material).
SEED_LOOKBACK_DAYS = int(os.getenv("SENTISENSE_MIRO_LOOKBACK", "7"))

# Fixed forecasting question (kept constant for comparability across days).
DEFAULT_QUESTION = (
    "Based only on the Israeli news below, will the TA-125 index close UP or DOWN tomorrow? "
    "Give each agent's view, then the crowd consensus."
)
