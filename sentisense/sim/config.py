"""MiroFish service config (env-overridable). The MiroFish backend runs as a SEPARATE
local service (Flask); SentiSense only talks to it over HTTP — it never imports MiroFish
code in-process. The LLM (Gemma-4) + Zep are configured inside MiroFish's own .env on the
box; here we only need where the service lives + polling/agent knobs.
"""

from __future__ import annotations

import os

def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")


MIRO_URL = os.getenv("SENTISENSE_MIRO_URL", "http://localhost:5001").rstrip("/")
POLL_SECONDS = int(os.getenv("SENTISENSE_MIRO_POLL", "5"))
TIMEOUT_SECONDS = int(os.getenv("SENTISENSE_MIRO_TIMEOUT", "3600"))   # per async stage

# Refuse a non-loopback MiroFish URL unless explicitly allowed — keeps the SentiSense↔MiroFish
# hop on-box by default (see sentisense.sim.preflight.assert_local). The MiroFish service's OWN
# egress (LLM, Zep, HF) is pinned local via its .env — see docs/miro/LOCAL_ONLY.md.
ALLOW_REMOTE_MIRO = _flag("SENTISENSE_MIRO_ALLOW_REMOTE", "false")

# Recorded with each cached run (reproducibility); must match MiroFish's .env LLM_MODEL_NAME.
LLM_MODEL = os.getenv("SENTISENSE_MIRO_LLM", "gemma-4")

# OFF by default: the Twitter OASIS path auto-downloads a recommender model from huggingface.co
# (TwHIN-BERT) and spawns extra LLM traffic. Opt in explicitly once your local stack is verified.
ENABLE_TWITTER = _flag("SENTISENSE_MIRO_TWITTER", "false")
ENABLE_REDDIT = _flag("SENTISENSE_MIRO_REDDIT", "false")

# Lookback window for the per-day causal seed (days of headlines ≤ T fed as seed material).
SEED_LOOKBACK_DAYS = int(os.getenv("SENTISENSE_MIRO_LOOKBACK", "7"))

# Fixed forecasting question (kept constant for comparability across days).
DEFAULT_QUESTION = (
    "Based only on the Israeli news below, will the TA-125 index close UP or DOWN tomorrow? "
    "Give each agent's view, then the crowd consensus."
)
