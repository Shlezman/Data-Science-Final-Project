"""
processing_engine.config
========================
Centralised configuration for the SentiSense Processing Engine.

All tuneable knobs live here so nothing is hard-coded deep inside
graph nodes or prompt templates.  Values can be overridden via
environment variables prefixed with ``SENTISENSE_``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(key: str, default: str) -> str:
    """Read an environment variable with a ``SENTISENSE_`` prefix."""
    return os.getenv(f"SENTISENSE_{key}", default)


# ---------------------------------------------------------------------------
# Relevancy categories — single source of truth
# ---------------------------------------------------------------------------

RELEVANCY_CATEGORIES: list[str] = [
    "politics_government",
    "economy_finance",
    "security_military",
    "health_medicine",
    "science_climate",
    "technology",
]

CATEGORY_DISPLAY_NAMES: dict[str, str] = {
    "politics_government": "Politics & Government",
    "economy_finance": "Economy & Finance",
    "security_military": "Security & Military",
    "health_medicine": "Health & Medicine",
    "science_climate": "Science & Climate",
    "technology": "Technology",
}

# ---------------------------------------------------------------------------
# Score bounds
# ---------------------------------------------------------------------------

RELEVANCY_MIN: int = 0
RELEVANCY_MAX: int = 10
SENTIMENT_MIN: int = -10
SENTIMENT_MAX: int = 10


# ---------------------------------------------------------------------------
# Ollama / LLM settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class OllamaConfig:
    """Configuration for the local Ollama inference backend."""

    base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "qwen2.5:14b"))
    temperature: float = field(default_factory=lambda: float(_env("OLLAMA_TEMPERATURE", "0.1")))
    request_timeout: float = field(default_factory=lambda: float(_env("OLLAMA_TIMEOUT", "120")))
    num_ctx: int = field(default_factory=lambda: int(_env("OLLAMA_NUM_CTX", "8192")))


# ---------------------------------------------------------------------------
# Agent settings
# ---------------------------------------------------------------------------

AGENT_RECURSION_LIMIT: int = int(_env("AGENT_RECURSION_LIMIT", "10"))

# Maximum number of agents that may call Ollama concurrently.
# Default: 7 (all agents run in parallel — fastest on machines with ample RAM).
# Set to 1 to run agents sequentially — required on memory-constrained machines
# where simultaneous requests cause Ollama to report OOM errors.
# Example: SENTISENSE_AGENT_CONCURRENCY=1 uv run python -m processing_engine
AGENT_CONCURRENCY: int = int(_env("AGENT_CONCURRENCY", "7"))


# ---------------------------------------------------------------------------
# Retry / resilience settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RetryConfig:
    """Tenacity retry policy for LLM calls."""

    max_attempts: int = field(default_factory=lambda: int(_env("RETRY_MAX_ATTEMPTS", "3")))
    wait_min: float = field(default_factory=lambda: float(_env("RETRY_WAIT_MIN", "2")))
    wait_max: float = field(default_factory=lambda: float(_env("RETRY_WAIT_MAX", "10")))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL: str = _env("LOG_LEVEL", "DEBUG")

# Directory where rotating log files are written.
# Set SENTISENSE_LOG_DIR="" to disable file logging entirely.
LOG_DIR: str = _env("LOG_DIR", "logs")
