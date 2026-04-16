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
# LLM backend selector
# ---------------------------------------------------------------------------

LLM_BACKEND: str = _env("LLM_BACKEND", "ollama")  # "ollama" | "openai"


# ---------------------------------------------------------------------------
# Ollama settings (default backend)
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
# OpenAI-compatible settings (external APIs — Mistral, vLLM, etc.)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class OpenAIConfig:
    """Configuration for OpenAI-compatible inference endpoints."""

    base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL", "https://10.10.248.21/v1"))
    model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "mistral-large-2"))
    temperature: float = field(default_factory=lambda: float(_env("OPENAI_TEMPERATURE", "0.1")))
    api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", "not-needed"))
    request_timeout: float = field(default_factory=lambda: float(_env("OPENAI_TIMEOUT", "600")))
    max_retries: int = field(default_factory=lambda: int(_env("OPENAI_MAX_RETRIES", "2")))
    verify_ssl: bool = field(default_factory=lambda: _env("OPENAI_VERIFY_SSL", "false").lower() in ("true", "1", "yes"))
    host_header: str = field(default_factory=lambda: _env("OPENAI_HOST_HEADER", ""))


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

# Rate limit for external API calls (requests per minute, 0 = unlimited).
# Only relevant for OpenAI-compatible backends; Ollama has no rate limits.
RATE_LIMIT_RPM: int = int(_env("RATE_LIMIT_RPM", "0"))

# Force the text-based ManualToolAgent for ALL models, regardless of name.
# Required when the inference server (e.g. vLLM on RunAI) does not support
# native tool/function calling (--enable-auto-tool-choice not set).
# Default: auto-detect (only Nemotron/Dicta use ManualToolAgent).
FORCE_MANUAL_TOOLS: bool = _env("FORCE_MANUAL_TOOLS", "false").lower() in ("true", "1", "yes")


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
