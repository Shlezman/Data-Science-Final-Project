"""
processing_engine.nodes
=======================
Async LangGraph node functions for the parent pipeline graph.

Each node is a thin wrapper that bridges the parent ``PipelineState``
with the internal message-based state of the ReAct sub-agents.

Node taxonomy
-------------
* **ingestion_node** — validates the raw observation.
* **agent node factory** — wraps a ReAct sub-agent: extracts the
  headline from state → invokes the agent → writes the structured
  result back to state.  Includes tenacity retry and graceful fallback.
* **validation_node** — post-fan-in quality gate.
* **aggregation_node** — flattens state into a serialisable ``output``.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from loguru import logger
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import (
    CATEGORY_DISPLAY_NAMES,
    RELEVANCY_CATEGORIES,
    RELEVANCY_MAX,
    RELEVANCY_MIN,
    SENTIMENT_MAX,
    SENTIMENT_MIN,
    RetryConfig,
)
from .models import (
    AgentResult,
    ObservationInput,
    PipelineState,
)

_retry_cfg = RetryConfig()


# ═══════════════════════════════════════════════════════════════════════
# 1.  INGESTION NODE
# ═══════════════════════════════════════════════════════════════════════


async def ingestion_node(state: PipelineState) -> dict[str, Any]:
    """
    Validate the incoming observation and seed the pipeline state.

    Reads the raw observation from state, validates it against
    ``ObservationInput``, and writes the cleaned fields back.
    """
    logger.info("Ingestion node — validating observation")

    try:
        obs = ObservationInput(
            date=state.get("date", ""),
            source=state.get("source", ""),
            hour=state.get("hour", ""),
            popularity=state.get("popularity", ""),
            headline=state.get("headline", ""),
        )
    except ValidationError as exc:
        logger.error("Observation validation failed: {}", exc)
        return {
            "errors": [f"ingestion_validation: {exc}"],
            "validation_passed": False,
        }

    logger.debug("Observation validated — headline: {}", obs.headline[:60])
    return {
        "date": obs.date,
        "source": obs.source,
        "hour": obs.hour,
        "popularity": obs.popularity,
        "headline": obs.headline,
        "errors": [],
    }


# ═══════════════════════════════════════════════════════════════════════
# 2.  AGENT NODE FACTORY
# ═══════════════════════════════════════════════════════════════════════


def make_agent_node(agent, state_key: str, display_name: str):
    """
    Create an async node function that wraps a ReAct sub-agent.

    The wrapper:
      1. Extracts the headline from the parent ``PipelineState``.
      2. Invokes the ReAct agent with a ``HumanMessage``.
      3. Extracts ``structured_response`` from the agent output.
      4. Writes an ``AgentResult`` to ``state[state_key]``.

    If the agent fails after retries, a fallback ``AgentResult``
    with ``score=0`` and the error message is written instead.

    Parameters
    ----------
    agent : CompiledGraph
        A compiled ``create_react_agent`` instance.
    state_key : str
        The key in ``PipelineState`` to write results to
        (e.g. ``"relevancy_economy_finance"`` or ``"sentiment"``).
    display_name : str
        Human-readable name for logging.
    """

    async def _invoke_agent(headline: str):
        """Invoke the ReAct agent and return the structured response."""
        result = await agent.ainvoke({
            "messages": [
                HumanMessage(content=f"Analyze this Hebrew news headline:\n\n{headline}")
            ],
        })
        structured = result.get("structured_response")
        if structured is None:
            raise ValueError(
                f"Agent '{display_name}' did not produce a structured_response. "
                f"Last message: {result.get('messages', [{}])[-1]}"
            )
        return structured

    _invoke_with_retry = retry(
        stop=stop_after_attempt(_retry_cfg.max_attempts),
        wait=wait_exponential(
            min=_retry_cfg.wait_min,
            max=_retry_cfg.wait_max,
        ),
        retry=retry_if_exception_type((ValueError, ValidationError, KeyError, TypeError)),
        reraise=True,
    )(_invoke_agent)

    async def agent_node(state: PipelineState) -> dict[str, Any]:
        headline = state["headline"]
        logger.info("[{}] Starting ReAct agent for: {}…", display_name, headline[:50])

        try:
            structured = await _invoke_with_retry(headline)
            logger.info(
                "[{}] score={} confidence={:.2f}",
                display_name,
                structured.score,
                structured.confidence,
            )
            return {
                state_key: AgentResult(
                    score=structured.score,
                    confidence=structured.confidence,
                    chain_of_thought=structured.chain_of_thought,
                    error=None,
                )
            }
        except Exception as exc:
            logger.error("[{}] Agent failed after retries: {}", display_name, exc)
            return {
                state_key: AgentResult(
                    score=0,
                    confidence=0.0,
                    chain_of_thought="",
                    error=str(exc),
                ),
                "errors": [f"{state_key}: {exc}"],
            }

    agent_node.__name__ = f"{state_key}_node"
    agent_node.__qualname__ = f"{state_key}_node"
    return agent_node


# ═══════════════════════════════════════════════════════════════════════
# 3.  VALIDATION NODE
# ═══════════════════════════════════════════════════════════════════════


async def validation_node(state: PipelineState) -> dict[str, Any]:
    """
    Post-fan-in quality gate.

    Checks:
      - Every relevancy score is in [RELEVANCY_MIN, RELEVANCY_MAX].
      - Sentiment score is in [SENTIMENT_MIN, SENTIMENT_MAX].
      - No agent reported an error.

    Sets ``validation_passed`` to ``True`` only if everything is clean.
    """
    logger.info("Validation node — checking all agent results")
    errors: list[str] = []

    for cat in RELEVANCY_CATEGORIES:
        key = f"relevancy_{cat}"
        result: AgentResult | None = state.get(key)  # type: ignore[assignment]
        if result is None:
            errors.append(f"{key}: missing from state")
            continue
        if result.get("error"):
            errors.append(f"{key}: agent error — {result['error']}")
        score = result.get("score", 0)
        if not (RELEVANCY_MIN <= score <= RELEVANCY_MAX):
            errors.append(
                f"{key}: score {score} out of bounds "
                f"[{RELEVANCY_MIN}, {RELEVANCY_MAX}]"
            )

    sentiment: AgentResult | None = state.get("sentiment")  # type: ignore[assignment]
    if sentiment is None:
        errors.append("sentiment: missing from state")
    else:
        if sentiment.get("error"):
            errors.append(f"sentiment: agent error — {sentiment['error']}")
        s_score = sentiment.get("score", 0)
        if not (SENTIMENT_MIN <= s_score <= SENTIMENT_MAX):
            errors.append(
                f"sentiment: score {s_score} out of bounds "
                f"[{SENTIMENT_MIN}, {SENTIMENT_MAX}]"
            )

    passed = len(errors) == 0
    level = "PASSED" if passed else "FAILED"
    logger.info("Validation {} — {} error(s)", level, len(errors))
    for e in errors:
        logger.warning("  ↳ {}", e)

    return {"validation_passed": passed, "errors": errors}


# ═══════════════════════════════════════════════════════════════════════
# 4.  AGGREGATION NODE
# ═══════════════════════════════════════════════════════════════════════


async def aggregation_node(state: PipelineState) -> dict[str, Any]:
    """
    Flatten all agent results into a single ``output`` dict suitable
    for direct Postgres insertion or CSV export.

    Output schema::

        {
          "date", "source", "hour", "popularity", "headline",
          "relevancy_{cat}_score", "relevancy_{cat}_confidence",
          "relevancy_{cat}_reasoning",   (× 6 categories)
          "sentiment_score", "sentiment_confidence",
          "sentiment_reasoning",
          "validation_passed", "errors"
        }
    """
    logger.info("Aggregation node — building flat output dict")

    output: dict[str, Any] = {
        "date": state.get("date", ""),
        "source": state.get("source", ""),
        "hour": state.get("hour", ""),
        "popularity": state.get("popularity", ""),
        "headline": state.get("headline", ""),
    }

    for cat in RELEVANCY_CATEGORIES:
        key = f"relevancy_{cat}"
        result: AgentResult = state.get(key, {})  # type: ignore[assignment]
        output[f"{key}_score"] = result.get("score", 0)
        output[f"{key}_confidence"] = result.get("confidence", 0.0)
        output[f"{key}_reasoning"] = result.get("chain_of_thought", "")

    sentiment: AgentResult = state.get("sentiment", {})  # type: ignore[assignment]
    output["sentiment_score"] = sentiment.get("score", 0)
    output["sentiment_confidence"] = sentiment.get("confidence", 0.0)
    output["sentiment_reasoning"] = sentiment.get("chain_of_thought", "")

    output["validation_passed"] = state.get("validation_passed", False)
    output["errors"] = state.get("errors", [])

    logger.debug("Aggregated output keys: {}", list(output.keys()))
    return {"output": output}
