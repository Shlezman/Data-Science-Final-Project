"""
processing_engine.models
========================
Pydantic V2 models and LangGraph state definitions.

Three validation layers:

1. **Input** — ``ObservationInput`` validates the raw dict from the caller.
2. **LLM output** — ``RelevancyOutput`` / ``SentimentOutput`` enforce that
   every agent response conforms to the expected schema.
3. **Graph state** — ``PipelineState`` (TypedDict) is the mutable envelope
   that LangGraph passes through every node, with ``Annotated`` reducers
   for safe parallel merging.

Output contract (7 columns per headline):
  relevance_category_1 … relevance_category_6  — integer scores 0–10
  global_sentiment                              — integer score -10..+10
                                                  (0 = neutral tone)
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field, field_validator

from .config import (
    RELEVANCY_MAX,
    RELEVANCY_MIN,
    SENTIMENT_MAX,
    SENTIMENT_MIN,
)


# ═══════════════════════════════════════════════════════════════════════
# 1.  INPUT VALIDATION
# ═══════════════════════════════════════════════════════════════════════


class ObservationInput(BaseModel):
    """
    Validates a single scraped observation before it enters the graph.

    Expected keys mirror the scraper CSV columns:
      - date        (str, "YYYY-MM-DD")
      - source      (str, publisher name)
      - hour        (str, "HH:MM" or similar)
      - popularity  (str, importance CSS class)
      - headline    (str, the Hebrew headline text)
    """

    date: str = Field(..., description="Publication date in YYYY-MM-DD format.")
    source: str = Field(..., description="News publisher name.")
    hour: str = Field(..., description="Time of publication.")
    popularity: str = Field(default="", description="Importance level from the HTML class attribute.")
    headline: str = Field(..., min_length=1, description="The Hebrew news headline to analyse.")

    model_config = {"extra": "allow"}


# ═══════════════════════════════════════════════════════════════════════
# 2.  LLM OUTPUT SCHEMAS  (used as response_format in create_react_agent)
# ═══════════════════════════════════════════════════════════════════════


class RelevancyOutput(BaseModel):
    """
    Structured output expected from every relevancy agent.

    ``chain_of_thought`` captures the CoT reasoning the agent produced
    *during* the ReAct loop (summarised at the final structured step).
    No confidence field — the score alone is the deliverable.
    """

    chain_of_thought: str = Field(
        ...,
        description=(
            "Your step-by-step reasoning in English.  Structure it as: "
            "(1) What is the headline about?  "
            "(2) Which entities/keywords did your tools detect?  "
            "(3) How does this evidence map to the scoring rubric?  "
            "(4) Why did you choose this specific score?  "
            "Reference concrete tool results — do not make claims "
            "without evidence from your analysis tools."
        ),
    )
    score: int = Field(
        ...,
        ge=RELEVANCY_MIN,
        le=RELEVANCY_MAX,
        description=(
            f"Relevancy score from {RELEVANCY_MIN} to {RELEVANCY_MAX}.  "
            f"{RELEVANCY_MIN}=completely unrelated, 1-3=tangential, "
            f"4-6=moderate overlap, 7-9=strongly related, "
            f"{RELEVANCY_MAX}=quintessential.  Use the full range; "
            f"most headlines are NOT 0 or {RELEVANCY_MAX}."
        ),
    )

    @field_validator("score", mode="before")
    @classmethod
    def _coerce_score(cls, v: Any) -> int:
        """Accept stringified ints and floats from sloppy LLM output."""
        if isinstance(v, str):
            v = float(v)
        if isinstance(v, float):
            v = round(v)
        return int(v)


class SentimentOutput(BaseModel):
    """
    Structured output expected from the sentiment agent.

    Score semantics (tone of the text, NOT a financial prediction):
      -10  = extremely negative tone (catastrophic, devastating language)
        0  = neutral tone (factual, no clear positive or negative valence)
      +10  = extremely positive tone (celebratory, triumphant language)
    """

    chain_of_thought: str = Field(
        ...,
        description=(
            "Your step-by-step reasoning in English.  Structure it as: "
            "(1) What is the headline about?  "
            "(2) Which tone signals did your tools detect (positive words, "
            "negative words, conflict language, achievement language)?  "
            "(3) What is the overall emotional valence of the text?  "
            "(4) Why did you choose this specific score?  "
            "Reference concrete tool results."
        ),
    )
    score: int = Field(
        ...,
        ge=SENTIMENT_MIN,
        le=SENTIMENT_MAX,
        description=(
            f"Tone score from {SENTIMENT_MIN} (extremely negative) "
            f"to {SENTIMENT_MAX} (extremely positive).  "
            f"0=neutral/no clear emotional valence.  Most headlines score "
            f"between -5 and +5.  Scores beyond ±7 require overwhelming "
            f"evidence of extreme positive or negative language."
        ),
    )

    @field_validator("score", mode="before")
    @classmethod
    def _coerce_score(cls, v: Any) -> int:
        if isinstance(v, str):
            v = float(v)
        if isinstance(v, float):
            v = round(v)
        return int(v)


# ═══════════════════════════════════════════════════════════════════════
# 3.  GRAPH STATE  (LangGraph TypedDict with reducers)
# ═══════════════════════════════════════════════════════════════════════


class HeadlineScores(BaseModel):
    """
    All 7 scores from a single LLM call (fast pipeline mode).

    Used by ``fast_pipeline.py`` to get all relevancy + sentiment scores
    in one inference call instead of 7 separate ReAct agent invocations.
    """

    chain_of_thought: str = Field(
        ...,
        description=(
            "Brief reasoning for each score.  Reference the pre-computed "
            "tool analysis results to justify your scores."
        ),
    )
    politics_government: int = Field(
        ..., ge=RELEVANCY_MIN, le=RELEVANCY_MAX,
        description="Relevancy to Politics & Government (0-10).",
    )
    economy_finance: int = Field(
        ..., ge=RELEVANCY_MIN, le=RELEVANCY_MAX,
        description="Relevancy to Economy & Finance (0-10).",
    )
    security_military: int = Field(
        ..., ge=RELEVANCY_MIN, le=RELEVANCY_MAX,
        description="Relevancy to Security & Military (0-10).",
    )
    health_medicine: int = Field(
        ..., ge=RELEVANCY_MIN, le=RELEVANCY_MAX,
        description="Relevancy to Health & Medicine (0-10).",
    )
    science_climate: int = Field(
        ..., ge=RELEVANCY_MIN, le=RELEVANCY_MAX,
        description="Relevancy to Science & Climate (0-10).",
    )
    technology: int = Field(
        ..., ge=RELEVANCY_MIN, le=RELEVANCY_MAX,
        description="Relevancy to Technology (0-10).",
    )
    global_sentiment: int = Field(
        ..., ge=SENTIMENT_MIN, le=SENTIMENT_MAX,
        description="Overall tone of the text (-10 to +10, 0=neutral).",
    )

    @field_validator(
        "politics_government", "economy_finance", "security_military",
        "health_medicine", "science_climate", "technology", "global_sentiment",
        mode="before",
    )
    @classmethod
    def _coerce_scores(cls, v: Any) -> int:
        """Accept stringified ints and floats from sloppy LLM output."""
        if isinstance(v, str):
            v = float(v)
        if isinstance(v, float):
            v = round(v)
        return int(v)


class HeadlineScoreEntry(BaseModel):
    """One headline's scores inside a batch response."""

    headline_index: int = Field(
        ...,
        ge=0,
        description="0-based index matching the headline's position in the input list.",
    )
    chain_of_thought: str = Field(
        ...,
        description="Brief reasoning for this headline's scores.",
    )
    politics_government: int = Field(
        ..., ge=RELEVANCY_MIN, le=RELEVANCY_MAX,
        description="Relevancy to Politics & Government (0-10).",
    )
    economy_finance: int = Field(
        ..., ge=RELEVANCY_MIN, le=RELEVANCY_MAX,
        description="Relevancy to Economy & Finance (0-10).",
    )
    security_military: int = Field(
        ..., ge=RELEVANCY_MIN, le=RELEVANCY_MAX,
        description="Relevancy to Security & Military (0-10).",
    )
    health_medicine: int = Field(
        ..., ge=RELEVANCY_MIN, le=RELEVANCY_MAX,
        description="Relevancy to Health & Medicine (0-10).",
    )
    science_climate: int = Field(
        ..., ge=RELEVANCY_MIN, le=RELEVANCY_MAX,
        description="Relevancy to Science & Climate (0-10).",
    )
    technology: int = Field(
        ..., ge=RELEVANCY_MIN, le=RELEVANCY_MAX,
        description="Relevancy to Technology (0-10).",
    )
    global_sentiment: int = Field(
        ..., ge=SENTIMENT_MIN, le=SENTIMENT_MAX,
        description="Overall tone of the text (-10 to +10, 0=neutral).",
    )

    @field_validator(
        "politics_government", "economy_finance", "security_military",
        "health_medicine", "science_climate", "technology", "global_sentiment",
        mode="before",
    )
    @classmethod
    def _coerce_scores(cls, v: Any) -> int:
        if isinstance(v, str):
            v = float(v)
        if isinstance(v, float):
            v = round(v)
        return int(v)


class BatchHeadlineScores(BaseModel):
    """
    Structured output for scoring multiple headlines in a single LLM call.

    Each entry in ``results`` corresponds to one headline from the input,
    identified by ``headline_index`` (0-based).
    """

    results: list[HeadlineScoreEntry] = Field(
        ...,
        description="One scoring entry per headline, in order of input.",
    )


class AgentResult(TypedDict, total=False):
    """Result payload written by a single agent node."""

    score: int          # used by both relevancy and sentiment agents
    chain_of_thought: str
    error: str | None


class PipelineState(TypedDict, total=False):
    """
    The mutable state envelope carried through every LangGraph node.

    ``errors`` uses an ``operator.add`` reducer so that parallel agent
    nodes can safely append errors without overwriting each other.
    All other keys are written by exactly one node, so no reducer is
    needed.

    Final ``output`` contains exactly 7 data columns:
      relevance_category_1 … relevance_category_6  (int, 0–10)
      global_sentiment                              (int, -10..+10)
    """

    # --- Original observation (immutable after ingestion) ----
    date: str
    source: str
    hour: str
    popularity: str
    headline: str

    # --- Agent results (written during fan-out, one key per agent) ----
    relevancy_politics_government: AgentResult
    relevancy_economy_finance: AgentResult
    relevancy_security_military: AgentResult
    relevancy_health_medicine: AgentResult
    relevancy_science_climate: AgentResult
    relevancy_technology: AgentResult
    sentiment: AgentResult

    # --- Post-validation flags ----
    validation_passed: bool
    errors: Annotated[list[str], operator.add]

    # --- Final flat output (populated by the aggregator) ----
    output: dict[str, Any]
