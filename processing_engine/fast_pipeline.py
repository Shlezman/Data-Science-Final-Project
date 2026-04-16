"""
processing_engine.fast_pipeline
================================
Single-prompt scoring pipeline — all 7 scores in ONE LLM call.

Instead of 7 separate ReAct agents (~21 LLM round-trips per headline),
this module:

1. Pre-computes ALL tool results locally (instant Python execution).
2. Builds a single prompt with the headline + tool evidence + rubric.
3. Makes exactly ONE LLM call that returns all 7 scores as structured JSON.

Speedup: ~10-15x compared to the multi-agent pipeline.

Usage::

    from processing_engine.fast_pipeline import score_headline, score_headlines_concurrent

    # Single headline
    scores = await score_headline("בנק ישראל הכריז על העלאת הריבית", llm)

    # Multiple headlines concurrently
    results = await score_headlines_concurrent(headlines, llm, concurrency=4)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from loguru import logger

from .config import (
    CATEGORY_DISPLAY_NAMES,
    RELEVANCY_CATEGORIES,
    RELEVANCY_MAX,
    RELEVANCY_MIN,
    SENTIMENT_MAX,
    SENTIMENT_MIN,
)
from .models import HeadlineScores
from .prompts import build_llm
from .tools import (
    SHARED_TOOLS,
    SENTIMENT_TOOLS,
    TOOLS_BY_CATEGORY,
)


# ═══════════════════════════════════════════════════════════════════════
# Tool pre-computation
# ═══════════════════════════════════════════════════════════════════════


def _run_tool(tool_fn, text: str) -> str:
    """Invoke a LangChain tool and return its string result."""
    try:
        return str(tool_fn.invoke({"text": text}))
    except Exception:
        # Some tools have different param names; try positional
        try:
            return str(tool_fn.invoke(text))
        except Exception as exc:
            return f"(tool error: {exc})"


def precompute_tool_evidence(headline: str) -> str:
    """
    Run ALL tools locally on a headline and format results.

    Returns a formatted text block with evidence from every tool
    (shared + all 6 categories + sentiment tools).  Tools that find
    nothing return "No ... detected" which is still useful signal.
    """
    sections: list[str] = []

    # Shared tools
    sections.append("### General Text Analysis")
    for tool_fn in SHARED_TOOLS:
        result = _run_tool(tool_fn, headline)
        sections.append(f"**{tool_fn.name}:** {result}")

    # Category-specific tools
    for cat in RELEVANCY_CATEGORIES:
        display = CATEGORY_DISPLAY_NAMES[cat]
        sections.append(f"\n### {display} Scan")
        for tool_fn in TOOLS_BY_CATEGORY[cat]:
            result = _run_tool(tool_fn, headline)
            sections.append(f"**{tool_fn.name}:** {result}")

    # Sentiment tools
    sections.append("\n### Sentiment / Tone Signals")
    for tool_fn in SENTIMENT_TOOLS:
        # Skip duplicates already covered in category scans
        if tool_fn.name in ("scan_financial_entities", "detect_economic_indicators"):
            continue
        result = _run_tool(tool_fn, headline)
        sections.append(f"**{tool_fn.name}:** {result}")

    return "\n".join(sections)


# ═══════════════════════════════════════════════════════════════════════
# Single-prompt system prompt
# ═══════════════════════════════════════════════════════════════════════

_FAST_SYSTEM_PROMPT = f"""\
You are an expert Hebrew news headline analyst. Your task is to score a \
headline on 7 dimensions in a single assessment.

## Scoring Dimensions

**6 Relevancy Categories (each {RELEVANCY_MIN}-{RELEVANCY_MAX}):**
1. **Politics & Government** — parliament, legislation, elections, diplomacy
2. **Economy & Finance** — markets, central bank, inflation, employment, trade
3. **Security & Military** — IDF, terror, conflict, defence, intelligence
4. **Health & Medicine** — hospitals, diseases, drugs, public health
5. **Science & Climate** — research, discoveries, environment, climate
6. **Technology** — startups, AI, cyber, apps, tech companies

**1 Sentiment Score ({SENTIMENT_MIN} to {SENTIMENT_MAX}):**
- Global tone of the text (NOT financial prediction)
- {SENTIMENT_MIN} = catastrophic/devastating, 0 = neutral/factual, \
{SENTIMENT_MAX} = triumphant/celebratory
- Most headlines score between -5 and +5

## Scoring Rubric (for each relevancy category)
- 0 = completely unrelated to the category
- 1-3 = tangentially related, indirect connection
- 4-6 = moderately related, partial overlap
- 7-9 = strongly related, core topic of the category
- 10 = quintessential example of the category

## Calibration
- Most headlines score 0 on most categories (not everything is about everything)
- Scores of 10 are rare — reserved for textbook examples
- Use the FULL range; 4-6 is valid for partial relevance
- When in doubt between two adjacent scores, prefer the lower one
- Base scores on the pre-computed tool evidence below, not guesswork

## Few-Shot Examples

**Headline:** "בנק ישראל הכריז על העלאת הריבית ב-0.25% לאחר עלייה באינפלציה"
(Bank of Israel announced 0.25% interest rate hike after inflation rise)
→ politics=3, economy=10, security=0, health=0, science=0, technology=0, sentiment=-3

**Headline:** "צה״ל תקף מטרות בדרום לבנון בתגובה לירי רקטות"
(IDF attacked targets in southern Lebanon in response to rocket fire)
→ politics=6, economy=2, security=9, health=0, science=0, technology=0, sentiment=-8

**Headline:** "חוקרים ישראלים פיתחו תרופה חדשה לסרטן הלבלב"
(Israeli researchers developed new drug for pancreatic cancer)
→ politics=0, economy=1, security=0, health=10, science=9, technology=3, sentiment=+8

**Headline:** "הכנסת אישרה את תקציב המדינה לשנת 2025 ברוב של 61 חברי כנסת"
(Knesset approved state budget for 2025 with 61 MK majority)
→ politics=10, economy=7, security=0, health=0, science=0, technology=0, sentiment=+3

## Instructions
1. Read the headline and the pre-computed tool analysis below.
2. For each dimension, cite the relevant tool evidence.
3. Assign integer scores based on the rubric.
4. Write a brief chain_of_thought summarising your reasoning.\
"""


# ═══════════════════════════════════════════════════════════════════════
# Core scoring functions
# ═══════════════════════════════════════════════════════════════════════


async def score_headline(
    headline: str,
    llm=None,
    *,
    _structured_llm=None,
) -> HeadlineScores:
    """
    Score a single headline using the fast single-prompt pipeline.

    Parameters
    ----------
    headline : str
        The Hebrew news headline text.
    llm : BaseChatModel, optional
        Override the default LLM instance.

    Returns
    -------
    HeadlineScores
        All 7 scores + chain_of_thought.
    """
    if _structured_llm is None:
        llm = llm or build_llm()
        _structured_llm = llm.with_structured_output(HeadlineScores)

    # Pre-compute all tool evidence locally (instant)
    evidence = precompute_tool_evidence(headline)

    user_message = f"## Headline\n{headline}\n\n## Pre-computed Tool Analysis\n{evidence}"

    result = await _structured_llm.ainvoke([
        {"role": "system", "content": _FAST_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ])

    return result


async def score_headlines_concurrent(
    headlines: list[dict[str, Any]],
    llm=None,
    *,
    concurrency: int = 4,
) -> list[dict[str, Any]]:
    """
    Score multiple headlines concurrently using the fast pipeline.

    Parameters
    ----------
    headlines : list[dict]
        List of observation dicts (must have 'headline' key).
    llm : BaseChatModel, optional
        Override the default LLM instance.
    concurrency : int
        Max simultaneous LLM calls (default: 4).

    Returns
    -------
    list[dict]
        One result dict per headline with all 7 scores + metadata.
    """
    llm = llm or build_llm()
    structured_llm = llm.with_structured_output(HeadlineScores)
    semaphore = asyncio.Semaphore(concurrency)

    async def _process_one(obs: dict[str, Any]) -> dict[str, Any]:
        headline = obs.get("headline", "")
        t0 = time.perf_counter()

        try:
            async with semaphore:
                scores = await score_headline(
                    headline, _structured_llm=structured_llm,
                )
            elapsed = time.perf_counter() - t0

            return {
                "date": obs.get("date", ""),
                "source": obs.get("source", ""),
                "hour": obs.get("hour", ""),
                "popularity": obs.get("popularity", ""),
                "headline": headline,
                "relevance_category_1": scores.politics_government,
                "relevance_category_2": scores.economy_finance,
                "relevance_category_3": scores.security_military,
                "relevance_category_4": scores.health_medicine,
                "relevance_category_5": scores.science_climate,
                "relevance_category_6": scores.technology,
                "global_sentiment": scores.global_sentiment,
                "validation_passed": True,
                "errors": [],
                "processing_time_seconds": round(elapsed, 3),
            }
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error("Fast pipeline failed for '{}…': {}", headline[:50], exc)
            return {
                "date": obs.get("date", ""),
                "source": obs.get("source", ""),
                "hour": obs.get("hour", ""),
                "popularity": obs.get("popularity", ""),
                "headline": headline,
                "relevance_category_1": 0,
                "relevance_category_2": 0,
                "relevance_category_3": 0,
                "relevance_category_4": 0,
                "relevance_category_5": 0,
                "relevance_category_6": 0,
                "global_sentiment": 0,
                "validation_passed": False,
                "errors": [f"fast_pipeline: {exc}"],
                "processing_time_seconds": round(elapsed, 3),
            }

    tasks = [_process_one(obs) for obs in headlines]
    return await asyncio.gather(*tasks)
