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

Three modes (from slowest to fastest):

- **score_headline()** — 1 headline → 1 LLM call
- **score_headlines_concurrent()** — N headlines → N parallel LLM calls
- **score_headlines_batch()** — N headlines → 1 LLM call (batch mode)

Usage::

    from processing_engine.fast_pipeline import (
        score_headline,
        score_headlines_concurrent,
        score_headlines_batch,
    )

    # Single headline
    scores = await score_headline("בנק ישראל הכריז על העלאת הריבית", llm)

    # Multiple headlines concurrently (1 call each)
    results = await score_headlines_concurrent(headlines, llm, concurrency=4)

    # Batch mode — multiple headlines in a single LLM call
    results = await score_headlines_batch(headlines, llm, batch_size=15)
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
from .models import BatchHeadlineScores, HeadlineScoreEntry, HeadlineScores
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


# ═══════════════════════════════════════════════════════════════════════
# Batch mode — multiple headlines per single LLM call
# ═══════════════════════════════════════════════════════════════════════

# Token budget constants (for Mistral-Large-2 / 32K context)
_CONTEXT_WINDOW = 32_768
_SYSTEM_PROMPT_TOKENS = 800       # measured ~750, padded
_PER_HEADLINE_INPUT_TOKENS = 650  # evidence + headline text + formatting
_PER_HEADLINE_OUTPUT_TOKENS = 250  # scores + chain_of_thought
_SAFETY_FACTOR = 0.75             # 25% safety margin

MAX_BATCH_SIZE = int(
    (_CONTEXT_WINDOW * _SAFETY_FACTOR - _SYSTEM_PROMPT_TOKENS)
    / (_PER_HEADLINE_INPUT_TOKENS + _PER_HEADLINE_OUTPUT_TOKENS)
)  # ≈ 25


_BATCH_SYSTEM_PROMPT = f"""\
You are an expert Hebrew news headline analyst. You will receive MULTIPLE \
headlines to score in a single assessment. Score each headline independently \
on 7 dimensions.

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
- Most headlines score 0 on most categories
- Scores of 10 are rare — reserved for textbook examples
- Use the FULL range; 4-6 is valid for partial relevance
- When in doubt between two adjacent scores, prefer the lower one
- Base scores on the pre-computed tool evidence, not guesswork

## Instructions
1. For EACH headline (identified by its index), read the text and tool analysis.
2. Score each headline INDEPENDENTLY — do not let one headline influence another.
3. Return one entry per headline in the ``results`` array, preserving the \
``headline_index`` from the input.\
"""


def _build_batch_user_message(headlines: list[str], evidences: list[str]) -> str:
    """Format N headlines + their tool evidence into a single user message."""
    parts: list[str] = []
    for i, (headline, evidence) in enumerate(zip(headlines, evidences)):
        parts.append(
            f"---\n## Headline [{i}]\n{headline}\n\n"
            f"### Tool Analysis [{i}]\n{evidence}"
        )
    return "\n\n".join(parts)


def _entry_to_result_dict(
    entry: HeadlineScoreEntry,
    obs: dict[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    """Convert a HeadlineScoreEntry + observation into the standard result dict."""
    return {
        "date": obs.get("date", ""),
        "source": obs.get("source", ""),
        "hour": obs.get("hour", ""),
        "popularity": obs.get("popularity", ""),
        "headline": obs.get("headline", ""),
        "relevance_category_1": entry.politics_government,
        "relevance_category_2": entry.economy_finance,
        "relevance_category_3": entry.security_military,
        "relevance_category_4": entry.health_medicine,
        "relevance_category_5": entry.science_climate,
        "relevance_category_6": entry.technology,
        "global_sentiment": entry.global_sentiment,
        "validation_passed": True,
        "errors": [],
        "processing_time_seconds": round(elapsed, 3),
    }


def _error_result_dict(obs: dict[str, Any], error: str, elapsed: float) -> dict[str, Any]:
    """Return a zeroed-out result dict for a failed headline."""
    return {
        "date": obs.get("date", ""),
        "source": obs.get("source", ""),
        "hour": obs.get("hour", ""),
        "popularity": obs.get("popularity", ""),
        "headline": obs.get("headline", ""),
        "relevance_category_1": 0,
        "relevance_category_2": 0,
        "relevance_category_3": 0,
        "relevance_category_4": 0,
        "relevance_category_5": 0,
        "relevance_category_6": 0,
        "global_sentiment": 0,
        "validation_passed": False,
        "errors": [error],
        "processing_time_seconds": round(elapsed, 3),
    }


async def score_headlines_batch(
    headlines: list[dict[str, Any]],
    llm=None,
    *,
    batch_size: int = 15,
    concurrency: int = 4,
) -> list[dict[str, Any]]:
    """
    Score multiple headlines by batching them into fewer LLM calls.

    Instead of 1 LLM call per headline, this packs up to ``batch_size``
    headlines + their pre-computed tool evidence into a single prompt.
    Multiple batches run concurrently (controlled by ``concurrency``).

    Parameters
    ----------
    headlines : list[dict]
        List of observation dicts (must have 'headline' key).
    llm : BaseChatModel, optional
        Override the default LLM instance.
    batch_size : int
        Max headlines per LLM call (default: 15, max: MAX_BATCH_SIZE).
    concurrency : int
        Max simultaneous batch LLM calls (default: 4).

    Returns
    -------
    list[dict]
        One result dict per headline with all 7 scores + metadata.
    """
    batch_size = min(batch_size, MAX_BATCH_SIZE)
    llm = llm or build_llm()
    structured_llm = llm.with_structured_output(BatchHeadlineScores)
    semaphore = asyncio.Semaphore(concurrency)

    async def _process_batch(
        batch_obs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Score one batch of headlines in a single LLM call."""
        headline_texts = [obs.get("headline", "") for obs in batch_obs]
        t0 = time.perf_counter()

        # Pre-compute all tool evidence locally (instant)
        evidences = [precompute_tool_evidence(h) for h in headline_texts]
        user_message = _build_batch_user_message(headline_texts, evidences)

        try:
            async with semaphore:
                batch_result = await structured_llm.ainvoke([
                    {"role": "system", "content": _BATCH_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ])
            elapsed = time.perf_counter() - t0

            # Map results by headline_index for safe ordering
            entries_by_idx: dict[int, HeadlineScoreEntry] = {
                entry.headline_index: entry for entry in batch_result.results
            }

            results: list[dict[str, Any]] = []
            for i, obs in enumerate(batch_obs):
                entry = entries_by_idx.get(i)
                if entry is not None:
                    results.append(_entry_to_result_dict(entry, obs, elapsed))
                else:
                    logger.warning(
                        "Batch missing index {} for '{}'",
                        i, obs.get("headline", "")[:50],
                    )
                    results.append(_error_result_dict(
                        obs, f"batch_missing_index_{i}", elapsed,
                    ))
            return results

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error(
                "Batch scoring failed ({} headlines): {}",
                len(batch_obs), exc,
            )
            return [
                _error_result_dict(obs, f"batch_error: {exc}", elapsed)
                for obs in batch_obs
            ]

    # Split into batches and run concurrently
    batches = [
        headlines[i : i + batch_size]
        for i in range(0, len(headlines), batch_size)
    ]

    logger.info(
        "Batch mode: {} headlines → {} batches (size ≤{}), concurrency={}",
        len(headlines), len(batches), batch_size, concurrency,
    )

    batch_results = await asyncio.gather(*[_process_batch(b) for b in batches])

    # Flatten batch results into a single ordered list
    return [result for batch in batch_results for result in batch]
