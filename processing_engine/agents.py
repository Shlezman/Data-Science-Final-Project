"""
processing_engine.agents
========================
ReAct agent construction using ``langgraph.prebuilt.create_react_agent``.

Each of the 7 agents (6 relevancy + 1 sentiment) is a fully autonomous
ReAct agent that:

1. Receives a Hebrew headline as a ``HumanMessage``.
2. Decides which of its bound tools to call (keyword scanners, entity
   detectors, numeric extractors, etc.).
3. Reasons step-by-step using the tool results.
4. Produces a structured Pydantic response via the ``response_format``
   parameter:
     - ``RelevancyOutput`` — integer score 0–10 for one category.
     - ``SentimentOutput`` — integer score -10..+10 reflecting the
       general tone of the text (0=neutral; no financial prediction,
       no confidence score).

The ``response_format`` triggers an extra LLM call after the ReAct loop
to extract a validated Pydantic model — but since all 7 agents execute
in parallel, the wall-clock overhead is just one extra call.

Architecture contract
---------------------
- Relevance is category-specific: evaluated 6 times per headline,
  once per category.
- Sentiment is global: evaluated exactly 1 time per headline.
- No agent emits a confidence score — the score/label alone is the
  deliverable.
"""

from __future__ import annotations

from loguru import logger
from langgraph.prebuilt import create_react_agent

from .config import (
    CATEGORY_DISPLAY_NAMES,
    RELEVANCY_CATEGORIES,
)
from .models import RelevancyOutput, SentimentOutput
from .prompts import (
    RELEVANCY_EXTRACTION_INSTRUCTION,
    SENTIMENT_EXTRACTION_INSTRUCTION,
    build_llm,
    build_relevancy_system_prompt,
    build_sentiment_system_prompt,
)
from .tools import SHARED_TOOLS, SENTIMENT_TOOLS, TOOLS_BY_CATEGORY


def build_relevancy_agent(category: str, llm=None):
    """
    Build a ReAct agent for a single relevancy category.

    Parameters
    ----------
    category : str
        One of the 6 category slugs (e.g. ``"economy_finance"``).
    llm : BaseChatModel, optional
        Override the default ChatOllama instance.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph agent that accepts
        ``{"messages": [HumanMessage(...)]}`` and returns a state
        containing ``structured_response: RelevancyOutput``
        (fields: ``score`` int 0–10, ``chain_of_thought`` str).
    """
    llm = llm or build_llm()
    display = CATEGORY_DISPLAY_NAMES[category]

    tools = SHARED_TOOLS + TOOLS_BY_CATEGORY[category]
    prompt = build_relevancy_system_prompt(category)

    # The tuple form (instruction_str, PydanticModel) passes extra
    # instructions to the structured-output extraction step that runs
    # after the ReAct loop completes.
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,
        response_format=(RELEVANCY_EXTRACTION_INSTRUCTION, RelevancyOutput),
        name=f"relevancy_{category}_agent",
    )

    logger.debug("Built ReAct agent: relevancy_{} ({} tools)", category, len(tools))
    return agent


def build_sentiment_agent(llm=None):
    """
    Build the ReAct agent for global text-tone sentiment classification.

    The agent scores the *general tone* of the headline on a scale
    from -10 (extremely negative) to +10 (extremely positive), with
    0 meaning neutral.  It does NOT predict financial market movements
    or TA-125 index impact.

    Parameters
    ----------
    llm : BaseChatModel, optional
        Override the default ChatOllama instance.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph agent that accepts
        ``{"messages": [HumanMessage(...)]}`` and returns a state
        containing ``structured_response: SentimentOutput``
        (fields: ``score`` int -10..+10,
        ``chain_of_thought`` str).
    """
    llm = llm or build_llm()

    tools = SHARED_TOOLS + SENTIMENT_TOOLS
    prompt = build_sentiment_system_prompt()

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,
        response_format=(SENTIMENT_EXTRACTION_INSTRUCTION, SentimentOutput),
        name="sentiment_agent",
    )

    logger.debug("Built ReAct agent: sentiment ({} tools)", len(tools))
    return agent


def build_all_agents(llm=None) -> dict:
    """
    Build all 7 agents, sharing a single LLM instance.

    Returns
    -------
    dict
        Keys: 6 × ``"relevancy_{category}"`` + ``"sentiment"``.
        Values: compiled ReAct agent graphs.
    """
    llm = llm or build_llm()
    agents: dict = {}

    for cat in RELEVANCY_CATEGORIES:
        agents[f"relevancy_{cat}"] = build_relevancy_agent(cat, llm=llm)

    agents["sentiment"] = build_sentiment_agent(llm=llm)

    logger.info("All 7 ReAct agents built successfully")
    return agents
