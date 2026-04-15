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
4. Produces a structured Pydantic response:
     - ``RelevancyOutput`` — integer score 0–10 for one category.
     - ``SentimentOutput`` — integer score -10..+10 reflecting the
       general tone of the text (0=neutral).

Two execution paths
-------------------
* **Native tool-calling** (default) — uses ``create_react_agent`` with
  ``response_format`` for models that support the Ollama function-calling
  API (LLaMA, Qwen, Mistral, Command-R, …).

* **Manual tool-calling fallback** — ``ManualToolAgent`` for models in
  the Nemotron / Dicta family that lack native API tool support.  The
  agent loop is implemented manually: tool schemas are injected into the
  system prompt, the model emits ``tool_call`` / ``final_answer`` JSON
  blocks, and the class executes tools and feeds results back.

Architecture contract
---------------------
- Relevance is category-specific: evaluated 6 times per headline,
  once per category.
- Sentiment is global: evaluated exactly 1 time per headline.
- Both agent types expose the same ``ainvoke({"messages": [...]})``
  interface so ``make_agent_node`` in nodes.py requires zero changes.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from loguru import logger
from langgraph.prebuilt import create_react_agent

from .config import (
    CATEGORY_DISPLAY_NAMES,
    FORCE_MANUAL_TOOLS,
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


# ═══════════════════════════════════════════════════════════════════════
# Nemotron / Dicta detection
# ═══════════════════════════════════════════════════════════════════════

_NEMOTRON_KEYWORDS: tuple[str, ...] = ("dicta", "nemotron")


def is_nemotron_model(model_name: str) -> bool:
    """
    Return True for models that lack native Ollama tool-calling support
    and require the manual JSON protocol (Nemotron / Dicta family).
    """
    return any(kw in model_name.lower() for kw in _NEMOTRON_KEYWORDS)


def _needs_manual_tools(llm) -> bool:
    """
    Return True if the LLM requires the ManualToolAgent fallback.

    Triggers:
      - ``SENTISENSE_FORCE_MANUAL_TOOLS=true`` (e.g. vLLM without
        ``--enable-auto-tool-choice``)
      - Model name contains a Nemotron / Dicta keyword.
    """
    if FORCE_MANUAL_TOOLS:
        return True
    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", "")
    return is_nemotron_model(model_name)


# ═══════════════════════════════════════════════════════════════════════
# ManualToolAgent — fallback for Nemotron / Dicta models
# ═══════════════════════════════════════════════════════════════════════

_TOOL_CALL_RE = re.compile(
    r"```tool_call\s*\n?(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)
_FINAL_ANSWER_RE = re.compile(
    r"```final_answer\s*\n?(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)

# Appended to the existing system prompt when ManualToolAgent is used.
_MANUAL_PROTOCOL_TEMPLATE = """

## ⚠️ MANUAL TOOL CALLING PROTOCOL (NO NATIVE FUNCTION CALLING)

You do NOT have native tool-calling capability in this environment.
Use the following text-based protocol instead.

### Step 1 — Call a tool
To call a tool, output exactly ONE ```tool_call block:
```tool_call
{{"tool": "<tool_name>", "args": {{"text": "<the Hebrew headline>"}}}}
```
The system will execute the tool and return the result. You may call
tools multiple times sequentially. You MUST call at least one tool
before giving your final answer.

### Step 2 — Give your final answer
After gathering evidence from your tools, output a ```final_answer block:
```final_answer
{{"score": <integer>, "chain_of_thought": "<full reasoning referencing tool results>"}}
```

### Rules
- Do NOT mix a tool_call and a final_answer in the same response.
- Your chain_of_thought MUST reference specific findings from tools.
- Do not guess — let the tool results guide your score.

### Available tools
{tool_schemas}
"""


def _build_tool_schemas_text(tools: list) -> str:
    """Render a readable list of tool names, params, and descriptions."""
    lines = []
    for t in tools:
        try:
            props = t.args_schema.schema().get("properties", {})
            params = ", ".join(
                f"{k}: {v.get('type', 'string')}" for k, v in props.items()
            )
        except Exception:
            params = "text: string"
        # Only show the first sentence of the docstring to keep the prompt lean
        short_desc = t.description.split("\n")[0].strip()
        lines.append(f"- **{t.name}**({params}): {short_desc}")
    return "\n".join(lines)


def _invoke_tool_sync(tool_fn, raw_args: dict) -> str:
    """
    Invoke a (sync) LangChain tool, normalising args to match the
    tool's expected parameter names.  Since all SentiSense tools accept
    a single ``text: str`` argument, any single-value dict is remapped
    automatically.
    """
    try:
        expected = list(tool_fn.args_schema.schema().get("properties", {}).keys())
    except Exception:
        expected = ["text"]

    if len(expected) == 1 and expected[0] not in raw_args and raw_args:
        raw_args = {expected[0]: next(iter(raw_args.values()))}

    return str(tool_fn.invoke(raw_args))


class ManualToolAgent:
    """
    Drop-in replacement for ``create_react_agent`` for Nemotron / Dicta
    models that do not support native Ollama tool calling.

    The agent implements the ReAct loop manually:
      1. Augments the system prompt with tool schemas + a JSON protocol.
      2. Parses ``tool_call`` JSON blocks from raw LLM text.
      3. Executes tools locally (in a thread-pool executor) and feeds
         results back as ``HumanMessage`` replies.
      4. Extracts ``score`` + ``chain_of_thought`` from a ``final_answer``
         JSON block and validates them against the target Pydantic model.

    Interface: identical to ``create_react_agent`` —
    ``ainvoke({"messages": [...]})`` → ``{"structured_response": <model>}``
    """

    MAX_STEPS: int = 12  # max tool round-trips before raising

    def __init__(
        self,
        llm,
        tools: list,
        system_prompt: str,
        response_model: type,
        name: str,
    ) -> None:
        self.llm = llm
        self._tools: dict[str, Any] = {t.name: t for t in tools}
        self._response_model = response_model
        self.name = name

        # Build the augmented prompt once — reused on every ainvoke call
        self._system_prompt = system_prompt + _MANUAL_PROTOCOL_TEMPLATE.format(
            tool_schemas=_build_tool_schemas_text(tools)
        )

    async def ainvoke(self, input_dict: dict, **_kwargs) -> dict:
        """
        Run the manual ReAct loop.

        Parameters
        ----------
        input_dict : dict
            Must contain ``"messages"`` — a list with at least one
            ``HumanMessage`` carrying the Hebrew headline.

        Returns
        -------
        dict
            ``{"structured_response": RelevancyOutput | SentimentOutput}``
        """
        messages: list = [
            SystemMessage(content=self._system_prompt),
            *input_dict.get("messages", []),
        ]

        loop = asyncio.get_running_loop()

        for step in range(self.MAX_STEPS):
            response = await self.llm.ainvoke(messages)
            text: str = response.content
            messages.append(AIMessage(content=text))

            # ── Priority 1: final_answer block ───────────────────────────
            final_match = _FINAL_ANSWER_RE.search(text)
            if final_match:
                try:
                    data = json.loads(final_match.group(1))
                    structured = self._response_model(
                        score=data["score"],
                        chain_of_thought=str(data.get("chain_of_thought", "")),
                    )
                    logger.debug(
                        "[ManualAgent:{}] final_answer at step {}: score={}",
                        self.name, step, structured.score,
                    )
                    return {"structured_response": structured}
                except Exception as exc:
                    logger.warning(
                        "[ManualAgent:{}] malformed final_answer JSON at step {}: {}",
                        self.name, step, exc,
                    )
                    messages.append(HumanMessage(
                        content=f"Your final_answer block had a JSON error: {exc}. "
                        "Please output a corrected ```final_answer block."
                    ))
                    continue

            # ── Priority 2: tool_call block ───────────────────────────────
            tool_match = _TOOL_CALL_RE.search(text)
            if tool_match:
                try:
                    call_data = json.loads(tool_match.group(1))
                    tool_name: str = call_data.get("tool", "")
                    tool_args: dict = call_data.get("args", {})
                except json.JSONDecodeError as exc:
                    messages.append(HumanMessage(
                        content=f"JSON parse error in your tool_call block: {exc}. "
                        "Please fix and retry."
                    ))
                    continue

                tool_fn = self._tools.get(tool_name)
                if tool_fn is None:
                    messages.append(HumanMessage(
                        content=f'Unknown tool "{tool_name}". '
                        f'Available: {", ".join(self._tools)}. Try again.'
                    ))
                    continue

                try:
                    result = await loop.run_in_executor(
                        None, _invoke_tool_sync, tool_fn, tool_args
                    )
                    logger.debug(
                        "[ManualAgent:{}] tool '{}' → {!r:.120}",
                        self.name, tool_name, result,
                    )
                except Exception as exc:
                    result = f"Tool execution error: {exc}"
                    logger.warning(
                        "[ManualAgent:{}] tool '{}' failed: {}",
                        self.name, tool_name, exc,
                    )

                messages.append(HumanMessage(
                    content=(
                        f"**Tool result for `{tool_name}`:**\n{result}\n\n"
                        "Now call another tool or provide your ```final_answer block."
                    )
                ))
                continue

            # ── No structured block found — nudge the model ──────────────
            logger.debug(
                "[ManualAgent:{}] step {}: no block found, nudging", self.name, step
            )
            messages.append(HumanMessage(
                content=(
                    "You must output either a ```tool_call block to invoke a tool, "
                    "or a ```final_answer block with your score and reasoning. "
                    "Plain text is not accepted — use the structured JSON blocks."
                )
            ))

        raise ValueError(
            f"ManualToolAgent '{self.name}' did not produce a final_answer "
            f"after {self.MAX_STEPS} steps."
        )


# ═══════════════════════════════════════════════════════════════════════
# Agent builders
# ═══════════════════════════════════════════════════════════════════════


def build_relevancy_agent(category: str, llm=None):
    """
    Build a ReAct agent for a single relevancy category.

    Uses ``ManualToolAgent`` automatically for Nemotron / Dicta models;
    falls back to ``create_react_agent`` for all other models.

    Parameters
    ----------
    category : str
        One of the 6 category slugs (e.g. ``"economy_finance"``).
    llm : BaseChatModel, optional
        Override the default ChatOllama instance.

    Returns
    -------
    ManualToolAgent | CompiledGraph
        An agent that accepts ``{"messages": [HumanMessage(...)]}`` and
        returns ``{"structured_response": RelevancyOutput}``.
    """
    llm = llm or build_llm()
    tools = SHARED_TOOLS + TOOLS_BY_CATEGORY[category]
    prompt = build_relevancy_system_prompt(category)

    if _needs_manual_tools(llm):
        model_name = getattr(llm, "model", "unknown")
        reason = "FORCE_MANUAL_TOOLS" if FORCE_MANUAL_TOOLS else "Nemotron/Dicta"
        logger.info(
            "Model '{}' ({}) — using ManualToolAgent for category: {}",
            model_name, reason, category,
        )
        return ManualToolAgent(
            llm=llm,
            tools=tools,
            system_prompt=prompt,
            response_model=RelevancyOutput,
            name=f"relevancy_{category}_agent",
        )

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

    Uses ``ManualToolAgent`` automatically for Nemotron / Dicta models;
    falls back to ``create_react_agent`` for all other models.

    Parameters
    ----------
    llm : BaseChatModel, optional
        Override the default ChatOllama instance.

    Returns
    -------
    ManualToolAgent | CompiledGraph
        An agent that accepts ``{"messages": [HumanMessage(...)]}`` and
        returns ``{"structured_response": SentimentOutput}``.
    """
    llm = llm or build_llm()
    tools = SHARED_TOOLS + SENTIMENT_TOOLS
    prompt = build_sentiment_system_prompt()

    if _needs_manual_tools(llm):
        model_name = getattr(llm, "model", "unknown")
        reason = "FORCE_MANUAL_TOOLS" if FORCE_MANUAL_TOOLS else "Nemotron/Dicta"
        logger.info(
            "Model '{}' ({}) — using ManualToolAgent for sentiment",
            model_name, reason,
        )
        return ManualToolAgent(
            llm=llm,
            tools=tools,
            system_prompt=prompt,
            response_model=SentimentOutput,
            name="sentiment_agent",
        )

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
        Values: ``ManualToolAgent`` or compiled ReAct agent graphs,
        depending on the active model.
    """
    llm = llm or build_llm()
    agents: dict = {}

    for cat in RELEVANCY_CATEGORIES:
        agents[f"relevancy_{cat}"] = build_relevancy_agent(cat, llm=llm)

    agents["sentiment"] = build_sentiment_agent(llm=llm)

    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", "unknown")
    agent_type = "ManualToolAgent" if _needs_manual_tools(llm) else "create_react_agent"
    logger.info(
        "All 7 agents built ({}) for model '{}'", agent_type, model_name
    )
    return agents
