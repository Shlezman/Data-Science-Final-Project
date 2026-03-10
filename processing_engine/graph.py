"""
processing_engine.graph
=======================
LangGraph ``StateGraph`` construction and compilation.

Topology
--------
::

                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                    ┌──────▼──────┐
                    │  ingestion  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐   Fan-out (7 parallel)
              │            │            │
    ┌─────────▼──┐  ┌──────▼──┐  ┌──────▼──────┐
    │ relevancy  │  │ relev.  │  │  sentiment  │
    │ politics   │  │ economy │  │    agent    │
    │   agent    │  │  agent  │  │             │
    └─────────┬──┘  └──────┬──┘  └──────┬──────┘
              │            │            │
              └────────────┼────────────┘   Fan-in
                    ┌──────▼──────┐
                    │ validation  │
                    └──────┬──────┘
                    ┌──────▼──────┐
                    │ aggregation │
                    └──────┬──────┘
                    ┌──────▼──────┐
                    │     END     │
                    └─────────────┘

Uses LangGraph's ``START`` / ``END`` constants and static fan-out
via multiple ``add_edge`` calls from ``ingestion`` to all 7 agent
nodes.  The ``Annotated[list, operator.add]`` reducer on the
``errors`` field ensures safe parallel merging.
"""

from __future__ import annotations

from loguru import logger
from langgraph.graph import END, START, StateGraph

from .agents import build_all_agents
from .config import CATEGORY_DISPLAY_NAMES, RELEVANCY_CATEGORIES
from .models import PipelineState
from .nodes import (
    aggregation_node,
    ingestion_node,
    make_agent_node,
    validation_node,
)


def build_graph() -> StateGraph:
    """
    Construct and compile the processing pipeline graph.

    Builds all 7 ReAct agents, wraps each in a parent-graph node
    via ``make_agent_node``, and wires the fan-out / fan-in topology.

    Returns
    -------
    CompiledStateGraph
        A compiled async-capable graph that accepts a
        ``PipelineState`` dict and returns the final state.
    """
    logger.info("Building LangGraph pipeline…")

    # ── Build all ReAct sub-agents (shared LLM) ────────────────────
    agents = build_all_agents()

    # ── Construct the parent StateGraph ────────────────────────────
    graph = StateGraph(PipelineState)

    # Ingestion node
    graph.add_node("ingestion", ingestion_node)

    # 6 relevancy agent wrapper nodes
    fan_out_targets: list[str] = []
    for cat in RELEVANCY_CATEGORIES:
        node_name = f"relevancy_{cat}"
        display = CATEGORY_DISPLAY_NAMES[cat]
        wrapper = make_agent_node(
            agent=agents[node_name],
            state_key=node_name,
            display_name=display,
        )
        graph.add_node(node_name, wrapper)
        fan_out_targets.append(node_name)

    # Sentiment agent wrapper node
    sentiment_wrapper = make_agent_node(
        agent=agents["sentiment"],
        state_key="sentiment",
        display_name="Sentiment",
    )
    graph.add_node("sentiment", sentiment_wrapper)
    fan_out_targets.append("sentiment")

    # Post-processing nodes
    graph.add_node("validation", validation_node)
    graph.add_node("aggregation", aggregation_node)

    # ── Edges ──────────────────────────────────────────────────────

    # START → ingestion
    graph.add_edge(START, "ingestion")

    # Fan-out: ingestion → all 7 agent nodes (run in parallel)
    for target in fan_out_targets:
        graph.add_edge("ingestion", target)

    # Fan-in: all 7 agents → validation (waits for all to complete)
    for target in fan_out_targets:
        graph.add_edge(target, "validation")

    # validation → aggregation → END
    graph.add_edge("validation", "aggregation")
    graph.add_edge("aggregation", END)

    logger.info(
        "Graph built: {} nodes, {} fan-out branches",
        len(fan_out_targets) + 3,  # agents + ingestion + validation + aggregation
        len(fan_out_targets),
    )

    return graph.compile()
