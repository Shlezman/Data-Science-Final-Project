"""
SentiSense Processing Engine
=============================
AI pipeline for Hebrew news headline analysis.

Two execution modes:

* **Fast pipeline** (production, ``fast_pipeline.py``) — single-prompt
  scoring that produces all 7 scores in one structured LLM call.
  Compatible with both ``/v1/chat/completions`` and the raw
  ``/v1/completions`` endpoint (vLLM Mistral-Small-4).
  Preferred for batch scoring and the default for the batch CLI
  (``scripts/process_headlines.py --fast``).

* **Multi-agent pipeline** (Ollama-only, ``graph.py``) — 7 autonomous
  ReAct agents (6 relevancy + 1 sentiment) running in parallel via
  LangGraph.  Requires a backend that supports native tool-calling
  (``bind_tools``); incompatible with ``FORCE_COMPLETIONS_API=true``.

Usage — multi-agent pipeline::

    from processing_engine import process_single_observation

    result = await process_single_observation({
        "date": "2025-01-15",
        "source": "כאן חדשות",
        "hour": "14:30",
        "popularity": "important",
        "headline": "בנק ישראל הכריז על העלאת הריבית ב-0.25%",
    })

Usage — fast pipeline (batch)::

    from processing_engine.fast_pipeline import score_headlines_batch
    results = await score_headlines_batch(obs_list, batch_size=50)
"""

from .engine import process_single_observation, reset_graph

__all__ = ["process_single_observation", "reset_graph"]
