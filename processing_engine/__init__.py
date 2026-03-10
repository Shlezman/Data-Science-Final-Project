"""
SentiSense Processing Engine
=============================
Multi-Agent AI pipeline for Hebrew news headline analysis.

Each headline is processed by 7 autonomous ReAct agents (6 relevancy
+ 1 sentiment) running in parallel via LangGraph, with local Ollama
inference.

Usage::

    from processing_engine import process_single_observation

    result = await process_single_observation({
        "date": "2025-01-15",
        "source": "כאן חדשות",
        "hour": "14:30",
        "popularity": "important",
        "headline": "בנק ישראל הכריז על העלאת הריבית ב-0.25%",
    })
"""

from .engine import process_single_observation

__all__ = ["process_single_observation"]
