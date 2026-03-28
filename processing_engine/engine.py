"""
processing_engine.engine
========================
Production entrypoint for the SentiSense Processing Engine.

Exposes a single async function::

    async def process_single_observation(observation: dict) -> dict

Designed to be called from:

* A **FastAPI route**::

      @app.post("/process")
      async def process(obs: dict):
          return await process_single_observation(obs)

* A **batch loop**::

      import asyncio, pandas as pd
      from processing_engine import process_single_observation

      df = pd.read_csv("headlines.csv")
      for _, row in df.iterrows():
          result = asyncio.run(process_single_observation(row.to_dict()))

* The **CLI** via ``python -m processing_engine``.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger

from .config import AGENT_RECURSION_LIMIT, LOG_DIR, LOG_LEVEL
from .models import PipelineState

# ── Logging configuration ───────────────────────────────────────────

_LOG_FMT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "{message}"
)

logger.remove()

# Stderr sink (coloured for interactive use)
logger.add(
    sys.stderr,
    level=LOG_LEVEL,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    colorize=True,
    enqueue=True,
    backtrace=True,
    diagnose=True,
)

# File sink — rotating daily, kept for 7 days, plain text (no ANSI codes)
if LOG_DIR:
    _log_dir = Path(LOG_DIR)
    _log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        _log_dir / "sentisense_{time:YYYY-MM-DD}.log",
        level=LOG_LEVEL,
        format=_LOG_FMT,
        colorize=False,
        rotation="00:00",       # new file every midnight
        retention="7 days",     # keep the last 7 daily files
        compression="gz",       # compress old logs
        enqueue=True,
        backtrace=True,
        diagnose=True,
        encoding="utf-8",
    )

# ── Compiled graph singleton ────────────────────────────────────────

_compiled_graph = None


def _get_graph():
    """Lazy-compile the graph once and cache it."""
    global _compiled_graph
    if _compiled_graph is None:
        logger.info("Compiling LangGraph pipeline (first call)…")
        from .graph import build_graph

        _compiled_graph = build_graph()
        logger.info("Graph compiled successfully")
    return _compiled_graph


def reset_graph() -> None:
    """
    Discard the cached compiled graph so the next call to
    ``process_single_observation`` rebuilds it from scratch.

    Call this whenever you change ``SENTISENSE_OLLAMA_MODEL`` at runtime
    (e.g. between model evaluations) so the new model name is actually picked up.
    """
    global _compiled_graph
    _compiled_graph = None


# ═══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════


async def process_single_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """
    Process a single news observation through the full AI pipeline.

    Parameters
    ----------
    observation : dict
        A dictionary with at least:
          - ``headline`` (str): the Hebrew news headline
          - ``date`` (str): publication date (YYYY-MM-DD)
          - ``source`` (str): news publisher
          - ``hour`` (str): time of publication
          - ``popularity`` (str, optional): importance level

    Returns
    -------
    dict
        A flat dictionary containing the original observation metadata
        plus exactly 7 data columns — ready for direct PostgreSQL
        insertion or CSV export:

          relevance_category_1  int   Politics & Government score (0–10)
          relevance_category_2  int   Economy & Finance score     (0–10)
          relevance_category_3  int   Security & Military score   (0–10)
          relevance_category_4  int   Health & Medicine score     (0–10)
          relevance_category_5  int   Science & Climate score     (0–10)
          relevance_category_6  int   Technology score            (0–10)
          global_sentiment      int   text tone score             (-10..+10, 0=neutral)

        Plus metadata: ``validation_passed``, ``errors``,
        ``processing_time_seconds``.

    Raises
    ------
    No exceptions are raised.  If individual agents fail, their
    scores default to 0 with error details in the ``errors`` list.
    """
    t0 = time.perf_counter()
    logger.info(
        "━━━ Processing observation: {} ━━━",
        observation.get("headline", "<no headline>")[:60],
    )

    initial_state: PipelineState = {
        "date": observation.get("date", ""),
        "source": observation.get("source", ""),
        "hour": observation.get("hour", ""),
        "popularity": observation.get("popularity", ""),
        "headline": observation.get("headline", ""),
    }

    graph = _get_graph()
    final_state = await graph.ainvoke(
        initial_state,
        config={"recursion_limit": AGENT_RECURSION_LIMIT},
    )

    elapsed = time.perf_counter() - t0
    output: dict[str, Any] = final_state.get("output", {})
    output["processing_time_seconds"] = round(elapsed, 3)

    logger.info(
        "━━━ Done in {:.2f}s | validation={} | errors={} ━━━",
        elapsed,
        output.get("validation_passed"),
        len(output.get("errors", [])),
    )

    return output


# ═══════════════════════════════════════════════════════════════════════
# CLI entrypoint
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json as _json

    sample = {
        "date": "2025-01-15",
        "source": "כאן חדשות",
        "hour": "14:30",
        "popularity": "important",
        "headline": "בנק ישראל הכריז על העלאת הריבית ב-0.25% לאחר עלייה באינפלציה",
    }

    logger.info("Running smoke test with sample observation…")
    result = asyncio.run(process_single_observation(sample))
    print(_json.dumps(result, ensure_ascii=False, indent=2))
