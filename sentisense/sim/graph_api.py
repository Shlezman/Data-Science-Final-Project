"""Read cached MiroFish graphs/reports for the UI + explainability (no MiroFish import).

These functions serve the persisted agent graph (the future UI's data source) and the
qualitative reports straight from Postgres — independent of whether the MiroFish service
is up. The graph JSON follows the contract in docs/miro/UI_GRAPH_CONTRACT.md.
"""

from __future__ import annotations

import json

from sqlalchemy import text

from sentisense.db import get_engine

_BY_DATE = text("""
    SELECT sim_run_id, sim_date, graph_id, nodes, edges, meta
    FROM narrative_sim_graph WHERE sim_date = :d AND mode = :mode ORDER BY created_at DESC LIMIT 1
""")
_LATEST = text("""
    SELECT sim_run_id, sim_date, graph_id, nodes, edges, meta
    FROM narrative_sim_graph WHERE mode = :mode ORDER BY sim_date DESC, created_at DESC LIMIT 1
""")
_REPORT = text("""
    SELECT report_id, report_md, sections FROM narrative_sim_report
    WHERE sim_date = :d AND mode = :mode ORDER BY created_at DESC LIMIT 1
""")


def _j(v):
    """JSONB comes back parsed on psycopg3, or as a str on some drivers — handle both."""
    return v if not isinstance(v, str) else json.loads(v)


def _to_graph(row) -> dict:
    return {"sim_run_id": row.sim_run_id, "date": str(row.sim_date), "graph_id": row.graph_id,
            "nodes": _j(row.nodes) or [], "edges": _j(row.edges) or [], "meta": _j(row.meta) or {}}


def graph_for_date(date, engine=None, *, mode: str = "source") -> dict | None:
    """Agent graph ({nodes,edges,meta}) for a date/mode, or None. 'source' = the rich per-outlet graph."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        row = conn.execute(_BY_DATE, {"d": str(date), "mode": mode}).first()
    return _to_graph(row) if row else None


def latest_graph(engine=None, *, mode: str = "source") -> dict | None:
    """Most recent cached agent graph for ``mode`` (for the live UI panel)."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        row = conn.execute(_LATEST, {"mode": mode}).first()
    return _to_graph(row) if row else None


def report_for_date(date, engine=None, *, mode: str = "source") -> dict | None:
    """Qualitative report (markdown + sections) for a date/mode, or None."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        row = conn.execute(_REPORT, {"d": str(date), "mode": mode}).first()
    if not row:
        return None
    return {"report_id": row.report_id, "report_md": row.report_md, "sections": _j(row.sections)}
