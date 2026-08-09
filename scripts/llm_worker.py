"""LLM worker — answers queued UI questions with the local Ollama model.

The UI host cannot reach Ollama directly (the firewall only passes Postgres between the
machines), so the database is the transport: the UI inserts rows into ``llm_requests``
(migration 008), this worker polls them on the GPU box, builds a prompt from the day's
scored headlines, calls the local Ollama HTTP API, and writes the answer back.

Request kinds:
  * ``narrate``  — summarize the day's news narratives and give an UP/DOWN lean for the
    next TA-125 session, with rationale.
  * ``ask``      — answer a free-form question grounded in the day's headlines.
  * ``simulate`` — generate a persona simulation for the day (each news outlet = one
    persona) directly into the ``narrative_sim_graph`` / ``narrative_sim_report`` tables
    the Simulator tab renders — a local-LLM stand-in for the MiroFish service, which
    cannot run on this box (its Zep dependency needs Docker). The agent graph is built
    deterministically from per-source stats; only the report text comes from the LLM.

Run (GPU box, needs SENTISENSE_DATABASE_URL + Ollama on localhost):
    uv run python scripts/llm_worker.py                 # poll loop (2s)
    uv run python scripts/llm_worker.py --once          # drain queue and exit
    SENTISENSE_OLLAMA_MODEL=gemma4:latest uv run python scripts/llm_worker.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

from loguru import logger
from sqlalchemy import text

from sentisense.constants import REPO_ROOT
from sentisense.db import get_engine

_MODEL = os.environ.get("SENTISENSE_OLLAMA_MODEL", "gemma4:latest")
_OLLAMA = os.environ.get("SENTISENSE_OLLAMA_URL", "http://localhost:11434")
_MIGRATION = REPO_ROOT / "sentisense" / "db" / "migrations" / "008_llm_requests.sql"
_MAX_HEADLINES = 40
_POLL_SECONDS = 2.0

_CLAIM = text(
    """
    UPDATE llm_requests SET status = 'working'
    WHERE id = (SELECT id FROM llm_requests WHERE status = 'pending'
                ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED)
    RETURNING id, kind, date, question
    """
)
_FINISH = text(
    "UPDATE llm_requests SET status = :s, answer = :a, answered_at = NOW() WHERE id = :i"
)
_DAY_HEADLINES = text(
    """
    SELECT rh.hour, rh.source, rh.headline, nv.global_sentiment
    FROM raw_headlines rh
    LEFT JOIN LATERAL (
        SELECT v.global_sentiment FROM nlp_vectors v
        WHERE v.headline_id = rh.id AND v.validation_passed
        ORDER BY v.id DESC LIMIT 1
    ) nv ON TRUE
    WHERE rh.date = :d
    ORDER BY rh.hour DESC NULLS LAST
    LIMIT :cap
    """
)


def ensure_table(engine) -> None:
    """Apply migration 008 (idempotent)."""
    ddl = re.sub(r"--[^\n]*", "", _MIGRATION.read_text(encoding="utf-8"))
    with engine.begin() as conn:
        for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
            conn.execute(text(stmt))


def _ollama_generate(prompt: str, timeout: int = 240) -> str:
    """One non-streaming completion from the local Ollama server (stdlib only)."""
    req = urllib.request.Request(
        f"{_OLLAMA}/api/generate",
        data=json.dumps({"model": _MODEL, "prompt": prompt, "stream": False}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["response"].strip()


def _day_context(engine, day) -> str:
    """The day's headlines (Hebrew) with sentiment, formatted for the prompt."""
    with engine.connect() as conn:
        rows = conn.execute(_DAY_HEADLINES, {"d": day, "cap": _MAX_HEADLINES}).all()
    if not rows:
        return ""
    lines = []
    for hour, source, headline, sent in rows:
        tag = f" [sentiment {sent:+d}]" if isinstance(sent, int) else ""
        lines.append(f"- ({hour} · {source}) {headline}{tag}")
    return "\n".join(lines)


def _build_prompt(kind: str, day, question: str | None, context: str) -> str:
    """Grounded analyst prompt; answers in English, headlines stay Hebrew."""
    base = (
        "You are a financial news analyst for the Israeli market (TA-125 index). "
        f"Below are Hebrew breaking-news headlines from {day} with LLM sentiment scores "
        "(-10 very negative .. +10 very positive).\n\n"
        f"HEADLINES:\n{context or '(no headlines stored for this day)'}\n\n"
    )
    if kind == "narrate":
        return base + (
            "Task: In English, summarize the 2-4 dominant news narratives of the day, then state "
            "whether the overall news flow leans UP or DOWN for the next TA-125 session and why. "
            "Be concise (<= 250 words). End with one line: 'Lean: UP' or 'Lean: DOWN' or 'Lean: NEUTRAL'."
        )
    return base + (
        "Task: Answer the user's question in English, grounded ONLY in the headlines above; "
        "say plainly when they don't contain the answer. Be concise.\n\n"
        f"QUESTION: {question}"
    )


_PERSONA_STATS = text(
    """
    SELECT rh.source AS source, COUNT(*) AS n,
           AVG(nv.global_sentiment)::float AS mean_sentiment
    FROM raw_headlines rh
    JOIN LATERAL (
        SELECT v.global_sentiment FROM nlp_vectors v
        WHERE v.headline_id = rh.id AND v.validation_passed
          AND v.global_sentiment IS NOT NULL
        ORDER BY v.id DESC LIMIT 1
    ) nv ON TRUE
    WHERE rh.date = :d
    GROUP BY rh.source HAVING COUNT(*) >= 3
    ORDER BY COUNT(*) DESC LIMIT 12
    """
)
_SIM_GRAPH_UPSERT = text(
    """
    INSERT INTO narrative_sim_graph (sim_run_id, sim_date, graph_id, nodes, edges, meta, mode)
    VALUES (:rid, :d, :rid, :nodes, :edges, :meta, :mode)
    ON CONFLICT (sim_run_id) DO UPDATE
        SET nodes = EXCLUDED.nodes, edges = EXCLUDED.edges, meta = EXCLUDED.meta,
            created_at = NOW()
    """
)
_SIM_REPORT_UPSERT = text(
    """
    INSERT INTO narrative_sim_report (sim_run_id, report_id, sim_date, report_md, sections, mode)
    VALUES (:rid, :rid, :d, :md, :sections, :mode)
    ON CONFLICT (sim_run_id) DO UPDATE
        SET report_md = EXCLUDED.report_md, sections = EXCLUDED.sections, created_at = NOW()
    """
)


def _stance(mean_sentiment: float) -> str:
    if mean_sentiment >= 0.5:
        return "bullish"
    if mean_sentiment <= -0.5:
        return "bearish"
    return "neutral"


def _simulate_day(engine, day) -> str:
    """Persona simulation for one day → narrative_sim_graph + narrative_sim_report.

    Nodes = news outlets active that day (attrs carry volume/sentiment/stance); edges connect
    outlets with aligned or opposed stances (deterministic — the graph never depends on the
    LLM emitting valid JSON). The LLM writes only the markdown report: a simulated roundtable
    of those personas plus a consensus lean.
    """
    with engine.connect() as conn:
        stats = conn.execute(_PERSONA_STATS, {"d": day}).all()
    if not stats:
        raise RuntimeError(f"no scored headlines for {day} — cannot simulate")

    nodes, edges = [], []
    for source, n, mean_sent in stats:
        nodes.append({"id": source, "type": "agent", "label": source,
                      "attrs": {"headlines": int(n), "mean_sentiment": round(float(mean_sent), 2),
                                "stance": _stance(float(mean_sent))}})
    for i in range(len(stats)):
        for j in range(i + 1, len(stats)):
            si, sj = float(stats[i][2]), float(stats[j][2])
            if abs(si) < 0.5 or abs(sj) < 0.5:
                continue                                   # neutral personas don't argue
            kind = "agrees" if (si > 0) == (sj > 0) else "disagrees"
            edges.append({"src": stats[i][0], "dst": stats[j][0], "type": kind,
                          "weight": round(min(abs(si), abs(sj)) / 10, 3)})

    persona_lines = "\n".join(
        f"- {s} ({n} headlines, mean sentiment {m:+.2f}, {_stance(float(m))})"
        for s, n, m in stats)
    context = _day_context(engine, day)
    prompt = (
        "You are simulating a roundtable of Israeli news outlets, each an opinionated persona, "
        f"debating the market impact of the news of {day} on the TA-125 index.\n\n"
        f"PERSONAS (real per-outlet stats for the day):\n{persona_lines}\n\n"
        f"HEADLINES (Hebrew, with sentiment):\n{context}\n\n"
        "Write in English, markdown, <= 400 words, with EXACTLY these sections:\n"
        "## Dominant narratives\n## Persona positions\n(2-6 personas, one line each, "
        "consistent with their stats)\n## Consensus\n"
        "End with one line: 'Lean: UP' or 'Lean: DOWN' or 'Lean: NEUTRAL'."
    )
    report_md = _ollama_generate(prompt)
    lean = "NEUTRAL"
    m = re.search(r"Lean:\s*(UP|DOWN|NEUTRAL)", report_md, re.IGNORECASE)
    if m:
        lean = m.group(1).upper()

    rid = f"llmsim-{day}-source"
    meta = {"generator": f"local-llm:{_MODEL}", "n_agents": len(nodes), "lean": lean,
            "question": f"How does the news of {day} move the TA-125?"}
    with engine.begin() as conn:
        conn.execute(_SIM_GRAPH_UPSERT, {"rid": rid, "d": day, "nodes": json.dumps(nodes),
                                         "edges": json.dumps(edges), "meta": json.dumps(meta),
                                         "mode": "source"})
        conn.execute(_SIM_REPORT_UPSERT, {"rid": rid, "d": day, "md": report_md,
                                          "sections": json.dumps({"lean": lean}),
                                          "mode": "source"})
    return f"simulation ready for {day}: {len(nodes)} personas, {len(edges)} edges, lean {lean}"


def handle_one(engine) -> bool:
    """Claim and answer one pending request. Returns False when the queue is empty."""
    with engine.begin() as conn:
        row = conn.execute(_CLAIM).first()
    if not row:
        return False
    rid, kind, day, question = int(row[0]), row[1], row[2], row[3]
    logger.info("Request {}: kind={} date={} q={}", rid, kind, day, (question or "")[:60])
    try:
        if kind == "simulate":
            answer = _simulate_day(engine, day)
        else:
            context = _day_context(engine, day) if day else ""
            answer = _ollama_generate(_build_prompt(kind, day, question, context))
        status = "done"
    except Exception as exc:  # noqa: BLE001 — record the failure; never crash the loop
        answer, status = f"worker error: {str(exc)[:300]}", "error"
        logger.warning("Request {} failed: {}", rid, str(exc)[:200])
    with engine.begin() as conn:
        conn.execute(_FINISH, {"s": status, "a": answer, "i": rid})
    logger.info("Request {} -> {} ({} chars)", rid, status, len(answer))
    return True


def main() -> int:
    """Poll the queue forever (or drain once with --once)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--once", action="store_true", help="Drain the queue and exit.")
    args = ap.parse_args()

    engine = get_engine()
    ensure_table(engine)
    with engine.begin() as conn:   # reclaim rows orphaned by a previous crash
        conn.execute(text("UPDATE llm_requests SET status = 'pending' WHERE status = 'working'"))
    logger.info("LLM worker up — model={} ollama={} poll={}s", _MODEL, _OLLAMA, _POLL_SECONDS)
    while True:
        worked = True
        while worked:
            worked = handle_one(engine)
        if args.once:
            return 0
        time.sleep(_POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
