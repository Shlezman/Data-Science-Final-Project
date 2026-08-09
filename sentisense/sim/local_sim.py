"""Local-LLM persona simulation — the nightly stand-in for the MiroFish service.

Each news outlet active on a day becomes an agent persona. The agent graph (stances,
volumes, agree/disagree edges) is built deterministically from the day's scored headlines,
so it never depends on the LLM emitting valid structure; the LLM (local Ollama) contributes
the human layer: one in-character statement per persona (the "conversation") and a markdown
roundtable report with a consensus lean. Results are upserted into ``narrative_sim_graph`` /
``narrative_sim_report`` — the exact tables the Simulator tab renders.

Used by ``scripts/sim_daily.py`` (nightly cron) and ``scripts/llm_worker.py`` (queued runs).
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

from loguru import logger
from sqlalchemy import text

OLLAMA_MODEL = os.environ.get("SENTISENSE_OLLAMA_MODEL", "gemma4:latest")
OLLAMA_URL = os.environ.get("SENTISENSE_OLLAMA_URL", "http://localhost:11434")

_MAX_HEADLINES = 40
_MAX_STATEMENTS = 8

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
_GRAPH_UPSERT = text(
    """
    INSERT INTO narrative_sim_graph (sim_run_id, sim_date, graph_id, nodes, edges, meta, mode)
    VALUES (:rid, :d, :rid, :nodes, :edges, :meta, :mode)
    ON CONFLICT (sim_run_id) DO UPDATE
        SET nodes = EXCLUDED.nodes, edges = EXCLUDED.edges, meta = EXCLUDED.meta,
            created_at = NOW()
    """
)
_REPORT_UPSERT = text(
    """
    INSERT INTO narrative_sim_report (sim_run_id, report_id, sim_date, report_md, sections, mode)
    VALUES (:rid, :rid, :d, :md, :sections, :mode)
    ON CONFLICT (sim_run_id) DO UPDATE
        SET report_md = EXCLUDED.report_md, sections = EXCLUDED.sections, created_at = NOW()
    """
)


def ollama_generate(prompt: str, timeout: int = 240) -> str:
    """One non-streaming completion from the local Ollama server (stdlib only)."""
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["response"].strip()


def day_context(engine, day) -> str:
    """The day's headlines (Hebrew) with sentiment, formatted for prompts."""
    with engine.connect() as conn:
        rows = conn.execute(_DAY_HEADLINES, {"d": day, "cap": _MAX_HEADLINES}).all()
    lines = []
    for hour, source, headline, sent in rows:
        tag = f" [sentiment {sent:+d}]" if isinstance(sent, int) else ""
        lines.append(f"- ({hour} · {source}) {headline}{tag}")
    return "\n".join(lines)


def _stance(mean_sentiment: float) -> str:
    if mean_sentiment >= 0.5:
        return "bullish"
    if mean_sentiment <= -0.5:
        return "bearish"
    return "neutral"


def _persona_statements(stats, context: str) -> dict:
    """One in-character line per persona, via ONE strict-JSON LLM call.

    Lenient parse: grab the outermost JSON array; on any failure return {} —
    the map still renders, just without quotes.
    """
    top = stats[:_MAX_STATEMENTS]
    persona_lines = "\n".join(
        f"- {s} (mean sentiment {m:+.2f}, {_stance(float(m))})" for s, _n, m in top)
    prompt = (
        "You are scripting a panel of Israeli news outlets debating today's market impact "
        "on the TA-125 index.\n\n"
        f"PANELISTS (with their real measured stance):\n{persona_lines}\n\n"
        f"TODAY'S HEADLINES (Hebrew, with sentiment):\n{context}\n\n"
        "For EACH panelist write ONE short in-character statement in English (max 25 words) "
        "consistent with its stance and grounded in the headlines.\n"
        'Answer with ONLY a JSON array, no prose: '
        '[{"source": "<exact panelist name>", "statement": "<one line>"}]'
    )
    try:
        raw = ollama_generate(prompt)
        start, end = raw.find("["), raw.rfind("]")
        items = json.loads(raw[start:end + 1])
        return {str(it["source"]): str(it["statement"])[:300]
                for it in items if isinstance(it, dict) and it.get("source") and it.get("statement")}
    except Exception as exc:  # noqa: BLE001 — statements are decoration, never fatal
        logger.warning("persona statements unparseable ({}) — skipping quotes", str(exc)[:120])
        return {}


def simulate_day(engine, day) -> str:
    """Full persona simulation for one day → graph + report rows. Returns a summary line."""
    with engine.connect() as conn:
        stats = conn.execute(_PERSONA_STATS, {"d": day}).all()
    if not stats:
        raise RuntimeError(f"no scored headlines for {day} — cannot simulate")

    context = day_context(engine, day)
    statements = _persona_statements(stats, context)

    nodes, edges = [], []
    for source, n, mean_sent in stats:
        attrs = {"headlines": int(n), "mean_sentiment": round(float(mean_sent), 2),
                 "stance": _stance(float(mean_sent))}
        if source in statements:
            attrs["statement"] = statements[source]
        nodes.append({"id": source, "type": "agent", "label": source, "attrs": attrs})
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
    report_md = ollama_generate(
        "You are simulating a roundtable of Israeli news outlets, each an opinionated persona, "
        f"debating the market impact of the news of {day} on the TA-125 index.\n\n"
        f"PERSONAS (real per-outlet stats for the day):\n{persona_lines}\n\n"
        f"HEADLINES (Hebrew, with sentiment):\n{context}\n\n"
        "Write in English, markdown, <= 400 words, with EXACTLY these sections:\n"
        "## Dominant narratives\n## Persona positions\n(2-6 personas, one line each, "
        "consistent with their stats)\n## Consensus\n"
        "End with one line: 'Lean: UP' or 'Lean: DOWN' or 'Lean: NEUTRAL'.")

    lean = "NEUTRAL"
    m = re.search(r"Lean:\s*(UP|DOWN|NEUTRAL)", report_md, re.IGNORECASE)
    if m:
        lean = m.group(1).upper()
    consensus = ""
    cm = re.search(r"##\s*Consensus\s*\n(.+?)(?:\nLean:|\Z)", report_md, re.DOTALL | re.IGNORECASE)
    if cm:
        consensus = cm.group(1).strip()[:400]

    rid = f"llmsim-{day}-source"
    meta = {"generator": f"local-llm:{OLLAMA_MODEL}", "n_agents": len(nodes), "lean": lean,
            "consensus": consensus, "n_statements": len(statements),
            "question": f"How does the news of {day} move the TA-125?"}
    with engine.begin() as conn:
        conn.execute(_GRAPH_UPSERT, {"rid": rid, "d": day, "nodes": json.dumps(nodes),
                                     "edges": json.dumps(edges), "meta": json.dumps(meta),
                                     "mode": "source"})
        conn.execute(_REPORT_UPSERT, {"rid": rid, "d": day, "md": report_md,
                                      "sections": json.dumps({"lean": lean, "consensus": consensus}),
                                      "mode": "source"})
    return (f"simulation ready for {day}: {len(nodes)} personas "
            f"({len(statements)} with statements), {len(edges)} edges, lean {lean}")
