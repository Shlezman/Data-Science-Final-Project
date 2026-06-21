"""Causal per-day MiroFish runner + idempotent cache.

For decision day T: seed = headlines strictly ``> T-lookback`` and ``<= T`` (no future),
run the MiroFish pipeline, extract the numeric feature + graph + report, and upsert into
narrative_sim / _graph / _report. Re-runs skip already-cached (date, seed, llm, seed_idx).
Multi-seed for variance (mean±std consumed downstream).
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
from loguru import logger
from sqlalchemy import text

from sentisense.db import get_engine
from sentisense.sim.config import DEFAULT_QUESTION, LLM_MODEL, SEED_LOOKBACK_DAYS
from sentisense.sim.extract import sections_to_markdown, votes_to_features
from sentisense.sim.graph import normalize_graph

SEED_CAP = 250   # cap headlines per seed (bound MiroFish graph-build cost)

_SEED_SQL = text("""
    SELECT date::date AS date, source, hour, headline
    FROM raw_headlines
    WHERE date > :lo AND date <= :hi AND headline IS NOT NULL AND headline <> ''
    ORDER BY date, hour
    LIMIT :cap
""")
_HAS_SQL = text("SELECT 1 FROM narrative_sim WHERE sim_date=:d AND seed_hash=:h "
                "AND llm_model=:m AND seed_idx=:i")
_INS_SIM = text("""
    INSERT INTO narrative_sim (sim_date, seed_hash, llm_model, seed_idx, dir_score, confidence,
                               disagreement, magnitude, dominant_narrative, n_agents, n_steps, simulation_id)
    VALUES (:sim_date, :seed_hash, :llm_model, :seed_idx, :dir_score, :confidence,
            :disagreement, :magnitude, :dominant_narrative, :n_agents, :n_steps, :simulation_id)
    ON CONFLICT (sim_date, seed_hash, llm_model, seed_idx) DO NOTHING
""")
_INS_GRAPH = text("""
    INSERT INTO narrative_sim_graph (sim_run_id, sim_date, graph_id, nodes, edges, meta)
    VALUES (:sim_run_id, :sim_date, :graph_id, CAST(:nodes AS JSONB), CAST(:edges AS JSONB), CAST(:meta AS JSONB))
    ON CONFLICT (sim_run_id) DO NOTHING
""")
_INS_REPORT = text("""
    INSERT INTO narrative_sim_report (sim_run_id, sim_date, report_id, sections, report_md)
    VALUES (:sim_run_id, :sim_date, :report_id, CAST(:sections AS JSONB), :report_md)
    ON CONFLICT (sim_run_id) DO NOTHING
""")


def seed_window(sim_date, lookback: int = SEED_LOOKBACK_DAYS) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(lo, hi] window for the seed — hi = T (inclusive), lo = T - lookback. Strictly past."""
    hi = pd.Timestamp(sim_date).normalize()
    return hi - pd.Timedelta(days=lookback), hi


def build_sim_seed(engine, sim_date, *, lookback: int = SEED_LOOKBACK_DAYS, cap: int = SEED_CAP):
    """Build the leak-safe seed text (headlines in (T-lookback, T]) + a stable hash + count."""
    lo, hi = seed_window(sim_date, lookback)
    with engine.connect() as conn:
        df = pd.read_sql(_SEED_SQL, conn, params={"lo": lo.date(), "hi": hi.date(), "cap": cap})
    lines = [f"[{r.date} {r.source} {r.hour}] {r.headline}" for r in df.itertuples()]
    seed = f"Israeli news headlines, {lo.date()}..{hi.date()}:\n" + "\n".join(lines)
    return seed, hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16], len(df)


def run_day(client, engine, sim_date, *, seed_idx: int = 0,
            lookback: int = SEED_LOOKBACK_DAYS, llm_model: str = LLM_MODEL,
            question: str = DEFAULT_QUESTION):
    """Run one causal day-sim; cache features + graph + report. None if skipped/empty."""
    d = pd.Timestamp(sim_date).date()
    seed, seed_hash, n = build_sim_seed(engine, sim_date, lookback=lookback)
    if n == 0:
        logger.warning("no headlines ≤ {} (lookback {}d) — skip", d, lookback)
        return None
    with engine.connect() as conn:
        if conn.execute(_HAS_SQL, {"d": d, "h": seed_hash, "m": llm_model, "i": seed_idx}).first():
            logger.info("cached {} seed#{} — skip", d, seed_idx)
            return None

    art = client.run_day_sim(seed, question, name=f"ta125_{d}_{seed_idx}")
    feats = votes_to_features(art["votes"])
    g = normalize_graph(art["graph"])
    sim_id = art["simulation_id"]
    with engine.begin() as conn:
        conn.execute(_INS_SIM, {
            "sim_date": d, "seed_hash": seed_hash, "llm_model": llm_model, "seed_idx": seed_idx,
            "dir_score": feats["dir_score"], "confidence": feats["confidence"],
            "disagreement": feats["disagreement"], "magnitude": None, "dominant_narrative": None,
            "n_agents": feats["n_votes"], "n_steps": None, "simulation_id": sim_id})
        conn.execute(_INS_GRAPH, {
            "sim_run_id": sim_id, "sim_date": d, "graph_id": art["graph_id"],
            "nodes": json.dumps(g["nodes"]), "edges": json.dumps(g["edges"]),
            "meta": json.dumps(g["meta"])})
        conn.execute(_INS_REPORT, {
            "sim_run_id": sim_id, "sim_date": d, "report_id": art["report_id"],
            "sections": json.dumps(art["sections"]), "report_md": sections_to_markdown(art["sections"])})
    logger.info("sim cached {} seed#{}: dir={:.3f} conf={:.3f} (n={})",
                d, seed_idx, feats["dir_score"], feats["confidence"], feats["n_votes"])
    return feats


def run_window(dates, *, seeds: int = 1, lookback: int = SEED_LOOKBACK_DAYS,
               engine=None, base_url: str | None = None) -> int:
    """Run causal day-sims over ``dates`` × ``seeds``; idempotent. Returns sims written."""
    from sentisense.sim import MiroClient
    engine = engine or get_engine()
    client = MiroClient(base_url) if base_url else MiroClient()
    written = 0
    for d in dates:
        for s in range(seeds):
            try:
                if run_day(client, engine, d, seed_idx=s, lookback=lookback) is not None:
                    written += 1
            except Exception as exc:  # noqa: BLE001 — one bad day shouldn't sink the window
                logger.error("sim {} seed#{} failed: {}", pd.Timestamp(d).date(), s, str(exc)[:160])
    logger.info("window complete: {} new sims cached", written)
    return written
