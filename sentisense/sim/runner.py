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
from sentisense.sim.config import (
    DEFAULT_QUESTION,
    LLM_MODEL,
    SEED_FETCH_CAP,
    SEED_LOOKBACK_DAYS,
    SEED_PER_SOURCE_CAP,
    SEED_TOTAL_CAP,
    SIM_ENTITY_TYPES,
    SIM_MODES,
    SOURCE_AS_AGENTS,
)
from sentisense.sim.extract import sections_to_markdown, source_agent_coverage, votes_to_features
from sentisense.sim.graph import normalize_graph
from sentisense.sim.miro_client import MiroError

# Preamble that nudges GraphRAG to treat each outlet as a distinct voice (source-as-agent).
_SOURCE_PREAMBLE = (
    "Israeli financial/general news headlines by SOURCE, {lo}..{hi}. Each source below is a "
    "separate news outlet with its own editorial perspective and agenda — treat each as a "
    "distinct voice, not as one consensus.\n"
)
# Preamble for the provider-agnostic 'flat' mode: the day's news as one pooled feed.
_FLAT_PREAMBLE = (
    "All Israeli financial/general news for {lo}..{hi}, pooled into a single feed — source "
    "attribution removed and near-duplicate stories merged. Simulate the overall market/public "
    "crowd reaction to the day's events as a whole.\n"
)

# Fetch the NEWEST headlines in the window (recent news carries the most next-day signal);
# balancing per-source happens in pandas so a prolific outlet can't dominate the seed.
_SEED_SQL = text("""
    SELECT date::date AS date, source, hour, headline
    FROM raw_headlines
    WHERE date > :lo AND date <= :hi AND headline IS NOT NULL AND headline <> ''
    ORDER BY date DESC, hour DESC
    LIMIT :cap
""")
_HAS_SQL = text("SELECT 1 FROM narrative_sim WHERE sim_date=:d AND seed_hash=:h "
                "AND llm_model=:m AND seed_idx=:i AND mode=:mode")
_INS_SIM = text("""
    INSERT INTO narrative_sim (sim_date, seed_hash, llm_model, seed_idx, mode, dir_score, confidence,
                               disagreement, magnitude, dominant_narrative, n_agents, n_steps, simulation_id)
    VALUES (:sim_date, :seed_hash, :llm_model, :seed_idx, :mode, :dir_score, :confidence,
            :disagreement, :magnitude, :dominant_narrative, :n_agents, :n_steps, :simulation_id)
    ON CONFLICT (sim_date, seed_hash, llm_model, seed_idx) DO NOTHING
""")
_INS_GRAPH = text("""
    INSERT INTO narrative_sim_graph (sim_run_id, sim_date, mode, graph_id, nodes, edges, meta)
    VALUES (:sim_run_id, :sim_date, :mode, :graph_id, CAST(:nodes AS JSONB), CAST(:edges AS JSONB), CAST(:meta AS JSONB))
    ON CONFLICT (sim_run_id) DO NOTHING
""")
_INS_REPORT = text("""
    INSERT INTO narrative_sim_report (sim_run_id, sim_date, mode, report_id, sections, report_md)
    VALUES (:sim_run_id, :sim_date, :mode, :report_id, CAST(:sections AS JSONB), :report_md)
    ON CONFLICT (sim_run_id) DO NOTHING
""")


def seed_window(sim_date, lookback: int = SEED_LOOKBACK_DAYS) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(lo, hi] window for the seed — hi = T (inclusive), lo = T - lookback. Strictly past."""
    hi = pd.Timestamp(sim_date).normalize()
    return hi - pd.Timedelta(days=lookback), hi


def _balance_by_source(df: pd.DataFrame, *, per_source_cap: int, total_cap: int) -> pd.DataFrame:
    """Keep the newest ``per_source_cap`` rows per source, then the newest ``total_cap`` overall."""
    newest_first = df.sort_values(["date", "hour"], ascending=False)
    capped = newest_first.groupby("source", group_keys=False, sort=False).head(per_source_cap)
    return capped.head(total_cap)


def _compose_seed(df: pd.DataFrame, lo, hi, mode: str) -> str:
    """Render the (already shaped) frame as seed text for ``mode`` ('source' | 'flat')."""
    if mode == "flat":   # provider-agnostic: pooled feed, no source attribution
        lines = [f"[{r.date} {r.hour}] {r.headline}" for r in df.itertuples()]
        return _FLAT_PREAMBLE.format(lo=lo.date(), hi=hi.date()) + "\n".join(lines)

    if not SOURCE_AS_AGENTS:   # legacy flat-with-tag fallback for the 'source' mode
        lines = [f"[{r.date} {r.source} {r.hour}] {r.headline}" for r in df.itertuples()]
        return f"Israeli news headlines, {lo.date()}..{hi.date()}:\n" + "\n".join(lines)

    parts = [_SOURCE_PREAMBLE.format(lo=lo.date(), hi=hi.date())]
    by_volume = df.groupby("source", sort=False).size().sort_values(ascending=False)
    for source in by_volume.index:                       # densest outlet first
        rows = df[df["source"] == source].sort_values(["date", "hour"])
        parts.append(f"\n### Source: {source} ({len(rows)} headlines)")
        parts.extend(f"[{r.date} {r.hour}] {r.headline}" for r in rows.itertuples())
    return "\n".join(parts)


def _shape_for_mode(df: pd.DataFrame, mode: str, *, per_source_cap: int, total_cap: int) -> pd.DataFrame:
    """Cap/dedup the window per mode. 'source' balances per outlet; 'flat' pools + dedups."""
    if mode == "flat":
        deduped = df.drop_duplicates(subset="headline")   # merge cross-provider echo
        return deduped.sort_values(["date", "hour"], ascending=False).head(total_cap)
    return _balance_by_source(df, per_source_cap=per_source_cap, total_cap=total_cap)


def build_sim_seed(engine, sim_date, *, mode: str = "source", lookback: int = SEED_LOOKBACK_DAYS,
                   per_source_cap: int = SEED_PER_SOURCE_CAP, total_cap: int = SEED_TOTAL_CAP):
    """Build the leak-safe seed text (headlines in (T-lookback, T]) + a stable hash + count.

    ``mode='source'`` balances per outlet and renders one section per source (each outlet a
    distinct voice). ``mode='flat'`` pools the whole day, deduped and source-stripped. Both
    operate only on the ≤T window, so ``seed_hash`` stays a pure function of ≤T content.
    """
    lo, hi = seed_window(sim_date, lookback)
    with engine.connect() as conn:
        df = pd.read_sql(_SEED_SQL, conn, params={"lo": lo.date(), "hi": hi.date(), "cap": SEED_FETCH_CAP})
    if len(df) == SEED_FETCH_CAP:
        logger.warning("seed for {} hit FETCH_CAP={} before shaping — window is very dense", hi.date(), SEED_FETCH_CAP)
    df = _shape_for_mode(df, mode, per_source_cap=per_source_cap, total_cap=total_cap)
    seed = _compose_seed(df, lo, hi, mode)
    sources = df["source"].drop_duplicates().tolist()
    return seed, hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16], len(df), sources


def run_day(client, engine, sim_date, *, mode: str = "source", seed_idx: int = 0,
            lookback: int = SEED_LOOKBACK_DAYS, llm_model: str = LLM_MODEL,
            question: str = DEFAULT_QUESTION):
    """Run one causal day-sim for ``mode``; cache features + graph + report. None if skipped/empty."""
    d = pd.Timestamp(sim_date).date()
    seed, seed_hash, n, sources = build_sim_seed(engine, sim_date, mode=mode, lookback=lookback)
    if n == 0:
        logger.warning("no headlines ≤ {} (lookback {}d) — skip", d, lookback)
        return None
    with engine.connect() as conn:
        if conn.execute(_HAS_SQL, {"d": d, "h": seed_hash, "m": llm_model, "i": seed_idx, "mode": mode}).first():
            logger.info("cached {} [{}] seed#{} — skip", d, mode, seed_idx)
            return None

    # 'flat' mode is provider-agnostic, so source entity_types scoping only applies to 'source'.
    art = client.run_day_sim(seed, question, name=f"ta125_{mode}_{d}_{seed_idx}",
                             entity_types=SIM_ENTITY_TYPES if mode == "source" else None)
    sim_id = art.get("simulation_id")
    if not sim_id:   # PK of _graph/_report — a null here would silently corrupt the cache
        raise MiroError(f"MiroFish returned no simulation_id for {d} [{mode}] seed#{seed_idx}")
    feats = votes_to_features(art["votes"])
    g = normalize_graph(art["graph"])
    if mode == "source":   # A3 seam: did outlets become graph entities? (no-op for flat)
        cov = source_agent_coverage(sources, g)
        logger.info("source→graph coverage {}/{} ({:.0%}){}", cov["matched"], cov["n_sources"],
                    cov["coverage"], f" — missing {cov['missing'][:5]}" if cov["missing"] else "")
    with engine.begin() as conn:
        conn.execute(_INS_SIM, {
            "sim_date": d, "seed_hash": seed_hash, "llm_model": llm_model, "seed_idx": seed_idx,
            "mode": mode, "dir_score": feats["dir_score"], "confidence": feats["confidence"],
            "disagreement": feats["disagreement"], "magnitude": None, "dominant_narrative": None,
            "n_agents": feats["n_votes"], "n_steps": None, "simulation_id": sim_id})
        conn.execute(_INS_GRAPH, {
            "sim_run_id": sim_id, "sim_date": d, "mode": mode, "graph_id": art["graph_id"],
            "nodes": json.dumps(g["nodes"]), "edges": json.dumps(g["edges"]),
            "meta": json.dumps(g["meta"])})
        conn.execute(_INS_REPORT, {
            "sim_run_id": sim_id, "sim_date": d, "mode": mode, "report_id": art["report_id"],
            "sections": json.dumps(art["sections"]), "report_md": sections_to_markdown(art["sections"])})
    logger.info("sim cached {} [{}] seed#{}: dir={:.3f} conf={:.3f} (n={})",
                d, mode, seed_idx, feats["dir_score"], feats["confidence"], feats["n_votes"])
    return feats


def run_window(dates, *, modes: list[str] | None = None, seeds: int = 1,
               lookback: int = SEED_LOOKBACK_DAYS, engine=None, base_url: str | None = None) -> int:
    """Run causal day-sims over ``dates`` × ``modes`` × ``seeds``; idempotent. Returns sims written."""
    from sentisense.sim import MiroClient
    modes = modes or SIM_MODES
    engine = engine or get_engine()
    client = MiroClient(base_url) if base_url else MiroClient()
    written = 0
    for d in dates:
        for mode in modes:
            for s in range(seeds):
                try:
                    if run_day(client, engine, d, mode=mode, seed_idx=s, lookback=lookback) is not None:
                        written += 1
                except Exception as exc:  # noqa: BLE001 — one bad day shouldn't sink the window
                    logger.error("sim {} [{}] seed#{} failed: {}", pd.Timestamp(d).date(), mode, s, str(exc)[:160])
    logger.info("window complete: {} new sims cached", written)
    return written
