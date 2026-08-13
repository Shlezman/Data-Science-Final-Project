"""Archive simulation results (agent graphs + reports) from Postgres into MongoDB.

Postgres keeps ONE row per (day, mode) — re-runs overwrite it. This archiver copies every
simulation state into the front machine's MongoDB (``sentisense.sim_archive``), keyed by
``(sim_date, mode, pg_created_at)``, so history is preserved across re-runs and past days
stay browsable/manipulable (same story as the performance-JSON versions).

Run (front machine — Mongo is local there; nightly cron after sim generation):
    uv run --extra ui python scripts/archive_sims_to_mongo.py            # everything new
    uv run --extra ui python scripts/archive_sims_to_mongo.py --days 7   # recent window only

Needs SENTISENSE_DATABASE_URL + SENTISENSE_MONGO_URL in the environment.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

from loguru import logger
from sqlalchemy import text

from sentisense.db import get_engine

_ROWS = text(
    """
    SELECT g.sim_date, g.mode, g.created_at, g.sim_run_id, g.nodes, g.edges, g.meta,
           r.report_md, r.sections
    FROM narrative_sim_graph g
    LEFT JOIN narrative_sim_report r
           ON r.sim_run_id = g.sim_run_id AND r.mode = g.mode
    WHERE (CAST(:since AS date) IS NULL OR g.sim_date >= CAST(:since AS date))
    ORDER BY g.sim_date
    """
)


def _j(v):
    """JSONB arrives parsed on psycopg3 or as str elsewhere — normalize to Python."""
    return v if not isinstance(v, str) else json.loads(v)


def main() -> int:
    """Upsert every simulation row into Mongo; idempotent on (date, mode, created_at)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=0, help="Only archive the last N days (0 = all).")
    args = ap.parse_args()

    mongo_url = os.environ.get("SENTISENSE_MONGO_URL", "")
    if not mongo_url:
        raise SystemExit("SENTISENSE_MONGO_URL is not set.")
    from pymongo import MongoClient, UpdateOne

    coll = MongoClient(mongo_url, serverSelectionTimeoutMS=3000)["sentisense"]["sim_archive"]
    coll.create_index([("sim_date", 1), ("mode", 1), ("pg_created_at", 1)], unique=True)

    since = (str(dt.date.today() - dt.timedelta(days=args.days)) if args.days else None)
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(_ROWS, {"since": since}).mappings().all()
    if not rows:
        logger.info("No simulation rows to archive.")
        return 0

    ops = [UpdateOne(
        {"sim_date": str(r["sim_date"]), "mode": r["mode"], "pg_created_at": str(r["created_at"])},
        {"$set": {
            "sim_run_id": r["sim_run_id"],
            "nodes": _j(r["nodes"]), "edges": _j(r["edges"]), "meta": _j(r["meta"]),
            "report_md": r["report_md"], "sections": _j(r["sections"]),
            "archived_at": dt.datetime.utcnow(),
        }},
        upsert=True) for r in rows]
    res = coll.bulk_write(ops, ordered=False)
    logger.info("Archived {} sim rows → Mongo sim_archive (upserted {}, modified {}; total in archive: {}).",
                len(rows), res.upserted_count, res.modified_count, coll.count_documents({}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
