"""Phase 1 ingestion — historical backfill, cutoff-scoped scoring, coverage report.

These modules deliberately REUSE the battle-tested existing scripts
(``scripts/backfill_history.py``, ``scripts/process_headlines.py``) via subprocess
rather than re-implementing scraping/scoring. They add: a consistent
``python -m sentisense.ingest.X`` entry point, the hard ``<= 2023-10-07`` cutoff
where it matters, and a net-new DB coverage report (read through the SQLAlchemy
engine in :mod:`sentisense.db`).
"""
