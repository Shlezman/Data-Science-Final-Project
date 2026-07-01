"""Migrate the entire SentiSense database to another Postgres server (pg_dump → pg_restore).

Copies ALL tables (schema + data + indexes + constraints + sequences) from a SOURCE database
to a TARGET database on a different, more-persistent server that this host can reach. Uses
PostgreSQL's own tools (no per-table reinvention), then verifies row counts table-by-table.

After a successful migration, switch the live system to the new DB by exporting the new URL
as ``SENTISENSE_DATABASE_URL`` (shell + cron + pm2) — the application code reads only that env
var, so no code change is required.

Run (server-side, from repo root). Pass URLs as plain ``postgresql://USER:PASS@HOST:5432/db``
(secrets via the URL/env — nothing is hardcoded or logged):
    export SENTISENSE_DATABASE_URL='postgresql://sentisense:***@localhost:5432/sentisense'   # source
    uv run python scripts/migrate_db.py --target 'postgresql://USER:***@NEWHOST:5432/sentisense'
    uv run python scripts/migrate_db.py --target '...' --dry-run     # print the plan only
    uv run python scripts/migrate_db.py --target '...' --jobs 8      # parallel restore
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from loguru import logger


def _plain(url: str) -> str:
    """Normalise a SQLAlchemy-style URL to the plain libpq form pg_dump/psql expect."""
    return (url.replace("postgresql+psycopg://", "postgresql://")
               .replace("postgres+psycopg://", "postgresql://")
               .replace("postgresql+psycopg2://", "postgresql://"))


def _redact(url: str) -> str:
    """Hide the password before logging a connection URL."""
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, host = rest.split("@", 1)
    user = creds.split(":", 1)[0]
    return f"{scheme}://{user}:***@{host}"


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a subprocess, raising with a clear message on failure."""
    proc = subprocess.run(cmd, text=True, capture_output=True, **kw)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed (rc={proc.returncode}): {proc.stderr.strip()[-500:]}")
    return proc


def _psql_scalar(url: str, sql: str) -> str:
    """Return a single scalar from psql -tA."""
    return _run(["psql", url, "-tAc", sql]).stdout.strip()


def _table_list(url: str) -> list[str]:
    """All base tables in the public schema, qualified."""
    rows = _run(["psql", url, "-tAc",
                 "SELECT schemaname||'.'||tablename FROM pg_tables WHERE schemaname='public' "
                 "ORDER BY 1"]).stdout
    return [r.strip() for r in rows.splitlines() if r.strip()]


def _verify(source: str, target: str) -> bool:
    """Compare per-table row counts source vs target. Returns True if all match."""
    ok = True
    for tbl in _table_list(source):
        s = _psql_scalar(source, f"SELECT count(*) FROM {tbl}")
        try:
            t = _psql_scalar(target, f"SELECT count(*) FROM {tbl}")
        except RuntimeError:
            t = "MISSING"
        mark = "✓" if s == t else "✗"
        if s != t:
            ok = False
        logger.info("  {} {:<28} source={:>10} target={:>10}", mark, tbl, s, t)
    return ok


def main() -> int:
    """Dump the source DB and restore it into the target; then verify counts."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=os.environ.get("SENTISENSE_DATABASE_URL", ""),
                    help="Source DB URL (default: SENTISENSE_DATABASE_URL).")
    ap.add_argument("--target", required=True, help="Target DB URL on the new server.")
    ap.add_argument("--jobs", type=int, default=4, help="Parallel pg_restore workers.")
    ap.add_argument("--dump-file", default="", help="Dump path (default: a temp file).")
    ap.add_argument("--no-clean", action="store_true",
                    help="Do not DROP existing objects on the target before restore.")
    ap.add_argument("--no-verify", action="store_true", help="Skip the row-count verification.")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan; do nothing.")
    args = ap.parse_args()

    if not args.source:
        raise SystemExit("No source — set SENTISENSE_DATABASE_URL or pass --source.")
    source, target = _plain(args.source), _plain(args.target)
    dump = Path(args.dump_file) if args.dump_file else Path(tempfile.gettempdir()) / "sentisense_migrate.dump"

    logger.info("Source: {}", _redact(source))
    logger.info("Target: {}", _redact(target))
    logger.info("Dump  : {} | jobs={} | clean={}", dump, args.jobs, not args.no_clean)

    dump_cmd = ["pg_dump", source, "-Fc", "--no-owner", "--no-acl", "-f", str(dump)]
    restore_cmd = (["pg_restore", "--no-owner", "--no-acl", "-j", str(args.jobs)]
                   + ([] if args.no_clean else ["--clean", "--if-exists"])
                   + ["-d", target, str(dump)])
    if args.dry_run:
        logger.info("[dry-run] {}", " ".join(["pg_dump", _redact(source), "-Fc", "…", str(dump)]))
        logger.info("[dry-run] {}", " ".join(["pg_restore", "…", "-d", _redact(target), str(dump)]))
        return 0

    # Connectivity preflight (clear failure before a long dump).
    _psql_scalar(source, "SELECT 1")
    _psql_scalar(target, "SELECT 1")

    logger.info("Dumping source → {} …", dump)
    _run(dump_cmd)
    logger.info("Dump size: {:.1f} MB", dump.stat().st_size / 1e6)

    logger.info("Restoring into target (errors on DROP of non-existent objects are normal) …")
    proc = subprocess.run(restore_cmd, text=True, capture_output=True)
    # pg_restore returns non-zero on benign --clean warnings; surface stderr, don't hard-fail yet.
    if proc.returncode != 0:
        logger.warning("pg_restore exited rc={} (often benign --clean noise). Tail:\n{}",
                       proc.returncode, proc.stderr.strip()[-800:])

    if args.no_verify:
        logger.info("Restore done (verification skipped).")
        return 0
    logger.info("Verifying row counts source vs target …")
    if _verify(source, target):
        logger.info("✅ All tables match. Switch the app: export SENTISENSE_DATABASE_URL to the target.")
        return 0
    logger.error("❌ Row-count mismatch — do NOT switch the app yet; investigate above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
