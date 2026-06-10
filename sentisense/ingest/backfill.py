"""Phase 1.1 — historical headline backfill (backwards), idempotent + resumable.

Thin orchestrator over the proven ``scripts/backfill_history.py``, which scrapes
``raw_headlines`` BACKWARDS from the oldest stored date until the source is
exhausted (N consecutive empty/all-duplicate windows). That script already handles
Asia/Jerusalem tz anchoring, ``ON CONFLICT`` dedup, subprocess scraper isolation,
and resumability — we do not re-implement any of it.

Why a wrapper at all: a single ``python -m sentisense.ingest.backfill`` entry point
consistent with the rest of the package, plus explicit logging of the exact
delegated command for the paste-back gate.

Cutoff note: backfill extends history *earlier* than what already exists, so it
never produces data after ``2023-10-07``; the cutoff is enforced downstream at
scoring (:mod:`sentisense.ingest.score`) and at every modeling query. The optional
``--start-before`` lets the operator target a specific pre-cutoff window.

Run (server-side, operator):
    uv run python -m sentisense.ingest.backfill --window 7 --dry-run
    uv run python -m sentisense.ingest.backfill --window 7 --max-days 3650
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from loguru import logger

from sentisense.constants import REPO_ROOT, parse_iso_date

_BACKFILL_SCRIPT = REPO_ROOT / "scripts" / "backfill_history.py"
_PROCESSING_ENGINE = REPO_ROOT / "processing_engine"


def build_command(args: argparse.Namespace) -> list[str]:
    """Assemble the delegated ``backfill_history.py`` invocation.

    Args:
        args: Parsed CLI args.

    Returns:
        The argv list to execute (run with cwd=processing_engine via ``uv run``).
    """
    cmd = [
        "uv", "run", "--project", str(_PROCESSING_ENGINE),
        "python", str(_BACKFILL_SCRIPT),
        "--window", str(args.window),
        "--empty-streak", str(args.empty_streak),
        "--pages", str(args.pages),
    ]
    if args.max_days:
        cmd += ["--max-days", str(args.max_days)]
    if args.start_before:
        cmd += ["--start-before", args.start_before]
    if args.dry_run:
        cmd += ["--dry-run"]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1.1 backfill — extend raw_headlines history backwards "
        "(delegates to scripts/backfill_history.py).",
    )
    parser.add_argument("--window", type=int, default=7, help="Days per scrape window (default 7).")
    parser.add_argument("--empty-streak", type=int, default=2,
                        help="Stop after N consecutive empty/all-dup windows (default 2).")
    parser.add_argument("--pages", type=int, default=100, help="Max pages per date (default 100).")
    parser.add_argument("--max-days", type=int, default=0,
                        help="Cap total days walked back (0 = until exhausted).")
    parser.add_argument("--start-before", type=str, default="",
                        help="Begin backfill before this date YYYY-MM-DD (default: DB oldest).")
    parser.add_argument("--dry-run", action="store_true", help="Print plan; no scrape, no writes.")
    args = parser.parse_args()

    if args.window < 1:
        parser.error("--window must be >= 1")
    if args.empty_streak < 1:
        parser.error("--empty-streak must be >= 1")
    if args.start_before:
        try:
            parse_iso_date(args.start_before)
        except ValueError:
            parser.error("--start-before must be YYYY-MM-DD")

    cmd = build_command(args)
    logger.info("Phase 1.1 backfill → delegating to backfill_history.py")
    logger.info("  command: {}", " ".join(cmd))
    # SECURITY NOTE: scraping uses no credentials; the delegated script reads
    # SENTISENSE_DATABASE_URL from the inherited environment.
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT))
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
