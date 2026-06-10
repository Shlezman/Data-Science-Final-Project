"""Phase 1.2 — score unscored headlines (<= 2023-10-07) through the LLM pipeline.

Thin orchestrator over the proven ``scripts/process_headlines.py`` fast path. That
script already: queries only headlines lacking a row for the active model
(``LEFT JOIN nlp_vectors ... WHERE nv.id IS NULL``), runs the single-prompt fast
pipeline, writes ``nlp_vectors`` with ``ON CONFLICT DO NOTHING`` idempotency, and
resolves the model name. We add the hard ``--date-to 2023-10-07`` cutoff and the
package entry point.

Idempotency: re-runs skip already-scored rows. NOTE — a previously *failed*
(``validation_passed=FALSE``) row is NOT overwritten by this happy-path scorer; use
``scripts/retry_failed_headlines.py`` to re-score failures (documented for the operator).

Backend: the production vLLM ``mistral-small-4`` setup requires the fast path and
``SENTISENSE_FORCE_COMPLETIONS_API=true`` in the environment (the multi-agent path
sys.exit(2)s under that flag). This wrapper always passes ``--fast``.

Run (server-side, operator), with the production LLM env exported:
    uv run python -m sentisense.ingest.score --dry-run
    uv run python -m sentisense.ingest.score --headlines-per-call 20 --concurrency 50
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from loguru import logger

from sentisense.constants import CUTOFF_DATE, CUTOFF_DATE_ISO, REPO_ROOT, parse_iso_date

_PROCESS_SCRIPT = REPO_ROOT / "scripts" / "process_headlines.py"
_PROCESSING_ENGINE = REPO_ROOT / "processing_engine"


def build_command(args: argparse.Namespace) -> list[str]:
    """Assemble the delegated ``process_headlines.py`` invocation (fast, cutoff-scoped).

    Args:
        args: Parsed CLI args.

    Returns:
        The argv list to execute via ``uv run`` from the processing_engine project.
    """
    cmd = [
        "uv", "run", "--project", str(_PROCESSING_ENGINE),
        "python", str(_PROCESS_SCRIPT),
        "--fast",
        "--date-to", CUTOFF_DATE_ISO,  # HARD cutoff: never score past 2023-10-07
        "--concurrency", str(args.concurrency),
        "--headlines-per-call", str(args.headlines_per_call),
    ]
    if args.date_from:
        cmd += ["--date-from", args.date_from]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.dry_run:
        cmd += ["--dry-run"]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1.2 scoring — score unscored headlines <= 2023-10-07 "
        "(delegates to scripts/process_headlines.py --fast).",
    )
    parser.add_argument("--concurrency", type=int, default=50,
                        help="Concurrent headlines in fast mode (default 50, max 128).")
    parser.add_argument("--headlines-per-call", type=int, default=20,
                        help="Headlines packed per LLM call (default 20; 0 = one per call).")
    parser.add_argument("--date-from", type=str, default="",
                        help="Only score on/after this date YYYY-MM-DD (default: earliest).")
    parser.add_argument("--limit", type=int, default=0, help="Max headlines to score (0 = all).")
    parser.add_argument("--dry-run", action="store_true", help="Show count; no LLM calls, no writes.")
    args = parser.parse_args()

    if not (1 <= args.concurrency <= 128):
        parser.error("--concurrency must be between 1 and 128")
    if not (0 <= args.headlines_per_call <= 150):
        parser.error("--headlines-per-call must be between 0 and 150")
    if args.date_from:
        try:
            df = parse_iso_date(args.date_from)
        except ValueError:
            parser.error("--date-from must be YYYY-MM-DD")
        if df > CUTOFF_DATE:
            parser.error(f"--date-from {args.date_from} is after the hard cutoff {CUTOFF_DATE_ISO}")

    cmd = build_command(args)
    logger.info("Phase 1.2 scoring → delegating to process_headlines.py (cutoff <= {})", CUTOFF_DATE_ISO)
    logger.info("  command: {}", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT))
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
