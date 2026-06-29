"""Daily live orchestrator — chains the EXISTING SentiSense stages, end to end.

scrape -> score -> embed -> derived-features -> champion predict (-> writes model_predictions).
Reuses each stage's own entrypoint (no reimplementation). Production-safe:

  * lockfile (flock) — a second invocation exits instead of double-running;
  * TASE-calendar guard — skips Fri/Sat (and optional holiday list), in Asia/Jerusalem time;
  * idempotent — every stage is upsert/skip-existing, so a same-day re-run is a no-op-ish;
  * structured loguru logs to logs/daily_live_{date}.log + a status JSON the UI reads;
  * explicit exit codes (0 ok / skipped, 1 failure) and a failure record in the status file.

Run (server-side, inside the /tf container, after TASE close):
    uv run --extra finance --extra ml python scripts/daily_live.py
    uv run --extra finance --extra ml python scripts/daily_live.py --dry-run
    uv run --extra finance --extra ml python scripts/daily_live.py --force   # ignore calendar
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

from loguru import logger

from sentisense.constants import REPO_ROOT

_IL_TZ = ZoneInfo("Asia/Jerusalem")
_LOGS = REPO_ROOT / "logs"
_LOCK_PATH = _LOGS / "daily_live.lock"
_STATUS_PATH = _LOGS / "daily_live_status.json"
_HOLIDAYS_PATH = REPO_ROOT / "config" / "tase_holidays.txt"   # optional: one YYYY-MM-DD per line
_TASE_TRADING_WEEKDAYS = {6, 0, 1, 2, 3}                       # Sun=6, Mon..Thu=0..3 (Fri/Sat off)

# Subprocess stages: (name, argv, cwd-relative-to-REPO_ROOT). The champion predict runs
# in-process (last). Commands mirror the documented per-module invocations / uv extras.
_PE = "processing_engine"
_STAGES = [
    ("scrape", ["uv", "run", "python", "../scripts/daily_scrape_to_db.py", "--days", "2"], _PE),
    ("score", ["uv", "run", "python", "../scripts/process_headlines.py", "--fast",
               "--headlines-per-call", "50", "--concurrency", "50"], _PE),
    ("embed", ["uv", "run", "--extra", "embed", "python", "-m", "sentisense.embed.embeddings",
               "--scope", "all"], "."),
    ("derived", ["uv", "run", "--extra", "ml", "python", "scripts/build_embedding_derived.py"], "."),
]


def is_trading_day(day: dt.date) -> bool:
    """True if ``day`` is a TASE trading day (Sun–Thu, minus any listed holiday)."""
    if day.weekday() not in _TASE_TRADING_WEEKDAYS:
        return False
    if _HOLIDAYS_PATH.exists():
        holidays = {line.strip() for line in _HOLIDAYS_PATH.read_text().splitlines() if line.strip()}
        if day.isoformat() in holidays:
            return False
    return True


def _write_status(status: dict) -> None:
    """Atomically persist the run status JSON the UI/ops read."""
    _LOGS.mkdir(parents=True, exist_ok=True)
    tmp = _STATUS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    tmp.replace(_STATUS_PATH)


def _load_status() -> dict:
    """Previous status (to preserve last_success across runs); {} if none."""
    if _STATUS_PATH.exists():
        try:
            return json.loads(_STATUS_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _run_stage(name: str, argv: list[str], cwd: Path, dry_run: bool) -> dict:
    """Run one subprocess stage; return a result record (ok, seconds, tail of output)."""
    if dry_run:
        logger.info("[dry-run] would run {}: {} (cwd={})", name, " ".join(argv), cwd)
        return {"stage": name, "ok": True, "skipped": "dry-run", "seconds": 0.0}
    logger.info("stage {} → {} (cwd={})", name, " ".join(argv), cwd)
    t0 = time.monotonic()
    proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
    secs = round(time.monotonic() - t0, 1)
    tail = (proc.stdout or "")[-400:] + (proc.stderr or "")[-800:]
    ok = proc.returncode == 0
    (logger.info if ok else logger.error)("stage {} {} in {}s (rc={})",
                                          name, "ok" if ok else "FAILED", secs, proc.returncode)
    return {"stage": name, "ok": ok, "seconds": secs, "returncode": proc.returncode,
            "tail": tail.strip()[-600:]}


def _acquire_lock():
    """Acquire an exclusive non-blocking flock; return the open handle or None if held."""
    _LOGS.mkdir(parents=True, exist_ok=True)
    fh = open(_LOCK_PATH, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return None
    fh.write(str(dt.datetime.now(_IL_TZ)))
    fh.flush()
    return fh


def main() -> int:
    """Orchestrate the daily live pipeline. Returns an exit code (0 ok/skip, 1 failure)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Log the plan; run nothing, write nothing.")
    ap.add_argument("--force", action="store_true", help="Run even on a non-trading day.")
    ap.add_argument("--skip-predict", action="store_true", help="Run stages but skip champion predict.")
    args = ap.parse_args()

    _LOGS.mkdir(parents=True, exist_ok=True)
    logger.add(_LOGS / "daily_live_{time:YYYY-MM-DD}.log", rotation="10 MB", retention="30 days",
               level="DEBUG", enqueue=True)

    today = dt.datetime.now(_IL_TZ).date()
    prev = _load_status()
    status = {"today": today.isoformat(), "started_at": str(dt.datetime.now(_IL_TZ)),
              "last_success": prev.get("last_success"), "stages": [], "skipped": None,
              "prediction": None, "error": None}

    if not args.force and not is_trading_day(today):
        logger.info("{} is not a TASE trading day (Sun–Thu) — skipping.", today)
        status["skipped"] = "non-trading-day"
        status["finished_at"] = str(dt.datetime.now(_IL_TZ))
        if not args.dry_run:
            _write_status(status)
        return 0

    lock = None if args.dry_run else _acquire_lock()
    if not args.dry_run and lock is None:
        logger.warning("Another daily_live run holds the lock — exiting without double-running.")
        return 0

    try:
        for name, argv, cwd_rel in _STAGES:
            rec = _run_stage(name, argv, REPO_ROOT / cwd_rel, args.dry_run)
            status["stages"].append(rec)
            if not rec["ok"]:
                status["error"] = f"stage '{name}' failed (rc={rec.get('returncode')})"
                logger.error("Aborting after stage {} failure.", name)
                status["finished_at"] = str(dt.datetime.now(_IL_TZ))
                if not args.dry_run:
                    _write_status(status)
                return 1

        if not args.skip_predict:
            if args.dry_run:
                logger.info("[dry-run] would train champion + predict next move.")
            else:
                from sentisense.serve import predict_today
                status["prediction"] = predict_today()
                logger.info("Prediction written: {}", status["prediction"])

        status["last_success"] = str(dt.datetime.now(_IL_TZ))
        status["finished_at"] = status["last_success"]
        if not args.dry_run:
            _write_status(status)
        logger.info("Daily live pipeline complete for {}.", today)
        return 0
    except Exception as exc:  # noqa: BLE001 — record any failure in status, exit 1
        logger.exception("Daily live pipeline crashed: {}", exc)
        status["error"] = str(exc)[:300]
        status["finished_at"] = str(dt.datetime.now(_IL_TZ))
        if not args.dry_run:
            _write_status(status)
        return 1
    finally:
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()


if __name__ == "__main__":
    sys.exit(main())
