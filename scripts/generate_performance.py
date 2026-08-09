"""Generate the Model-performance panel JSON (``models/performance.json``).

The UI serves this file verbatim when it exists (file override > computed), so the panel can
be tuned by hand (edit the file) or regenerated methodically (re-run this script). Delete the
file to fall back to live computed values.

Run (any host with DB access):
    uv run --extra ui python scripts/generate_performance.py            # write the file
    uv run --extra ui python scripts/generate_performance.py --print    # stdout only
    uv run --extra ui python scripts/generate_performance.py --delete   # back to computed
"""

from __future__ import annotations

import argparse
import json
import sys

from loguru import logger

from sentisense.constants import REPO_ROOT

_OUT = REPO_ROOT / "models" / "performance.json"


def main() -> int:
    """Build the computed performance document and write/print/delete the override file."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--print", action="store_true", help="Print JSON to stdout; do not write.")
    ap.add_argument("--delete", action="store_true", help="Remove the override file.")
    args = ap.parse_args()

    if args.delete:
        if _OUT.exists():
            _OUT.unlink()
            logger.info("Removed {} — UI serves computed values again.", _OUT)
        else:
            logger.info("No override file present.")
        return 0

    from ui.app import _build_performance

    doc = _build_performance()
    doc["source"] = "file"
    text = json.dumps(doc, indent=2, ensure_ascii=False)
    if args.print:
        print(text)
        return 0
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(text, encoding="utf-8")
    logger.info("Wrote {} ({} core metrics). Edit freely — the UI serves it verbatim.",
                _OUT, len(doc.get("core", [])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
