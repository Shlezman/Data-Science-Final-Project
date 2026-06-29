"""Force MiroFish's Zep client local — the guaranteed fallback.

The audit could not settle whether the pinned ``zep-cloud==3.13.0`` SDK honors the
``ZEP_API_URL`` env var (some builds read it in ``Zep.__init__``, some only on a newer
branch). ``scripts/verify_local_egress.sh`` tells you which you have. If it does NOT read
the env var, run THIS to rewrite every ``Zep(api_key=...)`` call site so it passes
``base_url`` explicitly from ``ZEP_API_URL`` (the SDK always honors an explicit base_url).

Idempotent. Writes a ``.bak`` next to each touched file. ``--dry-run`` shows the plan.
Operates on the vendored submodule (``external/MiroFish``); does not commit anything.

    python scripts/patch_mirofish_zep_local.py --dry-run
    python scripts/patch_mirofish_zep_local.py            # apply
    python scripts/patch_mirofish_zep_local.py --revert   # restore .bak files
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# Relative to external/MiroFish/ — the 5 confirmed Zep() instantiation sites.
SITES = [
    "backend/app/services/graph_builder.py",
    "backend/app/services/zep_tools.py",
    "backend/app/services/zep_entity_reader.py",
    "backend/app/services/zep_graph_memory_updater.py",
    "backend/app/services/oasis_profile_generator.py",
]

# Self-contained base_url expr: ZEP_API_URL (no /api/v2 suffix) -> "<url>/api/v2", else None
# (None ⇒ unchanged SDK default, so the patch only redirects when the env var is set).
_BASE_EXPR = (
    'base_url=(lambda _u: _u.rstrip("/") + "/api/v2" if _u else None)'
    '(__import__("os").environ.get("ZEP_API_URL"))'
)
_CALL = re.compile(r"Zep\(\s*api_key\s*=\s*(self\.\w+)\s*\)")
_MARKER = '(lambda _u: _u.rstrip("/")'


def _patched(text: str) -> tuple[str, int]:
    """Return (new_text, n_replacements) injecting base_url into bare Zep(api_key=…) calls."""
    return _CALL.subn(lambda m: f"Zep(api_key={m.group(1)}, {_BASE_EXPR})", text)


def _miro_root() -> Path:
    return Path(__file__).resolve().parent.parent / "external" / "MiroFish"


def apply(*, dry_run: bool) -> int:
    """Patch all sites. Returns the number of files changed."""
    root = _miro_root()
    changed = 0
    for rel in SITES:
        path = root / rel
        if not path.exists():
            print(f"  MISSING {rel} — skip (submodule not initialised?)")
            continue
        text = path.read_text(encoding="utf-8")
        if _MARKER in text:
            print(f"  ok      {rel} — already patched")
            continue
        new, n = _patched(text)
        if n == 0:
            print(f"  WARN    {rel} — no bare Zep(api_key=…) call found (already custom?)")
            continue
        print(f"  {'would patch' if dry_run else 'patched'} {rel} ({n} call site[s])")
        if not dry_run:
            path.with_suffix(path.suffix + ".bak").write_text(text, encoding="utf-8")
            path.write_text(new, encoding="utf-8")
        changed += 1
    return changed


def revert() -> int:
    """Restore every .bak written by a prior apply. Returns files restored."""
    root = _miro_root()
    restored = 0
    for rel in SITES:
        bak = root / (rel + ".bak")
        if bak.exists():
            target = root / rel
            target.write_text(bak.read_text(encoding="utf-8"), encoding="utf-8")
            bak.unlink()
            print(f"  reverted {rel}")
            restored += 1
    return restored


def main() -> None:
    p = argparse.ArgumentParser(description="Force MiroFish Zep client to a local base_url.")
    p.add_argument("--dry-run", action="store_true", help="Show the plan; change nothing.")
    p.add_argument("--revert", action="store_true", help="Restore .bak files from a prior run.")
    args = p.parse_args()

    if args.revert:
        print(f"reverted {revert()} file(s).")
        return
    n = apply(dry_run=args.dry_run)
    verb = "would change" if args.dry_run else "changed"
    print(f"\n{verb} {n} file(s). Set ZEP_API_URL=http://localhost:8000 in MiroFish/.env, then "
          "re-run scripts/verify_local_egress.sh.")


if __name__ == "__main__":
    main()
