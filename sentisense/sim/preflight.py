"""Local-only guards for the MiroFish hop.

SentiSense talks to MiroFish over HTTP. This module enforces that the hop stays on-box
(loopback) unless the operator explicitly opts into a remote service. It does NOT (and
cannot) see MiroFish's OWN outbound calls (LLM / Zep / HuggingFace) — those are pinned
local via MiroFish's .env; see docs/miro/LOCAL_ONLY.md and scripts/verify_local_egress.sh
for the end-to-end proof.
"""

from __future__ import annotations

from urllib.parse import urlparse

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def is_loopback(url: str) -> bool:
    """True if ``url``'s host is a loopback/local address."""
    host = (urlparse(url).hostname or "").lower()
    return host in _LOOPBACK_HOSTS


def assert_local(url: str, *, allow_remote: bool) -> None:
    """Raise if ``url`` is not loopback and remote access wasn't explicitly allowed.

    Args:
        url: The MiroFish base URL the client is about to talk to.
        allow_remote: When True, a non-loopback host is permitted (operator opt-in).

    Raises:
        MiroError: If ``url`` is non-loopback and ``allow_remote`` is False.
    """
    if allow_remote or is_loopback(url):
        return
    from sentisense.sim.miro_client import MiroError

    raise MiroError(
        f"MiroFish URL {url!r} is not loopback. SentiSense runs MiroFish locally by default. "
        "If you really mean to reach a remote service, set SENTISENSE_MIRO_ALLOW_REMOTE=1 — "
        "but note MiroFish's own LLM/Zep/HF egress must still be pinned local "
        "(see docs/miro/LOCAL_ONLY.md)."
    )
