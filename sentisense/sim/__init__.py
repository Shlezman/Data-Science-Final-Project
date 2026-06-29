"""MiroFish narrative-simulation layer (arm's-length HTTP client + cache).

MiroFish (vendored at external/MiroFish, AGPL) runs as a SEPARATE local service; this
package only talks to it over HTTP and persists a numeric daily feature (mode A), the
agent graph (UI / mode C), and the qualitative report (mode B). It never imports
MiroFish in-process. See docs/miro/PLAN.md.
"""

from sentisense.sim.miro_client import MiroClient, MiroError

__all__ = ["MiroClient", "MiroError"]
