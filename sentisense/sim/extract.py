"""Turn MiroFish artifacts into a DETERMINISTIC numeric feature + explainability text.

The numeric signal comes from the **agent interview votes** (not NLP-parsing the prose
report): aggregate per-agent stances into a crowd consensus. Robust to a few plausible
response shapes (the exact MiroFish interview schema is confirmed on the box) — unknown
shapes yield NaN features + a logged warning, never a crash.
"""

from __future__ import annotations

import re

import numpy as np
from loguru import logger

_BULL = {"up", "buy", "bull", "bullish", "rise", "rising", "positive", "long", "gain", "increase", "higher"}
_BEAR = {"down", "sell", "bear", "bearish", "fall", "falling", "negative", "short", "loss", "decrease", "lower", "drop"}
_NEUTRAL = {"neutral", "flat", "hold", "unchanged", "unsure", "mixed"}


def _stance(x) -> float | None:
    """Map one answer to a stance in [-1, 1]: +1 up, -1 down, 0 neutral, None if unparseable."""
    if isinstance(x, bool):
        return 1.0 if x else -1.0
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(np.clip(x, -1.0, 1.0))
    if isinstance(x, str):
        t = x.strip().lower()
        try:
            return float(np.clip(float(t), -1.0, 1.0))   # explicit number
        except ValueError:
            pass
        toks = set(re.findall(r"[a-z]+", t))
        bull, bear, neut = toks & _BULL, toks & _BEAR, toks & _NEUTRAL
        if bull and not bear:
            return 1.0
        if bear and not bull:
            return -1.0
        if neut or (bull and bear):
            return 0.0
    return None


def _iter_votes(votes) -> list:
    """Find the list of per-agent answers in a flexible interview response."""
    if isinstance(votes, list):
        return votes
    if isinstance(votes, dict):
        for k in ("answers", "results", "agents", "votes", "interviews", "responses", "data"):
            v = votes.get(k)
            if isinstance(v, list):
                return v
            if isinstance(v, dict) and isinstance(v.get("answers"), list):
                return v["answers"]
    return []


def _extract_stance(item) -> float | None:
    if isinstance(item, (int, float, str, bool)):
        return _stance(item)
    if isinstance(item, dict):
        for k in ("stance", "vote", "direction", "answer", "prediction", "label",
                  "sentiment", "response", "text", "opinion"):
            if k in item:
                s = _stance(item[k])
                if s is not None:
                    return s
    return None


def votes_to_features(votes) -> dict:
    """Aggregate agent votes → {dir_score, confidence, disagreement, n_votes}.

    dir_score = mean stance ∈ [-1,1]; confidence = fraction agreeing with the majority
    sign; disagreement = stance std. NaN features (n_votes=0) when nothing parseable.
    """
    stances = [s for it in _iter_votes(votes) if (s := _extract_stance(it)) is not None]
    if not stances:
        logger.warning("MiroFish votes: no parseable agent stances — emitting NaN sim features.")
        return {"dir_score": float("nan"), "confidence": float("nan"),
                "disagreement": float("nan"), "n_votes": 0}
    a = np.asarray(stances, dtype=float)
    dir_score = float(a.mean())
    maj = 1.0 if dir_score >= 0 else -1.0
    confidence = float((np.sign(a) == maj).mean())
    return {"dir_score": dir_score, "confidence": confidence,
            "disagreement": float(a.std()), "n_votes": int(a.size)}


def sections_to_markdown(sections) -> str:
    """Join report sections (ordered) into one markdown string for explainability."""
    ordered = sorted(sections or [], key=lambda s: s.get("section_index", 0))
    return "\n\n".join(s.get("content", "") for s in ordered)
