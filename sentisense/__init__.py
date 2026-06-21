"""SentiSense Phase 2 & 3 — financial modeling over Hebrew-news sentiment features.

Predicts next-day **close-to-close** direction of the TA-125 (Tel Aviv 125) index
from LLM-scored Hebrew news headlines plus finance/market signals.

Hard project invariants (enforced throughout):

* **Cutoff:** only ``raw_headlines.date <= 2023-10-07`` is ever used. The regime
  breaks at that date (TASE), so training across it violates stationarity.
* **No leakage:** every transform (scaler, KMeans, imputer) is fit on the TRAIN
  fold only; validation is always ``TimeSeriesSplit`` / chronological — never random.
* **No hardcoded secrets:** the connection string is read from
  ``SENTISENSE_DATABASE_URL`` (see :mod:`sentisense.db.connection`); the module
  fails fast if it is unset rather than embedding a default password.

See ``docs/sentisense-understanding.md`` for the full schema + pipeline ground truth.
"""


def _load_dotenv_once() -> None:
    """Auto-load ``.env`` (then ``env``) from the repo root on first package import.

    Runs BEFORE :mod:`sentisense.constants` reads the environment, so every
    ``python -m sentisense.X`` entry point picks up the local config — and because
    load_dotenv populates ``os.environ``, the vars also propagate to the scoring
    subprocess (``scripts/process_headlines.py``). Real env vars win (override=False).
    No-op if python-dotenv is absent or no file exists.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for name in (".env", "env"):
        candidate = root / name
        if candidate.exists():
            load_dotenv(candidate, override=False)
            break


_load_dotenv_once()

from sentisense.constants import (  # noqa: E402  (must follow dotenv load)
    ACTIVE_MODEL_NAME,
    CUTOFF_DATE,
    DB_RELEVANCE_COLUMNS,
    SCORE_COLUMNS,
)

__all__ = [
    "ACTIVE_MODEL_NAME",
    "CUTOFF_DATE",
    "DB_RELEVANCE_COLUMNS",
    "SCORE_COLUMNS",
]
