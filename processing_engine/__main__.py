"""Allow ``python -m processing_engine`` to run a smoke test."""

import asyncio
import json
import sys

from loguru import logger

from .engine import process_single_observation

sample = {
    "date": "2025-01-15",
    "source": "כאן חדשות",
    "hour": "14:30",
    "popularity": "important",
    "headline": "בנק ישראל הכריז על העלאת הריבית ב-0.25% לאחר עלייה באינפלציה",
}


def main():
    logger.info("Running smoke test with sample observation…")
    result = asyncio.run(process_single_observation(sample))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
    sys.exit(0)
