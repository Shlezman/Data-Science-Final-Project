"""ETA helpers — formatting + stage clock. No DB required.

Run: uv run pytest tests/test_eta.py -v
"""

from __future__ import annotations

import re

from sentisense import eta


def test_fmt_duration_tiers():
    assert eta.fmt_duration(0) == "0s"
    assert eta.fmt_duration(45) == "45s"
    assert eta.fmt_duration(90) == "1m 30s"
    assert eta.fmt_duration(3661) == "1h 01m 01s"
    assert eta.fmt_duration(None) == "n/a"


def test_eta_clock_is_a_timestamp():
    s = eta.eta_clock(120)
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", s)
    assert eta.eta_clock(None) == "n/a"


def test_stage_clock_records_actuals_and_handles_remaining():
    estimates = {"features": 10.0, "baselines": 20.0, "tune": None}
    clock = eta.StageClock(estimates)
    clock.start_stage("features")
    clock.end_stage("features", remaining=["baselines", "tune"])  # must not raise
    assert "features" in clock.actual
    assert clock.actual["features"] >= 0.0
    # Final stage with no remaining work also fine.
    clock.start_stage("baselines")
    clock.end_stage("baselines", remaining=[])
    assert "baselines" in clock.actual
