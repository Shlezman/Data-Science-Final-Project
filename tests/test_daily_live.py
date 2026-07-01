"""WS1 orchestrator: TASE-calendar skip logic + the single-run flock guard."""

from __future__ import annotations

import datetime as dt
import importlib.util as _u
from pathlib import Path

import pytest

pytest.importorskip("loguru")
pytest.importorskip("sentisense")


def _load_daily_live():
    p = Path(__file__).resolve().parent.parent / "scripts" / "daily_live.py"
    spec = _u.spec_from_file_location("daily_live", p)
    m = _u.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_trading_day_skips_friday_saturday():
    m = _load_daily_live()
    assert m.is_trading_day(dt.date(2026, 6, 26)) is False   # Friday
    assert m.is_trading_day(dt.date(2026, 6, 27)) is False   # Saturday
    assert m.is_trading_day(dt.date(2026, 6, 28)) is True    # Sunday (TASE open)
    assert m.is_trading_day(dt.date(2026, 6, 29)) is True    # Monday


def test_holiday_file_excludes_a_trading_day(tmp_path, monkeypatch):
    m = _load_daily_live()
    hol = tmp_path / "tase_holidays.txt"
    hol.write_text("2026-06-29\n")
    monkeypatch.setattr(m, "_HOLIDAYS_PATH", hol)
    assert m.is_trading_day(dt.date(2026, 6, 29)) is False    # Monday, but listed holiday
    assert m.is_trading_day(dt.date(2026, 6, 28)) is True     # Sunday, not listed


def test_flock_guard_blocks_second_runner(tmp_path, monkeypatch):
    m = _load_daily_live()
    monkeypatch.setattr(m, "_LOGS", tmp_path)
    monkeypatch.setattr(m, "_LOCK_PATH", tmp_path / "daily_live.lock")
    first = m._acquire_lock()
    assert first is not None
    second = m._acquire_lock()          # same lock held → must refuse
    assert second is None
    import fcntl
    fcntl.flock(first, fcntl.LOCK_UN)
    first.close()
    third = m._acquire_lock()           # released → acquirable again
    assert third is not None
    fcntl.flock(third, fcntl.LOCK_UN)
    third.close()
