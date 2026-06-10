"""Phase 0 & 1 unit tests — cutoff enforcement + connection safety. No DB required.

Run (server-side or anywhere base deps installed):
    uv run pytest tests/test_phase0_phase1.py -v
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys

import pytest

from sentisense import constants


# ─────────────────────────────────────────────────────────────────────
# Cutoff constant
# ─────────────────────────────────────────────────────────────────────

def test_cutoff_constant_is_2023_10_07():
    assert constants.CUTOFF_DATE == _dt.date(2023, 10, 7)
    assert constants.CUTOFF_DATE_ISO == "2023-10-07"


def test_score_columns_contract():
    # DB-canonical order; sentiment last.
    assert constants.SCORE_COLUMNS[-1] == "global_sentiment"
    assert constants.DB_RELEVANCE_COLUMNS[0] == "relevance_politics"
    assert len(constants.DB_RELEVANCE_COLUMNS) == 6
    # Cross-layer maps are total over the 6 relevance cols.
    for col in constants.DB_RELEVANCE_COLUMNS:
        assert col in constants.DB_TO_ENGINE_KEY
        assert col in constants.DB_TO_GOLDEN_KEY


# ─────────────────────────────────────────────────────────────────────
# Scoring wrapper always pins the hard cutoff
# ─────────────────────────────────────────────────────────────────────

def test_score_command_always_enforces_cutoff():
    from sentisense.ingest import score

    args = argparse.Namespace(
        concurrency=50, headlines_per_call=20, date_from="", limit=0, dry_run=True,
    )
    cmd = score.build_command(args)
    # --date-to 2023-10-07 must be present and adjacent.
    assert "--date-to" in cmd
    assert cmd[cmd.index("--date-to") + 1] == "2023-10-07"
    assert "--fast" in cmd  # production vLLM requires the fast path
    assert "--dry-run" in cmd


def test_score_command_passes_optional_window():
    from sentisense.ingest import score

    args = argparse.Namespace(
        concurrency=32, headlines_per_call=0, date_from="2015-01-01", limit=100, dry_run=False,
    )
    cmd = score.build_command(args)
    assert cmd[cmd.index("--date-from") + 1] == "2015-01-01"
    assert cmd[cmd.index("--limit") + 1] == "100"
    assert "--dry-run" not in cmd


# ─────────────────────────────────────────────────────────────────────
# Backfill wrapper command shape
# ─────────────────────────────────────────────────────────────────────

def test_backfill_command_shape():
    from sentisense.ingest import backfill

    args = argparse.Namespace(
        window=7, empty_streak=2, pages=100, max_days=3650,
        start_before="2023-10-08", dry_run=True,
    )
    cmd = backfill.build_command(args)
    assert cmd[cmd.index("--window") + 1] == "7"
    assert cmd[cmd.index("--max-days") + 1] == "3650"
    assert cmd[cmd.index("--start-before") + 1] == "2023-10-08"
    assert "--dry-run" in cmd


# ─────────────────────────────────────────────────────────────────────
# Boundary validation of operator-supplied date args (fail fast)
# ─────────────────────────────────────────────────────────────────────

def test_score_rejects_postcutoff_date_from(monkeypatch):
    from sentisense.ingest import score

    monkeypatch.setattr(sys, "argv", ["score", "--date-from", "2024-01-01"])
    with pytest.raises(SystemExit):  # parser.error → SystemExit, before any subprocess
        score.main()


def test_score_rejects_malformed_date_from(monkeypatch):
    from sentisense.ingest import score

    monkeypatch.setattr(sys, "argv", ["score", "--date-from", "07-10-2023"])
    with pytest.raises(SystemExit):
        score.main()


def test_backfill_rejects_malformed_start_before(monkeypatch):
    from sentisense.ingest import backfill

    monkeypatch.setattr(sys, "argv", ["backfill", "--start-before", "not-a-date"])
    with pytest.raises(SystemExit):
        backfill.main()


# ─────────────────────────────────────────────────────────────────────
# Coverage report SQL is cutoff-bound (parameterized, never concatenated)
# ─────────────────────────────────────────────────────────────────────

def test_coverage_sql_is_cutoff_parameterized():
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("pandas")
    from sentisense.ingest import coverage_report as cov

    for sql in (cov._SUMMARY_SQL, cov._SCORED_SQL, cov._PER_MONTH_SQL,
                cov._NEWS_DATES_SQL, cov._CLASS_BALANCE_SQL):
        rendered = str(sql)
        assert ":cutoff" in rendered, f"cutoff bind missing in: {rendered[:60]}"
        # No literal cutoff string concatenated into SQL.
        assert "2023-10-07" not in rendered


def test_scored_sql_filters_active_model_and_validation():
    pytest.importorskip("sqlalchemy")
    from sentisense.ingest import coverage_report as cov

    rendered = str(cov._SCORED_SQL)
    assert ":model" in rendered
    assert "validation_passed = TRUE" in rendered


# ─────────────────────────────────────────────────────────────────────
# Connection: env-var driven, fail-fast, psycopg-v3 dialect
# ─────────────────────────────────────────────────────────────────────

def test_connection_requires_env(monkeypatch):
    pytest.importorskip("sqlalchemy")
    from sentisense.db import connection

    monkeypatch.delenv("SENTISENSE_DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="SENTISENSE_DATABASE_URL is not set"):
        connection.get_connection_url()


def test_connection_blank_env_fails(monkeypatch):
    pytest.importorskip("sqlalchemy")
    from sentisense.db import connection

    monkeypatch.setenv("SENTISENSE_DATABASE_URL", "   ")
    with pytest.raises(RuntimeError):
        connection.get_connection_url()


@pytest.mark.parametrize("raw,expected", [
    ("postgresql://u:p@h:5432/db", "postgresql+psycopg://u:p@h:5432/db"),
    ("postgres://u:p@h:5432/db", "postgresql+psycopg://u:p@h:5432/db"),
    ("postgresql+psycopg2://u:p@h:5432/db", "postgresql+psycopg://u:p@h:5432/db"),
    ("postgresql+psycopg://u:p@h:5432/db", "postgresql+psycopg://u:p@h:5432/db"),
])
def test_connection_url_normalises_to_psycopg_v3(monkeypatch, raw, expected):
    pytest.importorskip("sqlalchemy")
    from sentisense.db import connection

    monkeypatch.setenv("SENTISENSE_DATABASE_URL", raw)
    assert connection.get_connection_url() == expected


def test_connection_url_never_embeds_default_password(monkeypatch):
    """Regression guard: unset env must NOT silently return a dev-password DSN."""
    pytest.importorskip("sqlalchemy")
    from sentisense.db import connection

    monkeypatch.delenv("SENTISENSE_DATABASE_URL", raising=False)
    try:
        url = connection.get_connection_url()
    except RuntimeError:
        return  # correct: refused to fabricate a DSN
    assert "sentisense_dev" not in url, "must never embed the dev password"
