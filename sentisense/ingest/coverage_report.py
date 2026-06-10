"""Phase 1.3 — backfill coverage report (the Gate A artifact).

Reads the DB through the SQLAlchemy engine (:mod:`sentisense.db`) and emits a
markdown report covering, for data ``<= 2023-10-07`` only:

* earliest / latest ``raw_headlines.date`` reached (how far back backfill got),
* total raw headlines and total **successfully scored** rows (active model,
  ``validation_passed=TRUE``),
* per-month raw vs scored counts (so coverage gaps are visible),
* distinct news-date count (a proxy for trading-day coverage — the true TASE
  trading-day count is computed in Phase 2 after joining to the price calendar),
* class-balance preview from ``daily_features.ta125_up`` (often empty pre-Phase-2).

All queries are parameterized and cutoff-bounded. The operator runs this server-side
and pastes the report back (Gate A).

Run (server-side, operator):
    uv run python -m sentisense.ingest.coverage_report
"""

from __future__ import annotations

import datetime as _dt

import pandas as pd
from loguru import logger
from sqlalchemy import text

from sentisense.constants import (
    CUTOFF_DATE,
    CUTOFF_DATE_ISO,
    REPORTS_DIR,
    resolve_active_model,
)
from sentisense.db import get_engine

# Full per-model breakdown so the operator immediately sees an already-scored corpus.
_MODELS_SQL = text(
    """
    SELECT nv.model_name,
           COUNT(*) FILTER (WHERE nv.validation_passed) AS validated,
           COUNT(*)                                     AS total
    FROM nlp_vectors nv
    JOIN raw_headlines rh ON rh.id = nv.headline_id AND rh.date <= :cutoff
    GROUP BY nv.model_name
    ORDER BY validated DESC
    """
)

# Parameterized; :cutoff and :model are bound, never string-concatenated.
_SUMMARY_SQL = text(
    """
    SELECT
        MIN(rh.date) AS earliest_date,
        MAX(rh.date) AS latest_date,
        COUNT(*)     AS raw_total
    FROM raw_headlines rh
    WHERE rh.date <= :cutoff
    """
)

_SCORED_SQL = text(
    """
    SELECT COUNT(*) AS scored_total
    FROM raw_headlines rh
    JOIN nlp_vectors nv
        ON nv.headline_id = rh.id
       AND nv.model_name = :model
       AND nv.validation_passed = TRUE
    WHERE rh.date <= :cutoff
    """
)

_PER_MONTH_SQL = text(
    """
    SELECT
        to_char(date_trunc('month', rh.date), 'YYYY-MM') AS month,
        COUNT(*) AS raw_count,
        COUNT(nv.id) AS scored_count
    FROM raw_headlines rh
    LEFT JOIN nlp_vectors nv
        ON nv.headline_id = rh.id
       AND nv.model_name = :model
       AND nv.validation_passed = TRUE
    WHERE rh.date <= :cutoff
    GROUP BY 1
    ORDER BY 1
    """
)

_NEWS_DATES_SQL = text(
    """
    SELECT COUNT(DISTINCT rh.date) AS distinct_news_dates
    FROM raw_headlines rh
    WHERE rh.date <= :cutoff
    """
)

_CLASS_BALANCE_SQL = text(
    """
    SELECT
        SUM(CASE WHEN ta125_up IS TRUE  THEN 1 ELSE 0 END) AS up_days,
        SUM(CASE WHEN ta125_up IS FALSE THEN 1 ELSE 0 END) AS down_days,
        SUM(CASE WHEN ta125_up IS NULL  THEN 1 ELSE 0 END) AS unlabeled_days,
        COUNT(*) AS total_rows
    FROM daily_features
    WHERE date <= :cutoff
    """
)


def build_report() -> str:
    """Query the DB and render the coverage report as markdown.

    Returns:
        The full markdown report text.

    Raises:
        RuntimeError: If ``SENTISENSE_DATABASE_URL`` is unset (via get_engine).
    """
    engine = get_engine()
    active_model = resolve_active_model(engine)
    params = {"cutoff": CUTOFF_DATE, "model": active_model}

    with engine.connect() as conn:
        summary = conn.execute(_SUMMARY_SQL, params).mappings().one()
        scored = conn.execute(_SCORED_SQL, params).mappings().one()
        news_dates = conn.execute(_NEWS_DATES_SQL, {"cutoff": CUTOFF_DATE}).mappings().one()
        balance = conn.execute(_CLASS_BALANCE_SQL, {"cutoff": CUTOFF_DATE}).mappings().one()
        per_month = pd.read_sql(_PER_MONTH_SQL, conn, params=params)
        models = pd.read_sql(_MODELS_SQL, conn, params={"cutoff": CUTOFF_DATE})

    raw_total = int(summary["raw_total"] or 0)
    scored_total = int(scored["scored_total"] or 0)
    scored_pct = (scored_total / raw_total * 100) if raw_total else 0.0

    lines: list[str] = []
    lines.append(f"# SentiSense — Phase 1 Backfill Coverage Report (Gate A)\n")
    lines.append(f"- Dataset (read) model — auto-resolved: `{active_model}`")
    lines.append(f"- Hard cutoff: `<= {CUTOFF_DATE_ISO}` (applied to `raw_headlines.date`)\n")

    lines.append("## Scored-model breakdown (<= cutoff)")
    if models.empty:
        lines.append("_No nlp_vectors rows yet._\n")
    else:
        lines.append("| model_name | validated | total |")
        lines.append("|---|---:|---:|")
        for _, r in models.iterrows():
            lines.append(f"| `{r['model_name']}` | {int(r['validated'] or 0):,} | {int(r['total']):,} |")
        lines.append("\n_The pipeline models on the top-validated model above. If your corpus "
                     "is already scored, do NOT re-run the `score` stage locally (it would write "
                     "a different model_name); start at `--from embed`._\n")

    lines.append("## Corpus reach (<= cutoff)")
    lines.append(f"- Earliest headline date reached: **{summary['earliest_date']}**")
    lines.append(f"- Latest headline date (<= cutoff): **{summary['latest_date']}**")
    lines.append(f"- Raw headlines: **{raw_total:,}**")
    lines.append(f"- Successfully scored ({active_model}, validation_passed): "
                 f"**{scored_total:,}** ({scored_pct:.1f}% of raw)")
    lines.append(f"- Distinct news dates: **{int(news_dates['distinct_news_dates'] or 0):,}** "
                 "(proxy for trading-day coverage; true TASE Sun–Thu count computed in Phase 2)\n")

    lines.append("## Target label availability (daily_features.ta125_up, <= cutoff)")
    total_rows = int(balance["total_rows"] or 0)
    if total_rows == 0:
        lines.append("- `daily_features` has **no rows <= cutoff** — the target is not yet "
                     "materialised. Expected before Phase 2 feature engineering populates it.\n")
    else:
        up = int(balance["up_days"] or 0)
        down = int(balance["down_days"] or 0)
        unl = int(balance["unlabeled_days"] or 0)
        labeled = up + down
        up_pct = (up / labeled * 100) if labeled else 0.0
        lines.append(f"- Up days: **{up:,}**  |  Down days: **{down:,}**  |  Unlabeled (NULL): **{unl:,}**")
        lines.append(f"- Labeled total: **{labeled:,}**  (Up rate **{up_pct:.1f}%** — use for class weights)\n")

    lines.append("## Per-month coverage (raw vs scored)")
    if per_month.empty:
        lines.append("_No rows <= cutoff._\n")
    else:
        per_month["scored_pct"] = (
            per_month["scored_count"] / per_month["raw_count"].clip(lower=1) * 100
        ).round(1)
        lines.append("| month | raw | scored | scored % |")
        lines.append("|---|---:|---:|---:|")
        for _, r in per_month.iterrows():
            lines.append(f"| {r['month']} | {int(r['raw_count']):,} | "
                         f"{int(r['scored_count']):,} | {r['scored_pct']:.1f}% |")
        lines.append("")

    lines.append("## Gate A verification checklist")
    lines.append(f"- [ ] Latest date does not exceed {CUTOFF_DATE_ISO} (cutoff held).")
    lines.append("- [ ] Earliest date is as far back as the source allows (backfill saturated).")
    lines.append("- [ ] Scored % is high (>~95%); low months → run Phase 1.2 scoring or retry failures.")
    lines.append("- [ ] Distinct news dates sufficient for sequence modeling "
                 "(Phase 2 will report the true trading-day count vs the ~750 LSTM-viability bar).")
    return "\n".join(lines)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    # Static filename (no Date.now-style nondeterminism); overwrite on rerun.
    out = REPORTS_DIR / "phase1_coverage_report.md"
    out.write_text(report, encoding="utf-8")
    logger.info("Wrote coverage report → {}", out)
    print("\n" + report)


if __name__ == "__main__":
    main()
