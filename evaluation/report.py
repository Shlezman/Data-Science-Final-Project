"""
evaluation.report
==================
Leaderboard and per-model report generator.

Reads all ``*_metrics.json`` files from the results directory and
produces:
  1. A ranked leaderboard table (stdout + Markdown file).
  2. A per-model breakdown table (stdout + Markdown file).

Usage
-----
After running ``evaluate.py`` for one or more models::

    python -m evaluation.report \\
        --results evaluation/results/ \\
        --output  evaluation/results/leaderboard.md

The generated Markdown can be pasted directly into
``EVALUATION_REPORT.md`` Sections 5 and 6.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when run as a script
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ENGINE_ROOT = _HERE.parent
if str(_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENGINE_ROOT))

from evaluation.metrics import CATEGORY_COLUMNS, CATEGORY_NAMES  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
# JSON loader
# ═══════════════════════════════════════════════════════════════════════


def load_all_metrics(results_dir: Path) -> list[dict[str, Any]]:
    """
    Load all ``*_metrics.json`` files from ``results_dir``.

    Returns a list of payload dicts, each containing:
      - ``model``   (str)
      - ``metrics`` (dict — same structure as ``compute_all_metrics`` output)
    """
    json_files = sorted(results_dir.glob("*_metrics.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No *_metrics.json files found in {results_dir}.\n"
            "Run evaluate.py first to generate results."
        )

    payloads: list[dict[str, Any]] = []
    for path in json_files:
        with open(path, encoding="utf-8") as f:
            payloads.append(json.load(f))
        print(f"Loaded: {path.name}")

    return payloads


# ═══════════════════════════════════════════════════════════════════════
# Leaderboard
# ═══════════════════════════════════════════════════════════════════════


def build_leaderboard(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Build a ranked leaderboard from all model metrics payloads.

    Returns a list of dicts sorted by composite_score descending.
    Each dict has keys:
      rank, model, composite_score, avg_mae, avg_within1, avg_within2,
      avg_pearson_r, best_category, worst_category
    """
    rows: list[dict[str, Any]] = []

    for payload in payloads:
        model = payload["model"]
        avg = payload["metrics"]["average"]

        # Find best and worst category by Within-1 Accuracy
        cat_within1 = {
            col: payload["metrics"][col]["within1"]
            for col in CATEGORY_COLUMNS
        }
        best_col  = max(cat_within1, key=cat_within1.__getitem__)
        worst_col = min(cat_within1, key=cat_within1.__getitem__)

        col_to_name = dict(zip(CATEGORY_COLUMNS, CATEGORY_NAMES))

        rows.append({
            "model":           model,
            "composite_score": avg["composite_score"],
            "avg_mae":         avg["mae"],
            "avg_within1":     avg["within1"],
            "avg_within2":     avg["within2"],
            "avg_pearson_r":   avg["pearson_r"],
            "best_category":   col_to_name[best_col],
            "worst_category":  col_to_name[worst_col],
        })

    # Sort by composite score descending
    rows.sort(key=lambda r: r["composite_score"], reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i

    return rows


def format_leaderboard_markdown(rows: list[dict[str, Any]]) -> str:
    """Render the leaderboard as a Markdown table."""
    lines = [
        "## Model Leaderboard",
        "",
        "Ranked by **Composite Score** (average Within-1 Accuracy across 6 categories).",
        "",
        "| Rank | Model | Composite ↑ | Avg MAE ↓ | Avg W-1 ↑ | Avg W-2 ↑ | Avg Pearson r ↑ | Best category | Worst category |",
        "|------|-------|-------------|-----------|-----------|-----------|-----------------|---------------|----------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['rank']} "
            f"| `{r['model']}` "
            f"| {r['composite_score']:.1%} "
            f"| {r['avg_mae']:.3f} "
            f"| {r['avg_within1']:.1%} "
            f"| {r['avg_within2']:.1%} "
            f"| {r['avg_pearson_r']:.3f} "
            f"| {r['best_category']} "
            f"| {r['worst_category']} |"
        )
    return "\n".join(lines)


def print_leaderboard(rows: list[dict[str, Any]]) -> None:
    """Print the leaderboard to stdout."""
    print(f"\n{'═' * 100}")
    print("  MODEL LEADERBOARD  (ranked by Composite Score = avg Within-1 Accuracy)")
    print(f"{'═' * 100}")
    print(
        f"  {'Rank':>4}  {'Model':<22} {'Composite':>10} {'Avg MAE':>8} "
        f"{'Avg W-1':>8} {'Avg W-2':>8} {'Pearson r':>10}  "
        f"{'Best category':<26} {'Worst category'}"
    )
    print(f"  {'-' * 96}")
    for r in rows:
        print(
            f"  {r['rank']:>4}  {r['model']:<22} {r['composite_score']:>10.1%} "
            f"{r['avg_mae']:>8.3f} {r['avg_within1']:>8.1%} {r['avg_within2']:>8.1%} "
            f"{r['avg_pearson_r']:>10.3f}  {r['best_category']:<26} {r['worst_category']}"
        )
    print(f"{'═' * 100}\n")


# ═══════════════════════════════════════════════════════════════════════
# Per-model breakdown
# ═══════════════════════════════════════════════════════════════════════


def format_model_breakdown_markdown(payload: dict[str, Any]) -> str:
    """Render a per-category breakdown for one model as Markdown."""
    model = payload["model"]
    metrics = payload["metrics"]

    lines = [
        f"### `{model}`",
        "",
        "| Category | MAE ↓ | Within-1 ↑ | Within-2 ↑ | Pearson r ↑ |",
        "|----------|-------|------------|------------|-------------|",
    ]

    for col, name in zip(CATEGORY_COLUMNS, CATEGORY_NAMES):
        m = metrics[col]
        lines.append(
            f"| {name} "
            f"| {m['mae']:.3f} "
            f"| {m['within1']:.1%} "
            f"| {m['within2']:.1%} "
            f"| {m['pearson_r']:.3f} |"
        )

    avg = metrics["average"]
    lines += [
        f"| **Average** "
        f"| **{avg['mae']:.3f}** "
        f"| **{avg['within1']:.1%}** "
        f"| **{avg['within2']:.1%}** "
        f"| **{avg['pearson_r']:.3f}** |",
        "",
        f"**Composite Score:** {avg['composite_score']:.1%}",
        "",
    ]
    return "\n".join(lines)


def format_all_breakdowns_markdown(payloads: list[dict[str, Any]]) -> str:
    """Render per-model breakdowns for all models, sorted by composite score."""
    sorted_payloads = sorted(
        payloads,
        key=lambda p: p["metrics"]["average"]["composite_score"],
        reverse=True,
    )
    sections = ["## Per-Model Results", ""]
    for payload in sorted_payloads:
        sections.append(format_model_breakdown_markdown(payload))
    return "\n".join(sections)


# ═══════════════════════════════════════════════════════════════════════
# CLI entrypoint
# ═══════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a leaderboard and per-model breakdown from evaluation results. "
            "Reads all *_metrics.json files in the results directory."
        )
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("evaluation/results"),
        help="Directory containing *_metrics.json files from evaluate.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/leaderboard.md"),
        help="Output path for the generated Markdown report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load all model metrics
    payloads = load_all_metrics(args.results)

    # Build and print leaderboard
    leaderboard_rows = build_leaderboard(payloads)
    print_leaderboard(leaderboard_rows)

    # Generate Markdown
    leaderboard_md = format_leaderboard_markdown(leaderboard_rows)
    breakdowns_md  = format_all_breakdowns_markdown(payloads)

    full_report = "\n\n".join([
        "# SentiSense — Model Evaluation Results",
        f"*Generated automatically by `report.py` from {len(payloads)} model run(s).*",
        leaderboard_md,
        breakdowns_md,
    ])

    # Save Markdown
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(full_report, encoding="utf-8")
    print(f"Markdown report saved to {args.output}")


if __name__ == "__main__":
    main()
