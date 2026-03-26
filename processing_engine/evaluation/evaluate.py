"""
processing_engine.evaluation.evaluate
======================================
Main evaluation script for the SentiSense relevance scoring pipeline.

Usage — single model
---------------------
::

    python -m processing_engine.evaluation.evaluate \\
        --golden  evaluation/golden_dataset.csv \\
        --models  qwen2.5:14b \\
        --output  evaluation/results/

Usage — multiple models in one command
----------------------------------------
::

    python -m processing_engine.evaluation.evaluate \\
        --golden  evaluation/golden_dataset.csv \\
        --models  qwen2.5:14b llama3.1:8b mistral:7b \\
        --output  evaluation/results/

When multiple models are provided, each is evaluated sequentially and
a leaderboard is printed at the end comparing all models.

For each model this script will:
1. Load the golden dataset (headline + 6 gold relevance scores).
2. Run each headline through the full SentiSense pipeline.
3. Compute MAE, Within-1/2 Accuracy, and Pearson r per category.
4. Save a predictions CSV to ``results/<model>_predictions.csv``.
5. Save metrics to ``results/<model>_metrics.json``.
6. Print a per-category metrics summary to stdout.

The ``global_sentiment`` output from the pipeline is saved in the
predictions CSV for reference but is NOT included in any metric
computation.

Golden dataset CSV format
--------------------------
Required columns::

    headline, politics_government, economy_finance, security_military,
    health_medicine, science_climate, technology

Optional columns (passed through to the pipeline)::

    date, source, hour, popularity

If optional columns are absent, sensible defaults are used.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path when run as a script
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # processing_engine/evaluation/
_ENGINE_ROOT = _HERE.parent.parent               # project root
if str(_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENGINE_ROOT))

from processing_engine.engine import process_single_observation  # noqa: E402
from processing_engine.evaluation.metrics import (               # noqa: E402
    CATEGORY_COLUMNS,
    CATEGORY_NAMES,
    compute_all_metrics,
)


# ═══════════════════════════════════════════════════════════════════════
# CSV I/O helpers
# ═══════════════════════════════════════════════════════════════════════

# Golden dataset column names (match the real CSV schema exactly)
GOLD_COLUMNS = [
    "politics_government",
    "economy_finance",
    "security_military",
    "health_medicine",
    "science_climate",
    "technology",
]
# Predicted and error columns written to the predictions CSV
PRED_COLUMNS = [f"pred_{col}" for col in GOLD_COLUMNS]
ERR_COLUMNS  = [f"err_{col}"  for col in GOLD_COLUMNS]


def load_golden_dataset(path: Path) -> list[dict[str, Any]]:
    """
    Load the golden dataset CSV.

    Returns a list of dicts, one per headline.  Each dict contains at
    minimum ``headline`` and the 6 named category columns (as ints).
    """
    rows: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):  # row 1 = header
            # Validate required columns
            missing = [c for c in ["headline"] + GOLD_COLUMNS if c not in row]
            if missing:
                raise ValueError(
                    f"Row {i}: missing required columns: {missing}\n"
                    f"Expected: headline, {', '.join(GOLD_COLUMNS)}"
                )
            # Coerce gold scores to int
            for col in GOLD_COLUMNS:
                try:
                    row[col] = int(row[col])
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"Row {i}, column '{col}': expected integer, "
                        f"got {row[col]!r}"
                    ) from exc
                if not (0 <= row[col] <= 10):
                    raise ValueError(
                        f"Row {i}, column '{col}': value {row[col]} "
                        f"out of range [0, 10]"
                    )
            rows.append(row)

    if not rows:
        raise ValueError(f"Golden dataset is empty: {path}")

    print(f"Loaded {len(rows)} headlines from {path}")
    return rows


def save_predictions(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Save the full predictions CSV (gold + predicted + error per category).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = (
        ["headline"]
        + GOLD_COLUMNS
        + PRED_COLUMNS
        + ERR_COLUMNS
        + ["global_sentiment", "validation_passed", "pipeline_error"]
    )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Predictions saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Pipeline runner
# ═══════════════════════════════════════════════════════════════════════


async def run_pipeline_on_dataset(
    golden_rows: list[dict[str, Any]],
    model_name: str,
) -> list[dict[str, Any]]:
    """
    Run the SentiSense pipeline on every headline in the golden dataset.

    Sets ``SENTISENSE_OLLAMA_MODEL`` to ``model_name`` before running.
    Processes headlines sequentially to avoid overloading Ollama.

    Returns a list of result dicts, one per headline, containing:
      - All original golden dataset columns (headline + 6 gold scores)
      - ``pred_<category>`` columns (pipeline relevance scores)
      - ``err_<category>`` columns (|predicted − gold|)
      - ``global_sentiment`` (pipeline output, not evaluated)
      - ``validation_passed`` (pipeline flag)
      - ``pipeline_error`` (error string if pipeline failed, else "")
    """
    os.environ["SENTISENSE_OLLAMA_MODEL"] = model_name
    print(f"\nRunning pipeline with model: {model_name}")
    print(f"Processing {len(golden_rows)} headlines sequentially…\n")

    results: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    for idx, row in enumerate(golden_rows, start=1):
        observation = {
            "headline":   row.get("headline", ""),
            "date":       "2000-01-01",
            "source":     "golden_dataset",
            "hour":       "00:00",
            "popularity": "",
        }

        t0 = time.perf_counter()
        try:
            output = await process_single_observation(observation)
            elapsed = time.perf_counter() - t0

            result = dict(row)  # copy all original columns (headline + gold scores)

            # Map pipeline output relevance_category_N → named pred/err columns
            for i, col in enumerate(CATEGORY_COLUMNS, start=1):
                pred_val = output.get(f"relevance_category_{i}", 0)
                result[f"pred_{col}"] = pred_val
                result[f"err_{col}"]  = abs(pred_val - row[col])

            result["global_sentiment"]  = output.get("global_sentiment", 0)
            result["validation_passed"] = output.get("validation_passed", False)
            result["pipeline_error"]    = "; ".join(output.get("errors", []))

            status = "✓" if result["validation_passed"] else "✗"
            print(
                f"  [{idx:>4}/{len(golden_rows)}] {status} "
                f"({elapsed:.1f}s) {row['headline'][:60]}"
            )

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            result = dict(row)
            for col in CATEGORY_COLUMNS:
                result[f"pred_{col}"] = 0
                result[f"err_{col}"]  = row[col]  # max error = gold score
            result["global_sentiment"]  = 0
            result["validation_passed"] = False
            result["pipeline_error"]    = str(exc)
            print(
                f"  [{idx:>4}/{len(golden_rows)}] ✗ "
                f"({elapsed:.1f}s) ERROR: {exc} | {row['headline'][:40]}"
            )

        results.append(result)

    total = time.perf_counter() - t_start
    print(f"\nCompleted {len(results)} headlines in {total:.1f}s "
          f"({total/len(results):.1f}s avg)")
    return results


# ═══════════════════════════════════════════════════════════════════════
# Metrics computation
# ═══════════════════════════════════════════════════════════════════════


def extract_scores(
    results: list[dict[str, Any]],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """
    Extract predicted and gold score lists from result rows.

    Returns
    -------
    predictions : dict[str, list[float]]
        Keys: category slug names (e.g. ``"politics_government"``).
    gold_labels : dict[str, list[float]]
        Keys: category slug names (e.g. ``"politics_government"``).
    """
    predictions: dict[str, list[float]] = {c: [] for c in CATEGORY_COLUMNS}
    gold_labels: dict[str, list[float]] = {c: [] for c in CATEGORY_COLUMNS}

    for col in CATEGORY_COLUMNS:
        for row in results:
            predictions[col].append(float(row.get(f"pred_{col}", 0)))
            gold_labels[col].append(float(row.get(col, 0)))

    return predictions, gold_labels


def print_metrics_summary(
    metrics: dict[str, dict[str, float]],
    model_name: str,
) -> None:
    """Print a formatted metrics table to stdout."""
    print(f"\n{'═' * 80}")
    print(f"  RESULTS — {model_name}")
    print(f"{'═' * 80}")
    print(
        f"  {'Category':<28} {'MAE':>6} {'W-1 Acc':>8} {'W-2 Acc':>8} {'Pearson r':>10}"
    )
    print(f"  {'-' * 64}")

    for col, name in zip(CATEGORY_COLUMNS, CATEGORY_NAMES):
        m = metrics[col]
        print(
            f"  {name:<28} {m['mae']:>6.3f} {m['within1']:>8.1%} "
            f"{m['within2']:>8.1%} {m['pearson_r']:>10.3f}"
        )

    print(f"  {'-' * 64}")
    avg = metrics["average"]
    print(
        f"  {'AVERAGE':<28} {avg['mae']:>6.3f} {avg['within1']:>8.1%} "
        f"{avg['within2']:>8.1%} {avg['pearson_r']:>10.3f}"
    )
    print(f"\n  Composite Score (avg Within-1): {avg['composite_score']:.1%}")
    print(f"{'═' * 80}\n")


def save_metrics_json(
    metrics: dict[str, dict[str, float]],
    model_name: str,
    output_dir: Path,
) -> None:
    """Save metrics as JSON for later leaderboard aggregation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace(":", "_").replace("/", "_")
    path = output_dir / f"{safe_name}_metrics.json"

    payload = {
        "model": model_name,
        "metrics": metrics,
        "category_names": dict(zip(CATEGORY_COLUMNS, CATEGORY_NAMES)),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Metrics JSON saved to {path}")


# ═══════════════════════════════════════════════════════════════════════
# CLI entrypoint
# ═══════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the SentiSense pipeline against a golden dataset. "
            "Accepts one or more model names via --models. "
            "Computes MAE, Within-1/2 Accuracy, and Pearson r per category. "
            "When multiple models are provided, prints a leaderboard at the end."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Single model\n"
            "  python -m processing_engine.evaluation.evaluate --models qwen2.5:14b\n\n"
            "  # Multiple models — evaluated sequentially, leaderboard at the end\n"
            "  python -m processing_engine.evaluation.evaluate \\\n"
            "      --models qwen2.5:14b llama3.1:8b mistral:7b\n\n"
            "  # Dry run — validate CSV without running the pipeline\n"
            "  python -m processing_engine.evaluation.evaluate --dry-run\n"
        ),
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path("processing_engine/evaluation/golden_dataset.csv"),
        help="Path to the golden dataset CSV file.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[os.environ.get("SENTISENSE_OLLAMA_MODEL", "qwen2.5:14b")],
        metavar="MODEL",
        help=(
            "One or more Ollama model names to evaluate, separated by spaces. "
            "Example: --models qwen2.5:14b llama3.1:8b mistral:7b"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("processing_engine/evaluation/results"),
        help="Directory to save predictions CSVs, metrics JSONs, and leaderboard.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Load the golden dataset and validate it, but do not run "
            "the pipeline. Useful for checking CSV format."
        ),
    )
    return parser.parse_args()


async def evaluate_one_model(
    golden_rows: list[dict[str, Any]],
    model_name: str,
    output_dir: Path,
) -> dict[str, dict[str, float]]:
    """
    Run the full evaluation pipeline for a single model.

    Returns the metrics dict (same structure as ``compute_all_metrics``).
    """
    # Run pipeline
    results = await run_pipeline_on_dataset(golden_rows, model_name=model_name)

    # Save predictions CSV
    safe_name = model_name.replace(":", "_").replace("/", "_")
    predictions_path = output_dir / f"{safe_name}_predictions.csv"
    save_predictions(results, predictions_path)

    # Compute metrics
    predictions, gold_labels = extract_scores(results)
    metrics = compute_all_metrics(predictions, gold_labels)

    # Print per-model summary
    print_metrics_summary(metrics, model_name=model_name)

    # Save metrics JSON
    save_metrics_json(metrics, model_name=model_name, output_dir=output_dir)

    return metrics


async def main() -> None:
    args = parse_args()

    # 1. Load and validate golden dataset (once, shared across all models)
    golden_rows = load_golden_dataset(args.golden)

    if args.dry_run:
        print(f"\nDry run complete. {len(golden_rows)} headlines validated.")
        print("Sample row:")
        print(json.dumps(golden_rows[0], ensure_ascii=False, indent=2))
        return

    models = args.models
    print(f"\n{'━' * 60}")
    print(f"  Evaluating {len(models)} model(s): {', '.join(models)}")
    print(f"{'━' * 60}")

    # 2. Evaluate each model sequentially
    all_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for model_name in models:
        metrics = await evaluate_one_model(golden_rows, model_name, args.output)
        all_metrics[model_name] = metrics

    # 3. If multiple models were evaluated, print a final leaderboard
    if len(models) > 1:
        # Import here to avoid circular dependency at module level
        from processing_engine.evaluation.report import (
            build_leaderboard,
            print_leaderboard,
            format_leaderboard_markdown,
        )

        # Build payloads in the same format report.py expects
        payloads = [
            {
                "model": model_name,
                "metrics": all_metrics[model_name],
                "category_names": dict(zip(CATEGORY_COLUMNS, CATEGORY_NAMES)),
            }
            for model_name in models
        ]

        leaderboard_rows = build_leaderboard(payloads)
        print_leaderboard(leaderboard_rows)

        # Save leaderboard Markdown
        leaderboard_md = format_leaderboard_markdown(leaderboard_rows)
        leaderboard_path = args.output / "leaderboard.md"
        leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
        leaderboard_path.write_text(leaderboard_md, encoding="utf-8")
        print(f"Leaderboard saved to {leaderboard_path}")


if __name__ == "__main__":
    asyncio.run(main())
