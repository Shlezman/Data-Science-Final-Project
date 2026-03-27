"""
processing_engine.evaluation.evaluate
======================================
Main evaluation script for the SentiSense relevance scoring pipeline.

Usage — auto-discover all installed Ollama models (recommended)
----------------------------------------------------------------
::

    python -m processing_engine.evaluation.evaluate --all-models

    # With explicit output directory
    python -m processing_engine.evaluation.evaluate \\
        --all-models \\
        --output evaluation/results/

Usage — explicit model list
----------------------------
::

    python -m processing_engine.evaluation.evaluate \\
        --models qwen2.5:14b llama3.1:8b mistral:7b

Usage — single model
---------------------
::

    python -m processing_engine.evaluation.evaluate \\
        --golden  evaluation/golden_dataset.csv \\
        --models  qwen2.5:14b \\
        --output  evaluation/results/

Usage — dry run (validate CSV only, no LLM calls)
---------------------------------------------------
::

    python -m processing_engine.evaluation.evaluate --dry-run

Model discovery
---------------
When ``--all-models`` is passed (or when ``--models`` is omitted and the
``SENTISENSE_OLLAMA_MODEL`` env var is not set), the script calls
``ollama list`` and parses every model name from the output.  Models that
fail to start are skipped with a warning so the rest can still complete.

When multiple models are evaluated, a leaderboard is printed at the end
and saved to ``results/leaderboard.md``.

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
import shutil
import subprocess
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

from processing_engine.engine import process_single_observation, reset_graph  # noqa: E402
from processing_engine.evaluation.metrics import (                             # noqa: E402
    CATEGORY_COLUMNS,
    CATEGORY_NAMES,
    compute_all_metrics,
)


# ═══════════════════════════════════════════════════════════════════════
# Ollama model discovery
# ═══════════════════════════════════════════════════════════════════════


def discover_ollama_models() -> list[str]:
    """
    Return the list of models currently installed in the local Ollama instance.

    Runs ``ollama list`` and parses its tabular output.  The first column of
    every non-header line is treated as a model name.

    ``ollama list`` output example::

        NAME                    ID              SIZE      MODIFIED
        qwen2.5:14b             abc1234567ef    9.0 GB    3 weeks ago
        llama3.1:8b             def7654321ab    4.7 GB    5 days ago
        mistral:7b              aaa000111bbb    4.1 GB    2 months ago

    Returns
    -------
    list[str]
        Ordered list of model name strings (e.g. ``["qwen2.5:14b", ...]``).

    Raises
    ------
    RuntimeError
        If ``ollama`` is not found on PATH or the subprocess returns a
        non-zero exit code.
    """
    if shutil.which("ollama") is None:
        raise RuntimeError(
            "ollama executable not found on PATH. "
            "Install Ollama from https://ollama.com and ensure it is running."
        )

    try:
        proc = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("'ollama list' timed out after 15 seconds.") from exc

    if proc.returncode != 0:
        raise RuntimeError(
            f"'ollama list' exited with code {proc.returncode}.\n"
            f"stderr: {proc.stderr.strip()}"
        )

    models: list[str] = []
    lines = proc.stdout.splitlines()

    for line in lines[1:]:          # skip header row
        line = line.strip()
        if not line:
            continue
        # First whitespace-delimited token is the model name
        name = line.split()[0]
        if name:
            models.append(name)

    if not models:
        raise RuntimeError(
            "'ollama list' returned no models. "
            "Pull at least one model first, e.g.: ollama pull qwen2.5:14b"
        )

    return models


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


def check_model_supports_tools(model_name: str, base_url: str) -> tuple[bool, str]:
    """
    Send a minimal tool-calling request to Ollama and return whether the
    model supports function/tool calling.

    Uses the ``ollama`` Python client (already installed as a transitive
    dependency of ``langchain-ollama``) so the request format is guaranteed
    to match what Ollama expects.

    Returns
    -------
    (True, "")
        Model responded successfully — tool calling works.
    (False, reason)
        The model returned an error that mentions it does not support tools.
        ``reason`` contains the server's error message.

    Raises
    ------
    RuntimeError
        If Ollama is unreachable (connection refused, timeout, etc.) —
        this is a server problem, not a model capability problem.
    """
    import ollama  # transitive dep via langchain-ollama

    client = ollama.Client(host=base_url)

    try:
        client.chat(
            model=model_name,
            messages=[{"role": "user", "content": "ping"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "probe",
                        "description": "capability probe",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "input": {
                                    "type": "string",
                                    "description": "probe input",
                                }
                            },
                            "required": ["input"],
                        },
                    },
                }
            ],
        )
        return True, ""
    except ollama.ResponseError as exc:
        # Only hard-skip when Ollama explicitly says the model has no tool support.
        # Other errors (OOM, timeout, etc.) are treated as warnings — we still
        # attempt the evaluation and let per-headline error handling take over.
        if "does not support tools" in exc.error:
            return False, exc.error
        # Non-capability error: warn but allow the run to proceed.
        print(f"\n  ⚠  pre-flight warning for {model_name}: {exc.error}")
        print("     Proceeding anyway — per-headline errors will be recorded.")
        return True, ""


async def run_pipeline_on_dataset(
    golden_rows: list[dict[str, Any]],
    model_name: str,
) -> list[dict[str, Any]]:
    """
    Run the SentiSense pipeline on every headline in the golden dataset.

    Sets ``SENTISENSE_OLLAMA_MODEL`` to ``model_name`` before running and
    resets the compiled graph singleton so the new model is actually used.
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
    reset_graph()  # force LangGraph to rebuild with the new model
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
            "Use --all-models to benchmark every model installed in Ollama, "
            "or supply an explicit list with --models. "
            "Computes MAE, Within-1/2 Accuracy, and Pearson r per category. "
            "A leaderboard is printed and saved whenever more than one model "
            "is evaluated."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Auto-discover and benchmark all installed Ollama models\n"
            "  python -m processing_engine.evaluation.evaluate --all-models\n\n"
            "  # Explicit model list\n"
            "  python -m processing_engine.evaluation.evaluate \\\n"
            "      --models qwen2.5:14b llama3.1:8b mistral:7b\n\n"
            "  # Single model\n"
            "  python -m processing_engine.evaluation.evaluate --models qwen2.5:14b\n\n"
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

    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--models",
        type=str,
        nargs="+",
        metavar="MODEL",
        help=(
            "One or more Ollama model names to evaluate, separated by spaces. "
            "Mutually exclusive with --all-models. "
            "Example: --models qwen2.5:14b llama3.1:8b mistral:7b"
        ),
    )
    model_group.add_argument(
        "--all-models",
        action="store_true",
        dest="all_models",
        help=(
            "Discover all models installed in the local Ollama instance via "
            "'ollama list' and evaluate every one of them. "
            "Mutually exclusive with --models."
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
) -> dict[str, dict[str, float]] | None:
    """
    Run the full evaluation pipeline for a single model.

    Returns the metrics dict (same structure as ``compute_all_metrics``),
    or ``None`` if the model does not support tool calling.
    """
    # Pre-flight: verify the model supports tool calling before running 26 headlines
    base_url = os.environ.get("SENTISENSE_OLLAMA_BASE_URL", "http://localhost:11434")
    print(f"\n[pre-flight] Checking tool support for {model_name}…", end=" ", flush=True)
    supported, reason = check_model_supports_tools(model_name, base_url=base_url)
    if not supported:
        print(f"✗ SKIPPED ({reason})")
        print(
            f"  ⚠ {model_name} does not support tool calling ({reason}).\n"
            f"  The SentiSense pipeline requires tool-capable models.\n"
            f"  Skipping this model — no results written."
        )
        return None
    print("✓ supported")

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

    # 2. Resolve model list
    #    Priority: --all-models > --models > SENTISENSE_OLLAMA_MODEL env var > hardcoded default
    if args.all_models:
        print("\nDiscovering installed Ollama models via 'ollama list'…")
        try:
            models = discover_ollama_models()
            print(f"Found {len(models)} model(s): {', '.join(models)}")
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
    elif args.models:
        models = args.models
    else:
        # Neither flag given — try auto-discovery, fall back to env/default
        env_model = os.environ.get("SENTISENSE_OLLAMA_MODEL")
        if env_model:
            models = [env_model]
            print(f"\nUsing model from SENTISENSE_OLLAMA_MODEL: {env_model}")
        else:
            print("\nNo --models or --all-models specified. "
                  "Attempting auto-discovery via 'ollama list'…")
            try:
                models = discover_ollama_models()
                print(f"Found {len(models)} model(s): {', '.join(models)}")
            except RuntimeError:
                models = ["qwen2.5:14b"]
                print(
                    "Auto-discovery failed. Falling back to default model: "
                    f"{models[0]}\n"
                    "  Tip: pass --models <name> or --all-models explicitly."
                )

    print(f"\n{'━' * 60}")
    print(f"  Evaluating {len(models)} model(s): {', '.join(models)}")
    print(f"{'━' * 60}")

    # 2. Evaluate each model sequentially
    all_metrics: dict[str, dict[str, dict[str, float]]] = {}
    skipped: list[str] = []
    for model_name in models:
        metrics = await evaluate_one_model(golden_rows, model_name, args.output)
        if metrics is None:
            skipped.append(model_name)
        else:
            all_metrics[model_name] = metrics

    if skipped:
        print(f"\n⚠ Skipped {len(skipped)} model(s) (no tool support): {', '.join(skipped)}")

    # 3. If multiple models were evaluated, print a final leaderboard
    if len(all_metrics) > 1:
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
