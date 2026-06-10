"""End-to-end SentiSense Phase 1→6 orchestrator (the "run all features" script).

Stages, in order:
    backfill   → extend raw_headlines history backwards         (subprocess wrapper)
    score      → LLM-score unscored headlines <= cutoff         (subprocess wrapper)
    coverage   → Gate A coverage report                         (DB read)
    embed      → embed headlines (multilingual-e5) into cache   (embed extra)
    cluster    → causal per-day narrative features              (ml extra)
    features   → assemble leak-safe daily datasets (mt, ml)
    baselines  → majority / persistence / XGBoost (Phase 5)
    tune       → Optuna LSTM HPO (Phase 6, long-running, resumable)
    final      → Phase 7 sacred-holdout evaluation of the best trial

Examples (server-side, from repo root):
    uv run python -m sentisense.pipeline --dry-run
    uv run python -m sentisense.pipeline --stages embed,cluster,features,baselines
    uv run python -m sentisense.pipeline --from features          # skip ingest/embed
    uv run python -m sentisense.pipeline --only tune --trials 100

Heavy deps (torch, sentence-transformers, sklearn) import lazily per stage, so the
ingest-only stages run in the base env.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from loguru import logger

STAGES = ["backfill", "score", "coverage", "embed", "cluster",
          "features", "baselines", "tune", "final"]


def _run_module(mod: str, extra: list[str]) -> None:
    """Run a sentisense submodule as a subprocess (inherits env)."""
    cmd = [sys.executable, "-m", mod, *extra]
    logger.info("→ {}", " ".join(cmd))
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        raise RuntimeError(f"stage '{mod}' exited {completed.returncode}")


def _select_stages(args: argparse.Namespace) -> list[str]:
    if args.only:
        return [s.strip() for s in args.only.split(",")]
    if args.stages:
        return [s.strip() for s in args.stages.split(",")]
    start = STAGES.index(args.from_stage) if args.from_stage else 0
    end = STAGES.index(args.to_stage) + 1 if args.to_stage else len(STAGES)
    return STAGES[start:end]


def main() -> None:
    parser = argparse.ArgumentParser(description="SentiSense end-to-end pipeline orchestrator.")
    parser.add_argument("--stages", type=str, default="", help="Comma list to run (default: all).")
    parser.add_argument("--only", type=str, default="", help="Run exactly these comma-listed stages.")
    parser.add_argument("--from", dest="from_stage", choices=STAGES, default=None)
    parser.add_argument("--to", dest="to_stage", choices=STAGES, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Pass-through to ingest/embed stages.")
    parser.add_argument("--trials", type=int, default=0, help="Override Optuna trials for the tune stage.")
    parser.add_argument("--skip-final", action="store_true", help="Skip Phase 7 holdout eval.")
    args = parser.parse_args()

    selected = _select_stages(args)
    for s in selected:
        if s not in STAGES:
            parser.error(f"unknown stage '{s}'. Valid: {', '.join(STAGES)}")
    if args.skip_final and "final" in selected:
        selected.remove("final")
    logger.info("Pipeline stages: {}", " → ".join(selected))

    dry = ["--dry-run"] if args.dry_run else []
    narrative = None       # produced by 'cluster', consumed by 'features'
    mt = ml = None
    study = None

    for stage in selected:
        logger.info("══════ stage: {} ══════", stage)

        if stage == "backfill":
            _run_module("sentisense.ingest.backfill", dry)

        elif stage == "score":
            _run_module("sentisense.ingest.score", dry)

        elif stage == "coverage":
            _run_module("sentisense.ingest.coverage_report", [])

        elif stage == "embed":
            from sentisense.embed import embed_missing
            embed_missing(dry_run=args.dry_run)

        elif stage == "cluster":
            from sentisense.cluster import build_narrative_features
            narrative = build_narrative_features()
            if narrative is not None and not narrative.empty:
                logger.info("Narrative features: {} days", len(narrative))

        elif stage == "features":
            from sentisense.features import build_datasets
            mt, ml = build_datasets(extra_daily_features=narrative)

        elif stage == "baselines":
            if mt is None:
                from sentisense.features import build_datasets
                mt, ml = build_datasets(extra_daily_features=narrative)
            from sentisense.models.baselines import run_baselines
            run_baselines(mt)

        elif stage == "tune":
            if ml is None:
                from sentisense.features import build_datasets
                mt, ml = build_datasets(extra_daily_features=narrative)
            from sentisense.config import OPTUNA_TRIALS
            from sentisense.hpo import run_hpo
            n_trials = args.trials if args.trials > 0 else OPTUNA_TRIALS
            study = run_hpo(ml, n_trials=n_trials)

        elif stage == "final":
            if ml is None:
                from sentisense.features import build_datasets
                mt, ml = build_datasets(extra_daily_features=narrative)
            from sentisense.hpo import final_holdout_eval, run_hpo
            if study is None:
                # Resume/load the existing study to fetch best params without re-tuning.
                study = run_hpo(ml, n_trials=0)
            final_holdout_eval(ml, study.best_params)

    logger.info("Pipeline complete: {}", " → ".join(selected))


if __name__ == "__main__":
    main()
