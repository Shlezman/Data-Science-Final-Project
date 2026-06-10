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
    uv run python -m sentisense.pipeline --from features          # skip ingest/embed;
                                                                  # narrative auto-derives
                                                                  # from cached embeddings
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
    # backfill-stage knobs forwarded to sentisense.ingest.backfill
    parser.add_argument("--backfill-window", type=int, default=7,
                        help="Backfill: days per scrape window (default 7).")
    parser.add_argument("--backfill-batch-size", type=int, default=5,
                        help="Backfill: dates scraped CONCURRENTLY per batch (default 5).")
    parser.add_argument("--backfill-max-days", type=int, default=0,
                        help="Backfill: cap total days walked back (0 = until exhausted).")
    parser.add_argument("--score-concurrency", type=int, default=4,
                        help="Score stage: concurrent headlines (local Ollama default 4).")
    args = parser.parse_args()
    if args.backfill_window < 1 or args.backfill_batch_size < 1:
        parser.error("--backfill-window and --backfill-batch-size must be >= 1")

    selected = _select_stages(args)
    for s in selected:
        if s not in STAGES:
            parser.error(f"unknown stage '{s}'. Valid: {', '.join(STAGES)}")
    if args.skip_final and "final" in selected:
        selected.remove("final")
    logger.info("Pipeline stages: {}", " → ".join(selected))

    # Up-front ETA estimate (best-effort; needs the DB for score/embed counts).
    from sentisense.eta import StageClock, estimate

    estimates: dict = {s: None for s in selected}
    try:
        from sentisense.db import get_engine
        estimates = estimate(selected, get_engine())
    except Exception as exc:  # DB down / counts unavailable → run without the up-front ETA
        logger.warning("ETA estimate unavailable ({}). Continuing without it.", exc)
    clock = StageClock(estimates)

    dry = ["--dry-run"] if args.dry_run else []
    # Memoised so the narrative frame + datasets are built ONCE and consistently,
    # regardless of which stages are selected. This guarantees `tune` and a later
    # `--only final` see the SAME feature width (the narrative skew bug): narrative
    # is always derived from the cached embeddings (no GPU) before any features build.
    state: dict = {"narrative": None, "narrative_built": False, "mt": None, "ml": None,
                   "ml_emb": None, "ml_emb_built": False, "study_scores": None, "study_emb": None}

    def embedding_dataset():
        """Daily e5-centroid dataset (memoised). None if no embeddings cached."""
        if not state["ml_emb_built"]:
            from sentisense.features import build_embedding_dataset
            df = build_embedding_dataset()
            state["ml_emb"] = df if (df is not None and not df.empty) else None
            state["ml_emb_built"] = True
        return state["ml_emb"]

    def narrative_features():
        if not state["narrative_built"]:
            from sentisense.cluster import build_narrative_features
            nf = build_narrative_features()
            state["narrative"] = nf if (nf is not None and not nf.empty) else None
            state["narrative_built"] = True
            if state["narrative"] is not None:
                logger.info("Narrative features: {} days", len(state["narrative"]))
            else:
                logger.warning("No cached embeddings → modeling on base features only "
                               "(run the 'embed' stage to enable narrative clustering).")
        return state["narrative"]

    def datasets():
        if state["mt"] is None:
            from sentisense.features import build_datasets
            state["mt"], state["ml"] = build_datasets(extra_daily_features=narrative_features())
        return state["mt"], state["ml"]

    for i, stage in enumerate(selected):
        logger.info("══════ stage: {} ({}/{}) ══════", stage, i + 1, len(selected))
        clock.start_stage(stage)

        if stage == "backfill":
            bf = [*dry, "--window", str(args.backfill_window),
                  "--batch-size", str(args.backfill_batch_size)]
            if args.backfill_max_days:
                bf += ["--max-days", str(args.backfill_max_days)]
            _run_module("sentisense.ingest.backfill", bf)

        elif stage == "score":
            _run_module("sentisense.ingest.score", [*dry, "--concurrency", str(args.score_concurrency)])

        elif stage == "coverage":
            _run_module("sentisense.ingest.coverage_report", [])

        elif stage == "embed":
            from sentisense.embed import embed_missing
            embed_missing(dry_run=args.dry_run)

        elif stage == "cluster":
            narrative_features()  # force build + log (memoised for the feature stages)

        elif stage == "features":
            datasets()

        elif stage == "baselines":
            mt, _ = datasets()
            from sentisense.models.baselines import run_baselines
            run_baselines(mt)

        elif stage == "tune":
            _, ml = datasets()
            from sentisense.config import EMBED_PCA_COMPONENTS, OPTUNA_TRIALS
            from sentisense.hpo import run_hpo
            from sentisense.hpo.optuna_lstm import STUDY_EMB, STUDY_SCORES
            n_trials = args.trials if args.trials > 0 else OPTUNA_TRIALS

            logger.info("Tuning LSTM on SCORE features …")
            state["study_scores"] = run_hpo(ml, n_trials=n_trials, study_name=STUDY_SCORES)

            mle = embedding_dataset()
            if mle is not None:
                logger.info("Tuning LSTM on EMBEDDING features (PCA→{}) …", EMBED_PCA_COMPONENTS)
                state["study_emb"] = run_hpo(mle, n_trials=n_trials, study_name=STUDY_EMB,
                                             pca_components=EMBED_PCA_COMPONENTS)
            else:
                logger.warning("No embeddings cached → skipping the embedding-LSTM study "
                               "(run the 'embed' stage to enable it).")

        elif stage == "final":
            _, ml = datasets()
            from sentisense.config import EMBED_PCA_COMPONENTS
            from sentisense.hpo import final_holdout_eval, run_hpo
            from sentisense.hpo.optuna_lstm import STUDY_EMB, STUDY_SCORES, has_completed_trials

            s = state["study_scores"] or run_hpo(ml, n_trials=0, study_name=STUDY_SCORES)
            if not has_completed_trials(s):
                raise RuntimeError("No completed score-LSTM trials — run the 'tune' stage first.")
            logger.info("Final holdout — SCORE LSTM:")
            final_holdout_eval(ml, s.best_params)

            mle = embedding_dataset()
            if mle is not None:
                se = state["study_emb"] or run_hpo(mle, n_trials=0, study_name=STUDY_EMB,
                                                   pca_components=EMBED_PCA_COMPONENTS)
                if has_completed_trials(se):
                    logger.info("Final holdout — EMBEDDING LSTM:")
                    final_holdout_eval(mle, se.best_params, pca_components=EMBED_PCA_COMPONENTS)
                else:
                    logger.warning("No completed embedding-LSTM trials — skipping its final eval.")

        clock.end_stage(stage, remaining=selected[i + 1:])

    logger.info("Pipeline complete: {}", " → ".join(selected))


if __name__ == "__main__":
    main()
