"""Full-history comparison: an all-dates model vs the leak-safe (≤ 2023-10-07) model
whose post-cutoff decisions are forced to BUY.

  * Model A — "all history": features + target over EVERY date (cutoff lifted), the
    shared best hyperparameters, sacred-holdout eval. This is what you get if the model
    is actually trained and tested on the whole timeline.
  * Model B — "cutoff + buy": the project model trained ≤ 2023-10-07, with every
    post-cutoff trading day scored as a forced BUY (sentisense.hpo.postcutoff_buy_overlay).

Both reuse the tuned score-LSTM hyperparameters so the only difference is the data span
+ the forced-buy construction. The final stage prints both scorecards side by side.

Run (server-side, from repo root):
    uv run python scripts/full_compare.py
"""

from __future__ import annotations

import datetime as dt

from loguru import logger

_METRICS = ["accuracy@tuned", "balanced_accuracy@tuned", "roc_auc@tuned",
            "mcc@tuned", "brier_raw", "collapse_frac"]
_FAR_FUTURE = dt.date(2100, 1, 1)   # effectively "no cutoff" → every date


def _means(summary: dict) -> dict:
    return {k: v["mean"] for k, v in summary.items()}


def main() -> None:
    import optuna

    import sentisense  # noqa: F401 — loads .env
    from sentisense.db import get_connection_url
    from sentisense.features import build_datasets
    from sentisense.hpo import final_holdout_eval, postcutoff_buy_overlay
    from sentisense.hpo.optuna_lstm import STUDY_SCORES, has_completed_trials

    study = optuna.load_study(study_name=STUDY_SCORES, storage=get_connection_url())
    if not has_completed_trials(study):
        raise RuntimeError(f"Study '{STUDY_SCORES}' has no completed trials — run tune first.")
    best = study.best_params
    logger.info("Shared best hyperparameters: {}", best)

    # ── Model B: trained ≤ cutoff, post-cutoff forced BUY ───────────────────────
    logger.info("Building Model B (≤ cutoff) …")
    _, ml_cut = build_datasets()
    summ_b, proba_b, labels_b = final_holdout_eval(ml_cut, best)
    overlay = postcutoff_buy_overlay(proba_b, labels_b, threshold=0.5)
    b = _means(summ_b)

    # ── Model A: trained + tested over every date ───────────────────────────────
    logger.info("Building Model A (all history, cutoff lifted) …")
    _, ml_all = build_datasets(cutoff=_FAR_FUTURE)
    summ_a, _, _ = final_holdout_eval(ml_all, best)
    a = _means(summ_a)

    # ── Final comparison ────────────────────────────────────────────────────────
    logger.info("=" * 66)
    logger.info("COMPARISON — Model A (all dates) vs Model B (≤ cutoff + forced buy)")
    logger.info("  rows: A(all)={:,}  B(≤cutoff)={:,}", len(ml_all), len(ml_cut))
    logger.info("  {:<26s} {:>10s} {:>10s}", "metric", "A_all", "B_cutoff")
    for m in _METRICS:
        if m in a and m in b:
            logger.info("  {:<26s} {:>10.4f} {:>10.4f}", m, a[m], b[m])
    logger.info("-" * 66)
    logger.info("Model B forced-buy COMBINED scorecard (pre-cutoff model + post-cutoff BUY):")
    logger.info("  combined_accuracy        {:.4f}  ({} pre-cutoff model days + {} post-cutoff BUY days)",
                overlay["combined_accuracy"], overlay["n_pre"], overlay["n_post"])
    logger.info("  post-cutoff buy-only acc {:.4f}", overlay["postcutoff_buy_accuracy"])
    logger.info("  confusion [tn={} fp={} fn={} tp={}]",
                overlay["tn"], overlay["fp"], overlay["fn"], overlay["tp"])
    logger.info("=" * 66)
    logger.info("Note: Model B's headline combined_accuracy is inflated by forcing BUY on the "
                "post-2023 rise — it is NOT model skill. Compare against Model A's true "
                "all-dates holdout above.")


if __name__ == "__main__":
    main()
