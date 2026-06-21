# Model zoo — enriched leaderboard

`scripts/pipeline_compare.py` now compares **11+ model forms × 2 regimes** and writes
`leaderboard.md`. Every model is reduced to a uniform `(scores, labels)` on the same
last-15% out-of-sample window and scored by the shared `metrics_at` + backtest helpers, so
the rows are directly comparable. Each model is hyperparameter-tuned.

## Models + how each is tuned
| Model | Type | Framing | HPO | Extra |
|-------|------|---------|-----|-------|
| XGBoost | gradient-boost | classifier | **Optuna** wide space (`--xgb-trials`) | `ml` |
| LSTM | recurrent | classifier | pre-tuned Optuna study | `ml` |
| **GRU** | recurrent | classifier | Optuna (`optuna_seq`, `--seq-trials`) | `ml` |
| **TCN** | dilated causal conv | classifier | Optuna (`optuna_seq`) | `ml` |
| **PatchTST** | patched transformer | classifier | Optuna (`optuna_seq`) | `ml` |
| **TFT** | temporal fusion transformer | forecast→direction | Optuna (`--pf-trials`) | `ml` + `tft` |
| **N-HiTS** | hierarchical interpolation | forecast→direction | Optuna (`--pf-trials`) | `ml` + `tft` |
| **N-BEATS** | basis-expansion (univariate) | forecast→direction | Optuna (`--pf-trials`) | `ml` + `tft` |
| **Chronos** | foundation (Amazon) | forecast→direction | context-len + threshold sweep | `ml` + `chronos` |
| TimesFM | foundation (Google) | forecast→direction | context-len + threshold sweep | manual install |
| Buy&Hold | benchmark | — | — | — |

Classifiers (GRU/TCN/PatchTST) train on the per-source sequence dataset (`ml`) reusing the
existing windowed, train-only-scaled loaders + generic Optuna HPO in
`sentisense/hpo/optuna_seq.py` (the well-tuned LSTM path is untouched). TFT / N-HiTS /
N-BEATS (pytorch-forecasting, shared adapter in `tft_forecaster.py`; N-BEATS is univariate
so covariates are dropped) and Chronos / TimesFM forecast the next-day log-return and map
sign→direction with a validation-tuned threshold (the foundation models reuse the leak-safe
`walk_forward_directions` bridge). All HPO selects on a validation slice only; the test tail
stays sacred.

## Run (server-side, GPU)
```bash
uv sync --extra ml --extra tft --extra chronos      # torch zoo + TFT + Chronos
# TimesFM: manual install (see pyproject note)
uv run python scripts/pipeline_compare.py            # full board → leaderboard.md
# subsets / budgets:
uv run python scripts/pipeline_compare.py --no-timesfm --no-tft
uv run python scripts/pipeline_compare.py --seq-trials 40 --pf-trials 15 --xgb-trials 60 --regimes CUT
```

Search spaces are wide: the torch zoo (GRU/TCN/PatchTST) tunes window 5–60, capacity to
384 units / depth 4, dropout 0–0.7, lr 1e-5–3e-2, pooling/activation/optim choices; the
pytorch-forecasting models (TFT/N-HiTS/N-BEATS) tune a broadened capacity grid **plus the
encoder/context length** (15–60); Chronos/TimesFM sweep context length 64–1024; XGBoost
tunes a 9-dim space (trees/depth/lr/subsample/colsample/min-child/λ/α/γ). More trials =
deeper search (`--seq-trials`, `--pf-trials`, `--xgb-trials`).
The new classifier studies persist to the project DB (resumable). Each model is guarded —
a missing extra or a runtime failure skips just that row, so a partial stack still produces
a leaderboard. The "Ultimate model" line reports the best out-of-sample ROC-AUC.
