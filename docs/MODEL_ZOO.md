# Model zoo — enriched leaderboard

`scripts/pipeline_compare.py` now compares **11+ model forms × 2 regimes** and writes
`leaderboard.md`. Every model is reduced to a uniform `(scores, labels)` on the same
last-15% out-of-sample window and scored by the shared `metrics_at` + backtest helpers, so
the rows are directly comparable. Each model is hyperparameter-tuned.

## Models + how each is tuned
| Model | Type | Framing | HPO | Extra |
|-------|------|---------|-----|-------|
| XGBoost | gradient-boost | classifier | (existing fixed config) | `ml` |
| LSTM | recurrent | classifier | pre-tuned Optuna study | `ml` |
| **GRU** | recurrent | classifier | Optuna (`optuna_seq`, `--seq-trials`) | `ml` |
| **TCN** | dilated causal conv | classifier | Optuna (`optuna_seq`) | `ml` |
| **PatchTST** | patched transformer | classifier | Optuna (`optuna_seq`) | `ml` |
| **TFT** | temporal fusion transformer | forecast→direction | Optuna (`--tft-trials`) | `ml` + `tft` |
| **Chronos** | foundation (Amazon) | forecast→direction | context-len + threshold sweep | `ml` + `chronos` |
| TimesFM | foundation (Google) | forecast→direction | context-len + threshold sweep | manual install |
| Buy&Hold | benchmark | — | — | — |

Classifiers (GRU/TCN/PatchTST) train on the per-source sequence dataset (`ml`) reusing the
existing windowed, train-only-scaled loaders + generic Optuna HPO in
`sentisense/hpo/optuna_seq.py` (the well-tuned LSTM path is untouched). TFT/Chronos/TimesFM
forecast the next-day log-return and map sign→direction with a validation-tuned threshold,
reusing the leak-safe `walk_forward_directions` bridge. All HPO selects on a validation
slice only; the test tail stays sacred.

## Run (server-side, GPU)
```bash
uv sync --extra ml --extra tft --extra chronos      # torch zoo + TFT + Chronos
# TimesFM: manual install (see pyproject note)
uv run python scripts/pipeline_compare.py            # full board → leaderboard.md
# subsets / budgets:
uv run python scripts/pipeline_compare.py --no-timesfm --no-tft
uv run python scripts/pipeline_compare.py --seq-trials 40 --tft-trials 15 --regimes CUT
```
The new classifier studies persist to the project DB (resumable). Each model is guarded —
a missing extra or a runtime failure skips just that row, so a partial stack still produces
a leaderboard. The "Ultimate model" line reports the best out-of-sample ROC-AUC.
