# Model zoo — enriched leaderboard

`scripts/pipeline_compare.py` compares **every model × data-type × regime** and writes
`leaderboard.md`. Every cell is reduced to a uniform `(scores, labels)` on the same
last-15% out-of-sample window and scored by the shared metrics + backtest helpers, so all
rows are directly comparable. Each model is hyperparameter-tuned.

## The grid (three axes)
- **Model** — classifiers (XGBoost, LSTM, GRU, TCN, PatchTST) + forecasters (TFT, N-HiTS,
  N-BEATS, Chronos, TimesFM) + Buy&Hold.
- **Data-type** (`--data-types scored,embedded,fused`):
  - `scored` — daily LLM news scores (mean + per-source).
  - `embedded` — daily e5 embedding centroid (`embc_*`, 768-d) + finance.
  - `fused` — per-source scores ⊕ embedding centroid.
  **Classifiers run on all three**; embedded/fused need cached embeddings (`embed` stage).
  **Forecasters** use **scored covariates + univariate** only — the 768-d centroid is *not*
  fed as TFT/N-HiTS covariates (it would overwhelm them); ask if you want a PCA-reduced
  embedding-covariate variant.
- **Regime** (`--regimes CUT,FULL`): `CUT` ≤ 2023-10-07 (the Oct-7 regime break) vs `FULL`
  (entire timeline) — now applied to embedded/fused too (builders are cutoff-aware).

Row labels: `model [datatype/regime]` (classifiers), `model [cov=scored|none/regime]` /
`model [regime]` (forecasters). Each (model, data-type, regime) classifier cell gets its own
resumable Optuna study (`sentisense_<arch>_<dtype>_<regime>`).

## Resumable result cache
Each finished cell's metrics are written to `leaderboard_cache.json` the instant it
completes. A re-run (or crash-resume) **reuses cached cells and only computes new/changed
ones** — so adding a model or fixing one cell costs that cell, not the whole grid. When a
data-type's cells are all cached, its (heavy, esp. embedded/fused 3M-vector) dataset build
is skipped too. `--fresh` ignores the cache; `--cache PATH` changes the file. Note: changing
HPO budgets (`--seq-trials`/`--pf-trials`/`--xgb-trials`) does **not** auto-invalidate — use
`--fresh` (or delete the cache) to re-tune.

## Coverage report (no more silent skips)
`leaderboard.md` ends with a **Coverage** section listing every cell as ran or
`skipped — <reason>`, and the run logs the same. If a model is missing from the table, its
reason is right there (missing dep, runtime error, no embeddings, …).

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

## Run (server-side)
```bash
uv sync --extra ml --extra finance --extra embed --extra tft --extra chronos   # embed → embedded/fused
# TimesFM: manual install (see pyproject note); embedded/fused need the embed stage cached.

# fast smoke — proves every cell runs (tiny trials/epochs, one regime, scored only):
uv run python scripts/pipeline_compare.py --regimes CUT --data-types scored \
    --seq-trials 1 --pf-trials 1 --pf-epochs 3 --no-timesfm 2>&1 | tee smoke.log

# full grid → leaderboard.md (long; run in tmux):
uv run python scripts/pipeline_compare.py --seq-trials 30 --pf-trials 12 --xgb-trials 60

# subsets:
uv run python scripts/pipeline_compare.py --data-types scored,fused --regimes CUT
uv run python scripts/pipeline_compare.py --no-timesfm --no-tft --no-nhits --no-nbeats
```
The full grid is large (classifiers × 3 data-types × 2 regimes + forecasters) — many
HPO'd cells, hours of compute. Subset with `--data-types` / `--regimes` / `--no-*`. Building
embedded+fused loads the cached embeddings (~3M) into memory per regime — if RAM-bound, run
`--regimes CUT` and `--regimes FULL` separately. Read the **Coverage** section of
`leaderboard.md` to confirm exactly which cells ran.

Search spaces are wide: the torch zoo (GRU/TCN/PatchTST) tunes window 5–60, capacity to
384 units / depth 4, dropout 0–0.7, lr 1e-5–3e-2, pooling/activation/optim choices; the
pytorch-forecasting models (TFT/N-HiTS/N-BEATS) tune a broadened capacity grid **plus the
encoder/context length** (15–60); Chronos/TimesFM sweep context length 64–1024; XGBoost
tunes a 9-dim space (trees/depth/lr/subsample/colsample/min-child/λ/α/γ). More trials =
deeper search (`--seq-trials`, `--pf-trials`, `--xgb-trials`).
The new classifier studies persist to the project DB (resumable). Each model is guarded —
a missing extra or a runtime failure skips just that row, so a partial stack still produces
a leaderboard. The "Ultimate model" line reports the best out-of-sample ROC-AUC.

## Accuracy-boost additions (feat/accuracy-boost)
- **Overnight track** (`--overnight`): open(T+1) decision contract — adds the day-T global
  close returns (S&P/Nasdaq/VIX/Brent/USD-ILS, `ovn_` block) that drive TA-125's next open;
  leak-safe (one shift ahead of the close(T) baseline), cells tagged `+ovn`. Run baseline +
  `--overnight` and compare to isolate the overnight signal. Nasdaq added to the base assets.
- **Bootstrap ROC-AUC CI** (`auc_lo`/`auc_hi` columns): 95% CI per cell — if it straddles 0.5
  the cell is indistinguishable from chance (the honest read).
- **Soft-vote ensemble**: per track, rank-normalised mean of all member cells → `Ensemble [track]`
  row (a fair-comparison row; an ensemble of chance-level models stays ~chance).
- **Abstention**: accuracy when acting only on the most-confident fraction (acc@coverage),
  reported per ensemble in the Coverage section — trades coverage for accuracy.
