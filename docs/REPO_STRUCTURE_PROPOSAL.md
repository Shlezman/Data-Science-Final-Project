# Repo Structure Proposal — post-re-org cleanup (2026-08-13)

The `re-order` commit (`060637e`) gave the repo a clean top-level layout
(`sentisense/`, `ui/`, `scripts/`, `ops/`, `notebooks/`, `docs/`, `tests/`,
`evaluation/`, `external/`, `processing_engine/`, `mivzakim_scraper/`).
This document proposes the follow-up pass. Items are ordered by impact;
none are applied by the PR that adds this file — each is a deliberate,
separately-reviewable change.

## 1. Broken by the re-org (fix first)

| Issue | Where | Fix |
|---|---|---|
| Finance CSV paths broken on a clean checkout | `sentisense/constants.py:128-129` still resolves `TA 125 Historical Data.csv` / VTA35 CSV at repo root; commit `060637e` moved both into `evaluation/`. `sentisense/features/dataset.py:145,153` fails. | Create `data/`, move + rename `evaluation/TA 125 Historical Data.csv` → `data/ta125_historical.csv` and the VTA35 CSV → `data/vta35_historical.csv`, update `constants.py` (and `scripts/gap_fill.py:129` docstring). Spaces in filenames also break shell ergonomics. |
| Stale paths in onboarding docs | `README.md` (notebook links, repo tree, "Orchestration 🔜" status) and `.claude/CLAUDE.md` ("three notebooks live at the repo root") | Update to the `notebooks/` layout, list all 10 notebooks, mark module 5 (orchestration/dashboard/DevOps) ✅ shipped. |

## 2. Project-book assets

- Keep the **markdown source** (`Final_Project_book.md`, `Final_Project_book_v2.md`) in git under `docs/book/`.
- Move binary exports out of history going forward: `Final Project Book (2) (1).doc` (0.86 MB, Windows-download name) and `Final Project Book 1.pdf` (2.69 MB) are permanent history weight. Options: `docs/book/exports/` + Git LFS, or attach exports to GitHub Releases instead of tracking them.
- `FINAL_PROJECT_BOOK.html` is derivable from the md — regenerate on demand rather than track.

## 3. Delete / untrack

- `notebooks/compare_lstm_features_with_poc-Copy1.ipynb` — Jupyter auto-copy duplicate (merge any unique cells first).
- `mivzakim_scraper/.claude/settings.local.json` — local Claude settings, should be untracked + gitignored.
- `processing_engine/evaluation/results/*` — generated per-model eval artifacts; either keep only the summary leaderboard or gitignore. Fix the root-anchored `.gitignore` pattern `evaluation/results/` → `**/evaluation/results/`.
- On-disk only (already ignored, just delete): `processing_engine.egg-info/`, scattered `.DS_Store`.

## 4. Renames

- `docs/leaderboard-expirience-4.md` → `docs/leaderboard-experience-4.md` (typo; also consider folding into `docs/experiments/`).
- `docs/superpowers/specs/2026-07-02-dashboard-v2-design.md` → `docs/specs/` (tool-vendor-named dir hides a real design doc).

## 5. History & branch hygiene

- `.git` is 211 MB for a 219-file repo — notebook outputs dominate (`transformer_forecaster.ipynb` 830 KB, `sentisense_analysis.ipynb` ~1 MB × 3 revisions). Adopt **nbstripout** (or a pre-commit clear-outputs hook) for `notebooks/` going forward; optionally a one-time history rewrite if the team agrees.
- ~17 local branches are fully merged into `main` (`git branch --merged main`) — prune. Remote one-off notebook branches (`compare_lstm_features_with_poc1_.orian`, `…_copy.orian`, `golden_dataset`, `nidelsohn/poc`, `nidelsohn/transformer-notebook`) are merge-and-delete candidates.
- 6 old stashes; `stash@{0}` pins a 2.2 MB `data.csv` blob.

## 6. Config & packaging

- `.env.example` predates the live-UI era: defaults to `mistral-large-2`, missing `SENTISENSE_UI_PASSWORD`, `SENTISENSE_MONGO_URL`, `SENTISENSE_EMBED_MODEL`, `SENTISENSE_ACTIVE_MODEL`, `SENTISENSE_FORCE_COMPLETIONS_API`, `SENTISENSE_COMPLETIONS_MAX_TOKENS` — and ships `SENTISENSE_OPENAI_VERIFY_SSL=false`, which disables TLS verification and should not be the documented default. Refresh it (values as placeholders only, never real credentials).
- Three independent uv projects (root, `processing_engine/`, `mivzakim_scraper/`), each with its own lockfile and **no** `[tool.uv.workspace]`. Decide: keep isolation (current, fine) or consolidate into a uv workspace for one `uv sync` — recommend deciding once and documenting the choice in README.
- `.gitignore` duplicates worth collapsing: `**cookies/`+`/cookies`, `**sessions/`+`/sessions`, `**headlines.csv`+`/headlines.csv`, `.ipynb_checkpoints` twice.
- `config/tase_holidays.txt` is referenced by `scripts/daily_live.py:39` but server-side only — track an example (`config/tase_holidays.example.txt`) for reproducibility.

## 7. Target top-level tree

```
├── data/                  # market CSVs (renamed, no spaces)
├── docs/
│   ├── book/              # project book md source (+ exports via LFS/releases)
│   ├── specs/             # design specs (from docs/superpowers/)
│   └── miro/  …           # unchanged
├── evaluation/            # LLM-scoring eval harness ONLY (golden dataset)
├── external/MiroFish      # AGPL submodule, isolated as separate service
├── mivzakim_scraper/      # uv project
├── notebooks/             # 10 notebooks, outputs stripped going forward
├── ops/                   # crontab, pm2, nginx configs
├── processing_engine/     # uv project (LangGraph scoring)
├── scripts/               # operational CLIs
├── sentisense/            # core package
├── tests/
└── ui/                    # FastAPI + React SPA
```
