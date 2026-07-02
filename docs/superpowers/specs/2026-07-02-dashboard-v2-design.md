# Dashboard v2 — spec (2026-07-02)

## Goal
Upgrade the SentiSense live UI: prominent current-day up/down hero, a full-history
confusion matrix, EDA panels, a 3D daily-centroid drawer, and an eye-comfortable redesign.
All panels reflect the **currently-served model** (the pinned champion on main today; auto-
upgrades to the best-of-zoo when the registry branch merges). Ensemble comes later.

## Decisions (locked)
- Full-history confusion matrix = **in-sample** (fit champion on all labeled days, predict the
  same days). Panels are labelled by factual **scope** ("All days" vs "Live / settled") — no
  editorial quality flag.
- 3D daily centroid = first 3 leak-safe PCA components (`embpca_000..002`) from
  `daily_embedding_derived`, coloured by actual up/down.
- Charting + 3D = **Plotly** (`plotly.js-dist-min` + `react-plotly.js`, MIT) — one dep.
- Branch: `feat/dashboard-v2` off `main`.

## Architecture
Split-friendly: heavy compute (XGBoost fit) runs on the container (`ml` extra); the light UI box
reads a table. EDA + centroids are pandas/SQL over existing tables (no `ml` dep).

- **Migration `006_champion_full_eval.sql`** — table `champion_full_eval`:
  `model_version VARCHAR, date DATE, prediction BOOL, proba REAL, actual BOOL,
   created_at TIMESTAMPTZ`, PK `(model_version, date)`. Idempotent.
- **`scripts/compute_full_eval.py`** (container, has `ml`) — build the fused dataset
  (`keep_unlabeled=False`), fit the pinned champion on ALL labeled days, predict every day
  in-sample, upsert into `champion_full_eval`. Add to daily cron after predict; runnable now.

## API contracts (all read-only, in-process ~60 s cache)
- `GET /api/prediction/today` →
  `{date, up: bool, confidence: float, model_version: str}` (from latest `model_predictions`).
- `GET /api/confusion/full` →
  `{scope:"all", model_version, n, accuracy, precision, recall, matrix:{tp,fp,tn,fn}}`
  from `champion_full_eval`. Empty-safe (`{n:0,...}` if table empty).
- `GET /api/eda` →
  `{volume:[{date,count}], sentiment_ts:[{date,mean_sentiment}],
    sentiment_hist:[{bin,count}], relevance_hist:[{bin,count}],
    category_corr:{labels:[..6], matrix:[[..]]}, validation:{passed,failed,rate}}`
  (pandas over `raw_headlines` + `nlp_vectors`).
- `GET /api/centroids` →
  `{points:[{date, x, y, z, actual: bool|null, n_headlines}]}`
  from `daily_embedding_derived` (embpca_000..002) ⨝ `champion_full_eval` (actual).

## Frontend
- **Theme** (`styles.css` vars): dark low-glare (slate bg), muted surfaces, one green (up) /
  one red (down) accent, WCAG-AA text, more whitespace + line-height, readable sans stack.
- **Hero** (`Hero.jsx`): large green ▲ UP / red ▼ DOWN card — current-day dir + confidence % +
  date + served-model name. Top of the Dashboard tab.
- **FullConfusion** (`FullConfusion.jsx`): 2×2 heatmap + accuracy/precision/recall, scope
  "All days", shown beside the existing live/settled matrix.
- **EdaPanels** (`EdaPanels.jsx`): Plotly — volume line, sentiment time-series, sentiment +
  relevance histograms, 6×6 category-correlation heatmap, validation-rate gauge. Collapsible.
- **Centroids3D** (`Centroids3D.jsx`): right-side sliding drawer, Plotly 3D scatter (one point
  per day, green/red by actual), hover = date + headline count, date-range slider.
- **Wiring**: `api.js` adds the 4 GETs; `App.jsx`/`Dashboard.jsx` mount Hero + FullConfusion +
  EDA; a drawer toggle opens Centroids3D. `package.json` gains the Plotly deps.

## Error handling
Every endpoint degrades to an empty-but-valid payload (never 500) when a table is empty/absent,
matching the existing `sim/*` graceful pattern. Frontend renders "no data yet" states.

## Testing
- pytest: `champion_full_eval` upsert + in-sample fit/predict shape (`compute_full_eval`);
  `/api/confusion/full` matrix math on a fixture; `/api/eda` aggregation shape; `/api/centroids`
  join shape. Backend `py_compile`; frontend components `esbuild`-parse; `npm run build` on the
  DB machine at deploy.

## Deploy (DB machine + container)
1. Container: apply `006`, run `scripts/compute_full_eval.py` (populates `champion_full_eval`),
   add it to the daily cron after predict.
2. DB machine: `git pull`, `cd ui/frontend && npm install && npm run build`, `pm2 restart
   sentisense-ui`.

## Out of scope (later)
Registry best-of-zoo selection + soft-vote ensemble (separate branch); migrating phase-23
cached sim data.
