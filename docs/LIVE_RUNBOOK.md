# SentiSense live — deployment and operations runbook

The live web application runs on the database/UI host under PM2. The GPU/Jupyter checkout is a
separate machine and is not the deployment target. Secrets come from the server environment only;
never commit passwords, database URLs, API keys, or session cookies.

## 0. Prerequisite: Check Point VPN

SSH is available only through the internal address `10.10.248.109` while connected to the
Check Point VPN. Port 22 on the public address is blocked. The VPN requires the operator's personal
credentials and SMS approval, so only the operator can establish this connection.

```bash
ssh cs703@10.10.248.109
```

## 1. Identify the live checkout

There are three repository copies. Only one serves the live site:

| Location | Purpose |
|---|---|
| `/home/cs703/sentisense` | **Live checkout** and PM2 execution directory |
| `/home/cs703/Data-Science-Final-Project` | Old checkout; building here does not update the site |
| `/tf/Data-Science-Final-Project` | Jupyter/GPU checkout on a different machine |

Confirm the target before every deployment:

```bash
pm2 describe sentisense-ui
# Expected: exec cwd = /home/cs703/sentisense
cd /home/cs703/sentisense
pwd
```

If `exec cwd` is not `/home/cs703/sentisense`, stop and investigate instead of deploying.

## 2. How the live process runs

PM2 runs one process named `sentisense-ui` on port 3000. Its `watch` setting is disabled, so Python
changes are not loaded until the process is restarted. FastAPI serves files from
`ui/frontend/dist/` on every request, so a successful frontend build is visible without restarting
PM2.

Runtime configuration such as `SENTISENSE_DATABASE_URL`, `SENTISENSE_UI_PASSWORD`,
`SENTISENSE_ACTIVE_MODEL`, and `SENTISENSE_REPO` is injected through PM2/server configuration, not
stored in source control. Do not print or copy the PM2 environment into logs or documentation.

Useful read-only checks:

```bash
pm2 describe sentisense-ui
pm2 logs sentisense-ui --lines 100
```

## 3. Deploy an approved `main`

Deploy only after the pull request has been reviewed and merged into `main`.

```bash
cd /home/cs703/sentisense
git status --short --branch       # must not contain unexplained local changes
git fetch origin --prune
git switch main
git pull --ff-only origin main
git rev-parse --short HEAD

cd ui/frontend
npm ci --no-audit --no-fund
npm run build
cd ../..
```

Run `uv sync --extra ui --extra miro` only when `pyproject.toml` or `uv.lock` changed. A source-code
change alone does not require dependency synchronization.

Do not use `git reset --hard`, `git clean`, or overwrite unexplained server changes. Resolve them
before continuing.

## 4. When a restart is required

| Change | Required action |
|---|---|
| Only `.jsx`, `.css`, or other frontend source | `npm run build`; no PM2 restart |
| `package.json` or `package-lock.json` | `npm ci`, then `npm run build` |
| `ui/app.py`, `ui/queries.py`, or any Python module | `pm2 restart sentisense-ui` |
| Frontend and Python | Build the frontend, then restart PM2 |
| `pyproject.toml` or `uv.lock` | `uv sync --extra ui --extra miro`, then restart PM2 |
| PM2 environment or process configuration | Reload/restart with the approved environment procedure |

For a normal Python-code deployment:

```bash
pm2 restart sentisense-ui
pm2 status sentisense-ui
pm2 logs sentisense-ui --lines 100
```

The archive-filter deployment changed both `ui/queries.py` and `ui/app.py`; without a restart the
new controls render, but the old backend ignores their unfamiliar query parameters.

## 5. Verify that deployment reached the live site

Start with the checkout and process:

```bash
cd /home/cs703/sentisense
git status --short --branch
git rev-parse --short HEAD
pm2 describe sentisense-ui
```

Compare the newest built JavaScript bundle with the bundle served locally by FastAPI:

```bash
BUILT_ASSET=$(basename "$(ls -t ui/frontend/dist/assets/index-*.js | head -1)")
LOCAL_ASSET=$(curl -fsS http://127.0.0.1:3000/ \
  | sed -n 's/.*src="\/assets\/\([^"]*\.js\)".*/\1/p')
printf 'built=%s\nlocal=%s\n' "$BUILT_ASSET" "$LOCAL_ASSET"
test "$BUILT_ASSET" = "$LOCAL_ASSET"
```

Then bypass browser/proxy cache and compare the public asset hash:

```bash
PUBLIC_ASSET=$(curl -fsS -H 'Cache-Control: no-cache' \
  "https://sentisens.cs.colman.ac.il/?deploy=$(date +%s)" \
  | sed -n 's/.*src="\/assets\/\([^"]*\.js\)".*/\1/p')
printf 'built=%s\npublic=%s\n' "$BUILT_ASSET" "$PUBLIC_ASSET"
test "$BUILT_ASSET" = "$PUBLIC_ASSET"
```

If the hashes match, the public site is serving the checkout that was just built. A backend health
check requires authentication because all protected `/api/*` routes return HTTP 401 without the
session cookie:

```bash
COOKIE_JAR=$(mktemp)
trap 'rm -f "$COOKIE_JAR"; unset UI_PASSWORD LOGIN_JSON' EXIT
read -rsp 'Dashboard password: ' UI_PASSWORD; echo
LOGIN_JSON=$(UI_PASSWORD="$UI_PASSWORD" python -c \
  'import json, os; print(json.dumps({"password": os.environ["UI_PASSWORD"]}))')
curl -fsS -c "$COOKIE_JAR" -H 'Content-Type: application/json' \
  -d "$LOGIN_JSON" https://sentisens.cs.colman.ac.il/api/login
curl -fsS -b "$COOKIE_JAR" https://sentisens.cs.colman.ac.il/api/health \
  | python -m json.tool
```

## 6. Daily pipeline schedule

The live host uses the `Asia/Jerusalem` timezone. The installed crontab, rather than
`ops/crontab.txt`, is the source of truth for this machine. The relevant jobs currently run at
18:30 and 18:45 local time.

```bash
timedatectl
crontab -l
```

Do not install `ops/crontab.txt` on this host as-is: its `/tf` paths and UTC assumptions describe a
different environment. Manual pipeline checks from the live checkout are:

```bash
cd /home/cs703/sentisense
uv run --extra finance --extra ml python scripts/daily_live.py --dry-run
uv run --extra finance --extra ml python scripts/daily_live.py
```

## 7. Did a run succeed?

The single source of truth is `logs/daily_live_status.json`:

```bash
cd /home/cs703/sentisense
python -m json.tool logs/daily_live_status.json
```

- `skipped: "non-trading-day"` → Friday, Saturday, or a configured holiday; exit 0 is expected.
- `error: null`, `last_success` equal to today, and a populated `prediction` → success.
- `error: "stage 'X' failed..."` → inspect that stage's `tail` and
  `logs/daily_live_<date>.log` / `logs/cron_daily.log`.
- `/api/health` is useful only after obtaining the authenticated cookie shown above.

## 8. Champion / challenger (optional HPO)
- Served champion = `models/champion.json` (pinned XGBoost/fused/FULL/overnight). `daily_live`
  retrains it on all history and predicts; it does **not** re-tune.
- Enable the challenger only through an approved update to the installed crontab, or run it ad hoc:
  ```bash
  uv run --extra finance --extra ml python scripts/challenger_hpo.py --xgb-trials 80
  uv run --extra finance --extra ml python scripts/challenger_hpo.py --dry-run   # never promotes
  ```
- **Promotion gate**: a challenger replaces the champion only if `ΔROC-AUC ≥ 0.02` **and** MCC
  does not regress **and** the OOS window `n ≥ 200`, all on the same last-15% tail. On
  promotion, `models/champion.json` is overwritten (version bumped) and the decision is
  appended to `logs/promotions.jsonl`. Every evaluation is logged there regardless.
- Roll back a bad promotion by restoring the recorded previous `models/champion.json`, then run
  `pm2 restart sentisense-ui`.

## 9. Simulator (MiroFish)
The UI renders **cached** sims (`narrative_sim*`) with no extra service. To run *new* sims
(the "Run new simulation" button / `scripts/miro_daily.py`), the MiroFish HTTP service must be
up at `SENTISENSE_MIRO_BASE_URL`. If it's down, the UI probes `/api/sim/health` on the
Simulator tab, disables the run control, and shows a "historical (cached) simulations only"
banner — cached graphs still render.

### 9a. MiroFish is not deployed on the live UI host — this is intended
The live UI process never starts MiroFish. A request to port 5001 from the database/UI host
therefore **times out** when no approved external MiroFish service or tunnel is configured. This
is expected, not a UI defect. MiroFish is a heavy agent-simulation
sub-stack: it needs `zep-cloud` (Zep — local Zep requires Docker; the box has none) or Zep
Cloud (external SaaS), plus `camel-oasis`/`camel-ai` and an OpenAI-format LLM endpoint.

**Do not start MiroFish on the live UI host.** Zep Cloud egress would send data to a
third-party service (org data-handling policy — needs explicit approval), and there's no Docker
for local Zep anyway.

### 9b. Generating *new* simulations (batch, off-production)
Run MiroFish where it belongs — a box with **Docker + Zep + an LLM** (the phase-23 setup or a
development host), and write results into the **same Postgres** database the live UI reads. The UI
then serves them as cached graphs automatically; the live host never runs MiroFish.

```bash
# on a Docker+Zep host (not the live UI host):
cd external/MiroFish && docker compose up -d          # brings up MiroFish on :5001 (loopback)
export SENTISENSE_DATABASE_URL=postgresql://<user>:<pw>@10.10.248.109:5432/sentisense  # shared DB
export SENTISENSE_MIRO_URL=http://localhost:5001      # loopback → assert_local passes, no egress opened
uv run python scripts/miro_daily.py --date <YYYY-MM-DD>   # upserts narrative_sim* into the shared DB
```

If you genuinely need the live "Run new simulation" button (cross-machine, discouraged): keep
the port closed and use an SSH tunnel from the DB machine
(`ssh -L 5001:localhost:5001 <mirofish-host>`), then set `SENTISENSE_MIRO_BASE_URL=http://localhost:5001`
on the UI — tunnel stays encrypted and `assert_local` passes with no `SENTISENSE_MIRO_ALLOW_REMOTE`.
Opening `5001` directly is plaintext HTTP across a public↔private hop; avoid it.

> Env-var note: the UI reads **`SENTISENSE_MIRO_BASE_URL`** (`ui/app.py`); the pipeline client
> reads **`SENTISENSE_MIRO_URL`** (`sentisense/sim/config.py`). Different names — set the one
> that matches the process you're configuring.

## 10. Dashboard data prerequisites (per panel)
Each panel degrades to an explicit "no data" state until its producer has run. The live UI host
reads the resulting database records:

| Panel | Table | Producer |
|---|---|---|
| Hero + recent predictions | `model_predictions` | `daily_live.py` (cron) |
| "All days" confusion matrix | `champion_full_eval` | `scripts/compute_full_eval.py` |
| EDA panels | `raw_headlines` + `nlp_vectors` | scrape + score (cron) |
| 3D centroids — all days | `daily_embedding_derived` | `scripts/build_embedding_derived.py` |
| 3D centroids — single day | `embedding_pca_basis` + `headline_embeddings` | **rerun** `scripts/build_embedding_derived.py` (now also persists the PCA basis) |
| Personas (Simulator) | `nlp_vectors` per source | scrape + score (cron) |

After pulling approved frontend changes on the live host, build them under
`/home/cs703/sentisense/ui/frontend`. Restart PM2 only when the deployment also includes Python
changes, as described in section 4.

## 11. Honest note
The champion is the **best-available** cell, not a skillful one — daily TA-125 direction is
≈ chance (leaderboard ROC-AUC CIs span 0.5). The system is production-grade; the edge is not
claimed. The dashboard's accuracy/confusion matrix reflect that reality.
