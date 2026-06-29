# SentiSense live — ops runbook

Everything runs **inside the `/tf` container** (Postgres, GPU, uv env). The UI is a plain
process on `:3000` (no new docker container). Secrets come from env only — never commit DB URL
or LLM keys.

## 1. Deploy / update
```bash
cd /tf/Data-Science-Final-Project
git checkout -- uv.lock 2>/dev/null; git fetch origin && git checkout feat/go-live && git pull
git submodule update --init --recursive          # external/MiroFish (only for running sims)
uv sync --extra finance --extra ml --extra embed --extra ui    # resolves all serving deps
# build the SPA once (needs node/npm in the container):
cd ui/frontend && npm install && npm run build && cd ../..
```
Required env (export in the container profile / pm2 env — never commit):
`SENTISENSE_DATABASE_URL`, the scoring backend vars (`SENTISENSE_LLM_BACKEND` or
`SENTISENSE_OPENAI_*` + `SENTISENSE_FORCE_COMPLETIONS_API`), optional `SENTISENSE_EMBED_*`,
`SENTISENSE_MIRO_BASE_URL` (default `http://localhost:5001`), `SENTISENSE_UI_PORT` (default 3000).

## 2. Daily pipeline (cron)
Install the schedule **inside the container**:
```bash
crontab ops/crontab.txt        # daily_live 15:30 UTC, settle 15:45 UTC (self-skips Fri/Sat)
crontab -l                     # verify
```
Manual run / smoke:
```bash
uv run --extra finance --extra ml python scripts/daily_live.py --dry-run   # plan only
uv run --extra finance --extra ml python scripts/daily_live.py             # real run
```

## 3. Did a run succeed?  (the recurring question)
Single source of truth = `logs/daily_live_status.json`:
```bash
cat logs/daily_live_status.json | python -m json.tool   # last_success, stages[].ok, prediction, error
```
- `skipped: "non-trading-day"` → Fri/Sat/holiday, nothing to do (exit 0).
- `error: null` + `last_success` == today + `prediction` populated → success.
- `error: "stage 'X' failed..."` → that stage's `tail` has the cause; per-stage logs in
  `logs/daily_live_<date>.log`, `logs/cron_daily.log`.
- Or hit the UI: `GET http://localhost:3000/api/health`.

## 4. UI service (port 3000)
```bash
pm2 start ops/pm2.config.js     # start; survives logout
pm2 save && pm2 startup         # persist across reboot (run the printed command once)
pm2 logs sentisense-ui          # live logs (also logs/ui_*.log)
pm2 restart sentisense-ui       # after a code/SPA rebuild
pm2 stop sentisense-ui
curl -s localhost:3000/api/health
```
If port 3000 is reachable beyond localhost, put it behind the host's reverse proxy / firewall
or add auth — the app binds `0.0.0.0:3000` with no auth by design (internal use). No fallback:
without pm2 you can `nohup uv run --extra ui --extra finance --extra ml python -m ui.app &`.

## 5. Champion / challenger (optional HPO)
- Served champion = `models/champion.json` (pinned XGBoost/fused/FULL/overnight). `daily_live`
  retrains it on all history and predicts; it does **not** re-tune.
- Enable the challenger by uncommenting line 3 of `ops/crontab.txt` (or run it ad hoc):
  ```bash
  uv run --extra finance --extra ml python scripts/challenger_hpo.py --xgb-trials 80
  uv run --extra finance --extra ml python scripts/challenger_hpo.py --dry-run   # never promotes
  ```
- **Promotion gate**: a challenger replaces the champion only if `ΔROC-AUC ≥ 0.02` **and** MCC
  does not regress **and** the OOS window `n ≥ 200`, all on the same last-15% tail. On
  promotion, `models/champion.json` is overwritten (version bumped) and the decision is
  appended to `logs/promotions.jsonl`. Every evaluation is logged there regardless.
- Roll back a bad promotion: restore the previous `models/champion.json` (its `prev_version`
  is recorded) or `git checkout models/champion.json`, then `pm2 restart sentisense-ui`.

## 6. Simulator (mirofish)
The UI renders **cached** sims (`narrative_sim*`) with no extra service. To run *new* sims
(the "Run new simulation" button / `scripts/miro_daily.py`), the MiroFish HTTP service must be
up at `SENTISENSE_MIRO_BASE_URL`. If it's down, the UI shows an error event and still serves
cached graphs.

## 7. Honest note
The champion is the **best-available** cell, not a skillful one — daily TA-125 direction is
≈ chance (leaderboard ROC-AUC CIs span 0.5). The system is production-grade; the edge is not
claimed. The dashboard's accuracy/confusion matrix reflect that reality.
