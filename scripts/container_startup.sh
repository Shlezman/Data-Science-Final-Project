#!/usr/bin/env bash
# Bring SentiSense services up after a `docker restart`. Run from INSIDE the container.
#
# A Jupyter-style container's PID 1 is not systemd, so the `pm2 startup` systemd unit does
# not fire on boot, and native Postgres + cron don't auto-start either. This re-starts them.
# Idempotent: safe to run repeatedly. Wire it into the container entrypoint (host-side) for
# true hands-off recovery, or run it manually after each restart.
set -uo pipefail
cd /tf/Data-Science-Final-Project

echo "[startup] postgres…"
service postgresql start 2>/dev/null || pg_ctlcluster 14 main start 2>/dev/null || \
  su postgres -c '/usr/lib/postgresql/14/bin/pg_ctl -D /var/lib/postgresql/14/main -l /tmp/pg.log start' 2>/dev/null || true

echo "[startup] cron…"
service cron start 2>/dev/null || cron 2>/dev/null || true

echo "[startup] ui (pm2 :3000)…"
pm2 resurrect 2>/dev/null || pm2 start ops/pm2.config.js

echo "[startup] done — postgres + cron + ui(:3000) up."
pm2 status
