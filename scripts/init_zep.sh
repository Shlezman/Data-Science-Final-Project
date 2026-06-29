#!/usr/bin/env bash
# Self-host Zep (MiroFish's memory + knowledge-graph backend) on the remote box WITHOUT
# Docker — so Hebrew-news/internal data never leaves the machine (no Zep Cloud).
#
# Zep talks to: Postgres (store) + an embedder/LLM (OpenAI-compatible). We point the
# embedder at the SAME local Gemma-4 endpoint MiroFish uses (no external egress).
#
# ⚠️ CONFIRM-ON-BOX: Zep's exact build/run differs by version (the upstream default is
# docker-compose). This encodes the standard from-source path; verify the build target,
# config schema, and API port against the cloned repo's README the first time, then this
# script is the repeatable bootstrap. Re-runnable (idempotent-ish): skips clone/db if present.
set -euo pipefail

# ── config (override via env) ────────────────────────────────────────────────
ZEP_DIR="${ZEP_DIR:-$(cd "$(dirname "$0")/.." && pwd)/external/zep}"
ZEP_REPO="${ZEP_REPO:-https://github.com/getzep/zep.git}"
ZEP_PORT="${ZEP_PORT:-8000}"
ZEP_PG_DSN="${ZEP_PG_DSN:-postgres://postgres:postgres@localhost:5432/zep?sslmode=disable}"
LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:11434/v1}"   # local Gemma-4 (Ollama/vLLM)
LLM_MODEL="${LLM_MODEL:-gemma-4}"
LOG="${ZEP_DIR}/../zep_server.log"

echo "==> prereqs"
for bin in git go psql; do
  command -v "$bin" >/dev/null 2>&1 || { echo "MISSING: $bin — install it first."; exit 1; }
done

echo "==> Postgres database 'zep'"
# create the zep DB if absent (uses your local Postgres; adjust ZEP_PG_DSN as needed)
psql "${ZEP_PG_DSN%/zep*}/postgres" -tc "SELECT 1 FROM pg_database WHERE datname='zep'" \
  | grep -q 1 || psql "${ZEP_PG_DSN%/zep*}/postgres" -c "CREATE DATABASE zep;"

echo "==> clone + build Zep (from source, no docker)"
if [ ! -d "$ZEP_DIR/.git" ]; then
  git clone --depth 1 "$ZEP_REPO" "$ZEP_DIR"
fi
cd "$ZEP_DIR"
# CONFIRM: build target may be `make build` or `go build ./cmd/...` depending on version.
if [ -f Makefile ] && grep -q '^build:' Makefile; then make build; else go build -o ./zep ./... ; fi

echo "==> config (local Postgres + local Gemma-4 embedder; NO external services)"
# CONFIRM: key names against the cloned repo's config schema (config.yaml / zep.yaml / env).
# Auth is off (local-only dummy key), so the bind host MUST stay loopback — otherwise any
# host on the network could call Zep unauthenticated. CONFIRM the host key name per version.
cat > "$ZEP_DIR/.env.local" <<EOF
ZEP_STORE_TYPE=postgres
ZEP_STORE_POSTGRES_DSN=${ZEP_PG_DSN}
ZEP_SERVER_HOST=127.0.0.1
ZEP_SERVER_PORT=${ZEP_PORT}
ZEP_LLM_SERVICE=openai
ZEP_LLM_OPENAI_ENDPOINT=${LLM_BASE_URL}
ZEP_LLM_MODEL=${LLM_MODEL}
ZEP_OPENAI_API_KEY=local
ZEP_AUTH_REQUIRED=false
EOF
chmod 600 "$ZEP_DIR/.env.local"   # config file holds the DSN — not world-readable

echo "==> launch (nohup → ${LOG})"
ZEP_BIN="$([ -x ./zep ] && echo ./zep || ls ./bin/zep 2>/dev/null || echo ./zep)"
set -a; . "$ZEP_DIR/.env.local"; set +a
nohup "$ZEP_BIN" > "$LOG" 2>&1 &
echo "zep pid $!"

echo "==> healthcheck (port ${ZEP_PORT})"
for i in $(seq 1 30); do
  if curl -sf "http://localhost:${ZEP_PORT}/healthz" >/dev/null 2>&1; then
    echo "Zep is up on http://localhost:${ZEP_PORT}"
    echo "Put in MiroFish .env:  ZEP_API_URL=http://localhost:${ZEP_PORT}  ZEP_API_KEY=local"
    echo "⚠️  zep-cloud==3.13.0 may NOT honor ZEP_API_URL — confirm with:"
    echo "      bash scripts/verify_local_egress.sh"
    echo "    If it reports the SDK ignores the env var, run:"
    echo "      python scripts/patch_mirofish_zep_local.py   # forwards base_url in code"
    exit 0
  fi
  sleep 2
done
echo "Zep did not report healthy in 60s — check ${LOG} (likely a version-specific config/port key to adjust)."
exit 1
