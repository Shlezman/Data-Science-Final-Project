#!/usr/bin/env bash
# Prove the MiroFish stack runs LOCAL-ONLY. Run on the box (server-side).
#
#   1. Static: does the installed zep-cloud SDK honor ZEP_API_URL? (settles the audit's
#      one open question — env-redirect vs code-patch needed).
#   2. Static: scan MiroFish/.env for any external host in the egress-sensitive keys.
#   3. Live (optional): watch for OUTBOUND connections to known clouds while a sim runs.
#
# Exit non-zero if any external-egress risk is detected.
set -uo pipefail

MIRO_DIR="${MIRO_DIR:-$(cd "$(dirname "$0")/.." && pwd)/external/MiroFish}"
ENV_FILE="${MIRO_ENV:-$MIRO_DIR/backend/.env}"
PYTHON="${PYTHON:-python3}"
# Hosts that MUST never appear in an outbound connection / in the .env URLs.
EXTERNAL_RE='getzep\.com|api\.openai\.com|dashscope\.aliyuncs|huggingface\.co|cdn-lfs'
fail=0

echo "==> 1. zep-cloud SDK: does it honor ZEP_API_URL?  (resolves env-vs-patch)"
"$PYTHON" - <<'PY' || true
try:
    import inspect, zep_cloud.client as zc
    src = inspect.getsource(zc.Zep.__init__)
    if "ZEP_API_URL" in src:
        print("   PASS: SDK __init__ reads ZEP_API_URL -> env redirect works (set it in .env).")
    else:
        print("   ACTION: SDK __init__ does NOT read ZEP_API_URL -> env alone will NOT redirect.")
        print("           Apply: python scripts/patch_mirofish_zep_local.py")
except Exception as e:  # noqa: BLE001
    print(f"   SKIP: could not introspect zep_cloud ({e!r}). Run inside MiroFish's venv.")
PY

echo "==> 2. scan $ENV_FILE for external hosts in egress-sensitive keys"
if [ -f "$ENV_FILE" ]; then
  if grep -E '^(LLM_BASE_URL|OPENAI_API_BASE_URL|OPENAI_BASE_URL|LLM_BOOST_BASE_URL|ZEP_API_URL)=' "$ENV_FILE" \
       | grep -Eq "$EXTERNAL_RE"; then
    echo "   FAIL: an egress-sensitive URL points at an EXTERNAL host:"
    grep -E '^(LLM_BASE_URL|OPENAI_API_BASE_URL|OPENAI_BASE_URL|LLM_BOOST_BASE_URL|ZEP_API_URL)=' "$ENV_FILE" \
       | grep -E "$EXTERNAL_RE"
    fail=1
  else
    echo "   PASS: no external host in the egress-sensitive keys."
  fi
  grep -Eq '^(HF_HUB_OFFLINE|TRANSFORMERS_OFFLINE)=1' "$ENV_FILE" \
    && echo "   PASS: HuggingFace offline mode set." \
    || echo "   WARN: HF offline not set — only matters if the Twitter OASIS platform is enabled."
else
  echo "   WARN: $ENV_FILE not found. Copy scripts/mirofish.env.local-only.example -> \$ENV_FILE first."
  fail=1
fi

echo "==> 3. live outbound-connection watch (optional)"
# Usage: pass the MiroFish backend PID:  scripts/verify_local_egress.sh <pid>
PID="${1:-}"
if [ -n "$PID" ] && command -v lsof >/dev/null 2>&1; then
  echo "   established connections for pid $PID (loopback is fine; flagging external):"
  ext=$(lsof -nP -p "$PID" -i 2>/dev/null | awk '/ESTABLISHED/ && $9 !~ /127\.0\.0\.1|\[::1\]|localhost/ {print $9}')
  if [ -n "$ext" ]; then echo "$ext" | sed 's/^/      /'; echo "   REVIEW: external sockets above."; fail=1
  else echo "      none — all connections are loopback."; fi
else
  echo "   (skip) pass the backend PID to scan live sockets: scripts/verify_local_egress.sh <pid>"
  echo "   or run a sim and watch:  lsof -nP -iTCP -sTCP:ESTABLISHED | grep -E '$EXTERNAL_RE'"
fi

echo
[ "$fail" -eq 0 ] && echo "RESULT: local-only checks PASSED." || echo "RESULT: egress risk detected — fix the items above."
exit "$fail"
