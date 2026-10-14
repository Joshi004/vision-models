#!/usr/bin/env bash
# ensure.sh - Start the gateway only if it is not already answering.
#
# Unlike run.sh, which always restarts, this changes nothing while the gateway
# recorded in gateway.host answers, so scripts can call it as often as they
# like (the Mac's visio-gpu-tunnel.sh calls it when it finds no gateway).
# It checks three times, 10 s apart, before deciding the gateway is down, then
# runs run.sh. A lock in the data directory lets only one caller start it.
set -euo pipefail

API_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$API_DIR/.." && pwd)"
DATA_DIR="${VISION_DATA_DIR:-$(dirname "$REPO_ROOT")/vision-data}"
NODE_FILE="$DATA_DIR/gateway.host"
LOCK_FILE="$DATA_DIR/gateway.lock"

gateway_answers() {
  local node="" port=""
  [ -f "$NODE_FILE" ] || return 1
  read -r node port _ < "$NODE_FILE" || true
  [ -n "$node" ] || return 1
  curl -s -m 5 "http://$node:${port:-8012}/v1/health" | grep -q '"status":"ok"'
}

exec 9>"$LOCK_FILE"
if ! flock -w 300 9; then
  echo "Another start of the gateway is still running; try again later." >&2
  exit 1
fi

for attempt in 1 2 3; do
  if gateway_answers; then
    echo "The gateway is already answering on $(head -n 1 "$NODE_FILE"); nothing to do."
    exit 0
  fi
  if [ "$attempt" -lt 3 ]; then
    sleep 10
  fi
done

echo "The gateway is not answering; starting it with run.sh (code at commit $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown))."
# 9>&-: keep the lock out of the tmux server that run.sh starts, or it is never released.
"$API_DIR/run.sh" 9>&-
