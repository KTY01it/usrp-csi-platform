#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs

SESSION_TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/tx_clean_${SESSION_TS}.log"

echo "[TX-CLEAN] Starting WiFi TX baseline with clean environment..."
echo "[TX-CLEAN] Log: $LOG"

env -i \
  HOME="$HOME" \
  USER="$USER" \
  LOGNAME="$LOGNAME" \
  PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  DISPLAY="${DISPLAY:-:0}" \
  XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}" \
  XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-$USER}" \
  PYTHONPATH="$(pwd)/apps" \
  /usr/bin/python3 apps/wifi_tx.py 2>&1 | tee "$LOG"
