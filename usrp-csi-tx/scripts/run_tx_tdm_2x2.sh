#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs

SESSION_TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/tx_tdm_2x2_${SESSION_TS}.log"

echo "[TX-TDM-2X2] Starting true alternating 2Tx TDM WiFi..."
echo "[TX-TDM-2X2] even packet -> TX0"
echo "[TX-TDM-2X2] odd packet  -> TX1"
echo "[TX-TDM-2X2] Log: $LOG"

env -i \
  HOME="$HOME" \
  USER="$USER" \
  LOGNAME="$LOGNAME" \
  PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  DISPLAY="${DISPLAY:-:0}" \
  XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}" \
  XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-$USER}" \
  TDM_ACTIVE_TX="${TDM_ACTIVE_TX:-both}" \
  PYTHONPATH="$(pwd)/apps" \
  /usr/bin/python3 apps/wifi_tx_tdm_2x2.py 2>&1 | tee "$LOG"
