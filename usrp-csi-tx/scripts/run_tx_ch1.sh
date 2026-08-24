#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

mkdir -p logs

SESSION_TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/tx_ch1_${SESSION_TS}.log"

echo "[TX-CH1] Starting WiFi TX on RF channel 1..."
echo "[TX-CH1] Log: $LOG"

export PYTHONPATH="$(pwd)/apps:${PYTHONPATH:-}"

env -i \
  HOME="$HOME" \
  USER="$USER" \
  LOGNAME="$LOGNAME" \
  PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  DISPLAY="${DISPLAY:-:0}" \
  XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}" \
  XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-$USER}" \
  PYTHONPATH="$(pwd)/apps" \
  /usr/bin/python3 apps/wifi_tx_ch1.py 2>&1 | tee "$LOG"
