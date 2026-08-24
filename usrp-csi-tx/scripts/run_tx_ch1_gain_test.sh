#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

mkdir -p logs

SESSION_TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/tx_ch1_gain_test_${SESSION_TS}.log"

echo "[TX-CH1-GAIN-TEST] Starting WiFi TX channel [1] with adjusted gain..."
echo "[TX-CH1-GAIN-TEST] Log: $LOG"

env -i \
  HOME="$HOME" \
  USER="$USER" \
  LOGNAME="$LOGNAME" \
  PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  DISPLAY="${DISPLAY:-:0}" \
  XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}" \
  XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-$USER}" \
  PYTHONPATH="$(pwd)/apps" \
  /usr/bin/python3 apps/wifi_tx_ch1_gain_test.py 2>&1 | tee "$LOG"
