#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p logs

SESSION_TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/tx_dualport_ch1_${SESSION_TS}.log"

echo "[TX-DUALPORT-CH1] Starting WiFi TX on physical channel 1 using channels=[0,1]..."
echo "[TX-DUALPORT-CH1] TX0 port: zero"
echo "[TX-DUALPORT-CH1] TX1 port: WiFi waveform"
echo "[TX-DUALPORT-CH1] Log: $LOG"

env -i \
  HOME="$HOME" \
  USER="$USER" \
  LOGNAME="$LOGNAME" \
  PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  DISPLAY="${DISPLAY:-:0}" \
  XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}" \
  XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-$USER}" \
  PYTHONPATH="$(pwd)/apps" \
  /usr/bin/python3 apps/wifi_tx_dualport_ch1.py 2>&1 | tee "$LOG"
