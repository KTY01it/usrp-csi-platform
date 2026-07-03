#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

mkdir -p logs

SESSION_TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/tx_ch1_${SESSION_TS}.log"

echo "[TX-CH1] Starting WiFi TX on RF channel 1..."
echo "[TX-CH1] Log: $LOG"

export PYTHONPATH="$(pwd)/apps:${PYTHONPATH:-}"

python3 apps/wifi_tx_ch1.py 2>&1 | tee "$LOG"
