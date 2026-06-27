#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

mkdir -p data logs

echo "[RX] Starting WiFi Rx baseline..."
echo "[RX] Working directory: $(pwd)"
echo "[RX] Raw IQ output: data/rx_raw_iq.fc32"

python3 apps/wifi_rx.py
