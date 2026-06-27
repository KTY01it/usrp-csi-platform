#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

echo "[RX] Starting WiFi Rx baseline..."
echo "[RX] Working directory: $(pwd)"

python3 apps/wifi_rx.py
