#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

mkdir -p data logs

SESSION_TS="$(date +%Y%m%d_%H%M%S)"
RAW_IQ_FILE="data/rx_raw_iq_${SESSION_TS}.fc32"
META_FILE="logs/rx_session_${SESSION_TS}.json"

cat > "$META_FILE" <<META
{
  "role": "rx",
  "project": "usrp-csi-rx",
  "stage": "siso_raw_iq_baseline",
  "freq": 5890000000,
  "samp_rate": 10000000,
  "lo_offset": 0,
  "rx_gain": 0.75,
  "channels": [0],
  "raw_iq_output": "$RAW_IQ_FILE",
  "created_at": "$SESSION_TS"
}
META

echo "[RX] Starting WiFi Rx baseline..."
echo "[RX] Working directory: $(pwd)"
echo "[RX] Metadata: $META_FILE"
echo "[RX] Raw IQ output configured in wifi_rx.py: data/rx_raw_iq.fc32"
echo "[RX] Note: current wifi_rx.py still writes to fixed path data/rx_raw_iq.fc32"

python3 apps/wifi_rx.py
