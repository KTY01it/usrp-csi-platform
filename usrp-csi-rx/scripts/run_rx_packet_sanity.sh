#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

mkdir -p logs data

SESSION_TS="$(date +%Y%m%d_%H%M%S)"
META_FILE="logs/rx_packet_sanity_${SESSION_TS}.json"
CONSOLE_LOG="logs/rx_packet_sanity_${SESSION_TS}.log"

cat > "$META_FILE" <<META
{
  "role": "rx",
  "project": "usrp-csi-rx",
  "stage": "siso_packet_sanity_no_raw_iq",
  "freq": 5890000000,
  "samp_rate": 5000000,
  "lo_offset": 0,
  "rx_gain": 0.75,
  "channels": [0],
  "raw_iq_output": "/dev/null",
  "created_at": "$SESSION_TS"
}
META

echo "[RX] Starting packet sanity mode..."
echo "[RX] Working directory: $(pwd)"
echo "[RX] Metadata: $META_FILE"
echo "[RX] Console log: $CONSOLE_LOG"
echo "[RX] Raw IQ output: /dev/null"

export RX_RAW_IQ_FILE="/dev/null"

python3 apps/wifi_rx.py 2>&1 | tee "$CONSOLE_LOG"
