#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

mkdir -p logs csi

SESSION_TS="$(date +%Y%m%d_%H%M%S)"
CSI_DIR="csi/min_${SESSION_TS}"
META_FILE="logs/rx_csi_min_${SESSION_TS}.json"
CONSOLE_LOG="logs/rx_csi_min_${SESSION_TS}.log"

mkdir -p "$CSI_DIR"

cat > "$META_FILE" <<META
{
  "role": "rx",
  "project": "usrp-csi-rx",
  "stage": "siso_csi_min_from_working_wifi_rx",
  "freq": 5890000000,
  "samp_rate": 5000000,
  "rx_gain": 0.75,
  "csi_dir": "$CSI_DIR",
  "created_at": "$SESSION_TS"
}
META

echo "[RX-CSI-MIN] Starting minimal CSI capture..."
echo "[RX-CSI-MIN] CSI dir: $CSI_DIR"
echo "[RX-CSI-MIN] Log: $CONSOLE_LOG"

export RX_CSI_DIR="$CSI_DIR"
export PYTHONPATH="$(pwd)/tools:$(pwd)/apps:${PYTHONPATH:-}"

python3 apps/wifi_rx_csi_min.py 2>&1 | tee "$CONSOLE_LOG"
