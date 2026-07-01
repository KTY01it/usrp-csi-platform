#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

mkdir -p logs data csi

SESSION_TS="$(date +%Y%m%d_%H%M%S)"
META_FILE="logs/rx_siso_csi_${SESSION_TS}.json"
CONSOLE_LOG="logs/rx_siso_csi_${SESSION_TS}.log"
CSI_DIR="csi/siso_${SESSION_TS}"

mkdir -p "$CSI_DIR"

cat > "$META_FILE" <<META
{
  "role": "rx",
  "project": "usrp-csi-rx",
  "stage": "siso_csi_extraction",
  "freq": 5890000000,
  "samp_rate": 5000000,
  "rx_gain": 0.75,
  "channels": [0],
  "csi_dir": "$CSI_DIR",
  "created_at": "$SESSION_TS"
}
META

echo "[RX-CSI] Starting SISO CSI extraction..."
echo "[RX-CSI] Working directory: $(pwd)"
echo "[RX-CSI] Metadata: $META_FILE"
echo "[RX-CSI] Console log: $CONSOLE_LOG"
echo "[RX-CSI] CSI dir: $CSI_DIR"

export RX_CSI_DIR="$CSI_DIR"
export RX_CSI_SESSION="$SESSION_TS"

python3 apps/wifi_rx_siso_csi.py 2>&1 | tee "$CONSOLE_LOG"
