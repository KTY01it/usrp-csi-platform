#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

mkdir -p logs csi data

SESSION_TS="$(date +%Y%m%d_%H%M%S)"
CSI_DIR="csi/simo_${SESSION_TS}"
META_FILE="logs/rx_simo_csi_${SESSION_TS}.json"
CONSOLE_LOG="logs/rx_simo_csi_${SESSION_TS}.log"

mkdir -p "$CSI_DIR"

cat > "$META_FILE" <<META
{
  "role": "rx",
  "project": "usrp-csi-rx",
  "stage": "simo_stage1_ch0_csi_ch1_csi",
  "freq": 5890000000,
  "samp_rate": 5000000,
  "rx_gain": 0.75,
  "rx_channels": [0, 1],
  "tx_channels": [0],
  "nsub": 52,
  "csi_dir": "$CSI_DIR",
  "created_at": "$SESSION_TS"
}
META

echo "[RX-SIMO-CSI] Starting SIMO stage-1 capture..."
echo "[RX-SIMO-CSI] ch0 CSI dir: $CSI_DIR"
echo "[RX-SIMO-CSI] ch1 path: full WiFi sync/FFT/CSI path"
echo "[RX-SIMO-CSI] Log: $CONSOLE_LOG"

export RX_CSI_DIR="$CSI_DIR"
export RX_RAW_IQ_CH1_FILE="/dev/null"
export PYTHONPATH="$(pwd)/tools:$(pwd)/apps:${PYTHONPATH:-}"

python3 apps/wifi_rx_simo_csi.py 2>&1 | tee "$CONSOLE_LOG"
