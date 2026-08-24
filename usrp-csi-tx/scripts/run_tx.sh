#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

mkdir -p logs data

SESSION_TS="$(date +%Y%m%d_%H%M%S)"
META_FILE="logs/tx_session_${SESSION_TS}.json"

python3 - <<PYMETA
import json
from pathlib import Path

meta = {
    "role": "tx",
    "project": "usrp-csi-tx",
    "stage": "siso_tx_baseline",
    "freq": 5890000000,
    "samp_rate": 5000000,
    "lo_offset": 0,
    "tx_gain": 0.75,
    "channels": [0],
    "pdu_length": 100,
    "interval_ms": 1000,
    "created_at": "$SESSION_TS",
}
Path("$META_FILE").write_text(json.dumps(meta, indent=2) + "\n")
PYMETA

echo "[TX] Starting WiFi Tx baseline..."
echo "[TX] Working directory: $(pwd)"
echo "[TX] Metadata: $META_FILE"

export UHD_IMAGES_DIR="${UHD_IMAGES_DIR:-/usr/share/uhd/images}"
echo "[TX] UHD_IMAGES_DIR: $UHD_IMAGES_DIR"
echo "[TX] Python: /usr/bin/python3"

unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset CONDA_SHLVL
unset PYTHONHOME
unset PYTHONPATH

echo "[TX] Checking USRP USB mode..."
USB_MODE="$(uhd_usrp_probe 2>&1 | grep -i 'Operating over USB' || true)"
echo "[TX] $USB_MODE"

if ! echo "$USB_MODE" | grep -q "USB 3"; then
    echo "[TX][ERROR] USRP is not operating over USB 3. Fix cable/port before running Tx."
    exit 1
fi

env -u LD_LIBRARY_PATH \
    -u LD_PRELOAD \
    -u PYTHONPATH \
    -u PYTHONHOME \
    -u CONDA_PREFIX \
    -u CONDA_DEFAULT_ENV \
    -u CONDA_SHLVL \
    -u SNAP \
    -u SNAP_NAME \
    -u SNAP_INSTANCE_NAME \
    -u SNAP_REVISION \
    -u SNAP_ARCH \
    -u SNAP_VERSION \
    -u SNAP_COOKIE \
    /usr/bin/python3 apps/wifi_tx.py
