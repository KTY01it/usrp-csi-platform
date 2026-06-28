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
    "samp_rate": 10000000,
    "lo_offset": 0,
    "tx_gain": 0.75,
    "channels": [0],
    "created_at": "$SESSION_TS",
}
Path("$META_FILE").write_text(json.dumps(meta, indent=2) + "\n")
PYMETA

echo "[TX] Starting WiFi Tx baseline..."
echo "[TX] Working directory: $(pwd)"
echo "[TX] Metadata: $META_FILE"
export UHD_IMAGES_DIR="${UHD_IMAGES_DIR:-/usr/share/uhd/images}"
echo "[TX] UHD_IMAGES_DIR: $UHD_IMAGES_DIR"

# Remove common Snap/VSCode environment variables that can make Python load
# incompatible /snap/core20 libraries.
unset LD_LIBRARY_PATH
unset SNAP
unset SNAP_NAME
unset SNAP_ARCH
unset SNAP_VERSION
unset SNAP_LIBRARY_PATH
unset GTK_PATH
unset GIO_MODULE_DIR
unset GSETTINGS_SCHEMA_DIR
unset QT_PLUGIN_PATH
unset PYTHONPATH

python3 apps/wifi_tx.py
