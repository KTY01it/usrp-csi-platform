#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

mkdir -p logs data

SESSION_TS="$(date +%Y%m%d_%H%M%S)"
META_FILE="logs/tx_session_${SESSION_TS}.json"

python3 - <<PY
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
PY

echo "[TX] Starting WiFi Tx baseline..."
echo "[TX] Working directory: $(pwd)"
echo "[TX] Metadata: $META_FILE"

python3 apps/wifi_tx.py
