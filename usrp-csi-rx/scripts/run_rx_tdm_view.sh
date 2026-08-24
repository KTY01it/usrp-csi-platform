#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OUTPUT_DIR=""
DURATION="10"
SNR_MIN="10"
DRY_RUN="0"

usage() {
    cat <<EOF
Usage:
  $0 --output DIR [--duration SEC] [--snr-min DB] [--dry-run]

Options:
  --output DIR       CSI output directory for one view.
  --duration SEC     Capture duration in seconds. Default: 10.
  --snr-min DB       Minimum SNR for TDM tensor builder. Default: 10.
  --dry-run          Validate configuration without starting USRP.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --snr-min)
            SNR_MIN="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="1"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1"
            usage
            exit 2
            ;;
    esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "ERROR: --output is required"
    exit 2
fi

if ! [[ "$DURATION" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "ERROR: invalid duration: $DURATION"
    exit 2
fi

OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"

APP="$ROOT/apps/wifi_rx_simo_csi.py"
BUILDER="$ROOT/tools/build_tdm_mimo_tensor.py"

echo "=============================================================="
echo "CSI-SENSE RX 2x2 TDM VIEW BACKEND"
echo "=============================================================="
echo "Root        : $ROOT"
echo "Output      : $OUTPUT_DIR"
echo "Duration    : $DURATION s"
echo "SNR minimum : $SNR_MIN dB"
echo "RX channels : [0,1]"
echo "TX mapping  : even->TX0, odd->TX1"
echo

if [[ ! -f "$APP" ]]; then
    echo "FAIL: RX application not found:"
    echo "  $APP"
    exit 3
fi

if [[ ! -f "$BUILDER" ]]; then
    echo "FAIL: TDM tensor builder not found:"
    echo "  $BUILDER"
    exit 3
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "FAIL: python3 not found"
    exit 3
fi

if ! command -v uhd_find_devices >/dev/null 2>&1; then
    echo "FAIL: UHD not found"
    exit 3
fi

echo "[BACKEND]"
echo "RX app             PASS"
echo "TDM tensor builder PASS"
echo "Python              PASS"
echo "UHD                 PASS"

if [[ "$DRY_RUN" == "1" ]]; then
    echo
    echo "[DRY RUN]"
    echo "No USRP acquisition started."
    echo
    echo "Backend adapter: PASS"
    exit 0
fi

mkdir -p "$OUTPUT_DIR"

LOG_DIR="$(dirname "$OUTPUT_DIR")/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/rx_backend.log"
META_FILE="$LOG_DIR/rx_backend.json"

cat > "$META_FILE" <<EOF
{
  "backend": "wifi_rx_simo_csi",
  "mode": "physical_2x2_packet_tdm",
  "rx_channels": [0, 1],
  "tx_mapping": {
    "even_sequence": "TX0",
    "odd_sequence": "TX1"
  },
  "duration_s": $DURATION,
  "snr_min_db": $SNR_MIN,
  "output_dir": "$OUTPUT_DIR"
}
EOF

export RX_CSI_DIR="$OUTPUT_DIR"
export RX_RAW_IQ_FILE="/dev/null"
export RX_RAW_IQ_CH1_FILE="/dev/null"

export PYTHONPATH="$ROOT/tools:$ROOT/apps:${PYTHONPATH:-}"

echo
echo "[ACQUISITION]"
echo "Starting RX..."

set +e

timeout \
    --signal=INT \
    --kill-after=5s \
    "${DURATION}s" \
    python3 "$APP" \
    2>&1 | tee "$LOG_FILE"

ACQ_STATUS=${PIPESTATUS[0]}

set -e

#
# timeout normally returns 124 when duration expires.
# For our fixed-duration acquisition that is expected.
#
if [[ "$ACQ_STATUS" -ne 0 && "$ACQ_STATUS" -ne 124 && "$ACQ_STATUS" -ne 130 ]]; then
    echo
    echo "FAIL: RX acquisition exited with status $ACQ_STATUS"
    exit "$ACQ_STATUS"
fi

echo
echo "[CSI FILE CHECK]"

required=(
    "$OUTPUT_DIR/csi_pdu_ch0.bin"
    "$OUTPUT_DIR/csi_pdu_ch0.jsonl"
    "$OUTPUT_DIR/csi_pdu_ch1.bin"
    "$OUTPUT_DIR/csi_pdu_ch1.jsonl"
)

for f in "${required[@]}"; do
    if [[ ! -s "$f" ]]; then
        echo "FAIL: missing or empty:"
        echo "  $f"
        exit 4
    fi

    echo "PASS $(basename "$f")"
done

echo
echo "[TDM BUILD]"

python3 "$BUILDER" \
    "$OUTPUT_DIR" \
    --snr-min "$SNR_MIN" \
    --output "$OUTPUT_DIR/H_raw_tdm_physical_2x2.npz"

if [[ ! -s "$OUTPUT_DIR/H_raw_tdm_physical_2x2.npz" ]]; then
    echo "FAIL: 2x2 MIMO tensor was not produced"
    exit 5
fi

echo
echo "=============================================================="
echo "RX VIEW ACQUISITION: PASS"
echo "=============================================================="
echo "Tensor:"
echo "  $OUTPUT_DIR/H_raw_tdm_physical_2x2.npz"
