#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np

def npz_shape_summary(path):
    path = Path(path)
    if not path.exists():
        return {"exists": False, "path": str(path)}

    d = np.load(path)
    info = {"exists": True, "path": str(path), "keys": {}}
    for k in d.files:
        x = d[k]
        if hasattr(x, "shape"):
            info["keys"][k] = {
                "shape": list(x.shape),
                "dtype": str(x.dtype),
            }
        else:
            info["keys"][k] = str(x)
    return info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="csi/H_raw_mimo_pseudo.npz")
    parser.add_argument("--cal", default="csi/H_calibrated_mimo_pseudo.npz")
    parser.add_argument("--cir", default="csi/H_cir_pdp_mimo_pseudo.npz")
    parser.add_argument("--dd", default="csi/H_delay_doppler_mimo_pseudo.npz")
    parser.add_argument("--out", default="csi/mimo_dataset_manifest.json")
    args = parser.parse_args()

    manifest = {
        "dataset_type": "sequential_pseudo_mimo_2tx_2rx",
        "note": (
            "TX0 and TX1 are captured sequentially, not true simultaneous/TDM MIMO. "
            "Use tx-difference fields as virtual-TX/link evidence, not absolute AoD."
        ),
        "bandwidth_note": (
            "With 5 MHz bandwidth, CIR/PDP and delay-Doppler are delay-bin proxies, "
            "not radar-grade range/ToF measurements."
        ),
        "files": {
            "raw_mimo": npz_shape_summary(args.raw),
            "calibrated_mimo": npz_shape_summary(args.cal),
            "cir_pdp_mimo": npz_shape_summary(args.cir),
            "delay_doppler_mimo": npz_shape_summary(args.dd),
        },
        "available_fields": [
            "H_raw_mimo[frame, subcarrier, rx, tx]",
            "H_calibrated_mimo[frame, subcarrier, rx, tx]",
            "phase_diff_rx_raw[frame, subcarrier, tx]",
            "phase_diff_rx_calibrated[frame, subcarrier, tx]",
            "phase_diff_tx_raw[frame, subcarrier, rx]",
            "phase_diff_tx_calibrated[frame, subcarrier, rx]",
            "h_cir_mimo[frame, delay_bin, rx, tx]",
            "pdp_mimo[frame, delay_bin, rx, tx]",
            "pdp_mean_mimo[delay_bin, rx, tx]",
            "delay_doppler_mimo[doppler_bin, delay_bin, rx, tx]",
            "delay_doppler_power_mimo[doppler_bin, delay_bin, rx, tx]",
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(f"[OK] wrote: {out}")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
