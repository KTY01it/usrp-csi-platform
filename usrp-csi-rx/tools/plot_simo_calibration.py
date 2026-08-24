#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("calibrated_npz")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    path = Path(args.calibrated_npz)
    out_dir = Path(args.out_dir) if args.out_dir else path.parent / "plots_calibrated"
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(path)
    raw = d["phase_diff_raw"]
    cal = d["phase_diff_calibrated"]
    phi = d["phi_cal"]
    H_delta = d["H_delta"]

    raw_u = np.unwrap(raw, axis=1)
    cal_u = np.unwrap(cal, axis=1)

    plt.figure()
    plt.plot(phi)
    plt.xlabel("Subcarrier index")
    plt.ylabel("Phase offset phi_cal [rad]")
    plt.title("Inter-RX phase calibration offset")
    out1 = out_dir / "phi_cal_subcarrier.png"
    plt.savefig(out1, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(raw_u.mean(axis=0), label="raw")
    plt.plot(cal_u.mean(axis=0), label="calibrated")
    plt.xlabel("Subcarrier index")
    plt.ylabel("Mean unwrapped phase diff [rad]")
    plt.title("RX phase difference before/after calibration")
    plt.legend()
    out2 = out_dir / "phase_diff_raw_vs_cal_subcarrier.png"
    plt.savefig(out2, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(raw_u.mean(axis=1), label="raw")
    plt.plot(cal_u.mean(axis=1), label="calibrated")
    plt.xlabel("Frame index")
    plt.ylabel("Mean unwrapped phase diff [rad]")
    plt.title("RX phase difference over time")
    plt.legend()
    out3 = out_dir / "phase_diff_raw_vs_cal_time.png"
    plt.savefig(out3, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(np.abs(H_delta[:, :, 0, 0]).mean(axis=1), label="RX0")
    plt.plot(np.abs(H_delta[:, :, 1, 0]).mean(axis=1), label="RX1")
    plt.xlabel("Frame index")
    plt.ylabel("Mean |H_delta|")
    plt.title("Background-subtracted SIMO CSI magnitude")
    plt.legend()
    out4 = out_dir / "h_delta_magnitude_time.png"
    plt.savefig(out4, dpi=160, bbox_inches="tight")
    plt.close()

    print("[OK] plots:")
    print(out1)
    print(out2)
    print(out3)
    print(out4)

if __name__ == "__main__":
    main()
