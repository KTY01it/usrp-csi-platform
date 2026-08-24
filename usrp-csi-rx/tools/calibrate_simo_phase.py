#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

def circular_mean_phase(z, axis=0):
    return np.angle(np.mean(np.exp(1j * np.angle(z)), axis=axis))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", required=True, help="Background/static H_raw_simo.npz")
    parser.add_argument("--scene", required=True, help="Scene H_raw_simo.npz")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    bg = np.load(args.background)
    sc = np.load(args.scene)

    H_bg = bg["H_raw"]      # [N, K, 2, 1]
    H_sc = sc["H_raw"]

    if H_bg.shape[2] != 2 or H_sc.shape[2] != 2:
        raise SystemExit("[ERROR] Expected rx=2 in H_raw")

    Hb0 = H_bg[:, :, 0, 0]
    Hb1 = H_bg[:, :, 1, 0]
    Hs0 = H_sc[:, :, 0, 0]
    Hs1 = H_sc[:, :, 1, 0]

    # Per-subcarrier inter-RX phase offset from background
    ratio_bg = Hb1 / (Hb0 + 1e-12)
    phi_cal = circular_mean_phase(ratio_bg, axis=0)  # [K]

    # Raw and calibrated phase-difference for scene
    phase_diff_raw = np.angle(Hs1 / (Hs0 + 1e-12))
    phase_diff_cal = np.angle(np.exp(1j * (phase_diff_raw - phi_cal[None, :])))

    # Optional complex calibration: rotate RX1 by -phi_cal
    H_cal = H_sc.copy()
    H_cal[:, :, 1, 0] = H_cal[:, :, 1, 0] * np.exp(-1j * phi_cal[None, :])

    # Background-subtracted CSI
    H_bg_mean = np.mean(H_bg, axis=0, keepdims=True)
    H_delta = H_sc - H_bg_mean

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out,
        H_raw=H_sc.astype(np.complex64),
        H_calibrated=H_cal.astype(np.complex64),
        H_background_mean=H_bg_mean.astype(np.complex64),
        H_delta=H_delta.astype(np.complex64),
        phi_cal=phi_cal.astype(np.float32),
        phase_diff_raw=phase_diff_raw.astype(np.float32),
        phase_diff_calibrated=phase_diff_cal.astype(np.float32),
        background_file=str(args.background),
        scene_file=str(args.scene),
    )

    print(f"[OK] wrote: {out}")
    print("[CAL] H_raw:", H_sc.shape)
    print("[CAL] phi_cal shape:", phi_cal.shape)
    print("[CAL] phase raw mean/std:", float(phase_diff_raw.mean()), float(phase_diff_raw.std()))
    print("[CAL] phase cal mean/std:", float(phase_diff_cal.mean()), float(phase_diff_cal.std()))
    print("[CAL] H_delta abs mean:", float(np.abs(H_delta).mean()))

if __name__ == "__main__":
    main()
