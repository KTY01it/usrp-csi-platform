#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

def load_bin(path, nsub):
    x = np.fromfile(path, dtype=np.complex64)
    n_frames = x.size // nsub
    rem = x.size % nsub
    if n_frames == 0:
        raise SystemExit(f"[ERROR] No complete CSI frames in {path}")
    if rem != 0:
        print(f"[WARN] {path}: dropping trailing {rem} complex samples")
        x = x[:n_frames * nsub]
    return x.reshape(n_frames, nsub)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csi_dir")
    parser.add_argument("--nsub", type=int, default=52)
    parser.add_argument("--out", default=None)
    parser.add_argument("--center-freq", type=float, default=5890000000)
    parser.add_argument("--sample-rate", type=float, default=5000000)
    args = parser.parse_args()

    csi_dir = Path(args.csi_dir)
    ch0 = csi_dir / "csi_ch0.bin"
    ch1 = csi_dir / "csi_ch1.bin"

    H0 = load_bin(ch0, args.nsub)
    H1 = load_bin(ch1, args.nsub)

    n = min(H0.shape[0], H1.shape[0])
    if H0.shape[0] != H1.shape[0]:
        print(f"[WARN] frame mismatch: ch0={H0.shape[0]}, ch1={H1.shape[0]}, using n={n}")

    H0 = H0[:n]
    H1 = H1[:n]

    H = np.stack([H0, H1], axis=2)   # [frame, subcarrier, rx]
    H = H[:, :, :, None]             # [frame, subcarrier, rx, tx]

    phase_diff_rx = np.angle(H[:, :, 1, 0] / (H[:, :, 0, 0] + 1e-12))
    amp_ratio_rx = np.abs(H[:, :, 1, 0]) / (np.abs(H[:, :, 0, 0]) + 1e-12)

    out = Path(args.out) if args.out else csi_dir / "H_raw_simo.npz"

    np.savez_compressed(
        out,
        H_raw=H.astype(np.complex64),
        phase_diff_rx=phase_diff_rx.astype(np.float32),
        amp_ratio_rx=amp_ratio_rx.astype(np.float32),
        frame_index=np.arange(n, dtype=np.int64),
        subcarrier_index=np.arange(args.nsub, dtype=np.int64),
        center_freq=np.float64(args.center_freq),
        sample_rate=np.float64(args.sample_rate),
        nsub=np.int64(args.nsub),
        rx=np.int64(2),
        tx=np.int64(1),
        source_ch0=str(ch0),
        source_ch1=str(ch1),
    )

    print(f"[OK] wrote: {out}")
    print(f"[SIMO] H_raw shape: {H.shape}")
    print(f"[SIMO] ch0 abs mean: {np.abs(H[:, :, 0, 0]).mean():.6g}")
    print(f"[SIMO] ch1 abs mean: {np.abs(H[:, :, 1, 0]).mean():.6g}")
    print(f"[SIMO] phase_diff_rx mean/std: {phase_diff_rx.mean():.6g} / {phase_diff_rx.std():.6g}")
    print(f"[SIMO] amp_ratio_rx mean/std: {amp_ratio_rx.mean():.6g} / {amp_ratio_rx.std():.6g}")

if __name__ == "__main__":
    main()
