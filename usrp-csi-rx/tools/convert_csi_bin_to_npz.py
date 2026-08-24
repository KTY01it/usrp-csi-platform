#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bin_file")
    parser.add_argument("--nsub", type=int, default=52)
    parser.add_argument("--rx", type=int, default=1)
    parser.add_argument("--tx", type=int, default=1)
    parser.add_argument("--out", default=None)
    parser.add_argument("--center-freq", type=float, default=5890000000)
    parser.add_argument("--sample-rate", type=float, default=5000000)
    args = parser.parse_args()

    path = Path(args.bin_file)
    x = np.fromfile(path, dtype=np.complex64)

    per_frame = args.nsub * args.rx * args.tx
    n_frames = x.size // per_frame
    rem = x.size % per_frame

    if n_frames == 0:
        raise SystemExit(f"[ERROR] No complete CSI frame. complex_count={x.size}, per_frame={per_frame}")

    if rem != 0:
        print(f"[WARN] Dropping trailing {rem} complex samples")
        x = x[:n_frames * per_frame]

    H = x.reshape(n_frames, args.nsub, args.rx, args.tx)

    out = Path(args.out) if args.out else path.with_suffix(".npz")

    np.savez_compressed(
        out,
        H_raw=H,
        frame_index=np.arange(n_frames, dtype=np.int64),
        subcarrier_index=np.arange(args.nsub, dtype=np.int64),
        center_freq=np.float64(args.center_freq),
        sample_rate=np.float64(args.sample_rate),
        nsub=np.int64(args.nsub),
        rx=np.int64(args.rx),
        tx=np.int64(args.tx),
        source_bin=str(path),
    )

    print(f"[OK] wrote: {out}")
    print(f"[CSI] H_raw shape: {H.shape}")
    print(f"[CSI] abs mean/min/max: {np.abs(H).mean():.6g} / {np.abs(H).min():.6g} / {np.abs(H).max():.6g}")

if __name__ == "__main__":
    main()
