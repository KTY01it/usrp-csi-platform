#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bin_file")
    parser.add_argument("--nsub", type=int, default=64)
    args = parser.parse_args()

    path = Path(args.bin_file)
    x = np.fromfile(path, dtype=np.complex64)

    print(f"[CSI] file: {path}")
    print(f"[CSI] bytes: {path.stat().st_size}")
    print(f"[CSI] complex64_count: {x.size}")

    if x.size == 0:
        print("[CSI] empty")
        return

    n_frames_floor = x.size // args.nsub
    rem = x.size % args.nsub

    print(f"[CSI] assuming nsub={args.nsub}")
    print(f"[CSI] frames_floor: {n_frames_floor}")
    print(f"[CSI] remainder: {rem}")
    print(f"[CSI] abs_mean: {np.abs(x).mean():.6g}")
    print(f"[CSI] abs_min: {np.abs(x).min():.6g}")
    print(f"[CSI] abs_max: {np.abs(x).max():.6g}")

if __name__ == "__main__":
    main()
