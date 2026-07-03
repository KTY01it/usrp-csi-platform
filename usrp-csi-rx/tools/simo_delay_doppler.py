#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cir_npz")
    parser.add_argument("--out", default=None)
    parser.add_argument("--window", action="store_true")
    args = parser.parse_args()

    path = Path(args.cir_npz)
    d = np.load(path)

    h = d["h_cir"]  # [frame, delay, rx, tx]
    N = h.shape[0]

    if args.window:
        w = np.hanning(N).astype(np.float32)
        h_use = h * w[:, None, None, None]
    else:
        h_use = h

    DD = np.fft.fftshift(np.fft.fft(h_use, axis=0), axes=0)
    DD_power = np.abs(DD) ** 2

    out = Path(args.out) if args.out else path.parent / "H_delay_doppler_simo.npz"

    np.savez_compressed(
        out,
        delay_doppler=DD.astype(np.complex64),
        delay_doppler_power=DD_power.astype(np.float32),
        doppler_bin=np.arange(-N // 2, -N // 2 + N, dtype=np.int64),
        source=str(path),
        window=np.bool_(args.window),
    )

    print(f"[OK] wrote: {out}")
    print("[DD] delay_doppler:", DD.shape)
    print("[DD] power mean/max:", float(DD_power.mean()), float(DD_power.max()))

if __name__ == "__main__":
    main()
