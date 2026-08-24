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
    out = Path(args.out) if args.out else path.with_name("H_delay_doppler_mimo.npz")

    d = np.load(path)
    h = d["h_cir_mimo"]  # [N, delay, RX, TX]

    if h.ndim != 4 or h.shape[2] != 2 or h.shape[3] != 2:
        raise SystemExit(f"[ERROR] expected h_cir_mimo [N,D,2,2], got {h.shape}")

    N, D, R, T = h.shape

    x = h.copy()
    if args.window:
        win = np.hanning(N).astype(np.float32)
        x = x * win[:, None, None, None]

    # Doppler FFT along frame/time axis.
    DD = np.fft.fftshift(np.fft.fft(x, axis=0), axes=0).astype(np.complex64)
    DD_power = (np.abs(DD) ** 2).astype(np.float32)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        delay_doppler_mimo=DD,
        delay_doppler_power_mimo=DD_power,
        source=str(path),
        window=np.array(bool(args.window)),
    )

    print(f"[OK] wrote: {out}")
    print("[MIMO-DD] delay_doppler_mimo:", DD.shape)
    print("[MIMO-DD] delay_doppler_power_mimo:", DD_power.shape)
    print("[MIMO-DD] power mean/max:", float(DD_power.mean()), float(DD_power.max()))

    for tx in range(T):
        for rx in range(R):
            p = DD_power[:, :, rx, tx]
            idx = np.unravel_index(np.argmax(p), p.shape)
            print(
                f"[LINK] rx{rx}-tx{tx} peak_doppler_bin={idx[0]} "
                f"peak_delay_bin={idx[1]} peak={float(p[idx])} mean={float(p.mean())}"
            )

if __name__ == "__main__":
    main()
