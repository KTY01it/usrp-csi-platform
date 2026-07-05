#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mimo_npz")
    parser.add_argument("--out", default=None)
    parser.add_argument("--nfft", type=int, default=64)
    parser.add_argument("--key", default="H_calibrated_mimo")
    args = parser.parse_args()

    path = Path(args.mimo_npz)
    out = Path(args.out) if args.out else path.with_name("H_cir_pdp_mimo.npz")

    d = np.load(path)

    if args.key in d:
        H = d[args.key]
    elif "H_raw_mimo" in d:
        H = d["H_raw_mimo"]
    else:
        raise SystemExit(f"[ERROR] no key {args.key} or H_raw_mimo in {path}")

    if H.ndim != 4 or H.shape[2] != 2 or H.shape[3] != 2:
        raise SystemExit(f"[ERROR] expected H [N,K,2,2], got {H.shape}")

    N, K, R, T = H.shape
    nfft = args.nfft
    if nfft < K:
        raise SystemExit(f"[ERROR] nfft={nfft} must be >= K={K}")

    # Center 52 active subcarriers into a 64-bin spectrum before IFFT.
    H_pad = np.zeros((N, nfft, R, T), dtype=np.complex64)
    start = (nfft - K) // 2
    H_pad[:, start:start + K, :, :] = H

    # Convert frequency-domain CSI to CIR proxy.
    h_cir = np.fft.ifft(np.fft.ifftshift(H_pad, axes=1), axis=1).astype(np.complex64)

    pdp = (np.abs(h_cir) ** 2).astype(np.float32)
    pdp_mean = pdp.mean(axis=0).astype(np.float32)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        h_cir_mimo=h_cir,
        pdp_mimo=pdp,
        pdp_mean_mimo=pdp_mean,
        source=str(path),
        source_key=args.key,
        nfft=np.array(nfft),
    )

    print(f"[OK] wrote: {out}")
    print("[MIMO-CIR] H:", H.shape)
    print("[MIMO-CIR] h_cir_mimo:", h_cir.shape)
    print("[MIMO-CIR] pdp_mimo:", pdp.shape)
    print("[MIMO-CIR] pdp_mean_mimo:", pdp_mean.shape)
    print("[MIMO-CIR] pdp mean/max:", float(pdp.mean()), float(pdp.max()))

    for tx in range(T):
        for rx in range(R):
            x = pdp_mean[:, rx, tx]
            peak = int(np.argmax(x))
            print(f"[LINK] rx{rx}-tx{tx} pdp_mean peak_bin={peak} peak={float(x[peak])} mean={float(x.mean())}")

if __name__ == "__main__":
    main()
