#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("calibrated_npz")
    parser.add_argument("--out", default=None)
    parser.add_argument("--nfft", type=int, default=64)
    args = parser.parse_args()

    path = Path(args.calibrated_npz)
    d = np.load(path)

    H = d["H_calibrated"]  # [N, K=52, RX=2, TX=1]
    N, K, RX, TX = H.shape

    # Put 52 active subcarriers into a 64-bin vector.
    # This is a proxy mapping for current CSI extractor output.
    H64 = np.zeros((N, args.nfft, RX, TX), dtype=np.complex64)

    # Use centered placement: 26 left + 26 right around DC, leaving DC unused.
    if K != 52 or args.nfft != 64:
        raise SystemExit("[ERROR] This utility currently assumes K=52 and nfft=64")

    H64[:, 1:27, :, :] = H[:, 26:52, :, :]
    H64[:, -26:, :, :] = H[:, 0:26, :, :]

    h_cir = np.fft.ifft(np.fft.ifftshift(H64, axes=1), axis=1)
    pdp = np.abs(h_cir) ** 2
    pdp_mean = pdp.mean(axis=0)  # [delay, rx, tx]

    out = Path(args.out) if args.out else path.parent / "H_cir_pdp_simo.npz"

    np.savez_compressed(
        out,
        H_calibrated=H.astype(np.complex64),
        h_cir=h_cir.astype(np.complex64),
        pdp=pdp.astype(np.float32),
        pdp_mean=pdp_mean.astype(np.float32),
        nfft=np.int64(args.nfft),
        source=str(path),
    )

    print(f"[OK] wrote: {out}")
    print("[CIR] H:", H.shape)
    print("[CIR] h_cir:", h_cir.shape)
    print("[CIR] pdp_mean:", pdp_mean.shape)
    print("[CIR] pdp mean/max:", float(pdp.mean()), float(pdp.max()))

if __name__ == "__main__":
    main()
