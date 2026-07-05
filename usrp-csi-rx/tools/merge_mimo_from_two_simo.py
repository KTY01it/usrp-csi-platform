#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tx0", required=True)
    parser.add_argument("--tx1", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    d0 = np.load(args.tx0)
    d1 = np.load(args.tx1)

    H0 = d0["H_raw"]
    H1 = d1["H_raw"]

    if H0.shape[1] != H1.shape[1] or H0.shape[2] != 2 or H1.shape[2] != 2:
        raise SystemExit(f"[ERROR] Unexpected shapes: tx0={H0.shape}, tx1={H1.shape}")

    n = min(H0.shape[0], H1.shape[0])
    if H0.shape[0] != H1.shape[0]:
        print(f"[WARN] frame mismatch: tx0={H0.shape[0]}, tx1={H1.shape[0]}, using n={n}")

    H0 = H0[:n, :, :, 0]
    H1 = H1[:n, :, :, 0]

    H = np.stack([H0, H1], axis=3)

    phase_diff_rx_tx0 = np.angle(H[:, :, 1, 0] / (H[:, :, 0, 0] + 1e-12))
    phase_diff_rx_tx1 = np.angle(H[:, :, 1, 1] / (H[:, :, 0, 1] + 1e-12))
    phase_diff_tx_rx0 = np.angle(H[:, :, 0, 1] / (H[:, :, 0, 0] + 1e-12))
    phase_diff_tx_rx1 = np.angle(H[:, :, 1, 1] / (H[:, :, 1, 0] + 1e-12))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out,
        H_raw_mimo=H.astype(np.complex64),
        phase_diff_rx_tx0=phase_diff_rx_tx0.astype(np.float32),
        phase_diff_rx_tx1=phase_diff_rx_tx1.astype(np.float32),
        phase_diff_tx_rx0=phase_diff_tx_rx0.astype(np.float32),
        phase_diff_tx_rx1=phase_diff_tx_rx1.astype(np.float32),
        source_tx0=str(args.tx0),
        source_tx1=str(args.tx1),
    )

    print(f"[OK] wrote: {out}")
    print("[MIMO] H_raw_mimo:", H.shape)
    print("[MIMO] tx0 abs mean:", float(np.abs(H[:, :, :, 0]).mean()))
    print("[MIMO] tx1 abs mean:", float(np.abs(H[:, :, :, 1]).mean()))
    print("[MIMO] phase_diff_rx_tx0 mean/std:", float(phase_diff_rx_tx0.mean()), float(phase_diff_rx_tx0.std()))
    print("[MIMO] phase_diff_rx_tx1 mean/std:", float(phase_diff_rx_tx1.mean()), float(phase_diff_rx_tx1.std()))
    print("[MIMO] phase_diff_tx_rx0 mean/std:", float(phase_diff_tx_rx0.mean()), float(phase_diff_tx_rx0.std()))
    print("[MIMO] phase_diff_tx_rx1 mean/std:", float(phase_diff_tx_rx1.mean()), float(phase_diff_tx_rx1.std()))

if __name__ == "__main__":
    main()
