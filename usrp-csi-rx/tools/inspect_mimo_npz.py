#!/usr/bin/env python3
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz")
    args = parser.parse_args()

    d = np.load(args.npz)
    H = d["H_raw_mimo"]

    print("[MIMO] file:", args.npz)
    print("[MIMO] H_raw_mimo:", H.shape, H.dtype)

    N, K, R, T = H.shape
    print("[MIMO] frames:", N)
    print("[MIMO] subcarriers:", K)
    print("[MIMO] rx:", R)
    print("[MIMO] tx:", T)

    for tx in range(T):
        for rx in range(R):
            x = H[:, :, rx, tx]
            print(f"[LINK] rx{rx}-tx{tx} abs mean/std/min/max:",
                  float(np.abs(x).mean()),
                  float(np.abs(x).std()),
                  float(np.abs(x).min()),
                  float(np.abs(x).max()))

    # RX phase difference per TX: AoA proxy
    for tx in range(T):
        pd = np.angle(H[:, :, 1, tx] / (H[:, :, 0, tx] + 1e-12))
        print(f"[PHASE] rx phase diff tx{tx} mean/std:",
              float(pd.mean()), float(pd.std()))

    # TX phase difference per RX: AoD / virtual-TX proxy
    for rx in range(R):
        pd = np.angle(H[:, :, rx, 1] / (H[:, :, rx, 0] + 1e-12))
        print(f"[PHASE] tx phase diff rx{rx} mean/std:",
              float(pd.mean()), float(pd.std()))

if __name__ == "__main__":
    main()
