#!/usr/bin/env python3
import argparse
import numpy as np

def topk_indices(x, k):
    flat = x.ravel()
    k = min(k, flat.size)
    idx = np.argpartition(flat, -k)[-k:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    return [np.unravel_index(int(i), x.shape) for i in idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("volume_npz")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    d = np.load(args.volume_npz)
    V = d["rf_volume"]
    C = d["voxel_confidence"]
    xs = d["grid_x"]
    ys = d["grid_y"]
    zs = d["grid_z"]

    print("[RFV] file:", args.volume_npz)
    print("[RFV] rf_volume:", V.shape, V.dtype)
    print("[RFV] confidence:", C.shape, C.dtype)
    print("[RFV] value mean/std/min/max:",
          float(V.mean()), float(V.std()), float(V.min()), float(V.max()))
    print("[RFV] conf mean/std/min/max:",
          float(C.mean()), float(C.std()), float(C.min()), float(C.max()))

    print(f"[RFV] top-{args.topk} peaks:")
    for rank, idx in enumerate(topk_indices(V, args.topk), start=1):
        ix, iy, iz = idx
        print(
            f"{rank:02d}: idx={idx} "
            f"xyz=({float(xs[ix]):.3f}, {float(ys[iy]):.3f}, {float(zs[iz]):.3f}) "
            f"value={float(V[idx]):.6f} conf={float(C[idx]):.6f}"
        )

if __name__ == "__main__":
    main()
