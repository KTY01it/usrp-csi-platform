#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("volume_npz")
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--min_conf", type=float, default=0.20)
    parser.add_argument("--topk", type=int, default=500)
    args = parser.parse_args()

    d = np.load(args.volume_npz)
    V = d["rf_volume"]
    C = d["voxel_confidence"]
    xs = d["grid_x"]
    ys = d["grid_y"]
    zs = d["grid_z"]

    mask = (V >= args.threshold) & (C >= args.min_conf)
    idx = np.argwhere(mask)

    if idx.size == 0:
        print("[WARN] no voxels passed threshold; falling back to top-k by RF value")
        flat_idx = np.argsort(V.ravel())[::-1][:args.topk]
        idx = np.array([np.unravel_index(i, V.shape) for i in flat_idx], dtype=np.int64)
    else:
        values = V[tuple(idx.T)]
        order = np.argsort(values)[::-1][:args.topk]
        idx = idx[order]

    points = np.zeros((idx.shape[0], 3), dtype=np.float32)
    values = np.zeros((idx.shape[0],), dtype=np.float32)
    confs = np.zeros((idx.shape[0],), dtype=np.float32)

    for i, (ix, iy, iz) in enumerate(idx):
        points[i] = [xs[ix], ys[iy], zs[iz]]
        values[i] = V[ix, iy, iz]
        confs[i] = C[ix, iy, iz]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        points_xyz=points,
        values=values,
        confidence=confs,
        voxel_indices=idx.astype(np.int64),
        source=args.volume_npz,
        threshold=np.array(args.threshold, dtype=np.float32),
        min_conf=np.array(args.min_conf, dtype=np.float32),
    )

    print(f"[OK] wrote: {out}")
    print("[CAND] points:", points.shape)
    print("[CAND] value mean/max:", float(values.mean()), float(values.max()))
    print("[CAND] conf mean/max:", float(confs.mean()), float(confs.max()))
    print("[CAND] first 10:")
    for i in range(min(10, len(points))):
        print(i, points[i].tolist(), "value=", float(values[i]), "conf=", float(confs[i]))

if __name__ == "__main__":
    main()
