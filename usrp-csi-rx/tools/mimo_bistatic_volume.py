#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np

C = 299792458.0

def normalize01(x):
    x = np.asarray(x, dtype=np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx <= mn + 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cir", required=True, help="NPZ containing pdp_mean_mimo")
    parser.add_argument("--geom", required=True, help="JSON geometry metadata")
    parser.add_argument("--out", required=True)
    parser.add_argument("--key", default="pdp_mean_mimo")
    parser.add_argument("--subtract_min_path", action="store_true",
                        help="Use excess path length relative to direct TX-RX path")
    args = parser.parse_args()

    cir_path = Path(args.cir)
    geom_path = Path(args.geom)
    out_path = Path(args.out)

    d = np.load(cir_path)
    if args.key not in d:
        raise SystemExit(f"[ERROR] key {args.key} not found in {cir_path}. keys={d.files}")

    pdp = d[args.key]  # [delay, rx, tx]
    if pdp.ndim != 3 or pdp.shape[1] != 2 or pdp.shape[2] != 2:
        raise SystemExit(f"[ERROR] expected pdp_mean_mimo [D,2,2], got {pdp.shape}")

    with open(geom_path, "r") as f:
        geom = json.load(f)

    tx_pos = np.asarray(geom["tx_positions"], dtype=np.float64)  # [2,3]
    rx_pos = np.asarray(geom["rx_positions"], dtype=np.float64)  # [2,3]

    if tx_pos.shape != (2, 3) or rx_pos.shape != (2, 3):
        raise SystemExit(f"[ERROR] expected tx/rx positions [2,3], got tx={tx_pos.shape}, rx={rx_pos.shape}")

    bandwidth = float(geom.get("bandwidth_hz", 5e6))
    grid_cfg = geom["grid"]

    xs = np.arange(grid_cfg["x_min"], grid_cfg["x_max"] + 1e-9, grid_cfg["step"], dtype=np.float64)
    ys = np.arange(grid_cfg["y_min"], grid_cfg["y_max"] + 1e-9, grid_cfg["step"], dtype=np.float64)
    zs = np.arange(grid_cfg["z_min"], grid_cfg["z_max"] + 1e-9, grid_cfg["step"], dtype=np.float64)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_xyz = np.stack([X, Y, Z], axis=-1)  # [X,Y,Z,3]

    D, R, T = pdp.shape
    link_volume = np.zeros((len(xs), len(ys), len(zs), R, T), dtype=np.float32)

    # Normalize each link PDP before fusion so one strong link does not dominate completely.
    pdp_norm = np.zeros_like(pdp, dtype=np.float32)
    for rx in range(R):
        for tx in range(T):
            pdp_norm[:, rx, tx] = normalize01(pdp[:, rx, tx])

    for rx in range(R):
        for tx in range(T):
            txp = tx_pos[tx]
            rxp = rx_pos[rx]

            d_tx = np.linalg.norm(grid_xyz - txp[None, None, None, :], axis=-1)
            d_rx = np.linalg.norm(grid_xyz - rxp[None, None, None, :], axis=-1)
            path_len = d_tx + d_rx

            if args.subtract_min_path:
                direct_len = float(np.linalg.norm(txp - rxp))
                path_len = np.maximum(path_len - direct_len, 0.0)

            tau = path_len / C
            delay_bin = np.rint(tau * bandwidth).astype(np.int64)
            delay_bin = np.clip(delay_bin, 0, D - 1)

            link_volume[:, :, :, rx, tx] = pdp_norm[delay_bin, rx, tx]

    rf_volume = np.mean(link_volume, axis=(3, 4)).astype(np.float32)

    # Confidence: high when multiple links agree.
    link_mean = np.mean(link_volume, axis=(3, 4))
    link_std = np.std(link_volume, axis=(3, 4))
    voxel_confidence = (link_mean / (link_std + 1e-6)).astype(np.float32)
    voxel_confidence = normalize01(voxel_confidence)

    rf_volume_norm = normalize01(rf_volume)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        rf_volume=rf_volume_norm.astype(np.float32),
        rf_volume_raw=rf_volume.astype(np.float32),
        link_volume=link_volume.astype(np.float32),
        voxel_confidence=voxel_confidence.astype(np.float32),
        grid_x=xs.astype(np.float32),
        grid_y=ys.astype(np.float32),
        grid_z=zs.astype(np.float32),
        grid_xyz=grid_xyz.astype(np.float32),
        tx_positions=tx_pos.astype(np.float32),
        rx_positions=rx_pos.astype(np.float32),
        source_cir=str(cir_path),
        source_geom=str(geom_path),
        bandwidth_hz=np.array(bandwidth, dtype=np.float64),
        subtract_min_path=np.array(bool(args.subtract_min_path)),
    )

    peak_idx = np.unravel_index(np.argmax(rf_volume_norm), rf_volume_norm.shape)
    peak_xyz = grid_xyz[peak_idx]

    print(f"[OK] wrote: {out_path}")
    print("[VOL] pdp:", pdp.shape)
    print("[VOL] rf_volume:", rf_volume_norm.shape)
    print("[VOL] link_volume:", link_volume.shape)
    print("[VOL] voxel_confidence:", voxel_confidence.shape)
    print("[VOL] grid sizes:", len(xs), len(ys), len(zs))
    print("[VOL] peak_idx:", peak_idx)
    print("[VOL] peak_xyz_m:", [float(v) for v in peak_xyz])
    print("[VOL] peak_value:", float(rf_volume_norm[peak_idx]))
    print("[VOL] confidence_at_peak:", float(voxel_confidence[peak_idx]))

    for tx in range(T):
        for rx in range(R):
            lv = link_volume[:, :, :, rx, tx]
            idx = np.unravel_index(np.argmax(lv), lv.shape)
            print(f"[LINK] rx{rx}-tx{tx} peak_idx={idx} peak_xyz_m={[float(v) for v in grid_xyz[idx]]} peak={float(lv[idx])}")

if __name__ == "__main__":
    main()
