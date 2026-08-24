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
    parser.add_argument("--cir", required=True)
    parser.add_argument("--geom", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--key", default="pdp_mean_mimo")
    parser.add_argument(
        "--delay_mode",
        choices=["physical", "excess_physical", "normalized_grid"],
        default="normalized_grid",
    )
    parser.add_argument("--sigma_bins", type=float, default=1.5)
    args = parser.parse_args()

    cir_path = Path(args.cir)
    geom_path = Path(args.geom)
    out_path = Path(args.out)

    d = np.load(cir_path)

    if args.key not in d:
        raise SystemExit(
            f"[ERROR] key {args.key} not found in {cir_path}. keys={d.files}"
        )

    pdp = d[args.key]

    if pdp.ndim != 3 or pdp.shape[1:] != (2, 2):
        raise SystemExit(
            f"[ERROR] expected pdp_mean_mimo [D,2,2], got {pdp.shape}"
        )

    with open(geom_path, "r") as f:
        geom = json.load(f)

    tx_pos = np.asarray(geom["tx_positions"], dtype=np.float64)
    rx_pos = np.asarray(geom["rx_positions"], dtype=np.float64)

    if tx_pos.shape != (2, 3) or rx_pos.shape != (2, 3):
        raise SystemExit(
            f"[ERROR] expected tx/rx positions [2,3], "
            f"got tx={tx_pos.shape}, rx={rx_pos.shape}"
        )

    bandwidth = float(geom.get("bandwidth_hz", 5e6))
    grid_cfg = geom["grid"]

    step = float(grid_cfg["step"])

    xs = np.arange(
        float(grid_cfg["x_min"]),
        float(grid_cfg["x_max"]) + 1e-9,
        step,
        dtype=np.float64,
    )

    ys = np.arange(
        float(grid_cfg["y_min"]),
        float(grid_cfg["y_max"]) + 1e-9,
        step,
        dtype=np.float64,
    )

    zs = np.arange(
        float(grid_cfg["z_min"]),
        float(grid_cfg["z_max"]) + 1e-9,
        step,
        dtype=np.float64,
    )

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_xyz = np.stack([X, Y, Z], axis=-1)

    D, R, T = pdp.shape

    link_volume = np.zeros(
        (len(xs), len(ys), len(zs), R, T),
        dtype=np.float32,
    )

    delay_bin_volume = np.zeros_like(
        link_volume,
        dtype=np.float32,
    )

    pdp_norm = np.zeros_like(pdp, dtype=np.float32)

    peak_bins = np.zeros((R, T), dtype=np.int64)

    for rx in range(R):
        for tx in range(T):
            pdp_norm[:, rx, tx] = normalize01(
                pdp[:, rx, tx]
            )

            peak_bins[rx, tx] = int(
                np.argmax(pdp_norm[:, rx, tx])
            )

    sigma = max(float(args.sigma_bins), 1e-6)

    for rx in range(R):
        for tx in range(T):

            txp = tx_pos[tx]
            rxp = rx_pos[rx]

            d_tx = np.linalg.norm(
                grid_xyz - txp,
                axis=-1,
            )

            d_rx = np.linalg.norm(
                grid_xyz - rxp,
                axis=-1,
            )

            path_len = d_tx + d_rx

            direct_len = float(
                np.linalg.norm(txp - rxp)
            )

            excess_len = np.maximum(
                path_len - direct_len,
                0.0,
            )

            if args.delay_mode == "physical":

                delay_float = (
                    path_len / C
                ) * bandwidth

            elif args.delay_mode == "excess_physical":

                delay_float = (
                    excess_len / C
                ) * bandwidth

            else:
                mn = float(excess_len.min())
                mx = float(excess_len.max())

                if mx <= mn + 1e-12:
                    delay_float = np.zeros_like(
                        excess_len
                    )
                else:
                    delay_float = (
                        (excess_len - mn)
                        / (mx - mn)
                        * (D - 1)
                    )

            delay_bin_volume[:, :, :, rx, tx] = \
                delay_float.astype(np.float32)

            vol = np.zeros_like(
                delay_float,
                dtype=np.float32,
            )

            for b in range(D):

                w = np.exp(
                    -0.5
                    * ((delay_float - b) / sigma) ** 2
                ).astype(np.float32)

                vol += (
                    float(pdp_norm[b, rx, tx])
                    * w
                )

            link_volume[:, :, :, rx, tx] = \
                normalize01(vol)

    rf_volume_raw = np.mean(
        link_volume,
        axis=(3, 4),
    ).astype(np.float32)

    rf_volume = normalize01(
        rf_volume_raw
    )

    link_mean = np.mean(
        link_volume,
        axis=(3, 4),
    )

    link_std = np.std(
        link_volume,
        axis=(3, 4),
    )

    voxel_confidence = normalize01(
        link_mean / (link_std + 1e-6)
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        out_path,
        rf_volume=rf_volume.astype(np.float32),
        rf_volume_raw=rf_volume_raw.astype(np.float32),
        link_volume=link_volume.astype(np.float32),
        voxel_confidence=voxel_confidence.astype(np.float32),
        delay_bin_volume=delay_bin_volume.astype(np.float32),
        peak_bins=peak_bins.astype(np.int64),
        grid_x=xs.astype(np.float32),
        grid_y=ys.astype(np.float32),
        grid_z=zs.astype(np.float32),
        grid_xyz=grid_xyz.astype(np.float32),
        tx_positions=tx_pos.astype(np.float32),
        rx_positions=rx_pos.astype(np.float32),
        source_cir=str(cir_path),
        source_geom=str(geom_path),
        bandwidth_hz=np.array(
            bandwidth,
            dtype=np.float64
        ),
        delay_mode=np.array(args.delay_mode),
        sigma_bins=np.array(
            args.sigma_bins,
            dtype=np.float32
        ),
    )

    peak_idx = np.unravel_index(
        np.argmax(rf_volume),
        rf_volume.shape,
    )

    peak_xyz = grid_xyz[peak_idx]

    print(f"[OK] wrote: {out_path}")
    print("[VOL] delay_mode:", args.delay_mode)
    print("[VOL] pdp:", pdp.shape)
    print("[VOL] peak_bins[rx,tx]:")
    print(peak_bins)

    print(
        "[VOL] rf_volume:",
        rf_volume.shape
    )

    print(
        "[VOL] link_volume:",
        link_volume.shape
    )

    print(
        "[VOL] voxel_confidence:",
        voxel_confidence.shape
    )

    print(
        "[VOL] value mean/std/min/max:",
        float(rf_volume.mean()),
        float(rf_volume.std()),
        float(rf_volume.min()),
        float(rf_volume.max()),
    )

    print(
        "[VOL] conf mean/std/min/max:",
        float(voxel_confidence.mean()),
        float(voxel_confidence.std()),
        float(voxel_confidence.min()),
        float(voxel_confidence.max()),
    )

    print("[VOL] peak_idx:", peak_idx)

    print(
        "[VOL] peak_xyz_m:",
        [float(v) for v in peak_xyz]
    )

    print(
        "[VOL] peak_value:",
        float(rf_volume[peak_idx])
    )

    print(
        "[VOL] confidence_at_peak:",
        float(voxel_confidence[peak_idx])
    )

    for tx in range(T):
        for rx in range(R):

            lv = link_volume[:, :, :, rx, tx]

            idx = np.unravel_index(
                np.argmax(lv),
                lv.shape,
            )

            print(
                f"[LINK] rx{rx}-tx{tx} "
                f"peak_idx={idx} "
                f"peak_xyz_m="
                f"{[float(v) for v in grid_xyz[idx]]} "
                f"peak={float(lv[idx])}"
            )

if __name__ == "__main__":
    main()
