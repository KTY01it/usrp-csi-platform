#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bg", required=True)
    parser.add_argument("--obj", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--key", default="H_calibrated_mimo")
    args = parser.parse_args()

    bg_path = Path(args.bg)
    obj_path = Path(args.obj)
    out = Path(args.out)

    bg_d = np.load(bg_path)
    obj_d = np.load(obj_path)

    H_bg = bg_d[args.key]
    H_obj = obj_d[args.key]

    if H_bg.ndim != 4 or H_obj.ndim != 4:
        raise SystemExit(f"[ERROR] expected 4D tensors, got bg={H_bg.shape}, obj={H_obj.shape}")

    if H_bg.shape[1:] != H_obj.shape[1:]:
        raise SystemExit(f"[ERROR] incompatible non-frame shapes: bg={H_bg.shape}, obj={H_obj.shape}")

    n = min(H_bg.shape[0], H_obj.shape[0])
    if H_bg.shape[0] != H_obj.shape[0]:
        print(f"[WARN] frame mismatch: bg={H_bg.shape[0]}, obj={H_obj.shape[0]}, using n={n}")

    H_bg = H_bg[:n]
    H_obj = H_obj[:n]

    eps = 1e-12
    H_delta = H_obj - H_bg
    H_ratio = H_obj / (H_bg + eps)

    delta_power = (np.abs(H_delta) ** 2).astype(np.float32)
    ratio_mag = np.abs(H_ratio).astype(np.float32)
    ratio_phase = np.angle(H_ratio).astype(np.float32)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        H_background=H_bg.astype(np.complex64),
        H_object=H_obj.astype(np.complex64),
        H_delta_mimo=H_delta.astype(np.complex64),
        H_ratio_mimo=H_ratio.astype(np.complex64),
        delta_power_mimo=delta_power,
        ratio_mag_mimo=ratio_mag,
        ratio_phase_mimo=ratio_phase,
        source_bg=str(bg_path),
        source_obj=str(obj_path),
        source_key=args.key,
    )

    print(f"[OK] wrote: {out}")
    print("[DELTA] H_background:", H_bg.shape)
    print("[DELTA] H_object:", H_obj.shape)
    print("[DELTA] H_delta_mimo:", H_delta.shape)
    print("[DELTA] H_ratio_mimo:", H_ratio.shape)
    print("[DELTA] delta abs mean/max:", float(np.abs(H_delta).mean()), float(np.abs(H_delta).max()))
    print("[DELTA] ratio mag mean/std:", float(ratio_mag.mean()), float(ratio_mag.std()))

    R = H_delta.shape[2]
    T = H_delta.shape[3]
    for tx in range(T):
        for rx in range(R):
            x = H_delta[:, :, rx, tx]
            print(
                f"[LINK] rx{rx}-tx{tx} delta_abs_mean={float(np.abs(x).mean())} "
                f"delta_abs_max={float(np.abs(x).max())}"
            )

if __name__ == "__main__":
    main()
