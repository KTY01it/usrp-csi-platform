#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np


EPS = 1e-6


def wrap_phase(x):
    return np.angle(np.exp(1j * x))


def robust_summary(name, z, valid_mask):
    x = np.where(
        valid_mask[None, :, :],
        z,
        np.nan,
    )

    finite = np.isfinite(x)

    values = x[finite]

    print()
    print(name)

    if values.size == 0:
        print("no valid values")
        return

    print(
        "mean z:",
        float(np.mean(values)),
    )

    print(
        "median z:",
        float(np.median(values)),
    )

    print(
        "p90 z:",
        float(np.percentile(values, 90)),
    )

    print(
        "p95 z:",
        float(np.percentile(values, 95)),
    )

    print(
        "fraction z >= 2:",
        float(np.mean(values >= 2.0)),
    )

    print(
        "fraction z >= 3:",
        float(np.mean(values >= 3.0)),
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "object_calibrated",
    )

    ap.add_argument(
        "background_model",
    )

    ap.add_argument(
        "--output",
        default=None,
    )

    args = ap.parse_args()

    obj = np.load(
        args.object_calibrated
    )

    bg = np.load(
        args.background_model
    )

    H = obj["H_raw"]

    rx_phase = obj[
        "rx_phase_cal"
    ]

    tx_phase = obj[
        "tx_phase_cal"
    ]

    valid_rx = bg[
        "valid_rx_mask"
    ]

    valid_tx = bg[
        "valid_tx_mask"
    ]

    #
    # -------------------------
    # AMPLITUDE CHANGE
    # -------------------------
    #
    amp_obj = np.abs(H)

    amp_bg_mean = bg[
        "amp_mean"
    ]

    amp_bg_std = bg[
        "amp_std"
    ]

    amp_z = (
        np.abs(
            amp_obj
            -
            amp_bg_mean[None, :, :, :]
        )
        /
        (
            amp_bg_std[None, :, :, :]
            + EPS
        )
    )

    #
    # -------------------------
    # RX DIFFERENTIAL PHASE
    # -------------------------
    #
    rx_bg_mean = bg[
        "rx_phase_bg_mean"
    ]

    rx_bg_std = bg[
        "rx_phase_bg_circular_std"
    ]

    rx_delta = wrap_phase(
        rx_phase
        -
        rx_bg_mean[None, :, :]
    )

    rx_z = (
        np.abs(rx_delta)
        /
        (
            rx_bg_std[None, :, :]
            + EPS
        )
    )

    rx_z = np.where(
        valid_rx[None, :, :],
        rx_z,
        np.nan,
    )

    #
    # -------------------------
    # CROSS-TX PHASE
    # -------------------------
    #
    tx_bg_mean = bg[
        "tx_phase_bg_mean"
    ]

    tx_bg_std = bg[
        "tx_phase_bg_circular_std"
    ]

    tx_delta = wrap_phase(
        tx_phase
        -
        tx_bg_mean[None, :, :]
    )

    tx_z = (
        np.abs(tx_delta)
        /
        (
            tx_bg_std[None, :, :]
            + EPS
        )
    )

    tx_z = np.where(
        valid_tx[None, :, :],
        tx_z,
        np.nan,
    )

    #
    # Frame-level scores.
    #
    rx_score_frame = np.nanmean(
        rx_z,
        axis=(1, 2),
    )

    tx_score_frame = np.nanmean(
        tx_z,
        axis=(1, 2),
    )

    amp_score_frame = np.mean(
        amp_z,
        axis=(1, 2, 3),
    )

    #
    # Conservative fusion:
    # RX differential = primary
    # cross-TX = auxiliary
    # amplitude = auxiliary
    #
    fused_score_frame = (
        0.60 * rx_score_frame
        +
        0.25 * tx_score_frame
        +
        0.15 * amp_score_frame
    )

    if args.output is None:
        p = Path(
            args.object_calibrated
        )

        out = (
            p.parent
            /
            "mimo_object_change.npz"
        )
    else:
        out = Path(args.output)

    np.savez(
        out,

        amp_z=amp_z,

        rx_phase_delta=rx_delta,
        rx_phase_z=rx_z,

        tx_phase_delta=tx_delta,
        tx_phase_z=tx_z,

        rx_score_frame=rx_score_frame,
        tx_score_frame=tx_score_frame,
        amp_score_frame=amp_score_frame,

        fused_score_frame=
            fused_score_frame,

        valid_rx_mask=valid_rx,
        valid_tx_mask=valid_tx,

        source_object=np.array(
            str(args.object_calibrated)
        ),

        source_background=np.array(
            str(args.background_model)
        ),

        detector_type=np.array(
            "quality_aware_background_normalized_change"
        ),
    )

    print("======================================")
    print("MIMO OBJECT CHANGE DETECTOR")
    print("======================================")

    print(
        "object:",
        args.object_calibrated,
    )

    print(
        "background:",
        args.background_model,
    )

    print(
        "output:",
        out,
    )

    print(
        "cycles:",
        H.shape[0],
    )

    #
    # Per-channel statistics.
    #
    print()
    print(
        "===== RX DIFFERENTIAL PHASE ====="
    )

    for t in range(2):

        x = rx_z[:, :, t]

        print()
        print(f"TX{t}")

        print(
            "mean z:",
            float(
                np.nanmean(x)
            ),
        )

        print(
            "median z:",
            float(
                np.nanmedian(x)
            ),
        )

        print(
            "p95 z:",
            float(
                np.nanpercentile(
                    x,
                    95,
                )
            ),
        )

        print(
            "fraction z >= 2:",
            float(
                np.mean(
                    x[np.isfinite(x)] >= 2
                )
            ),
        )

        print(
            "fraction z >= 3:",
            float(
                np.mean(
                    x[np.isfinite(x)] >= 3
                )
            ),
        )

    print()
    print(
        "===== CROSS-TX PHASE ====="
    )

    for r in range(2):

        x = tx_z[:, :, r]

        print()
        print(f"RX{r}")

        print(
            "mean z:",
            float(
                np.nanmean(x)
            ),
        )

        print(
            "median z:",
            float(
                np.nanmedian(x)
            ),
        )

        print(
            "p95 z:",
            float(
                np.nanpercentile(
                    x,
                    95,
                )
            ),
        )

        print(
            "fraction z >= 2:",
            float(
                np.mean(
                    x[np.isfinite(x)] >= 2
                )
            ),
        )

        print(
            "fraction z >= 3:",
            float(
                np.mean(
                    x[np.isfinite(x)] >= 3
                )
            ),
        )

    print()
    print(
        "===== FRAME SCORES ====="
    )

    print(
        "RX score mean:",
        float(
            np.mean(
                rx_score_frame
            )
        ),
    )

    print(
        "TX score mean:",
        float(
            np.mean(
                tx_score_frame
            )
        ),
    )

    print(
        "Amplitude score mean:",
        float(
            np.mean(
                amp_score_frame
            )
        ),
    )

    print(
        "Fused score mean:",
        float(
            np.mean(
                fused_score_frame
            )
        ),
    )

    print(
        "Fused score median:",
        float(
            np.median(
                fused_score_frame
            )
        ),
    )

    print(
        "Fused score p95:",
        float(
            np.percentile(
                fused_score_frame,
                95,
            )
        ),
    )


if __name__ == "__main__":
    main()
