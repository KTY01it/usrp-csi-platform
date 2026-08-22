#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np


def circ_mean(phi, axis=0):
    z = np.exp(1j * phi)
    return np.angle(np.mean(z, axis=axis))


def circ_coherence(phi, axis=0):
    z = np.exp(1j * phi)
    return np.abs(np.mean(z, axis=axis))


def circ_std_from_coh(R):
    R = np.clip(R, 1e-8, 1.0)
    return np.sqrt(-2.0 * np.log(R))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "calibrated",
        help="H_reference_calibrated_2x2.npz",
    )

    ap.add_argument(
        "--output",
        default=None,
    )

    args = ap.parse_args()

    z = np.load(args.calibrated)

    H = z["H_raw"]

    rx_phase = z["rx_phase_cal"]
    tx_phase = z["tx_phase_cal"]

    valid_rx = z["valid_rx_mask"]
    valid_tx = z["valid_tx_mask"]

    #
    # Amplitude statistics.
    #
    amp = np.abs(H)

    amp_mean = np.mean(
        amp,
        axis=0,
    ).astype(np.float32)

    amp_std = np.std(
        amp,
        axis=0,
    ).astype(np.float32)

    #
    # Calibrated RX differential background.
    #
    rx_bg_mean = circ_mean(
        rx_phase,
        axis=0,
    ).astype(np.float32)

    rx_bg_coh = circ_coherence(
        rx_phase,
        axis=0,
    ).astype(np.float32)

    rx_bg_cstd = circ_std_from_coh(
        rx_bg_coh
    ).astype(np.float32)

    #
    # Calibrated cross-TX background.
    #
    tx_bg_mean = circ_mean(
        tx_phase,
        axis=0,
    ).astype(np.float32)

    tx_bg_coh = circ_coherence(
        tx_phase,
        axis=0,
    ).astype(np.float32)

    tx_bg_cstd = circ_std_from_coh(
        tx_bg_coh
    ).astype(np.float32)

    #
    # Mask unreliable dimensions.
    #
    rx_bg_cstd = np.where(
        valid_rx,
        rx_bg_cstd,
        np.nan,
    )

    tx_bg_cstd = np.where(
        valid_tx,
        tx_bg_cstd,
        np.nan,
    )

    if args.output is None:
        p = Path(args.calibrated)

        out = (
            p.parent /
            "mimo_background_model.npz"
        )
    else:
        out = Path(args.output)

    np.savez(
        out,

        amp_mean=amp_mean,
        amp_std=amp_std,

        rx_phase_bg_mean=rx_bg_mean,
        rx_phase_bg_coherence=rx_bg_coh,
        rx_phase_bg_circular_std=rx_bg_cstd,

        tx_phase_bg_mean=tx_bg_mean,
        tx_phase_bg_coherence=tx_bg_coh,
        tx_phase_bg_circular_std=tx_bg_cstd,

        valid_rx_mask=valid_rx,
        valid_tx_mask=valid_tx,

        source=np.array(
            str(args.calibrated)
        ),

        model_type=np.array(
            "static_reference_calibrated_background"
        ),
    )

    print("======================================")
    print("MIMO STATIC BACKGROUND MODEL")
    print("======================================")

    print("input :", args.calibrated)
    print("output:", out)
    print("cycles:", H.shape[0])

    print()
    print("RX differential residual noise:")

    for t in range(2):
        x = rx_bg_cstd[:, t]

        print(
            f"TX{t}: "
            f"valid={np.sum(valid_rx[:,t])}/52 "
            f"median_cstd="
            f"{np.nanmedian(x):.4f} rad "
            f"mean_cstd="
            f"{np.nanmean(x):.4f} rad"
        )

    print()
    print("Cross-TX residual noise:")

    for r in range(2):
        x = tx_bg_cstd[:, r]

        print(
            f"RX{r}: "
            f"valid={np.sum(valid_tx[:,r])}/52 "
            f"median_cstd="
            f"{np.nanmedian(x):.4f} rad "
            f"mean_cstd="
            f"{np.nanmean(x):.4f} rad"
        )


if __name__ == "__main__":
    main()
