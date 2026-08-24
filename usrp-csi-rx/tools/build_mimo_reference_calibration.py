#!/usr/bin/env python3

import argparse
import numpy as np


def circ_mean_and_coh(phi, axis=0):
    z = np.exp(1j * phi)
    m = np.mean(z, axis=axis)
    return np.angle(m), np.abs(m)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("tensor")

    ap.add_argument(
        "--output",
        default=None,
    )

    ap.add_argument(
        "--coh-threshold",
        type=float,
        default=0.8,
    )

    args = ap.parse_args()

    z = np.load(args.tensor)
    H = z["H_raw"]

    if H.ndim != 4 or H.shape[1:] != (52, 2, 2):
        raise RuntimeError(
            f"Unexpected H shape: {H.shape}"
        )

    N = H.shape[0]

    #
    # Mean amplitude and stability
    #
    amp = np.abs(H)

    amp_mean = np.mean(
        amp,
        axis=0,
    )

    amp_std = np.std(
        amp,
        axis=0,
    )

    amp_cv = np.divide(
        amp_std,
        amp_mean,
        out=np.full_like(
            amp_mean,
            np.nan,
            dtype=np.float32,
        ),
        where=amp_mean > 0,
    )

    #
    # RX differential:
    #
    # RX1 / RX0 for each TX
    #
    phi_rx = np.zeros(
        (52, 2),
        dtype=np.float32,
    )

    coh_rx = np.zeros(
        (52, 2),
        dtype=np.float32,
    )

    for t in range(2):
        c = (
            H[:, :, 1, t]
            * np.conj(H[:, :, 0, t])
        )

        p = np.angle(c)

        phi, coh = circ_mean_and_coh(
            p,
            axis=0,
        )

        phi_rx[:, t] = phi
        coh_rx[:, t] = coh

    #
    # Cross-TX differential:
    #
    # TX1 / TX0 for each RX
    #
    phi_tx = np.zeros(
        (52, 2),
        dtype=np.float32,
    )

    coh_tx = np.zeros(
        (52, 2),
        dtype=np.float32,
    )

    for r in range(2):
        c = (
            H[:, :, r, 1]
            * np.conj(H[:, :, r, 0])
        )

        p = np.angle(c)

        phi, coh = circ_mean_and_coh(
            p,
            axis=0,
        )

        phi_tx[:, r] = phi
        coh_tx[:, r] = coh

    valid_rx = (
        coh_rx >= args.coh_threshold
    )

    valid_tx = (
        coh_tx >= args.coh_threshold
    )

    if args.output is None:
        out = (
            args.tensor.replace(
                "H_raw_tdm_physical_2x2.npz",
                "mimo_reference_calibration.npz",
            )
        )
    else:
        out = args.output

    np.savez(
        out,

        phi_rx_ref=phi_rx,
        coh_rx_ref=coh_rx,

        phi_tx_ref=phi_tx,
        coh_tx_ref=coh_tx,

        valid_rx_mask=valid_rx,
        valid_tx_mask=valid_tx,

        amp_mean=amp_mean,
        amp_std=amp_std,
        amp_cv=amp_cv,

        cycles=np.int32(N),

        coherence_threshold=np.float32(
            args.coh_threshold
        ),

        calibration_type=np.array(
            "reference_based_relative_phase"
        ),

        note=np.array(
            "Reference/system calibration; "
            "not absolute hardware-only AoA/AoD calibration"
        ),
    )

    print("======================================")
    print("MIMO REFERENCE CALIBRATION")
    print("======================================")
    print("input :", args.tensor)
    print("output:", out)
    print("cycles:", N)

    print()
    print("RX differential valid subcarriers:")

    for t in range(2):
        print(
            f"TX{t}: "
            f"{np.sum(valid_rx[:,t])}/52 "
            f"(mean coherence="
            f"{np.mean(coh_rx[:,t]):.4f})"
        )

    print()
    print("Cross-TX valid subcarriers:")

    for r in range(2):
        print(
            f"RX{r}: "
            f"{np.sum(valid_tx[:,r])}/52 "
            f"(mean coherence="
            f"{np.mean(coh_tx[:,r]):.4f})"
        )

    print()
    print("Mean amplitude / CV:")

    for t in range(2):
        for r in range(2):
            print(
                f"TX{t}->RX{r}: "
                f"mean_amp="
                f"{np.mean(amp_mean[:,r,t]):.6f} "
                f"mean_CV="
                f"{np.nanmean(amp_cv[:,r,t]):.4f}"
            )


if __name__ == "__main__":
    main()
