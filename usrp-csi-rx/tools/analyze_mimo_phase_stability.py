#!/usr/bin/env python3

import argparse
import numpy as np


def circ_stats(phi):
    z = np.exp(1j * phi)

    m = np.mean(z, axis=0)

    mean_phase = np.angle(m)
    coherence = np.abs(m)

    return mean_phase, coherence


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "tensor",
        help="H_raw_tdm_physical_2x2.npz",
    )

    args = ap.parse_args()

    z = np.load(args.tensor)

    H = z["H_raw"]

    if H.ndim != 4 or H.shape[1:] != (52, 2, 2):
        raise RuntimeError(
            f"Unexpected H shape: {H.shape}"
        )

    print("======================================")
    print("MIMO CSI PHASE STABILITY")
    print("======================================")
    print("shape:", H.shape)
    print("cycles:", H.shape[0])

    print()
    print("===== MAGNITUDE STABILITY =====")

    for t in range(2):
        for r in range(2):

            a = np.mean(
                np.abs(H[:, :, r, t]),
                axis=1,
            )

            mean = np.mean(a)
            std = np.std(a)

            cv = (
                std / mean
                if mean > 0
                else np.nan
            )

            print(
                f"TX{t}->RX{r}: "
                f"mean={mean:.6f} "
                f"std={std:.6f} "
                f"CV={cv:.4f}"
            )

    #
    # Within one TX packet:
    #
    # RX1 / RX0
    #
    # Both RX chains observe the SAME transmitted packet.
    # Therefore common packet phase largely cancels.
    #
    print()
    print("===== RX DIFFERENTIAL PHASE =====")

    for t in range(2):

        cross = (
            H[:, :, 1, t]
            * np.conj(H[:, :, 0, t])
        )

        phi = np.angle(cross)

        mean_phase, coherence = circ_stats(phi)

        print()
        print(f"Physical TX{t}")

        print(
            "mean coherence:",
            float(np.mean(coherence)),
        )

        print(
            "median coherence:",
            float(np.median(coherence)),
        )

        print(
            "min coherence:",
            float(np.min(coherence)),
        )

        print(
            "max coherence:",
            float(np.max(coherence)),
        )

        good = np.sum(coherence >= 0.8)

        print(
            "subcarriers coherence >= 0.8:",
            f"{good}/52",
        )

        print(
            "mean differential phase [rad]:",
            float(
                np.angle(
                    np.mean(
                        np.exp(
                            1j * mean_phase
                        )
                    )
                )
            ),
        )

    #
    # Cross-TX diagnostic.
    #
    # TX0 and TX1 belong to DIFFERENT packets.
    # This quantity is NOT automatically physically coherent.
    #
    print()
    print("===== CROSS-TX ADJACENT-PACKET PHASE =====")
    print(
        "Diagnostic only: low coherence here means "
        "TX0/TX1 absolute phase cannot be used directly."
    )

    for r in range(2):

        cross = (
            H[:, :, r, 1]
            * np.conj(H[:, :, r, 0])
        )

        phi = np.angle(cross)

        _, coherence = circ_stats(phi)

        print()
        print(f"Physical RX{r}")

        print(
            "mean coherence:",
            float(np.mean(coherence)),
        )

        print(
            "median coherence:",
            float(np.median(coherence)),
        )

        print(
            "subcarriers coherence >= 0.8:",
            f"{np.sum(coherence >= 0.8)}/52",
        )

    #
    # Global per-cycle phase diagnostic.
    #
    print()
    print("===== PER-CYCLE PHASE CHANGE =====")

    for t in range(2):
        for r in range(2):

            # Complex average over subcarriers.
            q = np.mean(
                H[:, :, r, t],
                axis=1,
            )

            phase = np.unwrap(
                np.angle(q)
            )

            if len(phase) > 1:
                d = np.diff(phase)

                print(
                    f"TX{t}->RX{r}: "
                    f"phase-step std="
                    f"{np.std(d):.4f} rad "
                    f"mean_abs_step="
                    f"{np.mean(np.abs(d)):.4f} rad"
                )

    print()
    print("======================================")
    print("INTERPRETATION")
    print("======================================")
    print(
        "RX differential coherence high -> "
        "within-packet RX phase is usable."
    )
    print(
        "Cross-TX coherence low -> "
        "do NOT use raw TX0/TX1 phase for AoD."
    )


if __name__ == "__main__":
    main()
