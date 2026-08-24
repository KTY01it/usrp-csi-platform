#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np


EPS = 1e-8


def unit(z):
    return z / np.maximum(np.abs(z), EPS)


def circular_coherence(x, axis=0):
    return np.abs(
        np.mean(
            np.exp(1j * x),
            axis=axis
        )
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "target_npz",
        help="H_raw_tdm_physical_2x2.npz",
    )

    ap.add_argument(
        "reference_npz",
        help="Reference H_raw_tdm_physical_2x2.npz",
    )

    ap.add_argument(
        "--coherence-threshold",
        type=float,
        default=0.8,
    )

    ap.add_argument(
        "--output",
        default=None,
    )

    args = ap.parse_args()

    target = np.load(args.target_npz)
    ref = np.load(args.reference_npz)

    H = target["H_raw"]
    Href_all = ref["H_raw"]

    if H.shape[1:] != (52, 2, 2):
        raise RuntimeError(
            f"Unexpected target shape: {H.shape}"
        )

    if Href_all.shape[1:] != (52, 2, 2):
        raise RuntimeError(
            f"Unexpected reference shape: {Href_all.shape}"
        )

    #
    # Complex reference template.
    #
    Href = np.mean(
        Href_all,
        axis=0
    ).astype(np.complex64)

    #
    # Align each TX packet independently.
    #
    theta = np.zeros(
        (H.shape[0], 2),
        dtype=np.float32
    )

    H_aligned = np.empty_like(
        H,
        dtype=np.complex64
    )

    for t in range(2):
        X = H[:, :, :, t]
        R = Href[:, :, t]

        corr = np.sum(
            X
            * np.conj(R)[None, :, :],
            axis=(1, 2)
        )

        theta[:, t] = np.angle(
            corr
        ).astype(np.float32)

        H_aligned[:, :, :, t] = (
            X
            * np.exp(
                -1j * theta[:, t]
            )[:, None, None]
        )

    #
    # Residual relative to reference template.
    #
    residual_complex = (
        H_aligned
        * np.conj(
            Href[None, :, :, :]
        )
    ).astype(np.complex64)

    residual_phase = np.angle(
        residual_complex
    ).astype(np.float32)

    #
    # Unit-phase version.
    #
    residual_unit = unit(
        residual_complex
    ).astype(np.complex64)

    #
    # Coherence across target cycles.
    #
    coherence = np.abs(
        np.mean(
            residual_unit,
            axis=0
        )
    ).astype(np.float32)

    quality_mask = (
        coherence
        >= args.coherence_threshold
    )

    #
    # Also preserve amplitude-normalized aligned CSI.
    #
    ref_amp = np.maximum(
        np.abs(Href),
        EPS
    ).astype(np.float32)

    H_aligned_norm = (
        H_aligned
        / ref_amp[None, :, :, :]
    ).astype(np.complex64)

    delta_aligned = (
        H_aligned_norm
        - unit(Href)[None, :, :, :]
    ).astype(np.complex64)

    if args.output is None:
        p = Path(args.target_npz)

        out = (
            p.parent
            / "common_phase_aligned_mimo.npz"
        )
    else:
        out = Path(args.output)

    save_dict = {
        "H_aligned":
            H_aligned,

        "H_aligned_norm":
            H_aligned_norm,

        "theta_common":
            theta,

        "H_ref_template":
            Href,

        "residual_complex":
            residual_complex,

        "residual_phase":
            residual_phase,

        "residual_unit":
            residual_unit,

        "coherence_map":
            coherence,

        "quality_mask":
            quality_mask,

        "delta_aligned":
            delta_aligned,

        "coherence_threshold":
            np.array(
                args.coherence_threshold,
                dtype=np.float32
            ),

        "source_target":
            np.array(
                str(args.target_npz)
            ),

        "source_reference":
            np.array(
                str(args.reference_npz)
            ),

        "representation":
            np.array(
                "per_tx_common_phase_aligned_mimo_csi"
            ),

        "note":
            np.array(
                "One common phase offset is removed "
                "per cycle and TX. Frequency-dependent "
                "phase structure is preserved."
            ),
    }

    if "seq_pairs" in target.files:
        save_dict["seq_pairs"] = (
            target["seq_pairs"]
        )

    np.savez(
        out,
        **save_dict
    )

    print("======================================")
    print("COMMON-PHASE ALIGNED MIMO CSI")
    print("======================================")

    print("target   :", args.target_npz)
    print("reference:", args.reference_npz)
    print("output   :", out)
    print("cycles   :", H.shape[0])
    print()

    for t in range(2):
        c = coherence[:, :, t]

        p = np.abs(
            residual_phase[:, :, :, t]
        )

        print(f"TX{t}:")
        print(
            "  common phase std [rad]:",
            float(
                np.std(
                    np.unwrap(
                        theta[:, t]
                    )
                )
            )
        )

        print(
            "  coherence mean:",
            float(np.mean(c))
        )

        print(
            "  coherence median:",
            float(np.median(c))
        )

        print(
            "  coherence>=threshold:",
            int(
                np.sum(
                    c
                    >= args.coherence_threshold
                )
            ),
            "/",
            c.size
        )

        print(
            "  residual |phase| median [rad]:",
            float(np.median(p))
        )

        print()


if __name__ == "__main__":
    main()
