#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np


EPS = 1e-8


def unit_phasor(z):
    return z / np.maximum(np.abs(z), EPS)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "target_calibrated",
        help="Target H_reference_calibrated_2x2.npz",
    )

    ap.add_argument(
        "background_calibrated",
        help="Background H_reference_calibrated_2x2.npz",
    )

    ap.add_argument(
        "--output",
        default=None,
    )

    args = ap.parse_args()

    obj = np.load(args.target_calibrated)
    bg = np.load(args.background_calibrated)

    H_obj = obj["H_raw"]
    H_bg = bg["H_raw"]

    if H_obj.ndim != 4 or H_obj.shape[1:] != (52, 2, 2):
        raise RuntimeError(
            f"Unexpected target shape: {H_obj.shape}"
        )

    if H_bg.ndim != 4 or H_bg.shape[1:] != (52, 2, 2):
        raise RuntimeError(
            f"Unexpected background shape: {H_bg.shape}"
        )

    phi_ref_obj = obj["phi_rx_ref"]
    phi_ref_bg = bg["phi_rx_ref"]

    if not np.allclose(
        phi_ref_obj,
        phi_ref_bg,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Target and background do not use the same "
            "RX differential phase reference."
        )

    valid_obj = obj["valid_rx_mask"]
    valid_bg = bg["valid_rx_mask"]

    valid = valid_obj & valid_bg

    coh_obj = obj["coh_rx_ref"]
    coh_bg = bg["coh_rx_ref"]

    quality_weight = np.minimum(
        coh_obj,
        coh_bg,
    ).astype(np.float32)

    #
    # D_rx:
    #
    # shape [N, 52, 2]
    #
    # D = H_RX1 * conj(H_RX0)
    #
    D_obj_raw = np.empty(
        (H_obj.shape[0], 52, 2),
        dtype=np.complex64,
    )

    D_bg_raw = np.empty(
        (H_bg.shape[0], 52, 2),
        dtype=np.complex64,
    )

    for t in range(2):
        D_obj_raw[:, :, t] = (
            H_obj[:, :, 1, t]
            * np.conj(H_obj[:, :, 0, t])
        )

        D_bg_raw[:, :, t] = (
            H_bg[:, :, 1, t]
            * np.conj(H_bg[:, :, 0, t])
        )

    #
    # Remove reference differential phase.
    #
    ref_rot = np.exp(
        -1j * phi_ref_obj
    )[None, :, :]

    D_obj_cal = (
        D_obj_raw * ref_rot
    ).astype(np.complex64)

    D_bg_cal = (
        D_bg_raw * ref_rot
    ).astype(np.complex64)

    #
    # Background magnitude normalization.
    #
    # Median is used because amplitude was observed
    # to contain capture-to-capture outliers.
    #
    bg_amp_median = np.median(
        np.abs(D_bg_cal),
        axis=0,
    ).astype(np.float32)

    bg_amp_median = np.maximum(
        bg_amp_median,
        EPS,
    )

    D_obj_norm = (
        D_obj_cal
        / bg_amp_median[None, :, :]
    ).astype(np.complex64)

    D_bg_norm = (
        D_bg_cal
        / bg_amp_median[None, :, :]
    ).astype(np.complex64)

    #
    # Complex background template.
    #
    D_bg_template = np.mean(
        D_bg_norm,
        axis=0,
    ).astype(np.complex64)

    #
    # Phase-only background template.
    #
    U_bg = unit_phasor(
        D_bg_cal
    )

    U_bg_mean = np.mean(
        U_bg,
        axis=0,
    )

    U_bg_template = unit_phasor(
        U_bg_mean
    ).astype(np.complex64)

    U_obj = unit_phasor(
        D_obj_cal
    ).astype(np.complex64)

    #
    # Target-minus-background complex evidence.
    #
    delta_complex_norm = (
        D_obj_norm
        - D_bg_template[None, :, :]
    ).astype(np.complex64)

    #
    # Phase-only residual relative to BG.
    #
    phase_residual_complex = (
        U_obj
        * np.conj(
            U_bg_template[None, :, :]
        )
    ).astype(np.complex64)

    delta_phase = np.angle(
        phase_residual_complex
    ).astype(np.float32)

    delta_unit_complex = (
        U_obj
        - U_bg_template[None, :, :]
    ).astype(np.complex64)

    #
    # Background self-residual.
    #
    bg_delta_complex_norm = (
        D_bg_norm
        - D_bg_template[None, :, :]
    ).astype(np.complex64)

    bg_phase_residual = np.angle(
        U_bg
        * np.conj(
            U_bg_template[None, :, :]
        )
    ).astype(np.float32)

    #
    # Apply scientific quality masks.
    #
    cmask = valid[None, :, :]

    delta_complex_masked = np.where(
        cmask,
        delta_complex_norm,
        np.complex64(np.nan + 1j*np.nan),
    )

    delta_phase_masked = np.where(
        cmask,
        delta_phase,
        np.nan,
    ).astype(np.float32)

    delta_unit_masked = np.where(
        cmask,
        delta_unit_complex,
        np.complex64(np.nan + 1j*np.nan),
    )

    if args.output is None:
        p = Path(args.target_calibrated)

        out = (
            p.parent
            / "complex_differential_rf.npz"
        )
    else:
        out = Path(args.output)

    save_dict = {
        "D_rx_obj_raw": D_obj_raw,
        "D_rx_bg_raw": D_bg_raw,

        "D_rx_obj_cal": D_obj_cal,
        "D_rx_bg_cal": D_bg_cal,

        "D_rx_obj_norm": D_obj_norm,
        "D_rx_bg_norm": D_bg_norm,

        "D_bg_template": D_bg_template,
        "U_bg_template": U_bg_template,

        "delta_complex_norm":
            delta_complex_norm,

        "delta_complex_masked":
            delta_complex_masked,

        "delta_phase":
            delta_phase,

        "delta_phase_masked":
            delta_phase_masked,

        "delta_unit_complex":
            delta_unit_complex,

        "delta_unit_masked":
            delta_unit_masked,

        "bg_delta_complex_norm":
            bg_delta_complex_norm,

        "bg_phase_residual":
            bg_phase_residual,

        "bg_amp_median":
            bg_amp_median,

        "valid_rx_mask":
            valid,

        "quality_weight":
            quality_weight,

        "phi_rx_ref":
            phi_ref_obj,

        "source_target":
            np.array(
                str(args.target_calibrated)
            ),

        "source_background":
            np.array(
                str(args.background_calibrated)
            ),

        "representation":
            np.array(
                "reference_calibrated_rx_differential_rf"
            ),

        "note":
            np.array(
                "RX differential complex RF evidence. "
                "Amplitude normalized to median background "
                "per subcarrier/link. Phase-only evidence "
                "also retained."
            ),
    }

    if "seq_pairs" in obj.files:
        save_dict["target_seq_pairs"] = (
            obj["seq_pairs"]
        )

    if "seq_pairs" in bg.files:
        save_dict["background_seq_pairs"] = (
            bg["seq_pairs"]
        )

    np.savez(
        out,
        **save_dict,
    )

    print("======================================")
    print("COMPLEX DIFFERENTIAL RF")
    print("======================================")

    print(
        "target    :",
        args.target_calibrated,
    )

    print(
        "background:",
        args.background_calibrated,
    )

    print(
        "output    :",
        out,
    )

    print(
        "target cycles:",
        H_obj.shape[0],
    )

    print(
        "background cycles:",
        H_bg.shape[0],
    )

    print()

    for t in range(2):
        x_phase = (
            delta_phase_masked[:, :, t]
        )

        x_complex = (
            delta_complex_masked[:, :, t]
        )

        bg_phase = np.where(
            valid[None, :, t],
            bg_phase_residual[:, :, t],
            np.nan,
        )

        print(f"TX{t}:")
        print(
            "  valid subcarriers:",
            f"{np.sum(valid[:, t])}/52",
        )

        print(
            "  quality weight mean:",
            f"{np.mean(quality_weight[:, t]):.4f}",
        )

        print(
            "  BG differential amplitude median:",
            f"{np.median(bg_amp_median[:, t]):.6f}",
        )

        print(
            "  target |delta complex| mean:",
            f"{np.nanmean(np.abs(x_complex)):.6f}",
        )

        print(
            "  target |delta phase| median [rad]:",
            f"{np.nanmedian(np.abs(x_phase)):.6f}",
        )

        print(
            "  target |delta phase| mean [rad]:",
            f"{np.nanmean(np.abs(x_phase)):.6f}",
        )

        print(
            "  BG self |phase residual| median [rad]:",
            f"{np.nanmedian(np.abs(bg_phase)):.6f}",
        )

        print()


if __name__ == "__main__":
    main()
