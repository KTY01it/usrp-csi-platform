#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Build validated wideband angular RF evidence "
            "from pre-beta 2x2 TDM MIMO CSI."
        )
    )

    ap.add_argument(
        "tensor",
        type=Path,
    )

    #
    # Kept only for backwards CLI compatibility.
    # Post-beta calibration is NOT applied to H_spatial_raw.
    #
    ap.add_argument(
        "calibration",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Deprecated compatibility argument. "
            "Not applied to pre-beta spatial CSI."
        ),
    )

    ap.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    ap.add_argument(
        "--min-subcarriers",
        type=int,
        default=40,
    )

    ap.add_argument(
        "--min-confidence",
        type=float,
        default=0.70,
    )

    ap.add_argument(
        "--primary-tx",
        type=int,
        choices=(0, 1),
        default=0,
    )

    args = ap.parse_args()

    z = np.load(
        args.tensor,
        allow_pickle=True,
    )

    if "H_spatial_raw" not in z:
        raise RuntimeError(
            "Tensor does not contain H_spatial_raw. "
            "Absolute/angular processing must not silently "
            "fall back to post-beta H_raw."
        )

    H = np.asarray(
        z["H_spatial_raw"],
        dtype=np.complex64,
    )

    if (
        H.ndim != 4
        or H.shape[1:] != (52, 2, 2)
    ):
        raise RuntimeError(
            f"Unexpected H_spatial_raw shape: {H.shape}"
        )

    N = H.shape[0]

    if "spatial_packet_valid" in z:
        packet_valid = np.asarray(
            z["spatial_packet_valid"],
            dtype=bool,
        )

        if packet_valid.shape != (N, 2, 2):
            raise RuntimeError(
                "Unexpected spatial_packet_valid shape"
            )
    else:
        packet_valid = np.all(
            np.isfinite(H),
            axis=1,
        )

    #
    # --------------------------------------------------------
    # Wideband RX-differential spatial evidence
    # --------------------------------------------------------
    #
    # D[n,k,t] =
    #   H_spatial[n,k,RX1,t]
    #   * conj(H_spatial[n,k,RX0,t])
    #
    # No closed-form AoA assumption is made here.
    #
    D = (
        H[:, :, 1, :]
        * np.conj(
            H[:, :, 0, :]
        )
    ).astype(np.complex64)

    magnitude = np.abs(
        D
    ).astype(np.float32)

    phase = np.angle(
        D
    ).astype(np.float32)

    finite = (
        np.isfinite(D.real)
        & np.isfinite(D.imag)
        & np.isfinite(magnitude)
        & (magnitude > 0)
    )

    unit = np.zeros_like(
        D,
        dtype=np.complex64,
    )

    unit[finite] = (
        D[finite]
        / magnitude[finite]
    ).astype(np.complex64)

    #
    # Packet validity for one TX requires both RX chains.
    #
    tx_packet_valid = np.zeros(
        (N, 2),
        dtype=bool,
    )

    for t in range(2):
        tx_packet_valid[:, t] = (
            packet_valid[:, 0, t]
            & packet_valid[:, 1, t]
        )

    #
    # --------------------------------------------------------
    # Per-frame spectral evidence
    # --------------------------------------------------------
    #
    used_subcarrier_count = np.sum(
        finite,
        axis=1,
    ).astype(np.int32)

    subcarrier_coverage = (
        used_subcarrier_count.astype(
            np.float64
        )
        / 52.0
    )

    angular_scalar_phase_rad = np.full(
        (N, 2),
        np.nan,
        dtype=np.float64,
    )

    angular_spectral_coherence = np.full(
        (N, 2),
        np.nan,
        dtype=np.float64,
    )

    for t in range(2):
        for n in range(N):

            valid = finite[n, :, t]

            if not np.any(valid):
                continue

            w = magnitude[
                n,
                valid,
                t,
            ].astype(np.float64)

            u = unit[
                n,
                valid,
                t,
            ].astype(np.complex128)

            ws = np.sum(w)

            if ws <= 0:
                continue

            q = np.sum(
                w * u
            ) / ws

            angular_scalar_phase_rad[
                n,
                t,
            ] = np.angle(q)

            angular_spectral_coherence[
                n,
                t,
            ] = np.abs(q)

    #
    # Confidence is deliberately an evidence-quality score,
    # NOT AoA accuracy.
    #
    angular_confidence = (
        angular_spectral_coherence
        * subcarrier_coverage
    )

    angular_valid = (
        tx_packet_valid
        & np.isfinite(
            angular_confidence
        )
        & (
            used_subcarrier_count
            >= args.min_subcarriers
        )
        & (
            angular_confidence
            >= args.min_confidence
        )
    )

    #
    # Empirical validation showed TX0 is the primary
    # angular link; TX1 remains auxiliary/quality-gated.
    #
    primary_tx = int(
        args.primary_tx
    )

    primary_angular_valid = (
        angular_valid[
            :,
            primary_tx,
        ]
    )

    primary_angular_confidence = (
        angular_confidence[
            :,
            primary_tx,
        ]
    )

    #
    # --------------------------------------------------------
    # Absolute AoA intentionally disabled
    # --------------------------------------------------------
    #
    # Validation result:
    # - spatial phase is observable and temporally stable;
    # - angular response changes with physical angle;
    # - ideal 40-mm arcsin model failed;
    # - forward/reverse angular signatures were not
    #   repeatable enough for trusted absolute AoA.
    #
    absolute_aoa_deg = np.full(
        (N, 2),
        np.nan,
        dtype=np.float64,
    )

    absolute_aoa_valid = np.zeros(
        (N, 2),
        dtype=bool,
    )

    closed_form_aoa_deg = np.full(
        (N, 2),
        np.nan,
        dtype=np.float64,
    )

    closed_form_aoa_valid = np.zeros(
        (N, 2),
        dtype=bool,
    )

    if args.output is None:
        args.output = (
            args.tensor.parent.parent
            / "features"
            / "angular_features.npz"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out = {
        #
        # Core measured/derived angular evidence
        #
        "angular_complex":
            D,

        "angular_complex_normalized":
            unit,

        "angular_phase_rad":
            phase,

        "angular_magnitude":
            magnitude,

        "angular_scalar_phase_rad":
            angular_scalar_phase_rad,

        "angular_spectral_coherence":
            angular_spectral_coherence,

        "used_subcarrier_count":
            used_subcarrier_count,

        "subcarrier_coverage":
            subcarrier_coverage,

        "angular_confidence":
            angular_confidence,

        "angular_valid":
            angular_valid,

        #
        # Primary/auxiliary policy
        #
        "angular_primary_tx":
            np.int32(primary_tx),

        "angular_primary_tx_name":
            np.array(
                f"TX{primary_tx}"
            ),

        "primary_angular_valid":
            primary_angular_valid,

        "primary_angular_confidence":
            primary_angular_confidence,

        "tx0_role":
            np.array(
                "primary_angular_channel"
            ),

        "tx1_role":
            np.array(
                "auxiliary_quality_gated"
            ),

        #
        # Absolute AoA intentionally not claimed
        #
        "closed_form_aoa_deg":
            closed_form_aoa_deg,

        "closed_form_aoa_valid":
            closed_form_aoa_valid,

        "absolute_aoa_deg":
            absolute_aoa_deg,

        "absolute_aoa_valid":
            absolute_aoa_valid,

        "absolute_aoa_validation_status":
            np.array(
                "failed_physical_repeatability_validation"
            ),

        #
        # Physical geometry metadata
        #
        "fc_hz":
            np.float64(
                5.89e9
            ),

        "wavelength_m":
            np.float64(
                299792458.0
                / 5.89e9
            ),

        "physical_rx_spacing_m":
            np.float64(
                0.040
            ),

        "spatial_aliasing_present":
            np.bool_(True),

        #
        # Provenance
        #
        "angular_method":
            np.array(
                "pre_beta_wideband_rx_differential"
            ),

        "source_csi_field":
            np.array(
                "H_spatial_raw"
            ),

        "source_tensor":
            np.array(
                str(args.tensor)
            ),

        "post_beta_calibration_applied":
            np.bool_(False),

        "note":
            np.array(
                "Wideband angular RF evidence only. "
                "Pre-beta spatial CSI is validated as "
                "angle-sensitive and temporally stable, "
                "but closed-form absolute AoA failed "
                "forward/reverse physical repeatability. "
                "AoA must remain an estimator layer with "
                "independent calibration and validation."
            ),
    }

    #
    # Preserve useful timing / packet identity fields.
    #
    passthrough_keys = (
        "seq",
        "seq_unwrapped",
        "rf_sample_index",
        "rf_sample_rate_hz",
        "rf_reference_sample_index",
        "rf_time_relative_s",
        "rf_cycle_time_s",
        "rf_rx_sample_skew",
        "rf_sample_time_valid",
        "cycle_time_nominal_s",
        "packet_time_nominal_s",
        "timestamp_monotonic_ns",
        "timestamp_wall_ns",
        "physical_tx_mapping_valid",
    )

    for key in passthrough_keys:
        if key in z:
            out[key] = z[key]

    np.savez_compressed(
        args.output,
        **out,
    )

    print("=" * 68)
    print("ANGULAR EVIDENCE V2")
    print("=" * 68)

    print(
        "source:",
        args.tensor,
    )

    print(
        "H_spatial_raw:",
        H.shape,
    )

    print(
        "output:",
        args.output,
    )

    print()

    for t in range(2):
        print(
            f"TX{t}: "
            f"valid={int(np.sum(angular_valid[:, t]))}/{N} "
            f"median_confidence="
            f"{float(np.nanmedian(angular_confidence[:, t])):.4f} "
            f"median_coverage="
            f"{float(np.nanmedian(subcarrier_coverage[:, t])):.4f}"
        )

    print()

    print(
        "primary TX:",
        f"TX{primary_tx}",
    )

    print(
        "absolute AoA valid:",
        False,
    )

    print(
        "reason:",
        "failed_physical_repeatability_validation",
    )

    if args.calibration is not None:
        print()
        print(
            "NOTE: calibration argument was supplied but "
            "was NOT applied because it belongs to the "
            "post-beta relative-phase pipeline."
        )


if __name__ == "__main__":
    main()
