#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np


def wrap_phase(x):
    return np.angle(
        np.exp(1j * x)
    )


def weighted_circular_stats(
    phase,
    weight,
    mask,
):
    """
    phase  : [N,K]
    weight : [N,K]
    mask   : [K]

    Returns:
      mean_phase [N]
      coherence  [N]
      valid_count[N]
    """

    valid = (
        mask[None, :]
        & np.isfinite(phase)
        & np.isfinite(weight)
        & (weight > 0)
    )

    w = np.where(
        valid,
        weight,
        0.0,
    )

    z = np.where(
        valid,
        np.exp(1j * phase),
        0.0 + 0.0j,
    )

    wz = np.sum(
        w * z,
        axis=1,
    )

    ws = np.sum(
        w,
        axis=1,
    )

    mean_phase = np.full(
        phase.shape[0],
        np.nan,
        dtype=np.float64,
    )

    coherence = np.full(
        phase.shape[0],
        np.nan,
        dtype=np.float64,
    )

    good = ws > 0

    mean_phase[good] = np.angle(
        wz[good]
    )

    coherence[good] = (
        np.abs(wz[good])
        / ws[good]
    )

    valid_count = np.sum(
        valid,
        axis=1,
    ).astype(np.int32)

    return (
        mean_phase,
        coherence,
        valid_count,
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "tensor",
        type=Path,
    )

    ap.add_argument(
        "calibration",
        type=Path,
    )

    ap.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    ap.add_argument(
        "--min-subcarriers",
        type=int,
        default=12,
    )

    ap.add_argument(
        "--min-confidence",
        type=float,
        default=0.70,
    )

    args = ap.parse_args()

    z = np.load(
        args.tensor,
        allow_pickle=True,
    )

    c = np.load(
        args.calibration,
        allow_pickle=True,
    )

    H = z["H_raw"]

    if (
        H.ndim != 4
        or H.shape[1:] != (52, 2, 2)
    ):
        raise RuntimeError(
            f"Unexpected H shape: {H.shape}"
        )

    N = H.shape[0]

    phi_rx_ref = c[
        "phi_rx_ref"
    ].astype(np.float64)

    coh_rx_ref = c[
        "coh_rx_ref"
    ].astype(np.float64)

    valid_rx_mask = c[
        "valid_rx_mask"
    ].astype(bool)

    if phi_rx_ref.shape != (52, 2):
        raise RuntimeError(
            "Unexpected phi_rx_ref shape"
        )

    #
    # ----------------------------------------------------
    # RX differential:
    #
    # D = H_RX1 * conj(H_RX0)
    #
    # shape:
    # [N,52,TX]
    # ----------------------------------------------------
    #
    rx_differential_complex = np.zeros(
        (
            N,
            52,
            2,
        ),
        dtype=np.complex64,
    )

    rx_phase_raw = np.zeros(
        (
            N,
            52,
            2,
        ),
        dtype=np.float32,
    )

    rx_phase_relative = np.zeros_like(
        rx_phase_raw
    )

    rx_differential_magnitude = np.zeros(
        (
            N,
            52,
            2,
        ),
        dtype=np.float32,
    )

    #
    # Weight combines instantaneous differential
    # magnitude and reference coherence.
    #
    angular_weight = np.zeros(
        (
            N,
            52,
            2,
        ),
        dtype=np.float32,
    )

    for t in range(2):

        d = (
            H[:, :, 1, t]
            * np.conj(
                H[:, :, 0, t]
            )
        )

        rx_differential_complex[
            :, :, t
        ] = d

        raw = np.angle(
            d
        )

        rx_phase_raw[
            :, :, t
        ] = raw

        rx_phase_relative[
            :, :, t
        ] = wrap_phase(
            raw
            - phi_rx_ref[
                :, t
            ][None, :]
        )

        mag = np.abs(
            H[:, :, 0, t]
        ) * np.abs(
            H[:, :, 1, t]
        )

        rx_differential_magnitude[
            :, :, t
        ] = mag

        angular_weight[
            :, :, t
        ] = (
            mag
            * coh_rx_ref[
                :, t
            ][None, :]
        )

    #
    # ----------------------------------------------------
    # Aggregate subcarrier angular evidence
    # ----------------------------------------------------
    #
    relative_phase_mean_rad = np.full(
        (
            N,
            2,
        ),
        np.nan,
        dtype=np.float64,
    )

    angular_confidence = np.full(
        (
            N,
            2,
        ),
        np.nan,
        dtype=np.float64,
    )

    used_subcarrier_count = np.zeros(
        (
            N,
            2,
        ),
        dtype=np.int32,
    )

    for t in range(2):

        (
            relative_phase_mean_rad[:, t],
            angular_confidence[:, t],
            used_subcarrier_count[:, t],
        ) = weighted_circular_stats(
            rx_phase_relative[
                :, :, t
            ],
            angular_weight[
                :, :, t
            ],
            valid_rx_mask[
                :, t
            ],
        )

    #
    # Circular confidence measures agreement only among the
    # currently usable subcarriers. Also penalize sparse
    # reference support so TX1 cannot appear as reliable as TX0
    # when only a small fraction of the 52 carriers is valid.
    #
    subcarrier_coverage = (
        used_subcarrier_count.astype(np.float64)
        / 52.0
    )

    overall_angular_confidence = (
        angular_confidence
        * subcarrier_coverage
    )

    angular_valid = (
        np.isfinite(
            relative_phase_mean_rad
        )
        & np.isfinite(
            overall_angular_confidence
        )
        & (
            used_subcarrier_count
            >= args.min_subcarriers
        )
        & (
            overall_angular_confidence
            >= args.min_confidence
        )
    )

    #
    # ----------------------------------------------------
    # Geometry / AoA validity
    # ----------------------------------------------------
    #
    # Current platform geometry is not yet measured.
    # Therefore relative angular evidence is valid,
    # while absolute AoA is deliberately disabled.
    #
    absolute_aoa_valid = np.zeros(
        (
            N,
            2,
        ),
        dtype=bool,
    )

    aoa_rad = np.full(
        (
            N,
            2,
        ),
        np.nan,
        dtype=np.float64,
    )

    aoa_deg = np.full(
        (
            N,
            2,
        ),
        np.nan,
        dtype=np.float64,
    )

    relative_angular_evidence_valid = (
        angular_valid.copy()
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

    save_dict = {
        "rx_differential_complex":
            rx_differential_complex,

        "rx_differential_magnitude":
            rx_differential_magnitude,

        "rx_phase_raw_rad":
            rx_phase_raw,

        "rx_phase_relative_rad":
            rx_phase_relative,

        "relative_phase_mean_rad":
            relative_phase_mean_rad,

        "angular_weight":
            angular_weight,

        "angular_confidence":
            angular_confidence,

        "subcarrier_coverage":
            subcarrier_coverage,

        "overall_angular_confidence":
            overall_angular_confidence,

        "used_subcarrier_count":
            used_subcarrier_count,

        "angular_valid":
            angular_valid,

        "relative_angular_evidence_valid":
            relative_angular_evidence_valid,

        "aoa_rad":
            aoa_rad,

        "aoa_deg":
            aoa_deg,

        "absolute_aoa_valid":
            absolute_aoa_valid,

        "valid_subcarrier_mask":
            valid_rx_mask,

        "reference_coherence":
            coh_rx_ref,

        "reference_phase_rad":
            phi_rx_ref,

        "rx_differential_definition":
            np.array(
                "H_RX1_times_conj_H_RX0"
            ),

        "relative_phase_definition":
            np.array(
                "wrap(raw_rx_differential_phase"
                "_minus_reference_phase)"
            ),

        "angle_convention":
            np.array(
                "not_available_until_geometry_is_measured"
            ),

        "estimator":
            np.array(
                "reference_calibrated_rx_"
                "differential_phase"
            ),

        "calibration_type":
            np.array(
                "reference_based_relative_phase"
            ),

        "absolute_geometry_valid":
            np.bool_(False),

        "antenna_spacing_m":
            np.float64(np.nan),

        "wavelength_m":
            np.float64(
                299792458.0 / 5.89e9
            ),

        "fc_hz":
            np.float64(5.89e9),

        "source_tensor":
            np.array(
                str(args.tensor)
            ),

        "source_calibration":
            np.array(
                str(args.calibration)
            ),

        "note":
            np.array(
                "Relative angular evidence only. "
                "Absolute AoA requires measured "
                "RX phase-center geometry and "
                "known angle convention/reference."
            ),
    }

    #
    # Preserve useful tensor metadata.
    #
    for key in (
        "seq",
        "rf_cycle_time_s",
        "cycle_time_nominal_s",
        "rf_sample_time_valid",
        "sample_domain_time_valid",
    ):
        if key in z.files:
            save_dict[key] = z[key]

    np.savez_compressed(
        args.output,
        **save_dict,
    )

    print(
        "======================================"
    )
    print(
        "ANGULAR FEATURES"
    )
    print(
        "======================================"
    )

    print(
        "tensor :",
        args.tensor,
    )

    print(
        "cal    :",
        args.calibration,
    )

    print(
        "output :",
        args.output,
    )

    print(
        "cycles :",
        N,
    )

    print()

    for t in range(2):

        print(
            f"TX{t}:"
        )

        print(
            "  reference valid subcarriers:",
            int(
                np.sum(
                    valid_rx_mask[
                        :, t
                    ]
                )
            ),
            "/52",
        )

        print(
            "  median used subcarriers:",
            float(
                np.median(
                    used_subcarrier_count[
                        :, t
                    ]
                )
            ),
        )

        print(
            "  median angular confidence:",
            float(
                np.nanmedian(
                    angular_confidence[
                        :, t
                    ]
                )
            ),
        )


        print(
            "  median overall confidence:",
            float(
                np.nanmedian(
                    overall_angular_confidence[
                        :, t
                    ]
                )
            ),
        )

        print(
            "  valid frames:",
            int(
                np.sum(
                    angular_valid[
                        :, t
                    ]
                )
            ),
            "/",
            N,
        )

        print(
            "  median relative phase [rad]:",
            float(
                np.nanmedian(
                    relative_phase_mean_rad[
                        :, t
                    ]
                )
            ),
        )

    print()

    print(
        "absolute AoA valid: False"
    )

    print(
        "reason: RX array geometry "
        "not measured"
    )


if __name__ == "__main__":
    main()
