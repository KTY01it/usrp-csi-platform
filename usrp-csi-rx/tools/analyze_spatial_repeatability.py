#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np


def wrap_phase(x):
    return np.angle(
        np.exp(1j * x)
    )


def circular_mean(x, axis=None):
    return np.angle(
        np.mean(
            np.exp(1j * x),
            axis=axis,
        )
    )


def circular_coherence(x, axis=None):
    return np.abs(
        np.mean(
            np.exp(1j * x),
            axis=axis,
        )
    )


def pair_metrics(
    phase,
    a,
    b,
):
    delta = wrap_phase(
        phase[b]
        - phase[a]
    )

    tx_count = delta.shape[1]

    common_offset = np.zeros(
        tx_count,
        dtype=np.float64,
    )

    spectral_coherence = np.zeros(
        tx_count,
        dtype=np.float64,
    )

    residual = np.zeros_like(
        delta,
        dtype=np.float64,
    )

    median_abs_raw = np.zeros(
        tx_count,
        dtype=np.float64,
    )

    median_abs_residual = np.zeros(
        tx_count,
        dtype=np.float64,
    )

    p90_abs_residual = np.zeros(
        tx_count,
        dtype=np.float64,
    )

    rms_residual = np.zeros(
        tx_count,
        dtype=np.float64,
    )

    for tx in range(
        tx_count
    ):
        d = delta[
            :,
            tx,
        ]

        median_abs_raw[tx] = float(
            np.median(
                np.abs(d)
            )
        )

        common_offset[tx] = float(
            circular_mean(d)
        )

        spectral_coherence[tx] = float(
            circular_coherence(d)
        )

        r = wrap_phase(
            d
            - common_offset[tx]
        )

        residual[
            :,
            tx,
        ] = r

        ares = np.abs(r)

        median_abs_residual[tx] = float(
            np.median(ares)
        )

        p90_abs_residual[tx] = float(
            np.percentile(
                ares,
                90,
            )
        )

        rms_residual[tx] = float(
            np.sqrt(
                np.mean(
                    r ** 2
                )
            )
        )

    return {
        "delta":
            delta,

        "common_offset":
            common_offset,

        "spectral_coherence":
            spectral_coherence,

        "residual":
            residual,

        "median_abs_raw":
            median_abs_raw,

        "median_abs_residual":
            median_abs_residual,

        "p90_abs_residual":
            p90_abs_residual,

        "rms_residual":
            rms_residual,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "compatibility_file",
        help=(
            "rf_geometry_compatibility.npz "
            "from A1-B-A2 repeatability session"
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    path = Path(
        args.compatibility_file
    ).resolve()

    if not path.exists():
        raise SystemExit(
            f"ERROR: missing {path}"
        )

    z = np.load(
        path,
        allow_pickle=True,
    )

    view_id = np.asarray(
        z[
            "view_id"
        ]
    )

    phase = np.asarray(
        z[
            "view_phase_mean_rad"
        ],
        dtype=np.float64,
    )

    coherence = np.asarray(
        z[
            "view_phase_coherence"
        ],
        dtype=np.float64,
    )

    confidence = np.asarray(
        z[
            "view_angular_confidence"
        ],
        dtype=np.float64,
    )

    valid_fraction = np.asarray(
        z[
            "view_angular_valid_fraction"
        ],
        dtype=np.float64,
    )

    if phase.shape[0] != 3:
        raise SystemExit(
            "ERROR: A-B-A test requires exactly 3 views"
        )

    #
    # Expected semantics:
    #
    # view_000 = A1
    # view_001 = B
    # view_002 = A2
    #
    a1_b = pair_metrics(
        phase,
        0,
        1,
    )

    a1_a2 = pair_metrics(
        phase,
        0,
        2,
    )

    b_a2 = pair_metrics(
        phase,
        1,
        2,
    )

    #
    # Spatial discrimination ratio:
    #
    # > 1 means A1-B residual is larger than
    # same-pose A1-A2 residual.
    #
    eps = 1e-12

    spatial_ratio_median = (
        a1_b[
            "median_abs_residual"
        ]
        /
        np.maximum(
            a1_a2[
                "median_abs_residual"
            ],
            eps,
        )
    )

    spatial_ratio_rms = (
        a1_b[
            "rms_residual"
        ]
        /
        np.maximum(
            a1_a2[
                "rms_residual"
            ],
            eps,
        )
    )

    #
    # Conservative PASS:
    #
    # TX0 is primary.
    #
    # We require:
    #   - within-view RF coherence still strong
    #   - A1-B residual clearly larger than A1-A2
    #
    tx0_min_view_coherence = float(
        np.min(
            np.mean(
                coherence[
                    :,
                    :,
                    0,
                ],
                axis=1,
            )
        )
    )

    tx0_ratio = float(
        spatial_ratio_median[0]
    )

    tx0_repeatability_pass = bool(
        tx0_min_view_coherence >= 0.90
        and
        tx0_ratio >= 2.0
    )

    output = (
        Path(
            args.output
        )
        if args.output
        else path.parent
        / "spatial_repeatability_report.npz"
    )

    np.savez_compressed(
        output,

        schema_version=np.asarray(
            "1.0"
        ),

        package_type=np.asarray(
            "csi_sense_spatial_repeatability_diagnostic"
        ),

        view_id=view_id,

        expected_view_semantics=np.asarray(
            [
                "A1_same_pose",
                "B_displaced_pose",
                "A2_returned_same_pose",
            ]
        ),

        a1_b_common_offset_rad=
            a1_b[
                "common_offset"
            ],

        a1_b_spectral_coherence=
            a1_b[
                "spectral_coherence"
            ],

        a1_b_raw_median_abs_rad=
            a1_b[
                "median_abs_raw"
            ],

        a1_b_residual_median_abs_rad=
            a1_b[
                "median_abs_residual"
            ],

        a1_b_residual_p90_abs_rad=
            a1_b[
                "p90_abs_residual"
            ],

        a1_b_residual_rms_rad=
            a1_b[
                "rms_residual"
            ],

        a1_a2_common_offset_rad=
            a1_a2[
                "common_offset"
            ],

        a1_a2_spectral_coherence=
            a1_a2[
                "spectral_coherence"
            ],

        a1_a2_raw_median_abs_rad=
            a1_a2[
                "median_abs_raw"
            ],

        a1_a2_residual_median_abs_rad=
            a1_a2[
                "median_abs_residual"
            ],

        a1_a2_residual_p90_abs_rad=
            a1_a2[
                "p90_abs_residual"
            ],

        a1_a2_residual_rms_rad=
            a1_a2[
                "rms_residual"
            ],

        b_a2_residual_median_abs_rad=
            b_a2[
                "median_abs_residual"
            ],

        spatial_discrimination_ratio_median=
            spatial_ratio_median,

        spatial_discrimination_ratio_rms=
            spatial_ratio_rms,

        tx0_min_view_phase_coherence=
            np.asarray(
                tx0_min_view_coherence
            ),

        tx0_repeatability_pass=
            np.asarray(
                tx0_repeatability_pass
            ),

        absolute_aoa_valid=
            np.asarray(
                False
            ),

        phase_residual_is_absolute_angle=
            np.asarray(
                False
            ),

        semantic_policy=np.asarray(
            "A_B_A_phase_repeatability_is_RF_spatial_evidence_only_not_absolute_AoA_or_position_ground_truth"
        ),
    )

    print(
        "=" * 82
    )

    print(
        "A-B-A SPATIAL REPEATABILITY"
    )

    print(
        "=" * 82
    )

    print(
        "views:",
        view_id.tolist()
    )

    for tx in range(2):

        print()
        print(
            f"TX{tx}"
        )

        print(
            "  A1-B raw median |dphi| [rad]:",
            float(
                a1_b[
                    "median_abs_raw"
                ][tx]
            )
        )

        print(
            "  A1-B common offset [rad]:",
            float(
                a1_b[
                    "common_offset"
                ][tx]
            )
        )

        print(
            "  A1-B spectral coherence:",
            float(
                a1_b[
                    "spectral_coherence"
                ][tx]
            )
        )

        print(
            "  A1-B residual median [rad]:",
            float(
                a1_b[
                    "median_abs_residual"
                ][tx]
            )
        )

        print(
            "  A1-B residual p90 [rad]:",
            float(
                a1_b[
                    "p90_abs_residual"
                ][tx]
            )
        )

        print(
            "  A1-A2 raw median |dphi| [rad]:",
            float(
                a1_a2[
                    "median_abs_raw"
                ][tx]
            )
        )

        print(
            "  A1-A2 common offset [rad]:",
            float(
                a1_a2[
                    "common_offset"
                ][tx]
            )
        )

        print(
            "  A1-A2 spectral coherence:",
            float(
                a1_a2[
                    "spectral_coherence"
                ][tx]
            )
        )

        print(
            "  A1-A2 residual median [rad]:",
            float(
                a1_a2[
                    "median_abs_residual"
                ][tx]
            )
        )

        print(
            "  A1-A2 residual p90 [rad]:",
            float(
                a1_a2[
                    "p90_abs_residual"
                ][tx]
            )
        )

        print(
            "  spatial ratio median:",
            float(
                spatial_ratio_median[
                    tx
                ]
            )
        )

        print(
            "  spatial ratio RMS:",
            float(
                spatial_ratio_rms[
                    tx
                ]
            )
        )

    print()
    print(
        "TX0 minimum within-view coherence:",
        tx0_min_view_coherence
    )

    print(
        "TX0 spatial-repeatability PASS:",
        tx0_repeatability_pass
    )

    print(
        "Absolute AoA interpretation:",
        False
    )

    print(
        "output:",
        output
    )


if __name__ == "__main__":
    main()
