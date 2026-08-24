#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np


def circular_mean_phase(
    x,
    axis,
):
    return np.angle(
        np.mean(
            np.exp(
                1j * x
            ),
            axis=axis,
        )
    )


def circular_coherence(
    x,
    axis,
):
    return np.abs(
        np.mean(
            np.exp(
                1j * x
            ),
            axis=axis,
        )
    )


def normalize01(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    mn = np.nanmin(
        x
    )

    mx = np.nanmax(
        x
    )

    if not np.isfinite(
        mn
    ) or not np.isfinite(
        mx
    ):
        return np.zeros_like(
            x,
            dtype=np.float32,
        )

    if mx <= mn + 1e-12:
        return np.zeros_like(
            x,
            dtype=np.float32,
        )

    return (
        (
            x - mn
        )
        /
        (
            mx - mn
        )
    ).astype(
        np.float32
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "spatial_fusion_input",
    )

    parser.add_argument(
        "geometric_descriptors",
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    fusion_path = Path(
        args.spatial_fusion_input
    ).resolve()

    geom_path = Path(
        args.geometric_descriptors
    ).resolve()

    if not fusion_path.exists():
        raise SystemExit(
            f"ERROR: missing {fusion_path}"
        )

    if not geom_path.exists():
        raise SystemExit(
            f"ERROR: missing {geom_path}"
        )

    f = np.load(
        fusion_path,
        allow_pickle=True,
    )

    g = np.load(
        geom_path,
        allow_pickle=True,
    )

    view_id_f = np.asarray(
        f[
            "view_id"
        ]
    )

    view_id_g = np.asarray(
        g[
            "view_id"
        ]
    )

    if not np.array_equal(
        view_id_f,
        view_id_g,
    ):
        raise SystemExit(
            "ERROR: view mismatch"
        )

    view_count = int(
        f[
            "view_count"
        ]
    )

    candidate_xyz = np.asarray(
        g[
            "candidate_xyz_m"
        ],
        dtype=np.float64,
    )

    P = candidate_xyz.shape[0]

    cycle_view_index = np.asarray(
        f[
            "cycle_view_index"
        ],
        dtype=np.int32,
    )

    angular_complex = np.asarray(
        f[
            "angular_rx_differential_complex"
        ]
    )

    angular_confidence = np.asarray(
        f[
            "angular_confidence"
        ],
        dtype=np.float64,
    )

    angular_valid = np.asarray(
        f[
            "angular_valid"
        ],
        dtype=bool,
    )

    if angular_complex.ndim != 3:
        raise SystemExit(
            "ERROR: invalid angular evidence shape"
        )

    #
    # Measured phase evidence:
    #
    # [N,52,2]
    #
    measured_phase = np.angle(
        angular_complex
    )

    #
    # Collapse cycles belonging to each view.
    #
    view_phase_mean = np.full(
        (
            view_count,
            52,
            2,
        ),
        np.nan,
        dtype=np.float64,
    )

    view_phase_coherence = np.zeros(
        (
            view_count,
            52,
            2,
        ),
        dtype=np.float64,
    )

    view_angular_confidence = np.zeros(
        (
            view_count,
            2,
        ),
        dtype=np.float64,
    )

    view_valid_fraction = np.zeros(
        (
            view_count,
            2,
        ),
        dtype=np.float64,
    )

    for v in range(
        view_count
    ):
        mask = (
            cycle_view_index
            ==
            v
        )

        if not np.any(
            mask
        ):
            raise SystemExit(
                f"ERROR: no cycles for view {v}"
            )

        phase_v = measured_phase[
            mask
        ]

        view_phase_mean[v] = (
            circular_mean_phase(
                phase_v,
                axis=0,
            )
        )

        view_phase_coherence[v] = (
            circular_coherence(
                phase_v,
                axis=0,
            )
        )

        view_angular_confidence[v] = (
            np.nanmedian(
                angular_confidence[
                    mask
                ],
                axis=0,
            )
        )

        view_valid_fraction[v] = (
            np.mean(
                angular_valid[
                    mask
                ],
                axis=0,
            )
        )

    #
    # Reduce subcarrier evidence to robust view/TX
    # descriptors.
    #
    view_phase_scalar = (
        circular_mean_phase(
            view_phase_mean,
            axis=1,
        )
    )

    view_phase_scalar_coherence = (
        np.mean(
            view_phase_coherence,
            axis=1,
        )
    )

    #
    # Geometry descriptors already generated:
    #
    # [V,P]
    #
    axis_projection = np.asarray(
        g[
            "rx0_candidate_array_axis_projection"
        ],
        dtype=np.float64,
    )

    broadside_projection = np.asarray(
        g[
            "rx0_candidate_broadside_projection"
        ],
        dtype=np.float64,
    )

    excess_path = np.asarray(
        g[
            "excess_bistatic_path_hypothesis_m"
        ],
        dtype=np.float64,
    )

    #
    # IMPORTANT:
    #
    # No closed-form mapping from measured phase
    # to absolute angle is allowed.
    #
    # We construct feature tensors only.
    #
    angular_feature = np.zeros(
        (
            view_count,
            P,
            8,
        ),
        dtype=np.float32,
    )

    #
    # Candidate geometry features.
    #
    angular_feature[
        :,
        :,
        0,
    ] = axis_projection

    angular_feature[
        :,
        :,
        1,
    ] = broadside_projection

    #
    # View-level measured RF phase features.
    # TX0 primary.
    #
    angular_feature[
        :,
        :,
        2,
    ] = view_phase_scalar[
        :,
        0,
        None,
    ]

    angular_feature[
        :,
        :,
        3,
    ] = view_phase_scalar_coherence[
        :,
        0,
        None,
    ]

    angular_feature[
        :,
        :,
        4,
    ] = view_angular_confidence[
        :,
        0,
        None,
    ]

    angular_feature[
        :,
        :,
        5,
    ] = view_valid_fraction[
        :,
        0,
        None,
    ]

    #
    # TX1 auxiliary-quality features.
    #
    angular_feature[
        :,
        :,
        6,
    ] = view_angular_confidence[
        :,
        1,
        None,
    ]

    angular_feature[
        :,
        :,
        7,
    ] = view_valid_fraction[
        :,
        1,
        None,
    ]

    #
    # Geometry-only path descriptors are carried
    # separately, never interpreted as measured range.
    #
    excess_path_min = np.min(
        excess_path,
        axis=(
            2,
            3,
        ),
    )

    excess_path_mean = np.mean(
        excess_path,
        axis=(
            2,
            3,
        ),
    )

    #
    # Multi-view geometric support descriptors.
    #
    geometry_support = np.zeros(
        (
            P,
            4,
        ),
        dtype=np.float32,
    )

    geometry_support[
        :,
        0,
    ] = np.mean(
        np.abs(
            axis_projection
        ),
        axis=0,
    )

    geometry_support[
        :,
        1,
    ] = np.mean(
        broadside_projection,
        axis=0,
    )

    geometry_support[
        :,
        2,
    ] = np.std(
        excess_path_min,
        axis=0,
    )

    geometry_support[
        :,
        3,
    ] = np.std(
        excess_path_mean,
        axis=0,
    )

    #
    # Simple normalized diagnostic score only.
    #
    # This is NOT occupancy probability.
    #
    broadside_support = normalize01(
        np.mean(
            np.maximum(
                broadside_projection,
                0.0,
            ),
            axis=0,
        )
    )

    view_confidence_mean = float(
        np.mean(
            view_angular_confidence[
                :,
                0,
            ]
        )
    )

    diagnostic_candidate_score = (
        broadside_support
        * view_confidence_mean
    ).astype(
        np.float32
    )

    output = (
        Path(
            args.output
        )
        if args.output
        else fusion_path.parent
        / "rf_geometry_compatibility.npz"
    )

    np.savez_compressed(
        output,

        schema_version=np.asarray(
            "1.0"
        ),

        package_type=np.asarray(
            "csi_sense_rf_geometry_compatibility_features"
        ),

        view_id=view_id_f,

        candidate_xyz_m=
            candidate_xyz,

        view_phase_mean_rad=
            view_phase_mean,

        view_phase_coherence=
            view_phase_coherence,

        view_phase_scalar_rad=
            view_phase_scalar,

        view_phase_scalar_coherence=
            view_phase_scalar_coherence,

        view_angular_confidence=
            view_angular_confidence,

        view_angular_valid_fraction=
            view_valid_fraction,

        candidate_axis_projection=
            axis_projection,

        candidate_broadside_projection=
            broadside_projection,

        angular_geometry_feature=
            angular_feature,

        excess_path_min_hypothesis_m=
            excess_path_min,

        excess_path_mean_hypothesis_m=
            excess_path_mean,

        geometry_support_feature=
            geometry_support,

        diagnostic_candidate_score=
            diagnostic_candidate_score,

        feature_names=np.asarray(
            [
                "rx_array_axis_projection",
                "rx_broadside_projection",
                "TX0_measured_phase_summary",
                "TX0_phase_coherence",
                "TX0_angular_confidence",
                "TX0_valid_fraction",
                "TX1_angular_confidence",
                "TX1_valid_fraction",
            ]
        ),

        diagnostic_score_is_occupancy_probability=
            np.asarray(
                False
            ),

        validity_absolute_aoa=
            np.asarray(
                False
            ),

        validity_absolute_aod=
            np.asarray(
                False
            ),

        validity_measured_tof=
            np.asarray(
                False
            ),

        validity_absolute_range=
            np.asarray(
                False
            ),

        physical_backprojection_allowed=
            np.asarray(
                False
            ),

        semantic_policy=np.asarray(
            "RF_geometry_features_are_estimator_inputs_not_absolute_angle_range_or_occupancy_measurements"
        ),
    )

    print(
        "=" * 78
    )

    print(
        "RF-GEOMETRY COMPATIBILITY FEATURES"
    )

    print(
        "=" * 78
    )

    print(
        "views:",
        view_count
    )

    print(
        "candidates:",
        P
    )

    print(
        "view phase:",
        view_phase_mean.shape
    )

    print(
        "angular geometry feature:",
        angular_feature.shape
    )

    print(
        "geometry support:",
        geometry_support.shape
    )

    print(
        "diagnostic score:",
        diagnostic_candidate_score.shape
    )

    print()

    print(
        "Absolute AoA:",
        False
    )

    print(
        "Absolute range:",
        False
    )

    print(
        "Occupancy probability:",
        False
    )

    print()

    print(
        "output:",
        output
    )

    print(
        "RF-geometry compatibility: PASS"
    )


if __name__ == "__main__":
    main()
