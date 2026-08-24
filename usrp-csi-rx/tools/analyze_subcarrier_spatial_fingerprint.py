#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np


def wrap(x):
    return np.angle(
        np.exp(1j * x)
    )


def circular_mean(x):
    return np.angle(
        np.mean(
            np.exp(1j * x)
        )
    )


def remove_common_offset(delta):
    offset = circular_mean(
        delta
    )

    return (
        wrap(
            delta - offset
        ),
        offset,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "compatibility_file"
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    path = Path(
        args.compatibility_file
    ).resolve()

    z = np.load(
        path,
        allow_pickle=True,
    )

    phase = np.asarray(
        z[
            "view_phase_mean_rad"
        ],
        dtype=np.float64,
    )

    view_id = np.asarray(
        z[
            "view_id"
        ]
    )

    if phase.shape != (
        3,
        52,
        2,
    ):
        raise SystemExit(
            f"ERROR: expected [3,52,2], got {phase.shape}"
        )

    #
    # A1 = 0
    # B  = 1
    # A2 = 2
    #
    d_ab = wrap(
        phase[1]
        - phase[0]
    )

    d_aa = wrap(
        phase[2]
        - phase[0]
    )

    residual_ab = np.zeros_like(
        d_ab
    )

    residual_aa = np.zeros_like(
        d_aa
    )

    offset_ab = np.zeros(
        2,
        dtype=np.float64,
    )

    offset_aa = np.zeros(
        2,
        dtype=np.float64,
    )

    abs_ab = np.zeros_like(
        d_ab
    )

    abs_aa = np.zeros_like(
        d_aa
    )

    discrimination_ratio = np.zeros_like(
        d_ab
    )

    tx_fraction_ab_gt_aa = np.zeros(
        2,
        dtype=np.float64,
    )

    tx_fraction_ratio_gt_2 = np.zeros(
        2,
        dtype=np.float64,
    )

    tx_median_ratio = np.zeros(
        2,
        dtype=np.float64,
    )

    tx_weighted_effect = np.zeros(
        2,
        dtype=np.float64,
    )

    eps = 1e-6

    for tx in range(2):

        (
            residual_ab[
                :,
                tx,
            ],
            offset_ab[tx],
        ) = remove_common_offset(
            d_ab[
                :,
                tx,
            ]
        )

        (
            residual_aa[
                :,
                tx,
            ],
            offset_aa[tx],
        ) = remove_common_offset(
            d_aa[
                :,
                tx,
            ]
        )

        abs_ab[
            :,
            tx,
        ] = np.abs(
            residual_ab[
                :,
                tx,
            ]
        )

        abs_aa[
            :,
            tx,
        ] = np.abs(
            residual_aa[
                :,
                tx,
            ]
        )

        discrimination_ratio[
            :,
            tx,
        ] = (
            abs_ab[
                :,
                tx,
            ]
            /
            np.maximum(
                abs_aa[
                    :,
                    tx,
                ],
                eps,
            )
        )

        tx_fraction_ab_gt_aa[tx] = float(
            np.mean(
                abs_ab[
                    :,
                    tx,
                ]
                >
                abs_aa[
                    :,
                    tx,
                ]
            )
        )

        tx_fraction_ratio_gt_2[tx] = float(
            np.mean(
                discrimination_ratio[
                    :,
                    tx,
                ]
                >=
                2.0
            )
        )

        tx_median_ratio[tx] = float(
            np.median(
                discrimination_ratio[
                    :,
                    tx,
                ]
            )
        )

        #
        # Signed effect magnitude:
        # positive means A-B exceeds same-pose A-A.
        #
        tx_weighted_effect[tx] = float(
            np.mean(
                abs_ab[
                    :,
                    tx,
                ]
                -
                abs_aa[
                    :,
                    tx,
                ]
            )
        )

    #
    # Conservative fingerprint smoke criterion.
    #
    tx0_fingerprint_pass = bool(
        tx_fraction_ab_gt_aa[0]
        >= 0.70
        and
        tx_fraction_ratio_gt_2[0]
        >= 0.40
        and
        tx_weighted_effect[0]
        > 0.0
    )

    output = (
        Path(args.output)
        if args.output
        else path.parent
        / "subcarrier_spatial_fingerprint.npz"
    )

    np.savez_compressed(
        output,

        schema_version=np.asarray(
            "1.0"
        ),

        package_type=np.asarray(
            "csi_sense_subcarrier_spatial_fingerprint_diagnostic"
        ),

        view_id=view_id,

        a1_b_common_offset_rad=
            offset_ab,

        a1_a2_common_offset_rad=
            offset_aa,

        a1_b_residual_rad=
            residual_ab,

        a1_a2_residual_rad=
            residual_aa,

        a1_b_abs_residual_rad=
            abs_ab,

        a1_a2_abs_residual_rad=
            abs_aa,

        subcarrier_discrimination_ratio=
            discrimination_ratio,

        fraction_ab_greater_than_aa=
            tx_fraction_ab_gt_aa,

        fraction_ratio_ge_2=
            tx_fraction_ratio_gt_2,

        median_subcarrier_ratio=
            tx_median_ratio,

        mean_ab_minus_aa_rad=
            tx_weighted_effect,

        tx0_fingerprint_pass=
            np.asarray(
                tx0_fingerprint_pass
            ),

        absolute_aoa_valid=
            np.asarray(False),

        localization_valid=
            np.asarray(False),

        semantic_policy=np.asarray(
            "common_offset_rejected_subcarrier_phase_structure_is_RF_fingerprint_evidence_not_absolute_AoA_or_position"
        ),
    )

    print("=" * 82)
    print("SUBCARRIER SPATIAL FINGERPRINT")
    print("=" * 82)

    for tx in range(2):

        print()
        print(
            f"TX{tx}"
        )

        print(
            "  A1-B common offset [deg]:",
            float(
                np.rad2deg(
                    offset_ab[tx]
                )
            )
        )

        print(
            "  A1-A2 common offset [deg]:",
            float(
                np.rad2deg(
                    offset_aa[tx]
                )
            )
        )

        print(
            "  SC fraction A1-B > A1-A2:",
            float(
                tx_fraction_ab_gt_aa[
                    tx
                ]
            )
        )

        print(
            "  SC fraction ratio >= 2:",
            float(
                tx_fraction_ratio_gt_2[
                    tx
                ]
            )
        )

        print(
            "  median SC ratio:",
            float(
                tx_median_ratio[
                    tx
                ]
            )
        )

        print(
            "  mean |AB|-|AA| [rad]:",
            float(
                tx_weighted_effect[
                    tx
                ]
            )
        )

        order = np.argsort(
            discrimination_ratio[
                :,
                tx,
            ]
        )[::-1]

        print(
            "  top discriminative SC indices:",
            order[:10].tolist()
        )

        print(
            "  top ratios:",
            np.round(
                discrimination_ratio[
                    order[:10],
                    tx,
                ],
                3,
            ).tolist()
        )

    print()

    print(
        "TX0 fingerprint PASS:",
        tx0_fingerprint_pass
    )

    print(
        "Absolute AoA:",
        False
    )

    print(
        "Localization:",
        False
    )

    print(
        "output:",
        output
    )


if __name__ == "__main__":
    main()
