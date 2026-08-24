#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError:
    raise SystemExit(
        "ERROR: PyYAML required"
    )


def load_yaml(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def unit_vector(x):
    norm = np.linalg.norm(
        x,
        axis=-1,
        keepdims=True,
    )

    return x / np.maximum(
        norm,
        1e-12,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "spatial_fusion_input",
    )

    parser.add_argument(
        "--grid",
        required=True,
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    fusion_path = Path(
        args.spatial_fusion_input
    ).resolve()

    if not fusion_path.exists():
        raise SystemExit(
            f"ERROR: missing {fusion_path}"
        )

    grid_cfg_path = Path(
        args.grid
    ).resolve()

    if not grid_cfg_path.exists():
        raise SystemExit(
            f"ERROR: missing {grid_cfg_path}"
        )

    z = np.load(
        fusion_path,
        allow_pickle=True,
    )

    cfg = load_yaml(
        grid_cfg_path
    )

    view_count = int(
        z["view_count"]
    )

    view_id = np.asarray(
        z["view_id"]
    )

    tx = np.asarray(
        z[
            "view_tx_element_xyz_m"
        ],
        dtype=np.float64,
    )

    rx = np.asarray(
        z[
            "view_rx_element_xyz_m"
        ],
        dtype=np.float64,
    )

    if tx.shape != (
        view_count,
        2,
        3,
    ):
        raise SystemExit(
            f"ERROR: TX shape {tx.shape}"
        )

    if rx.shape != (
        view_count,
        2,
        3,
    ):
        raise SystemExit(
            f"ERROR: RX shape {rx.shape}"
        )

    if not np.all(
        np.isfinite(tx)
    ):
        raise SystemExit(
            "ERROR: non-finite nominal TX geometry"
        )

    if not np.all(
        np.isfinite(rx)
    ):
        raise SystemExit(
            "ERROR: non-finite nominal RX geometry"
        )

    grid = cfg[
        "grid"
    ]

    step = float(
        grid["step_m"]
    )

    xs = np.arange(
        float(
            grid["x_min_m"]
        ),
        float(
            grid["x_max_m"]
        ) + step * 0.5,
        step,
        dtype=np.float64,
    )

    ys = np.arange(
        float(
            grid["y_min_m"]
        ),
        float(
            grid["y_max_m"]
        ) + step * 0.5,
        step,
        dtype=np.float64,
    )

    zs = np.arange(
        float(
            grid["z_min_m"]
        ),
        float(
            grid["z_max_m"]
        ) + step * 0.5,
        step,
        dtype=np.float64,
    )

    X, Y, Z = np.meshgrid(
        xs,
        ys,
        zs,
        indexing="ij",
    )

    candidate_xyz = np.stack(
        [
            X,
            Y,
            Z,
        ],
        axis=-1,
    ).reshape(
        -1,
        3,
    )

    P = candidate_xyz.shape[0]

    #
    # Shapes:
    #
    # candidate_xyz         [P,3]
    # tx                    [V,2,3]
    # rx                    [V,2,3]
    #
    # tx_delta              [V,P,2,3]
    # rx_delta              [V,P,2,3]
    #
    tx_delta = (
        candidate_xyz[
            None,
            :,
            None,
            :,
        ]
        - tx[
            :,
            None,
            :,
            :,
        ]
    )

    rx_delta = (
        candidate_xyz[
            None,
            :,
            None,
            :,
        ]
        - rx[
            :,
            None,
            :,
            :,
        ]
    )

    tx_distance = np.linalg.norm(
        tx_delta,
        axis=-1,
    )

    rx_distance = np.linalg.norm(
        rx_delta,
        axis=-1,
    )

    tx_direction = unit_vector(
        tx_delta
    )

    rx_direction = unit_vector(
        rx_delta
    )

    #
    # Bistatic path hypothesis:
    #
    # [V,P,R,T]
    #
    path_length = (
        rx_distance[
            :,
            :,
            :,
            None,
        ]
        +
        tx_distance[
            :,
            :,
            None,
            :,
        ]
    )

    #
    # Direct TX-RX path:
    #
    # [V,R,T]
    #
    direct_path = np.linalg.norm(
        (
            rx[
                :,
                :,
                None,
                :,
            ]
            -
            tx[
                :,
                None,
                :,
                :,
            ]
        ),
        axis=-1,
    )

    excess_path = (
        path_length
        -
        direct_path[
            :,
            None,
            :,
            :,
        ]
    )

    excess_path = np.maximum(
        excess_path,
        0.0,
    )

    #
    # RX array orientation descriptors.
    #
    rx_axis = np.asarray(
        z[
            "view_rx_array_axis_global"
        ],
        dtype=np.float64,
    )

    rx_broadside = np.asarray(
        z[
            "view_rx_broadside_axis_global"
        ],
        dtype=np.float64,
    )

    #
    # Candidate direction relative to RX0 platform reference.
    #
    rx0_direction = rx_direction[
        :,
        :,
        0,
        :,
    ]

    rx0_array_projection = np.sum(
        rx0_direction
        * rx_axis[
            :,
            None,
            :,
        ],
        axis=-1,
    )

    rx0_broadside_projection = np.sum(
        rx0_direction
        * rx_broadside[
            :,
            None,
            :,
        ],
        axis=-1,
    )

    #
    # These are pure geometric descriptors.
    #
    # No conversion to ToF/range/Doppler.
    #

    output = (
        Path(
            args.output
        )
        if args.output
        else fusion_path.parent
        / "geometric_candidate_descriptors.npz"
    )

    np.savez_compressed(
        output,

        schema_version=np.asarray(
            "1.0"
        ),

        package_type=np.asarray(
            "csi_sense_geometric_candidate_descriptors"
        ),

        coordinate_system=np.asarray(
            "room_local"
        ),

        source_spatial_fusion_input=
            np.asarray(
                str(
                    fusion_path
                )
            ),

        source_grid_config=
            np.asarray(
                str(
                    grid_cfg_path
                )
            ),

        view_count=np.asarray(
            view_count,
            dtype=np.int32,
        ),

        candidate_count=np.asarray(
            P,
            dtype=np.int32,
        ),

        view_id=view_id,

        grid_x_m=xs,
        grid_y_m=ys,
        grid_z_m=zs,

        candidate_xyz_m=
            candidate_xyz,

        #
        # Candidate ↔ TX geometry.
        #
        candidate_tx_distance_m=
            tx_distance,

        candidate_tx_direction_unit=
            tx_direction,

        #
        # Candidate ↔ RX geometry.
        #
        candidate_rx_distance_m=
            rx_distance,

        candidate_rx_direction_unit=
            rx_direction,

        #
        # Bistatic hypotheses.
        #
        bistatic_path_length_hypothesis_m=
            path_length,

        direct_tx_rx_path_length_m=
            direct_path,

        excess_bistatic_path_hypothesis_m=
            excess_path,

        #
        # Angular geometric descriptors only.
        #
        rx0_candidate_array_axis_projection=
            rx0_array_projection,

        rx0_candidate_broadside_projection=
            rx0_broadside_projection,

        #
        # Validity policy.
        #
        validity_nominal_geometry=
            np.asarray(
                True
            ),

        validity_rf_phase_center_geometry=
            np.asarray(
                False
            ),

        validity_geometric_path_hypothesis=
            np.asarray(
                True
            ),

        validity_measured_tof=
            np.asarray(
                False
            ),

        validity_absolute_range=
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

        delay_to_path_length_mapping_allowed=
            np.asarray(
                False
            ),

        physical_backprojection_allowed=
            np.asarray(
                False
            ),

        semantic_policy=np.asarray(
            "candidate_descriptors_are_geometry_only_and_must_not_be_interpreted_as_measured_ToF_range_AoA_or_AoD"
        ),
    )

    print(
        "=" * 78
    )

    print(
        "GEOMETRIC CANDIDATE DESCRIPTORS"
    )

    print(
        "=" * 78
    )

    print(
        "views:",
        view_count
    )

    print(
        "grid shape:",
        (
            len(xs),
            len(ys),
            len(zs),
        )
    )

    print(
        "candidates:",
        P
    )

    print(
        "TX distance:",
        tx_distance.shape
    )

    print(
        "RX distance:",
        rx_distance.shape
    )

    print(
        "bistatic path:",
        path_length.shape
    )

    print(
        "direct path:",
        direct_path.shape
    )

    print(
        "excess path:",
        excess_path.shape
    )

    print()

    print(
        "measured ToF:",
        False
    )

    print(
        "absolute range:",
        False
    )

    print(
        "physical backprojection:",
        False
    )

    print()

    print(
        "output:",
        output
    )

    print(
        "geometric descriptors: PASS"
    )


if __name__ == "__main__":
    main()
