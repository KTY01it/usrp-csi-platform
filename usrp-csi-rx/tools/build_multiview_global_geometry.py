#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError:
    raise SystemExit(
        "ERROR: PyYAML is required"
    )


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_yaml(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return yaml.safe_load(f)


def rotation_matrix_ypr_deg(
    yaw_deg,
    pitch_deg,
    roll_deg,
):
    """
    Active right-handed rotation:

        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    Local platform vector -> room_local vector.

    This convention is explicitly recorded in the
    output package. It is a software convention and
    does not itself validate physical orientation.
    """

    yaw = np.deg2rad(
        float(yaw_deg)
    )

    pitch = np.deg2rad(
        float(pitch_deg)
    )

    roll = np.deg2rad(
        float(roll_deg)
    )

    cy = np.cos(yaw)
    sy = np.sin(yaw)

    cp = np.cos(pitch)
    sp = np.sin(pitch)

    cr = np.cos(roll)
    sr = np.sin(roll)

    rz = np.asarray(
        [
            [cy, -sy, 0.0],
            [sy,  cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    ry = np.asarray(
        [
            [ cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=np.float64,
    )

    rx = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0,  cr, -sr],
            [0.0,  sr,  cr],
        ],
        dtype=np.float64,
    )

    return (
        rz
        @ ry
        @ rx
    )


def parse_local_xyz(
    geometry,
    array_name,
    element_names,
):
    array = geometry.get(
        array_name,
        {}
    )

    antennas = array.get(
        "antennas",
        {}
    )

    xyz = []

    available = True
    phase_center_valid = True

    for name in element_names:
        entry = antennas.get(
            name,
            {}
        )

        p = entry.get(
            "xyz_m"
        )

        if (
            p is None
            or not isinstance(
                p,
                list
            )
            or len(p) != 3
        ):
            xyz.append(
                np.full(
                    3,
                    np.nan,
                    dtype=np.float64,
                )
            )

            available = False

        else:
            xyz.append(
                np.asarray(
                    p,
                    dtype=np.float64,
                )
            )

        if (
            entry.get(
                "phase_center_measured"
            )
            is not True
        ):
            phase_center_valid = False

    return (
        np.asarray(
            xyz,
            dtype=np.float64,
        ),
        bool(available),
        bool(phase_center_valid),
    )


def transform_elements(
    platform_xyz,
    rotation,
    local_xyz,
):
    out = np.full(
        local_xyz.shape,
        np.nan,
        dtype=np.float64,
    )

    for i in range(
        local_xyz.shape[0]
    ):
        p = local_xyz[i]

        if np.all(
            np.isfinite(p)
        ):
            out[i] = (
                platform_xyz
                + rotation @ p
            )

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "session_dir",
        help=(
            "Validated multi-view "
            "csi-sense session."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    session = Path(
        args.session_dir
    ).resolve()

    qc_path = (
        session
        / "dataset_qc.json"
    )

    index_path = (
        session
        / "dataset_index.npz"
    )

    geometry_path = (
        session
        / "config_snapshot"
        / "geometry.yaml"
    )

    if not qc_path.exists():
        raise SystemExit(
            "ERROR: dataset_qc.json missing"
        )

    qc = load_json(
        qc_path
    )

    if qc.get(
        "status"
    ) != "PASS":
        raise SystemExit(
            "ERROR: dataset QC is not PASS"
        )

    if not index_path.exists():
        raise SystemExit(
            "ERROR: dataset_index.npz missing"
        )

    if not geometry_path.exists():
        raise SystemExit(
            "ERROR: geometry snapshot missing"
        )

    index = np.load(
        index_path,
        allow_pickle=True,
    )

    geometry = load_yaml(
        geometry_path
    )

    view_id = np.asarray(
        index[
            "view_id"
        ]
    )

    tx_platform_xyz = np.asarray(
        index[
            "tx_xyz_m"
        ],
        dtype=np.float64,
    )

    rx_platform_xyz = np.asarray(
        index[
            "rx_xyz_m"
        ],
        dtype=np.float64,
    )

    tx_ypr = np.asarray(
        index[
            "tx_orientation_ypr_deg"
        ],
        dtype=np.float64,
    )

    rx_ypr = np.asarray(
        index[
            "rx_orientation_ypr_deg"
        ],
        dtype=np.float64,
    )

    view_count = len(
        view_id
    )

    (
        rx_local_xyz,
        rx_local_available,
        rx_phase_center_valid,
    ) = parse_local_xyz(
        geometry,
        "rx_array",
        [
            "RX0",
            "RX1",
        ],
    )

    (
        tx_local_xyz,
        tx_local_available,
        tx_phase_center_valid,
    ) = parse_local_xyz(
        geometry,
        "tx_array",
        [
            "TX0",
            "TX1",
        ],
    )

    rx_rotation = np.empty(
        (
            view_count,
            3,
            3,
        ),
        dtype=np.float64,
    )

    tx_rotation = np.empty(
        (
            view_count,
            3,
            3,
        ),
        dtype=np.float64,
    )

    rx_global_xyz = np.full(
        (
            view_count,
            2,
            3,
        ),
        np.nan,
        dtype=np.float64,
    )

    tx_global_xyz = np.full(
        (
            view_count,
            2,
            3,
        ),
        np.nan,
        dtype=np.float64,
    )

    rx_array_axis_global = np.full(
        (
            view_count,
            3,
        ),
        np.nan,
        dtype=np.float64,
    )

    rx_broadside_axis_global = np.full(
        (
            view_count,
            3,
        ),
        np.nan,
        dtype=np.float64,
    )

    rx_vertical_axis_global = np.full(
        (
            view_count,
            3,
        ),
        np.nan,
        dtype=np.float64,
    )

    local_x = np.asarray(
        [1.0, 0.0, 0.0],
        dtype=np.float64,
    )

    local_y = np.asarray(
        [0.0, 1.0, 0.0],
        dtype=np.float64,
    )

    local_z = np.asarray(
        [0.0, 0.0, 1.0],
        dtype=np.float64,
    )

    for i in range(
        view_count
    ):
        r_rx = rotation_matrix_ypr_deg(
            rx_ypr[i, 0],
            rx_ypr[i, 1],
            rx_ypr[i, 2],
        )

        r_tx = rotation_matrix_ypr_deg(
            tx_ypr[i, 0],
            tx_ypr[i, 1],
            tx_ypr[i, 2],
        )

        rx_rotation[i] = r_rx
        tx_rotation[i] = r_tx

        rx_global_xyz[i] = (
            transform_elements(
                rx_platform_xyz[i],
                r_rx,
                rx_local_xyz,
            )
        )

        tx_global_xyz[i] = (
            transform_elements(
                tx_platform_xyz[i],
                r_tx,
                tx_local_xyz,
            )
        )

        rx_array_axis_global[i] = (
            r_rx
            @ local_x
        )

        rx_broadside_axis_global[i] = (
            r_rx
            @ local_y
        )

        rx_vertical_axis_global[i] = (
            r_rx
            @ local_z
        )

    #
    # This is deliberately conservative:
    #
    # RX physical antenna centers are available,
    # but RF phase centers have not been measured.
    #
    # TX local geometry is currently absent.
    #
    rx_nominal_geometry_available = bool(
        rx_local_available
    )

    tx_nominal_geometry_available = bool(
        tx_local_available
    )

    absolute_spatial_geometry_valid = bool(
        rx_local_available
        and tx_local_available
        and rx_phase_center_valid
        and tx_phase_center_valid
        and geometry.get(
            "geometry_status",
            {}
        ).get(
            "ready_for_absolute_spatial_processing"
        )
        is True
    )

    output = (
        Path(
            args.output
        )
        if args.output
        else session
        / "global_geometry.npz"
    )

    np.savez_compressed(
        output,

        schema_version=np.asarray(
            "1.0"
        ),

        package_type=np.asarray(
            "csi_sense_multiview_global_geometry"
        ),

        session_id=np.asarray(
            str(
                index[
                    "session_id"
                ].item()
            )
        ),

        coordinate_system=np.asarray(
            "room_local"
        ),

        transform_convention=np.asarray(
            "active_right_handed_Rz_yaw_Ry_pitch_Rx_roll_local_to_room"
        ),

        geometry_interpretation=np.asarray(
            "nominal_physical_antenna_centers_not_validated_RF_phase_centers"
        ),

        view_count=np.asarray(
            view_count,
            dtype=np.int32,
        ),

        view_id=view_id,

        tx_platform_xyz_m=
            tx_platform_xyz,

        rx_platform_xyz_m=
            rx_platform_xyz,

        tx_orientation_ypr_deg=
            tx_ypr,

        rx_orientation_ypr_deg=
            rx_ypr,

        tx_rotation_local_to_room=
            tx_rotation,

        rx_rotation_local_to_room=
            rx_rotation,

        tx_local_element_xyz_m=
            tx_local_xyz,

        rx_local_element_xyz_m=
            rx_local_xyz,

        tx_element_global_xyz_m=
            tx_global_xyz,

        rx_element_global_xyz_m=
            rx_global_xyz,

        tx0_global_xyz_m=
            tx_global_xyz[:, 0, :],

        tx1_global_xyz_m=
            tx_global_xyz[:, 1, :],

        rx0_global_xyz_m=
            rx_global_xyz[:, 0, :],

        rx1_global_xyz_m=
            rx_global_xyz[:, 1, :],

        rx_array_axis_global=
            rx_array_axis_global,

        rx_broadside_axis_global=
            rx_broadside_axis_global,

        rx_vertical_axis_global=
            rx_vertical_axis_global,

        validity_rx_nominal_physical_center_geometry=
            np.asarray(
                rx_nominal_geometry_available
            ),

        validity_rx_rf_phase_center_geometry=
            np.asarray(
                rx_phase_center_valid
            ),

        validity_tx_nominal_physical_center_geometry=
            np.asarray(
                tx_nominal_geometry_available
            ),

        validity_tx_rf_phase_center_geometry=
            np.asarray(
                tx_phase_center_valid
            ),

        validity_absolute_spatial_geometry=
            np.asarray(
                absolute_spatial_geometry_valid
            ),

        validity_absolute_aoa_geometry=
            np.asarray(
                False
            ),

        validity_absolute_aod_geometry=
            np.asarray(
                False
            ),

        validity_absolute_backprojection_geometry=
            np.asarray(
                False
            ),

        scientific_warning=np.asarray(
            "RX global coordinates use nominal physical antenna centers; "
            "RF phase centers are not measured. "
            "TX element coordinates remain invalid until TX local geometry "
            "is measured. Absolute AoA/AoD/backprojection remain disabled."
        ),
    )

    print(
        "=" * 76
    )

    print(
        "MULTI-VIEW GLOBAL ANTENNA GEOMETRY"
    )

    print(
        "=" * 76
    )

    print(
        "session:",
        session
    )

    print(
        "views:",
        view_count
    )

    print(
        "output:",
        output
    )

    print()

    print(
        "RX local nominal geometry available:",
        rx_nominal_geometry_available
    )

    print(
        "RX RF phase-center geometry valid:",
        rx_phase_center_valid
    )

    print(
        "TX local nominal geometry available:",
        tx_nominal_geometry_available
    )

    print(
        "TX RF phase-center geometry valid:",
        tx_phase_center_valid
    )

    print(
        "Absolute spatial geometry valid:",
        absolute_spatial_geometry_valid
    )

    for i in range(
        view_count
    ):
        print()
        print(
            view_id[i]
        )

        print(
            "  RX0 global:",
            rx_global_xyz[
                i,
                0,
            ].tolist()
        )

        print(
            "  RX1 global:",
            rx_global_xyz[
                i,
                1,
            ].tolist()
        )

        print(
            "  RX array axis:",
            rx_array_axis_global[
                i
            ].tolist()
        )

        print(
            "  RX broadside:",
            rx_broadside_axis_global[
                i
            ].tolist()
        )

        print(
            "  TX0 global:",
            tx_global_xyz[
                i,
                0,
            ].tolist()
        )

        print(
            "  TX1 global:",
            tx_global_xyz[
                i,
                1,
            ].tolist()
        )

    print()
    print(
        "global geometry build: PASS"
    )

    if not absolute_spatial_geometry_valid:
        print(
            "absolute spatial processing: DISABLED"
        )


if __name__ == "__main__":
    main()
