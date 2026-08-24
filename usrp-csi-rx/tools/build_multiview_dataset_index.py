#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def scalar(z, key):
    v = np.asarray(
        z[key]
    )

    if v.ndim != 0:
        raise ValueError(
            f"{key} is not scalar"
        )

    return v.item()


def pose_xyz(block):
    xyz = block.get(
        "xyz_m"
    )

    if (
        not isinstance(
            xyz,
            list
        )
        or len(xyz) != 3
    ):
        raise ValueError(
            "Invalid xyz_m pose"
        )

    return np.asarray(
        xyz,
        dtype=np.float64,
    )


def pose_orientation(block):
    return np.asarray(
        [
            float(
                block.get(
                    "yaw_deg",
                    0.0,
                )
            ),
            float(
                block.get(
                    "pitch_deg",
                    0.0,
                )
            ),
            float(
                block.get(
                    "roll_deg",
                    0.0,
                )
            ),
        ],
        dtype=np.float64,
    )


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

    parser.add_argument(
        "--require-qc",
        action="store_true",
        default=True,
        help=(
            "Require dataset_qc.json "
            "with PASS status."
        ),
    )

    args = parser.parse_args()

    session = Path(
        args.session_dir
    ).resolve()

    manifest_path = (
        session
        / "manifest.json"
    )

    qc_path = (
        session
        / "dataset_qc.json"
    )

    if not manifest_path.exists():
        raise SystemExit(
            f"ERROR: missing {manifest_path}"
        )

    if args.require_qc:

        if not qc_path.exists():
            raise SystemExit(
                "ERROR: dataset_qc.json missing. "
                "Run validate_multiview_dataset.py first."
            )

        qc = load_json(
            qc_path
        )

        if qc.get(
            "status"
        ) != "PASS":
            raise SystemExit(
                "ERROR: dataset QC status is not PASS"
            )

    manifest = load_json(
        manifest_path
    )

    views = manifest.get(
        "views",
        []
    )

    if not views:
        raise SystemExit(
            "ERROR: session has no views"
        )

    view_ids = []

    tx_xyz = []
    rx_xyz = []

    tx_orientation = []
    rx_orientation = []

    cycle_count = []

    tensor_paths = []
    feature_paths = []

    rf_center_frequency_hz = []
    rf_sample_rate_hz = []
    active_subcarrier_count = []

    angular_present = []
    delay_present = []
    temporal_present = []
    differential_delay_present = []

    validity_physical_tx_mapping = []
    validity_rf_sample_time = []

    validity_absolute_aoa_global = []
    validity_absolute_aod_global = []
    validity_physical_tof_global = []
    validity_absolute_range_global = []
    validity_physical_doppler_global = []
    validity_physical_velocity_global = []

    feature_schema_version = []
    package_type = []

    quality_accepted = []

    for item in views:

        view_id = item[
            "view_id"
        ]

        view_dir = (
            session
            / item[
                "path"
            ]
        )

        vm_path = (
            view_dir
            / "view_manifest.json"
        )

        feature_path = (
            view_dir
            / "features"
            / "sensing_features.npz"
        )

        tensor_path = (
            view_dir
            / "csi"
            / "H_raw_tdm_physical_2x2.npz"
        )

        if not vm_path.exists():
            raise SystemExit(
                f"ERROR: {view_id}: "
                "view_manifest.json missing"
            )

        if not feature_path.exists():
            raise SystemExit(
                f"ERROR: {view_id}: "
                "sensing_features.npz missing"
            )

        if not tensor_path.exists():
            raise SystemExit(
                f"ERROR: {view_id}: "
                "TDM tensor missing"
            )

        vm = load_json(
            vm_path
        )

        if (
            vm.get(
                "state"
            )
            != "accepted"
        ):
            raise SystemExit(
                f"ERROR: {view_id}: "
                "view state not accepted"
            )

        if (
            vm.get(
                "quality",
                {}
            ).get(
                "accepted"
            )
            is not True
        ):
            raise SystemExit(
                f"ERROR: {view_id}: "
                "view QC not accepted"
            )

        tx_pose = vm.get(
            "tx_pose",
            {}
        )

        rx_pose = vm.get(
            "rx_pose",
            {}
        )

        if (
            tx_pose.get(
                "measured"
            )
            is not True
            or
            rx_pose.get(
                "measured"
            )
            is not True
        ):
            raise SystemExit(
                f"ERROR: {view_id}: "
                "pose not measured"
            )

        pose_registration = vm.get(
            "pose_registration",
            {}
        )

        if (
            pose_registration.get(
                "valid_for_spatial_fusion"
            )
            is not True
        ):
            raise SystemExit(
                f"ERROR: {view_id}: "
                "pose invalid for spatial fusion"
            )

        z = np.load(
            feature_path,
            allow_pickle=True,
        )

        view_ids.append(
            view_id
        )

        tx_xyz.append(
            pose_xyz(
                tx_pose
            )
        )

        rx_xyz.append(
            pose_xyz(
                rx_pose
            )
        )

        tx_orientation.append(
            pose_orientation(
                tx_pose
            )
        )

        rx_orientation.append(
            pose_orientation(
                rx_pose
            )
        )

        cycle_count.append(
            int(
                scalar(
                    z,
                    "cycle_count"
                )
            )
        )

        tensor_paths.append(
            str(
                tensor_path
            )
        )

        feature_paths.append(
            str(
                feature_path
            )
        )

        rf_center_frequency_hz.append(
            float(
                scalar(
                    z,
                    "rf_center_frequency_hz"
                )
            )
        )

        rf_sample_rate_hz.append(
            float(
                scalar(
                    z,
                    "timing_rf_sample_rate_hz"
                )
            )
        )

        active_subcarrier_count.append(
            int(
                scalar(
                    z,
                    "active_subcarrier_count"
                )
            )
        )

        angular_present.append(
            bool(
                scalar(
                    z,
                    "angular_present"
                )
            )
        )

        delay_present.append(
            bool(
                scalar(
                    z,
                    "delay_present"
                )
            )
        )

        temporal_present.append(
            bool(
                scalar(
                    z,
                    "temporal_present"
                )
            )
        )

        differential_delay_present.append(
            bool(
                scalar(
                    z,
                    "differential_delay_present"
                )
            )
        )

        validity_physical_tx_mapping.append(
            bool(
                scalar(
                    z,
                    "validity_physical_tx_mapping"
                )
            )
        )

        validity_rf_sample_time.append(
            bool(
                scalar(
                    z,
                    "validity_rf_sample_time"
                )
            )
        )

        validity_absolute_aoa_global.append(
            bool(
                scalar(
                    z,
                    "validity_absolute_aoa_global"
                )
            )
        )

        validity_absolute_aod_global.append(
            bool(
                scalar(
                    z,
                    "validity_absolute_aod_global"
                )
            )
        )

        validity_physical_tof_global.append(
            bool(
                scalar(
                    z,
                    "validity_physical_tof_global"
                )
            )
        )

        validity_absolute_range_global.append(
            bool(
                scalar(
                    z,
                    "validity_absolute_range_global"
                )
            )
        )

        validity_physical_doppler_global.append(
            bool(
                scalar(
                    z,
                    "validity_physical_doppler_global"
                )
            )
        )

        validity_physical_velocity_global.append(
            bool(
                scalar(
                    z,
                    "validity_physical_velocity_global"
                )
            )
        )

        feature_schema_version.append(
            str(
                scalar(
                    z,
                    "schema_version"
                )
            )
        )

        package_type.append(
            str(
                scalar(
                    z,
                    "package_type"
                )
            )
        )

        quality_accepted.append(
            True
        )

    tx_xyz = np.asarray(
        tx_xyz,
        dtype=np.float64,
    )

    rx_xyz = np.asarray(
        rx_xyz,
        dtype=np.float64,
    )

    tx_orientation = np.asarray(
        tx_orientation,
        dtype=np.float64,
    )

    rx_orientation = np.asarray(
        rx_orientation,
        dtype=np.float64,
    )

    cycle_count = np.asarray(
        cycle_count,
        dtype=np.int32,
    )

    rf_center_frequency_hz = np.asarray(
        rf_center_frequency_hz,
        dtype=np.float64,
    )

    rf_sample_rate_hz = np.asarray(
        rf_sample_rate_hz,
        dtype=np.float64,
    )

    active_subcarrier_count = np.asarray(
        active_subcarrier_count,
        dtype=np.int32,
    )

    output = (
        Path(
            args.output
        )
        if args.output
        else session
        / "dataset_index.npz"
    )

    np.savez_compressed(
        output,

        schema_version=np.asarray(
            "1.0"
        ),

        package_type=np.asarray(
            "csi_sense_multiview_dataset_index"
        ),

        session_id=np.asarray(
            manifest.get(
                "session_id",
                session.name,
            )
        ),

        session_name=np.asarray(
            manifest.get(
                "session_name",
                ""
            )
        ),

        coordinate_system=np.asarray(
            "room_local"
        ),

        view_count=np.asarray(
            len(
                view_ids
            ),
            dtype=np.int32,
        ),

        view_id=np.asarray(
            view_ids
        ),

        tx_xyz_m=tx_xyz,
        rx_xyz_m=rx_xyz,

        tx_orientation_ypr_deg=
            tx_orientation,

        rx_orientation_ypr_deg=
            rx_orientation,

        cycle_count=cycle_count,

        tensor_path=np.asarray(
            tensor_paths
        ),

        sensing_features_path=np.asarray(
            feature_paths
        ),

        rf_center_frequency_hz=
            rf_center_frequency_hz,

        rf_sample_rate_hz=
            rf_sample_rate_hz,

        active_subcarrier_count=
            active_subcarrier_count,

        angular_present=np.asarray(
            angular_present,
            dtype=bool,
        ),

        delay_present=np.asarray(
            delay_present,
            dtype=bool,
        ),

        temporal_present=np.asarray(
            temporal_present,
            dtype=bool,
        ),

        differential_delay_present=
            np.asarray(
                differential_delay_present,
                dtype=bool,
            ),

        quality_accepted=np.asarray(
            quality_accepted,
            dtype=bool,
        ),

        validity_physical_tx_mapping=
            np.asarray(
                validity_physical_tx_mapping,
                dtype=bool,
            ),

        validity_rf_sample_time=
            np.asarray(
                validity_rf_sample_time,
                dtype=bool,
            ),

        validity_absolute_aoa_global=
            np.asarray(
                validity_absolute_aoa_global,
                dtype=bool,
            ),

        validity_absolute_aod_global=
            np.asarray(
                validity_absolute_aod_global,
                dtype=bool,
            ),

        validity_physical_tof_global=
            np.asarray(
                validity_physical_tof_global,
                dtype=bool,
            ),

        validity_absolute_range_global=
            np.asarray(
                validity_absolute_range_global,
                dtype=bool,
            ),

        validity_physical_doppler_global=
            np.asarray(
                validity_physical_doppler_global,
                dtype=bool,
            ),

        validity_physical_velocity_global=
            np.asarray(
                validity_physical_velocity_global,
                dtype=bool,
            ),

        feature_schema_version=
            np.asarray(
                feature_schema_version
            ),

        feature_package_type=
            np.asarray(
                package_type
            ),

        scientific_policy=np.asarray(
            "RF_evidence_only_unless_absolute_physical_parameter_is_independently_validated"
        ),
    )

    print(
        "=" * 72
    )

    print(
        "MULTI-VIEW DATASET INDEX"
    )

    print(
        "=" * 72
    )

    print(
        "session:",
        session
    )

    print(
        "views:",
        len(
            view_ids
        )
    )

    print(
        "output:",
        output
    )

    print()

    for i, view_id in enumerate(
        view_ids
    ):

        baseline = float(
            np.linalg.norm(
                rx_xyz[i]
                - tx_xyz[i]
            )
        )

        print(
            view_id
        )

        print(
            "  TX:",
            tx_xyz[i].tolist()
        )

        print(
            "  RX:",
            rx_xyz[i].tolist()
        )

        print(
            "  TX-RX baseline [m]:",
            f"{baseline:.6f}"
        )

        print(
            "  cycles:",
            int(
                cycle_count[i]
            )
        )

        print(
            "  angular/delay/temporal:",
            bool(
                angular_present[i]
            ),
            bool(
                delay_present[i]
            ),
            bool(
                temporal_present[i]
            ),
        )

    print()
    print(
        "dataset index: PASS"
    )


if __name__ == "__main__":
    main()
