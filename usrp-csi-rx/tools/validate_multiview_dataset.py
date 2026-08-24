#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED_FEATURE_KEYS = [
    "schema_version",
    "package_type",
    "cycle_count",
    "active_subcarrier_count",
    "rx_count",
    "tx_count",
    "rf_center_frequency_hz",
    "timing_rf_sample_rate_hz",
    "angular_present",
    "delay_present",
    "temporal_present",
    "differential_delay_present",
    "validity_physical_tx_mapping",
    "validity_rf_sample_time",
    "validity_absolute_aoa_global",
    "validity_absolute_aod_global",
    "validity_physical_tof_global",
    "validity_absolute_range_global",
    "validity_physical_doppler_global",
    "validity_physical_velocity_global",
    "measured_H_raw",
]


def scalar(z, key):
    v = z[key]

    if np.asarray(v).ndim == 0:
        return np.asarray(v).item()

    raise ValueError(
        f"{key} is not scalar"
    )


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def pose_vector(block):
    xyz = block.get(
        "xyz_m"
    )

    if (
        not isinstance(xyz, list)
        or len(xyz) != 3
    ):
        return None

    try:
        return np.asarray(
            xyz,
            dtype=np.float64,
        )
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "session_dir",
        help="Multi-view csi-sense session directory.",
    )

    parser.add_argument(
        "--min-views",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--min-pose-separation-m",
        type=float,
        default=0.01,
        help=(
            "Warn when both TX and RX poses are "
            "near-duplicates."
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

    manifest_path = (
        session
        / "manifest.json"
    )

    if not manifest_path.exists():
        raise SystemExit(
            f"ERROR: missing {manifest_path}"
        )

    manifest = load_json(
        manifest_path
    )

    report = {
        "schema_version": 1,
        "session_dir": str(session),
        "session_id": manifest.get(
            "session_id"
        ),
        "status": "PASS",
        "errors": [],
        "warnings": [],
        "views": [],
        "consistency": {},
    }

    views = manifest.get(
        "views",
        []
    )

    if len(views) < args.min_views:
        report["errors"].append(
            "insufficient_view_count"
        )

    reference = None

    tx_poses = []
    rx_poses = []

    for item in views:
        view_id = item.get(
            "view_id"
        )

        view_dir = (
            session
            / item["path"]
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

        entry = {
            "view_id": view_id,
            "path": str(view_dir),
            "errors": [],
            "warnings": [],
        }

        if not vm_path.exists():
            entry["errors"].append(
                "missing_view_manifest"
            )
            report["views"].append(
                entry
            )
            continue

        vm = load_json(
            vm_path
        )

        if (
            vm.get("state")
            != "accepted"
        ):
            entry["errors"].append(
                "view_state_not_accepted"
            )

        quality = vm.get(
            "quality",
            {}
        )

        if (
            quality.get("accepted")
            is not True
        ):
            entry["errors"].append(
                "view_quality_not_accepted"
            )

        tx_pose = pose_vector(
            vm.get(
                "tx_pose",
                {}
            )
        )

        rx_pose = pose_vector(
            vm.get(
                "rx_pose",
                {}
            )
        )

        if tx_pose is None:
            entry["errors"].append(
                "invalid_tx_pose"
            )
        else:
            tx_poses.append(
                (
                    view_id,
                    tx_pose,
                )
            )

        if rx_pose is None:
            entry["errors"].append(
                "invalid_rx_pose"
            )
        else:
            rx_poses.append(
                (
                    view_id,
                    rx_pose,
                )
            )

        pose_reg = vm.get(
            "pose_registration",
            {}
        )

        if (
            pose_reg.get(
                "valid_for_spatial_fusion"
            )
            is not True
        ):
            entry["errors"].append(
                "pose_not_valid_for_spatial_fusion"
            )

        capture = vm.get(
            "capture",
            {}
        )

        if (
            capture.get("started_at")
            is None
            or capture.get("finished_at")
            is None
            or capture.get("duration_s")
            is None
        ):
            entry["warnings"].append(
                "capture_lifecycle_metadata_incomplete"
            )

        if not feature_path.exists():
            entry["errors"].append(
                "missing_sensing_features"
            )

            report["views"].append(
                entry
            )

            continue

        z = np.load(
            feature_path,
            allow_pickle=True,
        )

        missing = [
            key
            for key in REQUIRED_FEATURE_KEYS
            if key not in z
        ]

        if missing:
            entry["errors"].append(
                "missing_feature_keys:"
                + ",".join(missing)
            )

            report["views"].append(
                entry
            )

            continue

        H = z[
            "measured_H_raw"
        ]

        cycle_count = int(
            scalar(
                z,
                "cycle_count"
            )
        )

        expected_shape = (
            cycle_count,
            52,
            2,
            2,
        )

        if H.shape != expected_shape:
            entry["errors"].append(
                f"H_shape_mismatch:{H.shape}"
            )

        current = {
            "schema_version":
                str(
                    scalar(
                        z,
                        "schema_version"
                    )
                ),

            "package_type":
                str(
                    scalar(
                        z,
                        "package_type"
                    )
                ),

            "subcarriers":
                int(
                    scalar(
                        z,
                        "active_subcarrier_count"
                    )
                ),

            "rx_count":
                int(
                    scalar(
                        z,
                        "rx_count"
                    )
                ),

            "tx_count":
                int(
                    scalar(
                        z,
                        "tx_count"
                    )
                ),

            "rf_center_frequency_hz":
                float(
                    scalar(
                        z,
                        "rf_center_frequency_hz"
                    )
                ),

            "rf_sample_rate_hz":
                float(
                    scalar(
                        z,
                        "timing_rf_sample_rate_hz"
                    )
                ),
        }

        entry[
            "cycle_count"
        ] = cycle_count

        entry[
            "feature_schema"
        ] = current

        entry[
            "angular_present"
        ] = bool(
            scalar(
                z,
                "angular_present"
            )
        )

        entry[
            "delay_present"
        ] = bool(
            scalar(
                z,
                "delay_present"
            )
        )

        entry[
            "temporal_present"
        ] = bool(
            scalar(
                z,
                "temporal_present"
            )
        )

        if not entry[
            "angular_present"
        ]:
            entry["errors"].append(
                "angular_evidence_missing"
            )

        if not entry[
            "delay_present"
        ]:
            entry["errors"].append(
                "delay_evidence_missing"
            )

        if not entry[
            "temporal_present"
        ]:
            entry["errors"].append(
                "temporal_evidence_missing"
            )

        if (
            bool(
                scalar(
                    z,
                    "validity_physical_tx_mapping"
                )
            )
            is not True
        ):
            entry["errors"].append(
                "physical_tx_mapping_invalid"
            )

        if (
            bool(
                scalar(
                    z,
                    "validity_rf_sample_time"
                )
            )
            is not True
        ):
            entry["errors"].append(
                "rf_sample_time_invalid"
            )

        physical_flags = [
            "validity_absolute_aoa_global",
            "validity_absolute_aod_global",
            "validity_physical_tof_global",
            "validity_absolute_range_global",
            "validity_physical_doppler_global",
            "validity_physical_velocity_global",
        ]

        for key in physical_flags:
            if bool(
                scalar(
                    z,
                    key
                )
            ):
                entry["warnings"].append(
                    f"unexpected_physical_validity:{key}"
                )

        if reference is None:
            reference = current
        else:
            for key in current:
                if current[key] != reference[key]:
                    entry["errors"].append(
                        f"inconsistent_{key}"
                    )

        report["views"].append(
            entry
        )

    duplicate_pairs = []

    for i in range(
        len(views)
    ):
        for j in range(
            i + 1,
            len(views),
        ):
            vi = views[i]["view_id"]
            vj = views[j]["view_id"]

            tx_i = next(
                (
                    p
                    for name, p
                    in tx_poses
                    if name == vi
                ),
                None,
            )

            tx_j = next(
                (
                    p
                    for name, p
                    in tx_poses
                    if name == vj
                ),
                None,
            )

            rx_i = next(
                (
                    p
                    for name, p
                    in rx_poses
                    if name == vi
                ),
                None,
            )

            rx_j = next(
                (
                    p
                    for name, p
                    in rx_poses
                    if name == vj
                ),
                None,
            )

            if (
                tx_i is None
                or tx_j is None
                or rx_i is None
                or rx_j is None
            ):
                continue

            tx_sep = float(
                np.linalg.norm(
                    tx_i - tx_j
                )
            )

            rx_sep = float(
                np.linalg.norm(
                    rx_i - rx_j
                )
            )

            if (
                tx_sep
                < args.min_pose_separation_m
                and
                rx_sep
                < args.min_pose_separation_m
            ):
                duplicate_pairs.append(
                    {
                        "views": [
                            vi,
                            vj,
                        ],
                        "tx_separation_m":
                            tx_sep,
                        "rx_separation_m":
                            rx_sep,
                    }
                )

    if duplicate_pairs:
        report[
            "warnings"
        ].append(
            "near_duplicate_bistatic_views"
        )

    report[
        "consistency"
    ][
        "reference_feature_schema"
    ] = reference

    report[
        "consistency"
    ][
        "near_duplicate_pairs"
    ] = duplicate_pairs

    all_errors = list(
        report["errors"]
    )

    for entry in report[
        "views"
    ]:
        all_errors.extend(
            entry[
                "errors"
            ]
        )

    if all_errors:
        report["status"] = "FAIL"

    output = (
        Path(args.output)
        if args.output
        else session
        / "dataset_qc.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "=" * 72
    )
    print(
        "MULTI-VIEW DATASET QC"
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
        len(views)
    )

    for entry in report[
        "views"
    ]:
        print()
        print(
            entry["view_id"]
        )

        print(
            "  cycles:",
            entry.get(
                "cycle_count",
                "-"
            )
        )

        print(
            "  errors:",
            len(
                entry["errors"]
            )
        )

        print(
            "  warnings:",
            len(
                entry["warnings"]
            )
        )

        for x in entry[
            "errors"
        ]:
            print(
                "    ERROR:",
                x
            )

        for x in entry[
            "warnings"
        ]:
            print(
                "    WARN :",
                x
            )

    print()
    print(
        "dataset status:",
        report["status"]
    )

    print(
        "report:",
        output
    )

    raise SystemExit(
        0
        if report["status"] == "PASS"
        else 3
    )


if __name__ == "__main__":
    main()
