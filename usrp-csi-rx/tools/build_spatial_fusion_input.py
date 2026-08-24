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
    x = np.asarray(
        z[key]
    )

    if x.ndim != 0:
        raise ValueError(
            f"{key} is not scalar"
        )

    return x.item()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "session_dir",
        help=(
            "Validated multi-view session with "
            "dataset_index.npz and global_geometry.npz"
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

    geom_path = (
        session
        / "global_geometry.npz"
    )

    for p in (
        qc_path,
        index_path,
        geom_path,
    ):
        if not p.exists():
            raise SystemExit(
                f"ERROR: missing {p}"
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

    index = np.load(
        index_path,
        allow_pickle=True,
    )

    geom = np.load(
        geom_path,
        allow_pickle=True,
    )

    view_ids = np.asarray(
        index[
            "view_id"
        ]
    )

    if not np.array_equal(
        view_ids,
        geom[
            "view_id"
        ],
    ):
        raise SystemExit(
            "ERROR: dataset index and geometry "
            "view_id mismatch"
        )

    view_count = len(
        view_ids
    )

    if int(
        index[
            "view_count"
        ]
    ) != view_count:
        raise SystemExit(
            "ERROR: invalid index view_count"
        )

    #
    # Per-view geometry.
    #
    tx_global = np.asarray(
        geom[
            "tx_element_global_xyz_m"
        ],
        dtype=np.float64,
    )

    rx_global = np.asarray(
        geom[
            "rx_element_global_xyz_m"
        ],
        dtype=np.float64,
    )

    if tx_global.shape != (
        view_count,
        2,
        3,
    ):
        raise SystemExit(
            f"ERROR: TX geometry shape {tx_global.shape}"
        )

    if rx_global.shape != (
        view_count,
        2,
        3,
    ):
        raise SystemExit(
            f"ERROR: RX geometry shape {rx_global.shape}"
        )

    #
    # Current geometry is nominal physical-center
    # geometry only, not validated RF phase-center.
    #
    nominal_tx_geometry_valid = bool(
        geom[
            "validity_tx_nominal_physical_center_geometry"
        ]
    )

    nominal_rx_geometry_valid = bool(
        geom[
            "validity_rx_nominal_physical_center_geometry"
        ]
    )

    absolute_geometry_valid = bool(
        geom[
            "validity_absolute_spatial_geometry"
        ]
    )

    if not (
        nominal_tx_geometry_valid
        and nominal_rx_geometry_valid
    ):
        raise SystemExit(
            "ERROR: nominal TX/RX geometry unavailable"
        )

    #
    # Lists are used because cycle counts can differ
    # between views.
    #
    H_raw_list = []
    H_spatial_list = []

    angular_complex_list = []
    angular_confidence_list = []
    angular_valid_list = []

    delay_peak_bin_list = []
    delay_centroid_list = []
    delay_rms_list = []
    delay_confidence_list = []
    delay_valid_list = []

    temporal_dphi_list = []
    temporal_dphi_time_list = []

    rf_cycle_time_list = []

    cycle_view_index = []
    cycle_local_index = []

    cycle_counts = []

    per_view_summary = []

    for v, view_id in enumerate(
        view_ids
    ):
        feature_path = Path(
            str(
                index[
                    "sensing_features_path"
                ][v]
            )
        )

        if not feature_path.exists():
            raise SystemExit(
                f"ERROR: missing features for {view_id}"
            )

        z = np.load(
            feature_path,
            allow_pickle=True,
        )

        H = np.asarray(
            z[
                "measured_H_raw"
            ]
        )

        Hs = np.asarray(
            z[
                "measured_H_spatial_raw"
            ]
        )

        if (
            H.ndim != 4
            or H.shape[1:] != (
                52,
                2,
                2,
            )
        ):
            raise SystemExit(
                f"ERROR: {view_id} H shape {H.shape}"
            )

        if Hs.shape != H.shape:
            raise SystemExit(
                f"ERROR: {view_id} H_spatial shape mismatch"
            )

        N = H.shape[0]

        cycle_counts.append(
            N
        )

        H_raw_list.append(
            H.astype(
                np.complex64,
                copy=False,
            )
        )

        H_spatial_list.append(
            Hs.astype(
                np.complex64,
                copy=False,
            )
        )

        #
        # Angular RF evidence.
        #
        #
        # Reconstruct validated wideband angular RF
        # evidence directly from pre-beta spatial CSI:
        #
        #   D_t[n,k] =
        #       H_spatial[n,k,RX1,t]
        #       * conj(H_spatial[n,k,RX0,t])
        #
        # Shape: [N,52,2]
        #
        # Do not require a duplicated complex-angular
        # field in the unified package.
        #
        angular_complex = (
            Hs[:, :, 1, :]
            * np.conj(
                Hs[:, :, 0, :]
            )
        ).astype(
            np.complex64,
            copy=False,
        )

        angular_conf = np.asarray(
            z[
                "angular_confidence"
            ]
        )

        angular_valid = np.asarray(
            z[
                "angular_valid"
            ]
        )

        if angular_complex.shape != (
            N,
            52,
            2,
        ):
            raise SystemExit(
                f"ERROR: {view_id} angular complex "
                f"shape {angular_complex.shape}"
            )

        if angular_conf.shape != (
            N,
            2,
        ):
            raise SystemExit(
                f"ERROR: {view_id} angular confidence "
                f"shape {angular_conf.shape}"
            )

        if angular_valid.shape != (
            N,
            2,
        ):
            raise SystemExit(
                f"ERROR: {view_id} angular valid "
                f"shape {angular_valid.shape}"
            )

        angular_complex_list.append(
            angular_complex
        )

        angular_confidence_list.append(
            angular_conf
        )

        angular_valid_list.append(
            angular_valid
        )

        #
        # Delay-domain RF evidence.
        # These are proxies only.
        #
        delay_peak_bin_list.append(
            np.asarray(
                z[
                    "delay_peak_bin"
                ]
            )
        )

        delay_centroid_list.append(
            np.asarray(
                z[
                    "delay_peak_relative_centroid_bins"
                ]
            )
        )

        delay_rms_list.append(
            np.asarray(
                z[
                    "delay_rms_spread_bins"
                ]
            )
        )

        delay_confidence_list.append(
            np.asarray(
                z[
                    "delay_confidence"
                ]
            )
        )

        delay_valid_list.append(
            np.asarray(
                z[
                    "delay_valid"
                ]
            )
        )

        #
        # Temporal evidence has N-1 samples.
        #
        temporal_dphi_list.append(
            np.asarray(
                z[
                    "temporal_phase_increment_rad"
                ]
            )
        )

        temporal_dphi_time_list.append(
            np.asarray(
                z[
                    "temporal_phase_increment_time_s"
                ]
            )
        )

        rf_cycle_time = np.asarray(
            z[
                "timing_rf_cycle_time_s"
            ],
            dtype=np.float64,
        )

        rf_cycle_time_list.append(
            rf_cycle_time
        )

        cycle_view_index.append(
            np.full(
                N,
                v,
                dtype=np.int32,
            )
        )

        cycle_local_index.append(
            np.arange(
                N,
                dtype=np.int32,
            )
        )

        per_view_summary.append(
            {
                "view_id":
                    str(view_id),

                "cycles":
                    int(N),

                "angular_present":
                    bool(
                        scalar(
                            z,
                            "angular_present"
                        )
                    ),

                "delay_present":
                    bool(
                        scalar(
                            z,
                            "delay_present"
                        )
                    ),

                "temporal_present":
                    bool(
                        scalar(
                            z,
                            "temporal_present"
                        )
                    ),

                "absolute_aoa_valid":
                    bool(
                        scalar(
                            z,
                            "validity_absolute_aoa_global"
                        )
                    ),

                "physical_tof_valid":
                    bool(
                        scalar(
                            z,
                            "validity_physical_tof_global"
                        )
                    ),

                "physical_doppler_valid":
                    bool(
                        scalar(
                            z,
                            "validity_physical_doppler_global"
                        )
                    ),
            }
        )

    #
    # Flatten only arrays that have exactly one item
    # per RF cycle. Variable-length temporal N-1 evidence
    # stays stored as object arrays per view.
    #
    H_raw_flat = np.concatenate(
        H_raw_list,
        axis=0,
    )

    H_spatial_flat = np.concatenate(
        H_spatial_list,
        axis=0,
    )

    angular_complex_flat = np.concatenate(
        angular_complex_list,
        axis=0,
    )

    angular_confidence_flat = np.concatenate(
        angular_confidence_list,
        axis=0,
    )

    angular_valid_flat = np.concatenate(
        angular_valid_list,
        axis=0,
    )

    delay_peak_bin_flat = np.concatenate(
        delay_peak_bin_list,
        axis=0,
    )

    delay_centroid_flat = np.concatenate(
        delay_centroid_list,
        axis=0,
    )

    delay_rms_flat = np.concatenate(
        delay_rms_list,
        axis=0,
    )

    delay_confidence_flat = np.concatenate(
        delay_confidence_list,
        axis=0,
    )

    delay_valid_flat = np.concatenate(
        delay_valid_list,
        axis=0,
    )

    rf_cycle_time_flat = np.concatenate(
        rf_cycle_time_list,
        axis=0,
    )

    cycle_view_index = np.concatenate(
        cycle_view_index,
        axis=0,
    )

    cycle_local_index = np.concatenate(
        cycle_local_index,
        axis=0,
    )

    total_cycles = H_raw_flat.shape[0]

    #
    # Associate every RF cycle with its view geometry.
    #
    cycle_tx_global = tx_global[
        cycle_view_index
    ]

    cycle_rx_global = rx_global[
        cycle_view_index
    ]

    output = (
        Path(
            args.output
        )
        if args.output
        else session
        / "spatial_fusion_input.npz"
    )

    np.savez_compressed(
        output,

        schema_version=np.asarray(
            "1.0"
        ),

        package_type=np.asarray(
            "csi_sense_geometry_aware_spatial_fusion_input"
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

        view_count=np.asarray(
            view_count,
            dtype=np.int32,
        ),

        total_cycle_count=np.asarray(
            total_cycles,
            dtype=np.int32,
        ),

        view_id=view_ids,

        cycle_count_per_view=np.asarray(
            cycle_counts,
            dtype=np.int32,
        ),

        #
        # View-level nominal geometry.
        #
        view_tx_element_xyz_m=
            tx_global,

        view_rx_element_xyz_m=
            rx_global,

        view_rx_array_axis_global=
            np.asarray(
                geom[
                    "rx_array_axis_global"
                ]
            ),

        view_rx_broadside_axis_global=
            np.asarray(
                geom[
                    "rx_broadside_axis_global"
                ]
            ),

        #
        # Cycle -> view association.
        #
        cycle_view_index=
            cycle_view_index,

        cycle_local_index=
            cycle_local_index,

        cycle_tx_element_xyz_m=
            cycle_tx_global,

        cycle_rx_element_xyz_m=
            cycle_rx_global,

        cycle_rf_time_s=
            rf_cycle_time_flat,

        #
        # Raw measured CSI evidence.
        #
        measured_H_raw=
            H_raw_flat,

        measured_H_spatial_raw=
            H_spatial_flat,

        #
        # Angular evidence.
        #
        angular_rx_differential_complex=
            angular_complex_flat,

        angular_confidence=
            angular_confidence_flat,

        angular_valid=
            angular_valid_flat,

        #
        # Delay evidence — proxy only.
        #
        delay_peak_bin=
            delay_peak_bin_flat,

        delay_peak_relative_centroid_bins=
            delay_centroid_flat,

        delay_rms_spread_bins=
            delay_rms_flat,

        delay_confidence=
            delay_confidence_flat,

        delay_valid=
            delay_valid_flat,

        #
        # Temporal evidence remains per-view because
        # it is defined between consecutive cycles.
        #
        temporal_phase_increment_rad_per_view=
            np.asarray(
                temporal_dphi_list,
                dtype=object,
            ),

        temporal_phase_increment_time_s_per_view=
            np.asarray(
                temporal_dphi_time_list,
                dtype=object,
            ),

        #
        # Scientific validity.
        #
        validity_nominal_tx_geometry=
            np.asarray(
                nominal_tx_geometry_valid
            ),

        validity_nominal_rx_geometry=
            np.asarray(
                nominal_rx_geometry_valid
            ),

        validity_rf_phase_center_geometry=
            np.asarray(
                False
            ),

        validity_absolute_spatial_geometry=
            np.asarray(
                absolute_geometry_valid
            ),

        validity_absolute_aoa=
            np.asarray(
                False
            ),

        validity_absolute_aod=
            np.asarray(
                False
            ),

        validity_physical_tof=
            np.asarray(
                False
            ),

        validity_absolute_range=
            np.asarray(
                False
            ),

        validity_physical_doppler=
            np.asarray(
                False
            ),

        validity_physical_velocity=
            np.asarray(
                False
            ),

        geometric_path_hypothesis_allowed=
            np.asarray(
                True
            ),

        physical_range_backprojection_allowed=
            np.asarray(
                False
            ),

        delay_to_range_conversion_allowed=
            np.asarray(
                False
            ),

        semantic_policy=np.asarray(
            "associate_validated_RF_evidence_with_registered_nominal_geometry_without_promoting_delay_or_phase_to_unvalidated_absolute_physical_parameters"
        ),

        per_view_summary_json=np.asarray(
            json.dumps(
                per_view_summary,
                separators=(
                    ",",
                    ":",
                ),
            )
        ),
    )

    print(
        "=" * 78
    )

    print(
        "GEOMETRY-AWARE SPATIAL FUSION INPUT"
    )

    print(
        "=" * 78
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
        "cycles/view:",
        cycle_counts
    )

    print(
        "total cycles:",
        total_cycles
    )

    print(
        "H:",
        H_raw_flat.shape
    )

    print(
        "angular:",
        angular_complex_flat.shape
    )

    print(
        "delay:",
        delay_peak_bin_flat.shape
    )

    print(
        "cycle TX geometry:",
        cycle_tx_global.shape
    )

    print(
        "cycle RX geometry:",
        cycle_rx_global.shape
    )

    print()
    print(
        "Nominal TX geometry:",
        nominal_tx_geometry_valid
    )

    print(
        "Nominal RX geometry:",
        nominal_rx_geometry_valid
    )

    print(
        "RF phase-center geometry:",
        False
    )

    print(
        "Absolute spatial geometry:",
        absolute_geometry_valid
    )

    print(
        "Physical range backprojection:",
        False
    )

    print()
    print(
        "spatial fusion input: PASS"
    )


if __name__ == "__main__":
    main()
