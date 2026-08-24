#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "1.0"


def load_npz(path):
    if path is None or not path.exists():
        return None

    return np.load(
        path,
        allow_pickle=True,
    )


def copy_if_present(
    src,
    dst,
    mapping,
):
    if src is None:
        return

    for source_key, output_key in mapping.items():
        if source_key in src:
            dst[output_key] = src[source_key]


def scalar_string(value):
    return np.array(
        str(value)
    )


def main():

    ap = argparse.ArgumentParser(
        description=(
            "Build one scientifically conservative unified "
            "sensing feature package for a USRP CSI view."
        )
    )

    ap.add_argument(
        "view_dir",
        type=Path,
        help=(
            "Path to view directory, e.g. "
            "sessions/.../views/view_000"
        ),
    )

    ap.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = ap.parse_args()

    view_dir = args.view_dir.resolve()

    tensor_path = (
        view_dir
        / "csi"
        / "H_raw_tdm_physical_2x2.npz"
    )

    feature_dir = (
        view_dir
        / "features"
    )

    angular_path = (
        feature_dir
        / "angular_features.npz"
    )

    delay_path = (
        feature_dir
        / "delay_features.npz"
    )

    temporal_path = (
        feature_dir
        / "temporal_motion_features.npz"
    )

    #
    # Differential-delay can currently exist under
    # different historical names. Prefer v2.
    #
    differential_candidates = [
        feature_dir
        / "differential_delay_profile_v2.npz",

        feature_dir
        / "differential_delay_profile.npz",

        view_dir
        / "csi"
        / "differential_delay_profile_v2.npz",

        view_dir
        / "csi"
        / "differential_delay_profile.npz",
    ]

    differential_path = None

    for candidate in differential_candidates:
        if candidate.exists():
            differential_path = candidate
            break

    if not tensor_path.exists():
        raise RuntimeError(
            f"Required tensor not found: {tensor_path}"
        )

    tensor = load_npz(
        tensor_path
    )

    angular = load_npz(
        angular_path
    )

    delay = load_npz(
        delay_path
    )

    temporal = load_npz(
        temporal_path
    )

    differential = load_npz(
        differential_path
    )

    H = np.asarray(
        tensor["H_raw"]
    )

    if (
        H.ndim != 4
        or H.shape[1:] != (52, 2, 2)
    ):
        raise RuntimeError(
            f"Unexpected H_raw shape: {H.shape}"
        )

    N = H.shape[0]

    out = {}

    #
    # ====================================================
    # SCHEMA / IDENTITY
    # ====================================================
    #
    out[
        "schema_version"
    ] = scalar_string(
        SCHEMA_VERSION
    )

    out[
        "package_type"
    ] = scalar_string(
        "unified_usrp_mimo_csi_sensing_evidence"
    )

    out[
        "view_dir"
    ] = scalar_string(
        view_dir
    )

    out[
        "cycle_count"
    ] = np.int32(
        N
    )

    out[
        "active_subcarrier_count"
    ] = np.int32(
        52
    )

    out[
        "rx_count"
    ] = np.int32(
        2
    )

    out[
        "tx_count"
    ] = np.int32(
        2
    )

    #
    # ====================================================
    # MEASURED / PRIMARY CSI
    # ====================================================
    #
    out[
        "measured_H_raw"
    ] = tensor[
        "H_raw"
    ]

    if "H_spatial_raw" in tensor:
        out[
            "measured_H_spatial_raw"
        ] = tensor[
            "H_spatial_raw"
        ]

    #
    # ====================================================
    # TIMING / PACKET IDENTITY
    # ====================================================
    #
    copy_if_present(
        tensor,
        out,
        {
            "seq":
                "timing_seq",

            "seq_unwrapped":
                "timing_seq_unwrapped",

            "snr":
                "measured_snr",

            "cfo_hz":
                "measured_cfo_hz",

            "rf_sample_index":
                "timing_rf_sample_index",

            "rf_sample_rate_hz":
                "timing_rf_sample_rate_hz",

            "rf_reference_sample_index":
                "timing_rf_reference_sample_index",

            "rf_time_relative_s":
                "timing_rf_time_relative_s",

            "rf_cycle_time_s":
                "timing_rf_cycle_time_s",

            "rf_rx_sample_skew":
                "timing_rf_rx_sample_skew",

            "cycle_time_nominal_s":
                "timing_cycle_time_nominal_s",

            "tx_packet_interval_s":
                "timing_tx_packet_interval_s",

            "fc_hz":
                "rf_center_frequency_hz",
        },
    )

    #
    # ====================================================
    # ANGULAR EVIDENCE
    # ====================================================
    #
    angular_present = (
        angular is not None
    )

    out[
        "angular_present"
    ] = np.bool_(
        angular_present
    )

    if angular_present:

        copy_if_present(
            angular,
            out,
            {
                "rx_differential_complex":
                    "angular_rx_differential_complex",

                "rx_differential_magnitude":
                    "angular_rx_differential_magnitude",

                "rx_phase_raw_rad":
                    "angular_phase_raw_rad",

                "rx_phase_relative_rad":
                    "angular_phase_relative_rad",

                "relative_phase_mean_rad":
                    "angular_relative_phase_mean_rad",

                "angular_weight":
                    "angular_weight",

                "angular_confidence":
                    "angular_confidence",

                "subcarrier_coverage":
                    "angular_subcarrier_coverage",

                "overall_angular_confidence":
                    "angular_overall_confidence",

                "used_subcarrier_count":
                    "angular_used_subcarrier_count",

                "angular_valid":
                    "angular_valid",

                "relative_angular_evidence_valid":
                    "angular_relative_evidence_valid",

                "valid_subcarrier_mask":
                    "angular_valid_subcarrier_mask",

                "spatial_argument":
                    "angular_spatial_argument",

                "spatial_argument_valid":
                    "angular_spatial_argument_valid",

                "unambiguous_aoa_mask":
                    "angular_unambiguous_aoa_mask",

                "reference_coherence":
                    "angular_reference_coherence",

                "reference_phase_rad":
                    "angular_reference_phase_rad",
            },
        )

        #
        # Candidate diagnostic angle fields are preserved
        # only with explicit validity. They are NOT promoted
        # to trusted physical AoA.
        #
        if "aoa_rad" in angular:
            out[
                "angular_aoa_rad_candidate"
            ] = angular[
                "aoa_rad"
            ]

        if "aoa_deg" in angular:
            out[
                "angular_aoa_deg_candidate"
            ] = angular[
                "aoa_deg"
            ]

        copy_if_present(
            angular,
            out,
            {
                "absolute_aoa_valid":
                    "validity_absolute_aoa",

                "absolute_geometry_valid":
                    "validity_absolute_spatial_geometry",

                "spatial_aliasing_present":
                    "angular_spatial_aliasing_present",

                "max_unambiguous_aoa_deg":
                    "angular_max_unambiguous_aoa_deg",

                "antenna_spacing_m":
                    "angular_antenna_spacing_m",

                "wavelength_m":
                    "angular_wavelength_m",
            },
        )

        out[
            "provenance_angular_file"
        ] = scalar_string(
            angular_path
        )

        for key in (
            "estimator",
            "calibration_type",
            "rx_differential_definition",
            "relative_phase_definition",
            "angle_convention",
            "note",
        ):
            if key in angular:
                out[
                    f"provenance_angular_{key}"
                ] = angular[key]

    #
    # ====================================================
    # PER-LINK DELAY EVIDENCE
    # ====================================================
    #
    delay_present = (
        delay is not None
    )

    out[
        "delay_present"
    ] = np.bool_(
        delay_present
    )

    if delay_present:

        copy_if_present(
            delay,
            out,
            {
                "cir_proxy_complex":
                    "delay_cir_proxy_complex",

                "pdp_proxy":
                    "delay_pdp_proxy",

                "pdp_normalized":
                    "delay_pdp_normalized",

                "delay_peak_bin":
                    "delay_peak_bin",

                "peak_relative_centroid_bins":
                    "delay_peak_relative_centroid_bins",

                "rms_delay_spread_bins":
                    "delay_rms_spread_bins",

                "delay_peak_prominence":
                    "delay_peak_prominence",

                "delay_confidence":
                    "delay_confidence",

                "delay_valid":
                    "delay_valid",

                "delay_axis_s":
                    "delay_axis_s",

                "delay_axis_ns":
                    "delay_axis_ns",

                "delay_peak_s_proxy":
                    "delay_peak_s_proxy",

                "peak_relative_centroid_s_proxy":
                    "delay_peak_relative_centroid_s_proxy",

                "rms_delay_spread_s_proxy":
                    "delay_rms_spread_s_proxy",

                "native_delay_bin_spacing_s":
                    "delay_native_bin_spacing_s",

                "interpolated_delay_spacing_s":
                    "delay_interpolated_spacing_s",

                "active_bins_shifted":
                    "delay_active_bins_shifted",

                "physical_tof_s":
                    "delay_physical_tof_s",

                "physical_tof_valid":
                    "validity_physical_tof",

                "absolute_range_m":
                    "delay_absolute_range_m",

                "absolute_range_valid":
                    "validity_absolute_range",
            },
        )

        out[
            "provenance_delay_file"
        ] = scalar_string(
            delay_path
        )

        for key in (
            "delay_method",
            "source_csi_field",
            "representation",
            "warning",
        ):
            if key in delay:
                out[
                    f"provenance_delay_{key}"
                ] = delay[key]

    #
    # ====================================================
    # DIFFERENTIAL DELAY EVIDENCE — OPTIONAL
    # ====================================================
    #
    differential_present = (
        differential is not None
    )

    out[
        "differential_delay_present"
    ] = np.bool_(
        differential_present
    )

    if differential_present:

        copy_if_present(
            differential,
            out,
            {
                "power_complex_norm":
                    "diff_delay_power_complex_norm",

                "power_phase_norm":
                    "diff_delay_power_phase_norm",

                "peak_complex":
                    "diff_delay_peak_complex",

                "peak_phase":
                    "diff_delay_peak_phase",

                "peak_relative_centroid_complex_bins":
                    "diff_delay_centroid_complex_bins",

                "peak_relative_centroid_phase_bins":
                    "diff_delay_centroid_phase_bins",

                "circular_rms_complex_bins":
                    "diff_delay_circular_rms_complex_bins",

                "circular_rms_phase_bins":
                    "diff_delay_circular_rms_phase_bins",

                "peak_prominence_complex":
                    "diff_delay_peak_prominence_complex",

                "peak_prominence_phase":
                    "diff_delay_peak_prominence_phase",

                "delay_evidence_valid":
                    "diff_delay_valid",

                "primary_tx":
                    "diff_delay_primary_tx",

                "physical_tof_valid":
                    "diff_delay_physical_tof_valid",

                "absolute_range_valid":
                    "diff_delay_absolute_range_valid",
            },
        )

        out[
            "provenance_differential_delay_file"
        ] = scalar_string(
            differential_path
        )

        for key in (
            "preferred_representation",
            "validation_status",
            "representation",
        ):
            if key in differential:
                out[
                    f"provenance_diff_delay_{key}"
                ] = differential[key]

    #
    # ====================================================
    # TEMPORAL MOTION EVIDENCE
    # ====================================================
    #
    temporal_present = (
        temporal is not None
    )

    out[
        "temporal_present"
    ] = np.bool_(
        temporal_present
    )

    if temporal_present:

        copy_if_present(
            temporal,
            out,
            {
                "differential_complex":
                    "temporal_differential_complex",

                "differential_unit":
                    "temporal_differential_unit",

                "temporal_phase_increment_rad":
                    "temporal_phase_increment_rad",

                "temporal_phase_increment_abs_rad":
                    "temporal_phase_increment_abs_rad",

                "phase_increment_time_s":
                    "temporal_phase_increment_time_s",

                "frame_median_abs_phase_increment":
                    "temporal_frame_median_abs_phase_increment",

                "frame_p90_abs_phase_increment":
                    "temporal_frame_p90_abs_phase_increment",

                "frame_p95_abs_phase_increment":
                    "temporal_frame_p95_abs_phase_increment",

                "static_threshold_p95_rad":
                    "temporal_static_threshold_p95_rad",

                "static_threshold_p99_rad":
                    "temporal_static_threshold_p99_rad",

                "subcarrier_anomaly_mask_p95":
                    "temporal_anomaly_mask_p95",

                "subcarrier_anomaly_mask_p99":
                    "temporal_anomaly_mask_p99",

                "subcarrier_anomaly_fraction_p95":
                    "temporal_anomaly_fraction_p95",

                "subcarrier_anomaly_fraction_p99":
                    "temporal_anomaly_fraction_p99",

                "motion_score":
                    "temporal_motion_score",

                "motion_valid":
                    "temporal_motion_valid",

                "baseline_threshold_valid":
                    "temporal_baseline_threshold_valid",

                "fixed_motion_threshold_valid":
                    "temporal_fixed_motion_threshold_valid",

                "capture_motion_classification_valid":
                    "temporal_capture_classification_valid",

                "physical_doppler_valid":
                    "validity_physical_doppler",

                "physical_velocity_valid":
                    "validity_physical_velocity",

                "primary_tx":
                    "temporal_primary_tx",
            },
        )

        out[
            "provenance_temporal_file"
        ] = scalar_string(
            temporal_path
        )

        for key in (
            "temporal_method",
            "baseline_source",
            "method_validation_status",
            "warning",
        ):
            if key in temporal:
                out[
                    f"provenance_temporal_{key}"
                ] = temporal[key]

    #
    # ====================================================
    # GLOBAL VALIDITY POLICY
    # ====================================================
    #
    # Conservative defaults. Evidence != absolute
    # physical parameter.
    #
    out[
        "validity_absolute_aoa_global"
    ] = np.bool_(
        False
    )

    out[
        "validity_absolute_aod_global"
    ] = np.bool_(
        False
    )

    out[
        "validity_physical_tof_global"
    ] = np.bool_(
        False
    )

    out[
        "validity_absolute_range_global"
    ] = np.bool_(
        False
    )

    out[
        "validity_physical_doppler_global"
    ] = np.bool_(
        False
    )

    out[
        "validity_physical_velocity_global"
    ] = np.bool_(
        False
    )

    copy_if_present(
        tensor,
        out,
        {
            "rf_sample_time_valid":
                "validity_rf_sample_time",

            "sample_domain_time_valid":
                "validity_sample_domain_time",

            "physical_tx_mapping_valid":
                "validity_physical_tx_mapping",
        },
    )

    #
    # ====================================================
    # SEMANTIC POLICY
    # ====================================================
    #
    out[
        "semantic_policy"
    ] = scalar_string(
        "measured_and_derived_RF_evidence_are_kept_"
        "separate_from_unvalidated_absolute_physical_parameters"
    )

    out[
        "trusted_primary_angular_representation"
    ] = scalar_string(
        "TX0_relative_or_prebeta_angular_evidence"
    )

    out[
        "trusted_primary_delay_representation"
    ] = scalar_string(
        "TX0_phase_only_differential_delay_when_available"
    )

    out[
        "trusted_primary_temporal_representation"
    ] = scalar_string(
        "TX0_subcarrier_resolved_inter_rx_phase_increment"
    )

    out[
        "physical_parameter_policy"
    ] = scalar_string(
        "AoA_AoD_ToF_Doppler_velocity_require_"
        "separate_estimation_and_validation"
    )

    #
    # ====================================================
    # WRITE
    # ====================================================
    #
    if args.output is None:
        output = (
            feature_dir
            / "sensing_features.npz"
        )
    else:
        output = args.output

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output,
        **out,
    )

    print("=" * 76)
    print("UNIFIED SENSING FEATURES")
    print("=" * 76)

    print(
        "view:",
        view_dir
    )

    print(
        "output:",
        output
    )

    print(
        "cycles:",
        N
    )

    print(
        "angular present:",
        angular_present
    )

    print(
        "delay present:",
        delay_present
    )

    print(
        "differential delay present:",
        differential_present
    )

    print(
        "temporal present:",
        temporal_present
    )

    print(
        "absolute AoA valid:",
        False
    )

    print(
        "absolute AoD valid:",
        False
    )

    print(
        "physical ToF valid:",
        False
    )

    print(
        "physical Doppler valid:",
        False
    )

    print(
        "physical velocity valid:",
        False
    )

    print(
        "keys:",
        len(out)
    )


if __name__ == "__main__":
    main()
