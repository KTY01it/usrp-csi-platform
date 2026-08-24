#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def run_step(
    label,
    cmd,
):
    print()
    print(
        f"[FEATURE] {label}"
    )

    print(
        "Command:",
        " ".join(
            str(x)
            for x in cmd
        )
    )

    result = subprocess.run(
        [
            str(x)
            for x in cmd
        ],
        cwd=ROOT,
    )

    if result.returncode == 0:
        print(
            f"{label}: PASS"
        )
        return True

    print(
        f"{label}: FAIL "
        f"(exit={result.returncode})"
    )

    return False


def main():

    ap = argparse.ArgumentParser(
        description=(
            "Run validated automatic feature extraction "
            "for one accepted CSI view."
        )
    )

    ap.add_argument(
        "view_dir",
        type=Path,
    )

    args = ap.parse_args()

    view_dir = (
        args.view_dir.resolve()
    )

    csi_dir = (
        view_dir
        / "csi"
    )

    feature_dir = (
        view_dir
        / "features"
    )

    tensor_path = (
        csi_dir
        / "H_raw_tdm_physical_2x2.npz"
    )

    if not tensor_path.exists():

        print(
            "ERROR: tensor not found:"
        )

        print(
            tensor_path
        )

        return 2

    feature_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    #
    # Inspect tensor capabilities.
    #
    z = np.load(
        tensor_path,
        allow_pickle=True,
    )

    if "H_raw" not in z:
        print(
            "ERROR: H_raw missing"
        )
        return 2

    H = z["H_raw"]

    if (
        H.ndim != 4
        or H.shape[1:] != (52,2,2)
    ):
        print(
            "ERROR: invalid H_raw shape:",
            H.shape,
        )
        return 2

    spatial_available = (
        "H_spatial_raw" in z
    )

    rf_timing_available = (
        "rf_cycle_time_s" in z
        and bool(
            z.get(
                "rf_sample_time_valid",
                False
            )
        )
    )

    print(
        "=" * 72
    )

    print(
        "AUTOMATIC VIEW FEATURE PIPELINE"
    )

    print(
        "=" * 72
    )

    print(
        "view:",
        view_dir
    )

    print(
        "tensor:",
        tensor_path
    )

    print(
        "cycles:",
        H.shape[0]
    )

    print(
        "pre-beta spatial CSI:",
        spatial_available
    )

    print(
        "RF sample timing:",
        rf_timing_available
    )

    results = {}

    #
    # ==================================================
    # ANGULAR
    # ==================================================
    #
    if spatial_available:

        angular_output = (
            feature_dir
            / "angular_features.npz"
        )

        results["angular"] = run_step(
            "angular evidence",
            [
                sys.executable,
                ROOT
                / "tools"
                / "build_angular_features.py",

                tensor_path,

                "--output",
                angular_output,
            ],
        )

    else:

        results["angular"] = None

        print()
        print(
            "[FEATURE] angular evidence"
        )

        print(
            "angular evidence: SKIP"
        )

        print(
            "Reason: H_spatial_raw not available"
        )

    #
    # ==================================================
    # PER-LINK DELAY
    # ==================================================
    #
    delay_output = (
        feature_dir
        / "delay_features.npz"
    )

    results["delay"] = run_step(
        "per-link delay evidence",
        [
            sys.executable,
            ROOT
            / "tools"
            / "mimo_cir_pdp.py",

            tensor_path,

            "--out",
            delay_output,

            "--key",
            "H_raw",
        ],
    )

    #
    # ==================================================
    # TEMPORAL
    # ==================================================
    #
    # This automatic stage intentionally builds
    # threshold-independent temporal evidence only.
    #
    # Without a per-subcarrier static calibration:
    #
    #   baseline_threshold_valid = False
    #   motion_score             = NaN
    #   motion_valid             = False
    #
    # This is scientifically preferable to inventing a
    # universal motion threshold.
    #
    if rf_timing_available:

        temporal_output = (
            feature_dir
            / "temporal_motion_features.npz"
        )

        results["temporal"] = run_step(
            "temporal motion evidence",
            [
                sys.executable,
                ROOT
                / "tools"
                / "build_temporal_motion_features.py",

                tensor_path,

                "--output",
                temporal_output,

                "--primary-tx",
                "0",
            ],
        )

    else:

        results["temporal"] = None

        print()
        print(
            "[FEATURE] temporal motion evidence"
        )

        print(
            "temporal motion evidence: SKIP"
        )

        print(
            "Reason: valid RF sample-domain timing "
            "not available"
        )

    #
    # ==================================================
    # DIFFERENTIAL DELAY
    # ==================================================
    #
    # Deliberately NOT auto-generated.
    #
    # It requires calibrated target/background context,
    # so a single view is insufficient.
    #
    results[
        "differential_delay"
    ] = None

    print()
    print(
        "[FEATURE] differential delay evidence"
    )

    print(
        "differential delay evidence: SKIP"
    )

    print(
        "Reason: requires calibrated "
        "target/background context"
    )

    #
    # ==================================================
    # UNIFIED EXPORT
    # ==================================================
    #
    unified_output = (
        feature_dir
        / "sensing_features.npz"
    )

    results["unified"] = run_step(
        "unified sensing export",
        [
            sys.executable,
            ROOT
            / "tools"
            / "build_unified_sensing_features.py",

            view_dir,

            "--output",
            unified_output,
        ],
    )

    print()
    print(
        "=" * 72
    )

    print(
        "FEATURE PIPELINE SUMMARY"
    )

    print(
        "=" * 72
    )

    for name in (
        "angular",
        "delay",
        "temporal",
        "differential_delay",
        "unified",
    ):

        state = results[
            name
        ]

        if state is True:
            text = "PASS"

        elif state is False:
            text = "FAIL"

        else:
            text = "SKIP"

        print(
            f"{name:<24}",
            text,
        )

    print()

    print(
        "Automatic feature extraction does not "
        "promote RF evidence to unvalidated "
        "absolute AoA/AoD/ToF/Doppler/velocity."
    )

    #
    # Unified package is the required final artifact.
    #
    if results["unified"] is not True:
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(
        main()
    )
