#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import yaml


EPS = 1e-12


def main():

    ap = argparse.ArgumentParser(
        description=(
            "Build TX0 subcarrier-resolved differential "
            "temporal motion evidence from 2x2 TDM MIMO CSI."
        )
    )

    ap.add_argument(
        "tensor",
        type=Path,
    )

    ap.add_argument(
        "--baseline",
        type=Path,
        default=Path(
            "configs/calibration/"
            "doppler_static_baseline.yaml"
        ),
    )

    ap.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    ap.add_argument(
        "--primary-tx",
        type=int,
        choices=(0, 1),
        default=0,
    )

    ap.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.95,
    )

    ap.add_argument(
        "--baseline-start-cycle",
        type=int,
        default=None,
    )

    ap.add_argument(
        "--baseline-end-cycle",
        type=int,
        default=None,
    )

    args = ap.parse_args()

    z = np.load(
        args.tensor,
        allow_pickle=True,
    )

    H = np.asarray(
        z["H_raw"],
        dtype=np.complex64,
    )

    if H.shape[1:] != (52, 2, 2):
        raise RuntimeError(
            f"Unexpected H shape: {H.shape}"
        )

    if "rf_cycle_time_s" not in z:
        raise RuntimeError(
            "rf_cycle_time_s is required"
        )

    t = np.asarray(
        z["rf_cycle_time_s"],
        dtype=np.float64,
    )

    tx = int(
        args.primary_tx
    )

    #
    # Inter-RX differential channel.
    #
    D = (
        H[:, :, 1, tx]
        * np.conj(
            H[:, :, 0, tx]
        )
    )

    U = (
        D
        / (
            np.abs(D)
            + EPS
        )
    ).astype(np.complex64)

    #
    # Temporal phase increment per subcarrier.
    #
    dU = (
        U[1:]
        * np.conj(
            U[:-1]
        )
    )

    dphi = np.angle(
        dU
    ).astype(np.float32)

    dphi_abs = np.abs(
        dphi
    ).astype(np.float32)

    tmid = (
        t[1:]
        + t[:-1]
    ) / 2.0

    #
    # ----------------------------------------------------
    # Baseline threshold source
    # ----------------------------------------------------
    #
    # For scientific deployment this should normally come
    # from a separate static calibration capture.
    #
    baseline_external = False

    if (
        args.baseline_start_cycle is not None
        and args.baseline_end_cycle is not None
    ):

        a = args.baseline_start_cycle
        b = args.baseline_end_cycle

        if (
            a < 0
            or b > len(H)
            or b - a < 3
        ):
            raise RuntimeError(
                "Invalid baseline cycle range"
            )

        base = dphi_abs[
            a:b-1
        ]

        baseline_source = np.array(
            "current_capture_cycle_range"
        )

    else:

        #
        # YAML contains scalar baseline metadata only.
        # It does NOT contain per-subcarrier thresholds.
        #
        # Therefore no fake per-SC threshold is synthesized.
        #
        baseline_external = True
        base = None

        baseline_source = np.array(
            "external_static_threshold_required"
        )

    if base is not None:

        q95 = np.quantile(
            base,
            0.95,
            axis=0,
        ).astype(np.float32)

        q99 = np.quantile(
            base,
            0.99,
            axis=0,
        ).astype(np.float32)

        anomaly95 = (
            dphi_abs
            > q95[None, :]
        )

        anomaly99 = (
            dphi_abs
            > q99[None, :]
        )

        anomaly_fraction95 = np.mean(
            anomaly95,
            axis=1,
        ).astype(np.float32)

        anomaly_fraction99 = np.mean(
            anomaly99,
            axis=1,
        ).astype(np.float32)

        threshold_valid = True

    else:

        q95 = np.full(
            52,
            np.nan,
            dtype=np.float32,
        )

        q99 = np.full(
            52,
            np.nan,
            dtype=np.float32,
        )

        anomaly95 = np.zeros(
            dphi.shape,
            dtype=bool,
        )

        anomaly99 = np.zeros(
            dphi.shape,
            dtype=bool,
        )

        anomaly_fraction95 = np.full(
            len(dphi),
            np.nan,
            dtype=np.float32,
        )

        anomaly_fraction99 = np.full(
            len(dphi),
            np.nan,
            dtype=np.float32,
        )

        threshold_valid = False

    #
    # Threshold-independent temporal evidence.
    #
    frame_median_abs = np.median(
        dphi_abs,
        axis=1,
    ).astype(np.float32)

    frame_p90_abs = np.percentile(
        dphi_abs,
        90,
        axis=1,
    ).astype(np.float32)

    frame_p95_abs = np.percentile(
        dphi_abs,
        95,
        axis=1,
    ).astype(np.float32)

    #
    # Motion score is only valid when per-subcarrier
    # baseline thresholds exist.
    #
    if threshold_valid:

        motion_score = (
            0.5 * anomaly_fraction95
            + 0.5 * anomaly_fraction99
        ).astype(np.float32)

    else:

        motion_score = np.full(
            len(dphi),
            np.nan,
            dtype=np.float32,
        )

    motion_valid = np.zeros(
        len(dphi),
        dtype=bool,
    )

    #
    # No universal threshold has been validated yet.
    #
    fixed_motion_threshold_valid = False

    if args.output is None:

        out = (
            args.tensor.parent.parent
            / "features"
            / "temporal_motion_features.npz"
        )

    else:

        out = args.output

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save = {
        "differential_complex":
            D,

        "differential_unit":
            U,

        "temporal_phase_increment_rad":
            dphi,

        "temporal_phase_increment_abs_rad":
            dphi_abs,

        "phase_increment_time_s":
            tmid,

        "frame_median_abs_phase_increment":
            frame_median_abs,

        "frame_p90_abs_phase_increment":
            frame_p90_abs,

        "frame_p95_abs_phase_increment":
            frame_p95_abs,

        "static_threshold_p95_rad":
            q95,

        "static_threshold_p99_rad":
            q99,

        "subcarrier_anomaly_mask_p95":
            anomaly95,

        "subcarrier_anomaly_mask_p99":
            anomaly99,

        "subcarrier_anomaly_fraction_p95":
            anomaly_fraction95,

        "subcarrier_anomaly_fraction_p99":
            anomaly_fraction99,

        "motion_score":
            motion_score,

        "motion_valid":
            motion_valid,

        "baseline_threshold_valid":
            np.bool_(
                threshold_valid
            ),

        "fixed_motion_threshold_valid":
            np.bool_(
                fixed_motion_threshold_valid
            ),

        "primary_tx":
            np.int32(tx),

        "primary_tx_name":
            np.array(
                f"TX{tx}"
            ),

        "temporal_method":
            np.array(
                "subcarrier_resolved_inter_rx_phase_increment"
            ),

        "baseline_source":
            baseline_source,

        "method_validation_status":
            np.array(
                "initial_controlled_motion_sensitivity_pass"
            ),

        "capture_motion_classification_valid":
            np.bool_(False),

        "physical_doppler_valid":
            np.bool_(False),

        "physical_velocity_valid":
            np.bool_(False),

        "warning":
            np.array(
                "Temporal differential phase is validated "
                "as motion-sensitive RF evidence. "
                "Absolute Doppler frequency and velocity "
                "are not yet physically validated."
            ),

        "source_tensor":
            np.array(
                str(args.tensor)
            ),
    }

    for key in (
        "seq",
        "seq_unwrapped",
        "rf_cycle_time_s",
        "rf_sample_rate_hz",
        "fc_hz",
        "physical_tx_mapping_valid",
    ):
        if key in z:
            save[key] = z[key]

    np.savez_compressed(
        out,
        **save,
    )

    print("=" * 72)
    print("TEMPORAL MOTION EVIDENCE")
    print("=" * 72)

    print(
        "source:",
        args.tensor,
    )

    print(
        "cycles:",
        len(H),
    )

    print(
        "phase increments:",
        dphi.shape,
    )

    print(
        "primary TX:",
        tx,
    )

    print(
        "baseline threshold valid:",
        threshold_valid,
    )

    if threshold_valid:

        print(
            "baseline cycles:",
            args.baseline_start_cycle,
            args.baseline_end_cycle,
        )

        print(
            "mean anomaly fraction P95:",
            float(
                np.mean(
                    anomaly_fraction95
                )
            ),
        )

        print(
            "mean anomaly fraction P99:",
            float(
                np.mean(
                    anomaly_fraction99
                )
            ),
        )

    print(
        "fixed motion threshold valid:",
        False,
    )

    print(
        "physical Doppler valid:",
        False,
    )

    print(
        "physical velocity valid:",
        False,
    )


if __name__ == "__main__":
    main()
