#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np


NFFT_PHY = 64

# gr-ieee802-11 get_csi():
# FFT bins 6..31 and 33..58.
#
# With DC at index 32 this corresponds to:
# -26..-1, +1..+26.
ACTIVE_BINS_SHIFTED = np.r_[
    np.arange(6, 32),
    np.arange(33, 59),
]


def robust_summary(x):
    finite = x[np.isfinite(x)]

    if finite.size == 0:
        return np.nan, np.nan, np.nan

    return (
        float(np.median(finite)),
        float(np.mean(finite)),
        float(np.percentile(finite, 95)),
    )


def map_52_to_fft_grid(x52):
    """
    Input:
        [..., 52]

    Output:
        [..., 64]

    Grid is in fftshifted order:
        DC = bin 32.

    Before np.fft.ifft(), caller must apply ifftshift().
    """
    shape = x52.shape[:-1] + (NFFT_PHY,)

    grid = np.zeros(
        shape,
        dtype=np.complex64,
    )

    grid[..., ACTIVE_BINS_SHIFTED] = x52

    return grid



def circular_delay_descriptors(
    power,
    peak_bin,
):
    """
    power:
        [N,TX,D]

    peak_bin:
        [N,TX]

    Returns:
        peak-relative centroid [N,TX]
        circular RMS spread    [N,TX]

    Delay-domain IFFT bins are periodic, so bin D-1
    is adjacent to bin 0.
    """

    N, T, D = power.shape

    centroid = np.full(
        (N, T),
        np.nan,
        dtype=np.float64,
    )

    rms = np.full(
        (N, T),
        np.nan,
        dtype=np.float64,
    )

    bins = np.arange(
        D,
        dtype=np.float64,
    )

    for n in range(N):
        for t in range(T):

            p = power[n, t].astype(
                np.float64
            )

            total = np.sum(p)

            if (
                not np.isfinite(total)
                or total <= 0
            ):
                continue

            peak = int(
                peak_bin[n, t]
            )

            delta = (
                (
                    bins
                    - peak
                    + D / 2.0
                )
                % D
                - D / 2.0
            )

            mu = np.sum(
                p * delta
            ) / total

            var = np.sum(
                p
                * (
                    delta - mu
                ) ** 2
            ) / total

            centroid[n, t] = mu

            rms[n, t] = np.sqrt(
                max(var, 0.0)
            )

    return centroid, rms


def peak_prominence(power):
    """
    power:
        [N,TX,D]
    """

    peak = np.max(
        power,
        axis=-1,
    )

    floor = np.median(
        power,
        axis=-1,
    )

    return (
        peak
        / (
            floor
            + 1e-12
        )
    ).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "rf_npz",
        help="complex_differential_rf.npz",
    )

    ap.add_argument(
        "--n-delay",
        type=int,
        default=256,
        help=(
            "IFFT length for interpolated delay-domain "
            "representation. Default: 256."
        ),
    )

    ap.add_argument(
        "--sample-rate",
        type=float,
        default=5e6,
        help="Effective OFDM sample rate [Hz]. Default: 5 MHz.",
    )

    ap.add_argument(
        "--output",
        default=None,
    )

    args = ap.parse_args()

    if args.n_delay < 64:
        raise RuntimeError(
            "--n-delay must be >= 64"
        )

    z = np.load(args.rf_npz)

    delta_complex = z[
        "delta_complex_masked"
    ]

    delta_unit = z[
        "delta_unit_masked"
    ]

    valid = z[
        "valid_rx_mask"
    ].astype(bool)

    quality = z[
        "quality_weight"
    ].astype(np.float32)

    if delta_complex.ndim != 3:
        raise RuntimeError(
            f"Unexpected delta_complex shape: "
            f"{delta_complex.shape}"
        )

    if delta_complex.shape[1:] != (52, 2):
        raise RuntimeError(
            f"Expected [N,52,2], got "
            f"{delta_complex.shape}"
        )

    n_cycles = delta_complex.shape[0]

    #
    # We use two representations:
    #
    # 1) complex-normalized differential:
    #    includes amplitude + phase.
    #
    # 2) unit-phasor differential:
    #    phase-focused and more robust to amplitude drift.
    #
    # Mask invalid carriers and multiply by
    # reference coherence quality.
    #
    complex_weighted = np.zeros_like(
        delta_complex,
        dtype=np.complex64,
    )

    unit_weighted = np.zeros_like(
        delta_unit,
        dtype=np.complex64,
    )

    for t in range(2):
        m = valid[:, t]

        q = quality[:, t]

        x = delta_complex[:, :, t]
        u = delta_unit[:, :, t]

        complex_weighted[:, m, t] = (
            np.nan_to_num(
                x[:, m],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            * q[m][None, :]
        )

        unit_weighted[:, m, t] = (
            np.nan_to_num(
                u[:, m],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            * q[m][None, :]
        )

    #
    # Convert [N,52,TX] -> [N,TX,52]
    #
    complex_52 = np.transpose(
        complex_weighted,
        (0, 2, 1),
    )

    unit_52 = np.transpose(
        unit_weighted,
        (0, 2, 1),
    )

    #
    # Restore exact 64-bin PHY frequency grid.
    #
    complex_grid_shifted = map_52_to_fft_grid(
        complex_52
    )

    unit_grid_shifted = map_52_to_fft_grid(
        unit_52
    )

    #
    # NumPy IFFT expects DC at index 0,
    # therefore ifftshift is mandatory.
    #
    complex_grid = np.fft.ifftshift(
        complex_grid_shifted,
        axes=-1,
    )

    unit_grid = np.fft.ifftshift(
        unit_grid_shifted,
        axes=-1,
    )

    #
    # Zero-padding to n_delay does NOT improve
    # physical range resolution.
    # It only interpolates the delay-domain curve.
    #
    h_complex = np.fft.ifft(
        complex_grid,
        n=args.n_delay,
        axis=-1,
    ).astype(np.complex64)

    h_phase = np.fft.ifft(
        unit_grid,
        n=args.n_delay,
        axis=-1,
    ).astype(np.complex64)

    p_complex = (
        np.abs(h_complex) ** 2
    ).astype(np.float32)

    p_phase = (
        np.abs(h_phase) ** 2
    ).astype(np.float32)

    #
    # Normalize each frame/profile for shape analysis.
    #
    p_complex_norm = (
        p_complex
        / (
            np.sum(
                p_complex,
                axis=-1,
                keepdims=True,
            )
            + 1e-12
        )
    ).astype(np.float32)

    p_phase_norm = (
        p_phase
        / (
            np.sum(
                p_phase,
                axis=-1,
                keepdims=True,
            )
            + 1e-12
        )
    ).astype(np.float32)

    #
    # Delay axes.
    #
    # Original physical delay spacing = 1/Fs.
    #
    # Zero-padding to Ndelay interpolates by:
    # 64 / Ndelay.
    #
    dt_physical = 1.0 / args.sample_rate

    dt_interp = (
        NFFT_PHY
        / args.n_delay
        / args.sample_rate
    )

    delay_axis_s = (
        np.arange(args.n_delay)
        * dt_interp
    )

    delay_axis_ns = (
        delay_axis_s * 1e9
    ).astype(np.float64)

    c = 299792458.0

    path_axis_m = (
        delay_axis_s * c
    ).astype(np.float64)

    #
    # ----------------------------------------------------
    # Circular / peak-relative delay descriptors
    # ----------------------------------------------------
    #
    # Delay-domain IFFT is periodic, therefore a linear
    # centroid over [0,D-1] is not physically meaningful
    # when a peak straddles the D-1 <-> 0 boundary.
    #
    peak_complex = np.argmax(
        p_complex,
        axis=-1,
    ).astype(np.int32)

    peak_phase = np.argmax(
        p_phase,
        axis=-1,
    ).astype(np.int32)

    (
        centroid_complex,
        rms_complex,
    ) = circular_delay_descriptors(
        p_complex_norm,
        peak_complex,
    )

    (
        centroid_phase,
        rms_phase,
    ) = circular_delay_descriptors(
        p_phase_norm,
        peak_phase,
    )

    prominence_complex = peak_prominence(
        p_complex
    )

    prominence_phase = peak_prominence(
        p_phase
    )

    #
    # Validity here means usable delay-domain evidence,
    # NOT valid physical ToF.
    #
    delay_evidence_valid = (
        np.isfinite(
            centroid_phase
        )
        & np.isfinite(
            rms_phase
        )
        & np.isfinite(
            prominence_phase
        )
    )

    #
    # TX0 has 52/52 calibrated carriers and passed the
    # BG-vs-object validation. TX1 has only 24/52 valid
    # carriers and remains auxiliary.
    #
    primary_tx = np.int32(0)

    physical_tof_s = np.full(
        (n_cycles, 2),
        np.nan,
        dtype=np.float64,
    )

    physical_tof_valid = np.zeros(
        (n_cycles, 2),
        dtype=bool,
    )

    absolute_range_m = np.full(
        (n_cycles, 2),
        np.nan,
        dtype=np.float64,
    )

    absolute_range_valid = np.zeros(
        (n_cycles, 2),
        dtype=bool,
    )

    if args.output is None:
        p = Path(args.rf_npz)

        out = (
            p.parent
            / "differential_delay_profile.npz"
        )
    else:
        out = Path(args.output)

    np.savez(
        out,

        h_complex=h_complex,
        h_phase=h_phase,

        power_complex=p_complex,
        power_phase=p_phase,

        power_complex_norm=p_complex_norm,
        power_phase_norm=p_phase_norm,

        peak_complex=peak_complex,
        peak_phase=peak_phase,

        peak_relative_centroid_complex_bins=
            centroid_complex.astype(np.float32),

        peak_relative_centroid_phase_bins=
            centroid_phase.astype(np.float32),

        circular_rms_complex_bins=
            rms_complex.astype(np.float32),

        circular_rms_phase_bins=
            rms_phase.astype(np.float32),

        peak_prominence_complex=
            prominence_complex,

        peak_prominence_phase=
            prominence_phase,

        delay_evidence_valid=
            delay_evidence_valid,

        primary_tx=
            primary_tx,

        tx0_role=np.array(
            "primary_phase_only_differential_delay"
        ),

        tx1_role=np.array(
            "auxiliary_quality_gated"
        ),

        physical_tof_s=
            physical_tof_s,

        physical_tof_valid=
            physical_tof_valid,

        absolute_range_m=
            absolute_range_m,

        absolute_range_valid=
            absolute_range_valid,

        fft_grid_complex_shifted=
            complex_grid_shifted,

        fft_grid_phase_shifted=
            unit_grid_shifted,

        active_bins_shifted=
            ACTIVE_BINS_SHIFTED,

        valid_rx_mask=valid,
        quality_weight=quality,

        sample_rate_hz=np.array(
            args.sample_rate,
            dtype=np.float64,
        ),

        nfft_phy=np.array(
            NFFT_PHY,
            dtype=np.int32,
        ),

        n_delay=np.array(
            args.n_delay,
            dtype=np.int32,
        ),

        physical_delay_resolution_s=np.array(
            dt_physical,
            dtype=np.float64,
        ),

        interpolated_delay_spacing_s=np.array(
            dt_interp,
            dtype=np.float64,
        ),

        delay_axis_ns=delay_axis_ns,
        path_axis_m=path_axis_m,

        source_rf=np.array(
            str(args.rf_npz)
        ),

        representation=np.array(
            "calibrated_inter_rx_differential_delay_evidence"
        ),

        preferred_representation=np.array(
            "phase_only_TX0"
        ),

        validation_status=np.array(
            "TX0_phase_only_BG_object_separation_pass"
        ),

        warning=np.array(
            "This is NOT a physical per-link CIR. "
            "It is the IFFT of calibrated inter-RX "
            "differential channel evidence "
            "H_RX1*conj(H_RX0)."
        ),
    )

    print("======================================")
    print("DIFFERENTIAL DELAY PROFILE")
    print("======================================")

    print("input :", args.rf_npz)
    print("output:", out)
    print("cycles:", n_cycles)

    print(
        "active FFT bins:",
        ACTIVE_BINS_SHIFTED.tolist(),
    )

    print(
        "PHY FFT size:",
        NFFT_PHY,
    )

    print(
        "delay IFFT size:",
        args.n_delay,
    )

    print(
        "sample rate [MHz]:",
        args.sample_rate / 1e6,
    )

    print(
        "physical delay resolution [ns]:",
        dt_physical * 1e9,
    )

    print(
        "c/Fs path-equivalent [m]:",
        c * dt_physical,
    )

    print(
        "interpolated delay spacing [ns]:",
        dt_interp * 1e9,
    )

    print()

    for t in range(2):
        med_c, mean_c, p95_c = robust_summary(
            centroid_complex[:, t]
        )

        med_p, mean_p, p95_p = robust_summary(
            centroid_phase[:, t]
        )

        print(f"TX{t}:")
        print(
            "  valid carriers:",
            f"{np.sum(valid[:, t])}/52",
        )

        print(
            "  complex peak-bin median:",
            float(
                np.median(
                    peak_complex[:, t]
                )
            ),
        )

        print(
            "  phase peak-bin median:",
            float(
                np.median(
                    peak_phase[:, t]
                )
            ),
        )

        print(
            "  complex centroid bins "
            "(median/mean/p95):",
            f"{med_c:.3f} / "
            f"{mean_c:.3f} / "
            f"{p95_c:.3f}",
        )

        print(
            "  phase centroid bins "
            "(median/mean/p95):",
            f"{med_p:.3f} / "
            f"{mean_p:.3f} / "
            f"{p95_p:.3f}",
        )

        print()

    print(
        "NOTE: zero-padding interpolates the profile;"
    )

    print(
        "it does NOT improve the physical delay "
        "resolution."
    )

    print(
        "NOTE: output is an inter-RX differential "
        "delay representation, not radar range."
    )


if __name__ == "__main__":
    main()
