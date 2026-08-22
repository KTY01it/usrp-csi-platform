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
    # Profile centroids.
    #
    idx = np.arange(
        args.n_delay,
        dtype=np.float64,
    )

    centroid_complex = (
        np.sum(
            p_complex_norm * idx,
            axis=-1,
        )
    ).astype(np.float32)

    centroid_phase = (
        np.sum(
            p_phase_norm * idx,
            axis=-1,
        )
    ).astype(np.float32)

    peak_complex = np.argmax(
        p_complex,
        axis=-1,
    ).astype(np.int32)

    peak_phase = np.argmax(
        p_phase,
        axis=-1,
    ).astype(np.int32)

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

        centroid_complex=centroid_complex,
        centroid_phase=centroid_phase,

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
            "inter_rx_differential_delay_profile"
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
