#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np


NFFT_PHY = 64

ACTIVE_BINS_SHIFTED = np.r_[
    np.arange(6, 32),
    np.arange(33, 59),
]


def map_52_to_fft_grid(H52):
    """
    H52:
        [N,52,2,2]

    Return:
        [N,64,2,2]

    fftshifted PHY ordering:
        DC = bin 32
        active:
          6..31  -> k=-26..-1
          33..58 -> k=+1..+26
    """

    if H52.shape[1] != 52:
        raise RuntimeError(
            f"Expected 52 subcarriers, got {H52.shape}"
        )

    out = np.zeros(
        (
            H52.shape[0],
            NFFT_PHY,
            H52.shape[2],
            H52.shape[3],
        ),
        dtype=np.complex64,
    )

    out[
        :,
        ACTIVE_BINS_SHIFTED,
        :,
        :,
    ] = H52

    return out


def circular_delay_descriptors(
    power,
    peak_bin,
):
    """
    Compute delay-shape descriptors on a circular IFFT axis.

    power:
        [N,D,2,2]

    peak_bin:
        [N,2,2]

    Returns:
        relative_centroid_bins [N,2,2]
        rms_spread_bins        [N,2,2]

    Delay bins from an IFFT are periodic.  Therefore bin D-1
    is adjacent to bin 0.  We express every delay bin relative
    to the strongest peak in the interval [-D/2, D/2).
    """

    N, D, R, T = power.shape

    relative_centroid = np.full(
        (N, R, T),
        np.nan,
        dtype=np.float64,
    )

    rms_spread = np.full(
        (N, R, T),
        np.nan,
        dtype=np.float64,
    )

    bins = np.arange(
        D,
        dtype=np.float64,
    )

    for n in range(N):
        for r in range(R):
            for t in range(T):

                p = power[
                    n, :, r, t
                ].astype(np.float64)

                total = np.sum(p)

                if (
                    not np.isfinite(total)
                    or total <= 0
                ):
                    continue

                peak = int(
                    peak_bin[n, r, t]
                )

                #
                # Signed circular offset from peak:
                #
                #   ... -2 -1 0 +1 +2 ...
                #
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

                relative_centroid[
                    n, r, t
                ] = mu

                rms_spread[
                    n, r, t
                ] = np.sqrt(
                    max(
                        var,
                        0.0,
                    )
                )

    return (
        relative_centroid,
        rms_spread,
    )

def main():

    ap = argparse.ArgumentParser(
        description=(
            "Build per-link OFDM CSI delay-domain evidence. "
            "Output is a CIR/PDP proxy, not radar-grade ToF/range."
        )
    )

    ap.add_argument(
        "mimo_npz",
        type=Path,
    )

    ap.add_argument(
        "--out",
        type=Path,
        default=None,
    )

    ap.add_argument(
        "--key",
        default="H_raw",
        help=(
            "Tensor field to process. "
            "Default: H_raw."
        ),
    )

    ap.add_argument(
        "--n-delay",
        type=int,
        default=256,
        help=(
            "Delay-domain IFFT size. "
            "Values >64 interpolate the profile only."
        ),
    )

    ap.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        help=(
            "OFDM sample rate [Hz]. "
            "Default: tensor rf_sample_rate_hz."
        ),
    )

    ap.add_argument(
        "--min-peak-prominence",
        type=float,
        default=2.0,
    )

    args = ap.parse_args()

    if args.n_delay < NFFT_PHY:
        raise RuntimeError(
            "--n-delay must be >= 64"
        )

    z = np.load(
        args.mimo_npz,
        allow_pickle=True,
    )

    if args.key not in z:
        raise RuntimeError(
            f"Input does not contain {args.key}. "
            f"Available keys: {z.files}"
        )

    H = np.asarray(
        z[args.key],
        dtype=np.complex64,
    )

    if (
        H.ndim != 4
        or H.shape[1:] != (52, 2, 2)
    ):
        raise RuntimeError(
            f"Expected [N,52,2,2], got {H.shape}"
        )

    N = H.shape[0]

    if args.sample_rate is not None:
        fs = float(
            args.sample_rate
        )

    elif "rf_sample_rate_hz" in z:
        fs = float(
            z["rf_sample_rate_hz"]
        )

    else:
        fs = 5e6

    #
    # --------------------------------------------------------
    # Restore exact 64-bin IEEE 802.11 PHY frequency grid.
    # --------------------------------------------------------
    #
    H_shifted = map_52_to_fft_grid(
        H
    )

    H_fft_order = np.fft.ifftshift(
        H_shifted,
        axes=1,
    )

    #
    # --------------------------------------------------------
    # CIR-like delay representation
    # --------------------------------------------------------
    #
    # n_delay > 64 is zero-padding/interpolation only.
    #
    cir = np.fft.ifft(
        H_fft_order,
        n=args.n_delay,
        axis=1,
    ).astype(np.complex64)

    pdp = (
        np.abs(cir) ** 2
    ).astype(np.float32)

    pdp_sum = np.sum(
        pdp,
        axis=1,
        keepdims=True,
    )

    pdp_norm = (
        pdp
        / (
            pdp_sum
            + 1e-12
        )
    ).astype(np.float32)

    #
    # --------------------------------------------------------
    # Per-frame delay-shape descriptors
    # --------------------------------------------------------
    #
    delay_peak_bin = np.argmax(
        pdp,
        axis=1,
    ).astype(np.int32)

    (
        delay_centroid_bin,
        rms_delay_spread_bins,
    ) = circular_delay_descriptors(
        pdp_norm,
        delay_peak_bin,
    )

    #
    # Peak prominence:
    #
    # strongest PDP bin / median PDP level.
    #
    peak_power = np.max(
        pdp,
        axis=1,
    )

    median_power = np.median(
        pdp,
        axis=1,
    )

    peak_prominence = (
        peak_power
        / (
            median_power
            + 1e-12
        )
    ).astype(np.float32)

    finite_link = np.all(
        np.isfinite(H.real)
        & np.isfinite(H.imag),
        axis=1,
    )

    delay_valid = (
        finite_link
        & np.isfinite(
            delay_centroid_bin
        )
        & np.isfinite(
            rms_delay_spread_bins
        )
        & np.isfinite(
            peak_prominence
        )
        & (
            peak_prominence
            >= args.min_peak_prominence
        )
    )

    #
    # --------------------------------------------------------
    # Delay axes and physical semantics
    # --------------------------------------------------------
    #
    # Native 64-bin delay-bin spacing:
    #
    #   dt_native = 1/Fs
    #
    # Zero-padding from 64 -> n_delay changes only numerical
    # spacing of interpolated samples:
    #
    #   dt_interp = 64/n_delay/Fs
    #
    native_delay_bin_spacing_s = (
        1.0 / fs
    )

    interpolated_delay_spacing_s = (
        NFFT_PHY
        / args.n_delay
        / fs
    )

    delay_axis_s = (
        np.arange(
            args.n_delay,
            dtype=np.float64,
        )
        * interpolated_delay_spacing_s
    )

    delay_axis_ns = (
        delay_axis_s
        * 1e9
    )

    delay_peak_s_proxy = (
        delay_peak_bin.astype(
            np.float64
        )
        * interpolated_delay_spacing_s
    )

    delay_centroid_s_proxy = (
        delay_centroid_bin
        * interpolated_delay_spacing_s
    )

    rms_delay_spread_s_proxy = (
        rms_delay_spread_bins
        * interpolated_delay_spacing_s
    )

    #
    # Confidence is profile quality only.
    # It is NOT ToF accuracy.
    #
    prominence_score = np.clip(
        (
            peak_prominence
            - args.min_peak_prominence
        )
        / (
            10.0
            - args.min_peak_prominence
        ),
        0.0,
        1.0,
    )

    delay_confidence = np.where(
        delay_valid,
        prominence_score,
        0.0,
    ).astype(np.float32)

    #
    # Physical absolute ToF/range intentionally disabled.
    #
    physical_tof_s = np.full(
        (N, 2, 2),
        np.nan,
        dtype=np.float64,
    )

    physical_tof_valid = np.zeros(
        (N, 2, 2),
        dtype=bool,
    )

    absolute_range_m = np.full(
        (N, 2, 2),
        np.nan,
        dtype=np.float64,
    )

    absolute_range_valid = np.zeros(
        (N, 2, 2),
        dtype=bool,
    )

    if args.out is None:
        out = (
            args.mimo_npz.parent.parent
            / "features"
            / "delay_features.npz"
        )
    else:
        out = args.out

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save = {
        #
        # Delay-domain evidence
        #
        "cir_proxy_complex":
            cir,

        "pdp_proxy":
            pdp,

        "pdp_normalized":
            pdp_norm,

        #
        # Per-frame descriptors
        #
        "delay_peak_bin":
            delay_peak_bin,

        "peak_relative_centroid_bins":
            delay_centroid_bin.astype(
                np.float32
            ),

        "rms_delay_spread_bins":
            rms_delay_spread_bins.astype(
                np.float32
            ),

        "delay_peak_prominence":
            peak_prominence,

        "delay_confidence":
            delay_confidence,

        "delay_valid":
            delay_valid,

        #
        # Delay proxy coordinates
        #
        "delay_axis_s":
            delay_axis_s,

        "delay_axis_ns":
            delay_axis_ns,

        "delay_peak_s_proxy":
            delay_peak_s_proxy,

        "peak_relative_centroid_s_proxy":
            delay_centroid_s_proxy,

        "rms_delay_spread_s_proxy":
            rms_delay_spread_s_proxy,

        #
        # Resolution semantics
        #
        "sample_rate_hz":
            np.float64(fs),

        "phy_fft_size":
            np.int32(NFFT_PHY),

        "delay_ifft_size":
            np.int32(
                args.n_delay
            ),

        "native_delay_bin_spacing_s":
            np.float64(
                native_delay_bin_spacing_s
            ),

        "interpolated_delay_spacing_s":
            np.float64(
                interpolated_delay_spacing_s
            ),

        "active_bins_shifted":
            ACTIVE_BINS_SHIFTED.astype(
                np.int32
            ),

        #
        # Absolute ToF/range disabled
        #
        "physical_tof_s":
            physical_tof_s,

        "physical_tof_valid":
            physical_tof_valid,

        "absolute_range_m":
            absolute_range_m,

        "absolute_range_valid":
            absolute_range_valid,

        #
        # Provenance
        #
        "delay_method":
            np.array(
                "per_link_ofdm_csi_ifft_proxy"
            ),

        "source_csi_field":
            np.array(
                args.key
            ),

        "source_tensor":
            np.array(
                str(
                    args.mimo_npz
                )
            ),

        "representation":
            np.array(
                "per_link_cir_pdp_delay_evidence"
            ),

        "warning":
            np.array(
                "CIR/PDP and delay values are OFDM CSI "
                "delay-domain proxies. Zero-padding only "
                "interpolates the profile. They are not "
                "validated absolute ToF or radar range."
            ),
    }

    #
    # Preserve acquisition identity and timing.
    #
    passthrough = (
        "seq",
        "seq_unwrapped",
        "snr",
        "cfo_hz",
        "rf_sample_index",
        "rf_sample_rate_hz",
        "rf_reference_sample_index",
        "rf_time_relative_s",
        "rf_cycle_time_s",
        "rf_rx_sample_skew",
        "rf_sample_time_valid",
        "sample_domain_time_valid",
        "timestamp_monotonic_ns",
        "timestamp_wall_ns",
        "packet_time_nominal_s",
        "cycle_time_nominal_s",
        "tx_packet_interval_s",
        "fc_hz",
        "physical_tx_mapping_valid",
    )

    for key in passthrough:
        if key in z:
            save[key] = z[key]

    np.savez_compressed(
        out,
        **save,
    )

    print("=" * 72)
    print("DELAY EVIDENCE V2")
    print("=" * 72)

    print(
        "source:",
        args.mimo_npz,
    )

    print(
        "source CSI:",
        args.key,
    )

    print(
        "H:",
        H.shape,
    )

    print(
        "CIR proxy:",
        cir.shape,
    )

    print(
        "PDP proxy:",
        pdp.shape,
    )

    print(
        "output:",
        out,
    )

    print()

    print(
        "sample rate [MHz]:",
        fs / 1e6,
    )

    print(
        "native delay-bin spacing [ns]:",
        native_delay_bin_spacing_s
        * 1e9,
    )

    print(
        "interpolated spacing [ns]:",
        interpolated_delay_spacing_s
        * 1e9,
    )

    print(
        "delay IFFT size:",
        args.n_delay,
    )

    print()

    for tx in range(2):
        for rx in range(2):

            v = delay_valid[
                :,
                rx,
                tx,
            ]

            print(
                f"RX{rx}<-TX{tx}: "
                f"valid={int(np.sum(v))}/{N} "
                f"median_peak_bin="
                f"{float(np.median(delay_peak_bin[:,rx,tx])):.1f} "
                f"median_centroid="
                f"{float(np.nanmedian(delay_centroid_bin[:,rx,tx])):.2f} "
                f"median_rms_bins="
                f"{float(np.nanmedian(rms_delay_spread_bins[:,rx,tx])):.2f} "
                f"median_prominence="
                f"{float(np.nanmedian(peak_prominence[:,rx,tx])):.2f}"
            )

    print()

    print(
        "physical ToF valid:",
        False,
    )

    print(
        "absolute range valid:",
        False,
    )


if __name__ == "__main__":
    main()
