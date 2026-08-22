#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np


DF = 5e6 / 64.0

SUBCARRIER_INDEX = np.r_[
    np.arange(-26, 0),
    np.arange(1, 27)
].astype(np.float64)

FREQ = SUBCARRIER_INDEX * DF

EPS = 1e-8


def remove_affine_phase(phase, mask):
    """
    phase: [N, 52]
    mask : [52]

    Returns:
        residual [N,52]
        slope    [N]
        rmse     [N]
    """

    f = FREQ[mask]

    fc = f - np.mean(f)

    denom = np.sum(fc * fc)

    residual = np.full(
        phase.shape,
        np.nan,
        dtype=np.float32
    )

    slopes = np.empty(
        phase.shape[0],
        dtype=np.float64
    )

    rmses = np.empty(
        phase.shape[0],
        dtype=np.float64
    )

    for n in range(phase.shape[0]):

        y = phase[n, mask]

        y = np.unwrap(y)

        yc = y - np.mean(y)

        slope = np.sum(
            fc * yc
        ) / denom

        intercept = (
            np.mean(y)
            - slope * np.mean(f)
        )

        fit = (
            slope * f
            + intercept
        )

        err = y - fit

        residual[n, mask] = (
            err.astype(np.float32)
        )

        slopes[n] = slope

        rmses[n] = np.sqrt(
            np.mean(err ** 2)
        )

    return residual, slopes, rmses


def robust_mad(x, axis=0):
    med = np.nanmedian(
        x,
        axis=axis
    )

    mad = np.nanmedian(
        np.abs(
            x - np.expand_dims(
                med,
                axis=axis
            )
        ),
        axis=axis
    )

    sigma = 1.4826 * mad

    return med, np.maximum(
        sigma,
        1e-4
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "target_aligned",
        help="Target common_phase_aligned_mimo.npz",
    )

    ap.add_argument(
        "background_aligned",
        help="Background common_phase_aligned_mimo.npz",
    )

    ap.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Fixed coherence threshold from background.",
    )

    ap.add_argument(
        "--output",
        default=None,
    )

    args = ap.parse_args()

    target = np.load(
        args.target_aligned
    )

    background = np.load(
        args.background_aligned
    )

    phase_obj = target[
        "residual_phase"
    ]

    phase_bg = background[
        "residual_phase"
    ]

    bg_coherence = background[
        "coherence_map"
    ]

    if phase_obj.shape[1:] != (52, 2, 2):
        raise RuntimeError(
            f"Unexpected target shape: {phase_obj.shape}"
        )

    if phase_bg.shape[1:] != (52, 2, 2):
        raise RuntimeError(
            f"Unexpected background shape: {phase_bg.shape}"
        )

    #
    # Critical:
    # quality mask comes ONLY from background.
    #
    fixed_mask = (
        bg_coherence
        >= args.threshold
    )

    Nobj = phase_obj.shape[0]
    Nbg = phase_bg.shape[0]

    curvature_obj = np.full(
        phase_obj.shape,
        np.nan,
        dtype=np.float32
    )

    curvature_bg = np.full(
        phase_bg.shape,
        np.nan,
        dtype=np.float32
    )

    slope_obj = np.full(
        (Nobj, 2, 2),
        np.nan,
        dtype=np.float64
    )

    slope_bg = np.full(
        (Nbg, 2, 2),
        np.nan,
        dtype=np.float64
    )

    rmse_obj = np.full(
        (Nobj, 2, 2),
        np.nan,
        dtype=np.float64
    )

    rmse_bg = np.full(
        (Nbg, 2, 2),
        np.nan,
        dtype=np.float64
    )

    #
    # Per physical TX/RX link.
    #
    for t in range(2):
        for r in range(2):

            mask = fixed_mask[
                :, r, t
            ]

            if np.sum(mask) < 10:
                continue

            (
                residual,
                slope,
                rmse
            ) = remove_affine_phase(
                phase_obj[:, :, r, t],
                mask
            )

            curvature_obj[
                :, :, r, t
            ] = residual

            slope_obj[
                :, r, t
            ] = slope

            rmse_obj[
                :, r, t
            ] = rmse

            (
                residual,
                slope,
                rmse
            ) = remove_affine_phase(
                phase_bg[:, :, r, t],
                mask
            )

            curvature_bg[
                :, :, r, t
            ] = residual

            slope_bg[
                :, r, t
            ] = slope

            rmse_bg[
                :, r, t
            ] = rmse

    #
    # Robust background model per:
    # [subcarrier, RX, TX].
    #
    bg_median, bg_sigma = robust_mad(
        curvature_bg,
        axis=0
    )

    curvature_z = np.abs(
        (
            curvature_obj
            - bg_median[
                None, :, :, :
            ]
        )
        /
        bg_sigma[
            None, :, :, :
        ]
    ).astype(np.float32)

    #
    # Per-frame features.
    #
    frame_median_z = np.nanmedian(
        curvature_z,
        axis=1
    ).astype(np.float32)

    frame_mean_z = np.nanmean(
        curvature_z,
        axis=1
    ).astype(np.float32)

    frame_frac2 = np.nanmean(
        np.where(
            np.isfinite(curvature_z),
            curvature_z >= 2.0,
            np.nan
        ),
        axis=1
    ).astype(np.float32)

    frame_frac3 = np.nanmean(
        np.where(
            np.isfinite(curvature_z),
            curvature_z >= 3.0,
            np.nan
        ),
        axis=1
    ).astype(np.float32)

    #
    # Phase-slope delay proxy retained only
    # as diagnostic metadata.
    #
    tau_obj = (
        -slope_obj
        / (2.0 * np.pi)
    )

    tau_bg = (
        -slope_bg
        / (2.0 * np.pi)
    )

    if args.output is None:

        p = Path(
            args.target_aligned
        )

        out = (
            p.parent
            / "phase_curvature_evidence.npz"
        )

    else:
        out = Path(
            args.output
        )

    np.savez(
        out,

        curvature_obj=
            curvature_obj,

        curvature_bg=
            curvature_bg,

        curvature_bg_median=
            bg_median.astype(
                np.float32
            ),

        curvature_bg_sigma=
            bg_sigma.astype(
                np.float32
            ),

        curvature_z=
            curvature_z,

        frame_median_z=
            frame_median_z,

        frame_mean_z=
            frame_mean_z,

        frame_frac2=
            frame_frac2,

        frame_frac3=
            frame_frac3,

        rmse_obj=
            rmse_obj,

        rmse_bg=
            rmse_bg,

        tau_eff_obj_s=
            tau_obj,

        tau_eff_bg_s=
            tau_bg,

        fixed_quality_mask=
            fixed_mask,

        background_coherence=
            bg_coherence,

        subcarrier_index=
            SUBCARRIER_INDEX,

        subcarrier_spacing_hz=
            np.array(
                DF,
                dtype=np.float64
            ),

        source_target=
            np.array(
                str(
                    args.target_aligned
                )
            ),

        source_background=
            np.array(
                str(
                    args.background_aligned
                )
            ),

        representation=
            np.array(
                "background_normalized_per_link_phase_curvature"
            ),

        warning=
            np.array(
                "Affine frequency phase is removed. "
                "Curvature reflects nonlinear spectral "
                "phase / multipath structure, not "
                "physical range."
            ),
    )

    print("======================================")
    print("PHASE CURVATURE EVIDENCE")
    print("======================================")

    print(
        "target    :",
        args.target_aligned
    )

    print(
        "background:",
        args.background_aligned
    )

    print(
        "output    :",
        out
    )

    print(
        "target cycles:",
        Nobj
    )

    print(
        "background cycles:",
        Nbg
    )

    print()

    for t in range(2):
        for r in range(2):

            mask = fixed_mask[
                :, r, t
            ]

            fm = frame_median_z[
                :, r, t
            ]

            f2 = frame_frac2[
                :, r, t
            ]

            f3 = frame_frac3[
                :, r, t
            ]

            print(
                f"TX{t}->RX{r}:"
            )

            print(
                "  fixed carriers:",
                int(
                    np.sum(mask)
                ),
                "/52"
            )

            print(
                "  curvature RMSE median [rad]:",
                float(
                    np.nanmedian(
                        rmse_obj[
                            :, r, t
                        ]
                    )
                )
            )

            print(
                "  frame median-z median:",
                float(
                    np.nanmedian(fm)
                )
            )

            print(
                "  frame median-z mean:",
                float(
                    np.nanmean(fm)
                )
            )

            print(
                "  fraction>=2 median:",
                float(
                    np.nanmedian(f2)
                )
            )

            print(
                "  fraction>=3 median:",
                float(
                    np.nanmedian(f3)
                )
            )

            print(
                "  frames median-z>=2:",
                int(
                    np.sum(
                        fm >= 2.0
                    )
                ),
                "/",
                len(fm)
            )

            print()


if __name__ == "__main__":
    main()
