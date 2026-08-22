#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np


def wrap_phase(x):
    return np.angle(np.exp(1j * x))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("tensor")
    ap.add_argument("calibration")

    ap.add_argument(
        "--output",
        default=None,
    )

    args = ap.parse_args()

    z = np.load(args.tensor)
    c = np.load(args.calibration)

    H = z["H_raw"]

    if H.ndim != 4 or H.shape[1:] != (52, 2, 2):
        raise RuntimeError(
            f"Unexpected H shape: {H.shape}"
        )

    phi_rx_ref = c["phi_rx_ref"]
    phi_tx_ref = c["phi_tx_ref"]

    coh_rx_ref = c["coh_rx_ref"]
    coh_tx_ref = c["coh_tx_ref"]

    valid_rx = c["valid_rx_mask"]
    valid_tx = c["valid_tx_mask"]

    N = H.shape[0]

    #
    # RX differential calibrated phase
    #
    # shape:
    # [N, 52, TX]
    #
    rx_phase_raw = np.zeros(
        (N, 52, 2),
        dtype=np.float32,
    )

    rx_phase_cal = np.zeros_like(
        rx_phase_raw
    )

    for t in range(2):
        cross = (
            H[:, :, 1, t]
            * np.conj(H[:, :, 0, t])
        )

        raw = np.angle(cross)

        rx_phase_raw[:, :, t] = raw

        rx_phase_cal[:, :, t] = wrap_phase(
            raw - phi_rx_ref[:, t][None, :]
        )

    #
    # Cross-TX calibrated phase
    #
    # shape:
    # [N, 52, RX]
    #
    tx_phase_raw = np.zeros(
        (N, 52, 2),
        dtype=np.float32,
    )

    tx_phase_cal = np.zeros_like(
        tx_phase_raw
    )

    for r in range(2):
        cross = (
            H[:, :, r, 1]
            * np.conj(H[:, :, r, 0])
        )

        raw = np.angle(cross)

        tx_phase_raw[:, :, r] = raw

        tx_phase_cal[:, :, r] = wrap_phase(
            raw - phi_tx_ref[:, r][None, :]
        )

    #
    # Apply masks.
    #
    rx_phase_cal_masked = np.where(
        valid_rx[None, :, :],
        rx_phase_cal,
        np.nan,
    )

    tx_phase_cal_masked = np.where(
        valid_tx[None, :, :],
        tx_phase_cal,
        np.nan,
    )

    #
    # Quality weights from reference coherence.
    #
    rx_weight = np.broadcast_to(
        coh_rx_ref[None, :, :],
        rx_phase_cal.shape,
    ).copy()

    tx_weight = np.broadcast_to(
        coh_tx_ref[None, :, :],
        tx_phase_cal.shape,
    ).copy()

    if args.output is None:
        p = Path(args.tensor)

        out = (
            p.parent
            / "H_reference_calibrated_2x2.npz"
        )
    else:
        out = Path(args.output)

    save_dict = {
        "H_raw": H,

        "rx_phase_raw": rx_phase_raw,
        "rx_phase_cal": rx_phase_cal,
        "rx_phase_cal_masked":
            rx_phase_cal_masked,

        "tx_phase_raw": tx_phase_raw,
        "tx_phase_cal": tx_phase_cal,
        "tx_phase_cal_masked":
            tx_phase_cal_masked,

        "rx_weight": rx_weight,
        "tx_weight": tx_weight,

        "valid_rx_mask": valid_rx,
        "valid_tx_mask": valid_tx,

        "phi_rx_ref": phi_rx_ref,
        "phi_tx_ref": phi_tx_ref,

        "coh_rx_ref": coh_rx_ref,
        "coh_tx_ref": coh_tx_ref,

        "calibration_type": np.array(
            "reference_based_relative_phase"
        ),

        "source_tensor": np.array(
            str(args.tensor)
        ),

        "source_calibration": np.array(
            str(args.calibration)
        ),

        "note": np.array(
            "Relative/system phase calibration. "
            "Not absolute hardware-only AoA/AoD calibration."
        ),
    }

    if "seq_pairs" in z.files:
        save_dict["seq_pairs"] = z["seq_pairs"]

    np.savez(
        out,
        **save_dict,
    )

    print("======================================")
    print("APPLY MIMO REFERENCE CALIBRATION")
    print("======================================")
    print("input :", args.tensor)
    print("cal   :", args.calibration)
    print("output:", out)
    print("cycles:", N)

    print()
    print("RX calibrated phase:")

    for t in range(2):
        x = rx_phase_cal_masked[:, :, t]

        mean_abs = np.nanmean(
            np.abs(x)
        )

        rms = np.sqrt(
            np.nanmean(x ** 2)
        )

        print(
            f"TX{t}: "
            f"valid={np.sum(valid_rx[:,t])}/52 "
            f"mean_abs={mean_abs:.4f} rad "
            f"rms={rms:.4f} rad"
        )

    print()
    print("Cross-TX calibrated phase:")

    for r in range(2):
        x = tx_phase_cal_masked[:, :, r]

        mean_abs = np.nanmean(
            np.abs(x)
        )

        rms = np.sqrt(
            np.nanmean(x ** 2)
        )

        print(
            f"RX{r}: "
            f"valid={np.sum(valid_tx[:,r])}/52 "
            f"mean_abs={mean_abs:.4f} rad "
            f"rms={rms:.4f} rad"
        )


if __name__ == "__main__":
    main()
