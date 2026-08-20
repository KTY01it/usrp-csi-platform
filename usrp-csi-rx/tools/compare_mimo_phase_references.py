#!/usr/bin/env python3

import argparse
import numpy as np


def circ_mean(phi, axis=0):
    return np.angle(
        np.mean(
            np.exp(1j * phi),
            axis=axis,
        )
    )


def circ_coh(phi, axis=0):
    return np.abs(
        np.mean(
            np.exp(1j * phi),
            axis=axis,
        )
    )


def phase_error(a, b):
    return np.angle(
        np.exp(1j * (a - b))
    )


def extract_templates(path):
    z = np.load(path)
    H = z["H_raw"]

    if H.ndim != 4 or H.shape[1:] != (52, 2, 2):
        raise RuntimeError(
            f"{path}: invalid H shape {H.shape}"
        )

    out = {}

    # RX differential phase:
    # RX1 / RX0 for each TX.
    for t in range(2):
        c = (
            H[:, :, 1, t]
            * np.conj(H[:, :, 0, t])
        )

        phi = np.angle(c)

        out[f"rx_diff_tx{t}"] = {
            "phase": circ_mean(phi),
            "coh": circ_coh(phi),
        }

    # Cross-TX phase:
    # TX1 / TX0 for each RX.
    for r in range(2):
        c = (
            H[:, :, r, 1]
            * np.conj(H[:, :, r, 0])
        )

        phi = np.angle(c)

        out[f"tx_diff_rx{r}"] = {
            "phase": circ_mean(phi),
            "coh": circ_coh(phi),
        }

    return out


def compare(name, A, B):
    pa = A[name]["phase"]
    pb = B[name]["phase"]

    e = phase_error(pb, pa)

    print()
    print(name)
    print(
        "mean abs phase difference [rad]:",
        float(np.mean(np.abs(e))),
    )
    print(
        "median abs phase difference [rad]:",
        float(np.median(np.abs(e))),
    )
    print(
        "max abs phase difference [rad]:",
        float(np.max(np.abs(e))),
    )

    print(
        "subcarriers |error| <= 0.10 rad:",
        f"{np.sum(np.abs(e) <= 0.10)}/52",
    )

    print(
        "subcarriers |error| <= 0.20 rad:",
        f"{np.sum(np.abs(e) <= 0.20)}/52",
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("reference")
    ap.add_argument("test")

    args = ap.parse_args()

    A = extract_templates(args.reference)
    B = extract_templates(args.test)

    print("REFERENCE:", args.reference)
    print("TEST     :", args.test)

    print()
    print("===== RX DIFFERENTIAL REPEATABILITY =====")

    compare(
        "rx_diff_tx0",
        A,
        B,
    )

    compare(
        "rx_diff_tx1",
        A,
        B,
    )

    print()
    print("===== CROSS-TX REPEATABILITY =====")

    compare(
        "tx_diff_rx0",
        A,
        B,
    )

    compare(
        "tx_diff_rx1",
        A,
        B,
    )


if __name__ == "__main__":
    main()
