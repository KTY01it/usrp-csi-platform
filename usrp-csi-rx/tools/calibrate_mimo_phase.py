#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

def wrap(x):
    return np.angle(np.exp(1j * x))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mimo_npz")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    path = Path(args.mimo_npz)
    out = Path(args.out) if args.out else path.with_name("H_calibrated_mimo.npz")

    d = np.load(path)
    H = d["H_raw_mimo"]  # [N, K, RX=2, TX=2]

    if H.ndim != 4 or H.shape[2] != 2 or H.shape[3] != 2:
        raise SystemExit(f"[ERROR] expected H [N,K,2,2], got {H.shape}")

    eps = 1e-12

    # RX-chain phase difference per TX: rx1/rx0
    phase_diff_rx_raw = np.angle(H[:, :, 1, :] / (H[:, :, 0, :] + eps))  # [N,K,TX]

    # Estimate RX-chain phase calibration per subcarrier and TX.
    phi_rx_cal = np.angle(np.mean(np.exp(1j * phase_diff_rx_raw), axis=0))  # [K,TX]

    # Apply RX-chain calibration to rx1 for each TX.
    H_rxcal = H.copy()
    for tx in range(2):
        H_rxcal[:, :, 1, tx] *= np.exp(-1j * phi_rx_cal[:, tx])[None, :]

    phase_diff_rx_calibrated = np.angle(
        H_rxcal[:, :, 1, :] / (H_rxcal[:, :, 0, :] + eps)
    )

    # TX-link phase difference per RX: tx1/tx0.
    # For sequential pseudo-MIMO this is virtual-TX/link phase, not true simultaneous AoD.
    phase_diff_tx_raw = np.angle(H_rxcal[:, :, :, 1] / (H_rxcal[:, :, :, 0] + eps))  # [N,K,RX]
    phi_tx_cal = np.angle(np.mean(np.exp(1j * phase_diff_tx_raw), axis=0))  # [K,RX]

    H_cal = H_rxcal.copy()
    for rx in range(2):
        H_cal[:, :, rx, 1] *= np.exp(-1j * phi_tx_cal[:, rx])[None, :]

    phase_diff_tx_calibrated = np.angle(
        H_cal[:, :, :, 1] / (H_cal[:, :, :, 0] + eps)
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        H_raw_mimo=H.astype(np.complex64),
        H_calibrated_mimo=H_cal.astype(np.complex64),
        phi_rx_cal=phi_rx_cal.astype(np.float32),
        phi_tx_cal=phi_tx_cal.astype(np.float32),
        phase_diff_rx_raw=phase_diff_rx_raw.astype(np.float32),
        phase_diff_rx_calibrated=phase_diff_rx_calibrated.astype(np.float32),
        phase_diff_tx_raw=phase_diff_tx_raw.astype(np.float32),
        phase_diff_tx_calibrated=phase_diff_tx_calibrated.astype(np.float32),
        source=str(path),
    )

    print(f"[OK] wrote: {out}")
    print("[MIMO-CAL] H_raw:", H.shape)
    print("[MIMO-CAL] H_calibrated:", H_cal.shape)
    print("[MIMO-CAL] phi_rx_cal:", phi_rx_cal.shape)
    print("[MIMO-CAL] phi_tx_cal:", phi_tx_cal.shape)

    for tx in range(2):
        raw = phase_diff_rx_raw[:, :, tx]
        cal = phase_diff_rx_calibrated[:, :, tx]
        print(f"[RX-PHASE] tx{tx} raw mean/std:", float(raw.mean()), float(raw.std()))
        print(f"[RX-PHASE] tx{tx} cal mean/std:", float(cal.mean()), float(cal.std()))

    for rx in range(2):
        raw = phase_diff_tx_raw[:, :, rx]
        cal = phase_diff_tx_calibrated[:, :, rx]
        print(f"[TX-PHASE] rx{rx} raw mean/std:", float(raw.mean()), float(raw.std()))
        print(f"[TX-PHASE] rx{rx} cal mean/std:", float(cal.mean()), float(cal.std()))

if __name__ == "__main__":
    main()
