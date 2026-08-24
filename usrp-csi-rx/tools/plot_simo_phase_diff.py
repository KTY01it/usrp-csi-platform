#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_file")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    path = Path(args.npz_file)
    out_dir = Path(args.out_dir) if args.out_dir else path.parent / "plots_simo"
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(path)
    H = d["H_raw"]
    phase_diff = d["phase_diff_rx"]
    amp_ratio = d["amp_ratio_rx"]

    H0 = H[:, :, 0, 0]
    H1 = H[:, :, 1, 0]

    plt.figure()
    plt.plot(np.abs(H0).mean(axis=0), label="RX0")
    plt.plot(np.abs(H1).mean(axis=0), label="RX1")
    plt.xlabel("Subcarrier index")
    plt.ylabel("Mean |H|")
    plt.title("SIMO CSI amplitude by RX")
    plt.legend()
    out1 = out_dir / "simo_amp_by_rx.png"
    plt.savefig(out1, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(np.unwrap(phase_diff, axis=1).mean(axis=0))
    plt.xlabel("Subcarrier index")
    plt.ylabel("Mean unwrapped phase diff")
    plt.title("RX phase difference over subcarriers")
    out2 = out_dir / "simo_phase_diff_subcarrier.png"
    plt.savefig(out2, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(np.unwrap(phase_diff, axis=1).mean(axis=1))
    plt.xlabel("Frame index")
    plt.ylabel("Mean unwrapped phase diff")
    plt.title("RX phase difference over time")
    out3 = out_dir / "simo_phase_diff_time.png"
    plt.savefig(out3, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(amp_ratio.mean(axis=1))
    plt.xlabel("Frame index")
    plt.ylabel("Mean |H1| / |H0|")
    plt.title("RX amplitude ratio over time")
    out4 = out_dir / "simo_amp_ratio_time.png"
    plt.savefig(out4, dpi=160, bbox_inches="tight")
    plt.close()

    print("[OK] plots:")
    print(out1)
    print(out2)
    print(out3)
    print(out4)
    print("[SIMO] H shape:", H.shape)

if __name__ == "__main__":
    main()
