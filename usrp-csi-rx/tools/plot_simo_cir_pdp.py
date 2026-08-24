#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cir_npz")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    path = Path(args.cir_npz)
    out_dir = Path(args.out_dir) if args.out_dir else path.parent / "plots_cir_pdp"
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(path)
    h_cir = d["h_cir"]          # [frame, delay, rx, tx]
    pdp = d["pdp"]              # [frame, delay, rx, tx]
    pdp_mean = d["pdp_mean"]    # [delay, rx, tx]

    plt.figure()
    plt.plot(pdp_mean[:, 0, 0], label="RX0")
    plt.plot(pdp_mean[:, 1, 0], label="RX1")
    plt.xlabel("Delay bin")
    plt.ylabel("Mean PDP")
    plt.title("SIMO mean PDP by RX")
    plt.legend()
    out1 = out_dir / "simo_pdp_mean_by_rx.png"
    plt.savefig(out1, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(pdp[:, :, 0, 0], aspect="auto", origin="lower")
    plt.xlabel("Delay bin")
    plt.ylabel("Frame index")
    plt.title("PDP over time - RX0")
    plt.colorbar(label="Power")
    out2 = out_dir / "simo_pdp_time_rx0.png"
    plt.savefig(out2, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.imshow(pdp[:, :, 1, 0], aspect="auto", origin="lower")
    plt.xlabel("Delay bin")
    plt.ylabel("Frame index")
    plt.title("PDP over time - RX1")
    plt.colorbar(label="Power")
    out3 = out_dir / "simo_pdp_time_rx1.png"
    plt.savefig(out3, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(np.abs(h_cir[:, :, 0, 0]).mean(axis=1), label="RX0")
    plt.plot(np.abs(h_cir[:, :, 1, 0]).mean(axis=1), label="RX1")
    plt.xlabel("Frame index")
    plt.ylabel("Mean |CIR|")
    plt.title("Mean CIR magnitude over time")
    plt.legend()
    out4 = out_dir / "simo_cir_magnitude_time.png"
    plt.savefig(out4, dpi=160, bbox_inches="tight")
    plt.close()

    print("[OK] plots:")
    print(out1)
    print(out2)
    print(out3)
    print(out4)

if __name__ == "__main__":
    main()
