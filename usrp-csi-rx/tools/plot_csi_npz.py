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
    out_dir = Path(args.out_dir) if args.out_dir else path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(path)
    H = d["H_raw"][:, :, 0, 0]

    amp = np.abs(H)
    phase = np.angle(H)
    phase_unwrapped = np.unwrap(phase, axis=1)

    plt.figure()
    plt.plot(amp.mean(axis=0))
    plt.xlabel("Subcarrier index")
    plt.ylabel("Mean |H|")
    plt.title("CSI amplitude over subcarriers")
    out1 = out_dir / "csi_amp_subcarrier.png"
    plt.savefig(out1, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(phase_unwrapped.mean(axis=0))
    plt.xlabel("Subcarrier index")
    plt.ylabel("Mean unwrapped phase")
    plt.title("CSI phase over subcarriers")
    out2 = out_dir / "csi_phase_subcarrier.png"
    plt.savefig(out2, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(amp.mean(axis=1))
    plt.xlabel("Frame index")
    plt.ylabel("Mean |H|")
    plt.title("CSI amplitude over time")
    out3 = out_dir / "csi_amp_time.png"
    plt.savefig(out3, dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(phase_unwrapped.mean(axis=1))
    plt.xlabel("Frame index")
    plt.ylabel("Mean unwrapped phase")
    plt.title("CSI phase over time")
    out4 = out_dir / "csi_phase_time.png"
    plt.savefig(out4, dpi=160, bbox_inches="tight")
    plt.close()

    print("[OK] plots:")
    print(out1)
    print(out2)
    print(out3)
    print(out4)
    print("[CSI] H shape:", H.shape)

if __name__ == "__main__":
    main()
