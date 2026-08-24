#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file")
    args = parser.parse_args()

    path = Path(args.log_file)
    text = path.read_text(errors="ignore")

    seqs = [int(x) for x in re.findall(r"seq nr:\s*(\d+)", text)]
    fers = [float(x) for x in re.findall(r"instantaneous fer:\s*([0-9.]+)", text)]

    print(f"[RX-LOG] file: {path}")
    print(f"[RX-LOG] decoded_frames: {len(seqs)}")

    if seqs:
        missing = []
        for a, b in zip(seqs, seqs[1:]):
            if b > a + 1:
                missing.extend(range(a + 1, b))
        print(f"[RX-LOG] first_seq: {seqs[0]}")
        print(f"[RX-LOG] last_seq: {seqs[-1]}")
        print(f"[RX-LOG] seq_gaps: {len(missing)}")
        print(f"[RX-LOG] missing_seq_preview: {missing[:20]}")

    if fers:
        print(f"[RX-LOG] fer_samples: {len(fers)}")
        print(f"[RX-LOG] fer_mean: {sum(fers) / len(fers):.4f}")
        print(f"[RX-LOG] fer_min: {min(fers):.4f}")
        print(f"[RX-LOG] fer_max: {max(fers):.4f}")

    if "Ousrp_source" in text or "overflow" in text.lower():
        print("[RX-LOG] warning: overflow detected")
    else:
        print("[RX-LOG] overflow: none detected")

if __name__ == "__main__":
    main()
