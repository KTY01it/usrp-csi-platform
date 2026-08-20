#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import numpy as np


def load_channel(root, ch):
    meta_path = root / f"csi_pdu_ch{ch}.jsonl"
    bin_path = root / f"csi_pdu_ch{ch}.bin"

    records = []

    with open(meta_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    H = np.fromfile(
        bin_path,
        dtype=np.complex64,
    )

    if len(H) % 52 != 0:
        raise RuntimeError(
            f"{bin_path}: CSI size is not divisible by 52"
        )

    H = H.reshape(-1, 52)

    if len(records) != len(H):
        raise RuntimeError(
            f"RX{ch}: metadata={len(records)}, CSI={len(H)}"
        )

    out = {}

    for rec, h in zip(records, H):
        seq = int(rec["seq"])

        if seq in out:
            raise RuntimeError(
                f"RX{ch}: duplicate seq {seq}"
            )

        out[seq] = {
            "meta": rec,
            "H": h,
        }

    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "capture_dir",
        type=Path,
    )

    ap.add_argument(
        "--snr-min",
        type=float,
        default=10.0,
    )

    ap.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = ap.parse_args()

    root = args.capture_dir

    rx0 = load_channel(root, 0)
    rx1 = load_channel(root, 1)

    common = sorted(
        set(rx0) & set(rx1)
    )

    # Quality gate.
    valid = []

    for seq in common:
        s0 = float(
            rx0[seq]["meta"].get("snr", -999)
        )
        s1 = float(
            rx1[seq]["meta"].get("snr", -999)
        )

        if s0 >= args.snr_min and s1 >= args.snr_min:
            valid.append(seq)

    if not valid:
        raise RuntimeError(
            "No valid common CSI packets"
        )

    #
    # Find adjacent even/odd sequence pairs.
    #
    # For now:
    #   tensor[..., tx=0] := even-sequence slot
    #   tensor[..., tx=1] := odd-sequence slot
    #
    # These are LOGICAL TDM slots, not yet calibrated
    # to physical USRP TX0/TX1.
    #
    valid_set = set(valid)

    cycles = []

    for even_seq in valid:
        if even_seq % 2 != 0:
            continue

        odd_seq = (even_seq + 1) % 4096

        if odd_seq not in valid_set:
            continue

        cycles.append(
            (even_seq, odd_seq)
        )

    if not cycles:
        raise RuntimeError(
            "No complete even/odd TDM cycles found"
        )

    H = np.zeros(
        (
            len(cycles),
            52,
            2,
            2,
        ),
        dtype=np.complex64,
    )

    seq_table = np.zeros(
        (len(cycles), 2),
        dtype=np.int32,
    )

    snr = np.zeros(
        (
            len(cycles),
            2,
            2,
        ),
        dtype=np.float32,
    )

    cfo = np.zeros(
        (
            len(cycles),
            2,
            2,
        ),
        dtype=np.float32,
    )

    for n, (seq_even, seq_odd) in enumerate(cycles):

        seq_table[n] = [
            seq_even,
            seq_odd,
        ]

        # Logical TX slot 0 = even seq
        H[n, :, 0, 0] = rx0[seq_even]["H"]
        H[n, :, 1, 0] = rx1[seq_even]["H"]

        # Logical TX slot 1 = odd seq
        H[n, :, 0, 1] = rx0[seq_odd]["H"]
        H[n, :, 1, 1] = rx1[seq_odd]["H"]

        for rx in (0, 1):
            src = rx0 if rx == 0 else rx1

            for txslot, seq in enumerate(
                (seq_even, seq_odd)
            ):
                m = src[seq]["meta"]

                snr[n, rx, txslot] = float(
                    m.get("snr", np.nan)
                )

                cfo[n, rx, txslot] = float(
                    m.get(
                        "frequency_offset",
                        np.nan,
                    )
                )

    if args.output is None:
        args.output = (
            root
            / "H_raw_tdm_logical_slots.npz"
        )

    np.savez_compressed(
        args.output,
        H_raw=H,
        seq=seq_table,
        snr=snr,
        cfo_hz=cfo,
        fc_hz=np.float64(5.89e9),
        nsub=np.int32(52),
        rx_count=np.int32(2),
        tx_slot_count=np.int32(2),
        tx_slot_mapping=np.array(
            [
                "logical_even_seq",
                "logical_odd_seq",
            ]
        ),
        physical_tx_mapping_valid=np.bool_(False),
    )

    print("capture:", root)
    print("common packets:", len(common))
    print("quality-valid packets:", len(valid))
    print("complete TDM cycles:", len(cycles))
    print("H_raw shape:", H.shape)
    print("sequences:")
    print(seq_table)
    print("output:", args.output)

    print()
    print(
        "WARNING: tx dimension currently means "
        "[even-seq slot, odd-seq slot], NOT yet "
        "physical [TX0, TX1]."
    )


if __name__ == "__main__":
    main()
