#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import numpy as np


def load_channel(root, ch):
    meta_path = root / f"csi_pdu_ch{ch}.jsonl"
    bin_path = root / f"csi_pdu_ch{ch}.bin"
    spatial_bin_path = (
        root / f"csi_spatial_pdu_ch{ch}.bin"
    )

    records = []

    with open(meta_path) as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(
                    json.loads(line)
                )

    H = np.fromfile(
        bin_path,
        dtype=np.complex64,
    )

    if H.size % 52 != 0:
        raise RuntimeError(
            f"{bin_path}: CSI size is not divisible by 52"
        )

    H = H.reshape(-1, 52)

    if len(records) != len(H):
        raise RuntimeError(
            f"RX{ch}: metadata={len(records)}, "
            f"CSI={len(H)}"
        )

    #
    # --------------------------------------------------------
    # Optional pre-beta spatial CSI
    # --------------------------------------------------------
    #
    spatial_vectors = None

    if spatial_bin_path.exists():

        spatial_flat = np.fromfile(
            spatial_bin_path,
            dtype=np.complex64,
        )

        if spatial_flat.size % 52 != 0:
            raise RuntimeError(
                f"{spatial_bin_path}: spatial CSI size "
                "is not divisible by 52"
            )

        spatial_vectors = spatial_flat.reshape(
            -1,
            52,
        )

        expected_spatial = sum(
            bool(
                rec.get(
                    "spatial_csi_present",
                    False,
                )
            )
            for rec in records
        )

        if (
            len(spatial_vectors)
            != expected_spatial
        ):
            raise RuntimeError(
                f"RX{ch}: spatial metadata="
                f"{expected_spatial}, "
                f"spatial CSI="
                f"{len(spatial_vectors)}"
            )

    #
    # --------------------------------------------------------
    # Unwrap 12-bit MAC sequence number
    # --------------------------------------------------------
    #
    # Raw IEEE 802.11 MAC sequence:
    #
    #   0 ... 4095 -> 0 ... 4095
    #
    # Long captures can therefore contain the same raw seq
    # more than once.  Use an epoch counter to obtain a
    # monotonically increasing packet identity:
    #
    #   raw:       4094 4095    0    1
    #   unwrapped: 4094 4095 4096 4097
    #
    # Because 4096 is even, even/odd TDM parity is preserved.
    #
    seq_unwrapped = []

    epoch = 0
    prev_raw = None

    for rec in records:

        raw_seq = int(
            rec["seq"]
        ) & 0x0FFF

        if prev_raw is not None:

            delta_raw = (
                raw_seq
                - prev_raw
            )

            #
            # Forward 12-bit wrap:
            #
            # e.g. 4095 -> 0
            #
            if delta_raw < -2048:
                epoch += 4096

        unwrapped = (
            raw_seq
            + epoch
        )

        seq_unwrapped.append(
            int(unwrapped)
        )

        prev_raw = raw_seq

    #
    # Sanity check.
    #
    # Small packet losses are fine.  What we must not have
    # after unwrapping is an exact duplicate packet identity.
    #
    if (
        len(set(seq_unwrapped))
        != len(seq_unwrapped)
    ):
        raise RuntimeError(
            f"RX{ch}: duplicate unwrapped sequence "
            "identity remains after 12-bit unwrap"
        )

    out = {}
    spatial_index = 0

    for rec, h, seq_u in zip(
        records,
        H,
        seq_unwrapped,
    ):

        raw_seq = int(
            rec["seq"]
        ) & 0x0FFF

        h_spatial = None

        if bool(
            rec.get(
                "spatial_csi_present",
                False,
            )
        ):

            if spatial_vectors is None:
                raise RuntimeError(
                    f"RX{ch}: seq={raw_seq} "
                    "metadata says spatial CSI is "
                    "present but spatial binary "
                    "file is missing"
                )

            h_spatial = spatial_vectors[
                spatial_index
            ]

            spatial_index += 1

        out[int(seq_u)] = {
            "meta": rec,
            "H": h,
            "H_spatial": h_spatial,
            "seq_raw": int(raw_seq),
            "seq_unwrapped": int(seq_u),
        }

    if spatial_vectors is not None:

        if spatial_index != len(
            spatial_vectors
        ):
            raise RuntimeError(
                f"RX{ch}: consumed spatial CSI="
                f"{spatial_index}, available="
                f"{len(spatial_vectors)}"
            )

    if out:
        keys = sorted(out)

        print(
            f"RX{ch} sequence range: "
            f"{keys[0]} -> {keys[-1]} "
            f"(unwrapped)"
        )

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
    # Physical TDM mapping established by TX0-only
    # calibration:
    #
    #   even MAC sequence -> physical TX0
    #   odd  MAC sequence -> physical TX1
    #
    valid_set = set(valid)

    cycles = []

    for even_seq in valid:
        if even_seq % 2 != 0:
            continue

        odd_seq = even_seq + 1

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

    #
    # Pre-beta spatial CSI tensor.
    #
    # Same physical layout as H_raw:
    #   [cycle, subcarrier, rx, tx]
    #
    H_spatial = np.full(
        (
            len(cycles),
            52,
            2,
            2,
        ),
        np.complex64(
            np.nan + 1j * np.nan
        ),
        dtype=np.complex64,
    )

    spatial_packet_valid = np.zeros(
        (
            len(cycles),
            2,
            2,
        ),
        dtype=bool,
    )

    seq_table = np.zeros(
        (len(cycles), 2),
        dtype=np.int32,
    )

    #
    # Absolute packet identity across 12-bit MAC
    # sequence wraps.
    #
    seq_unwrapped_table = np.zeros(
        (len(cycles), 2),
        dtype=np.int64,
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

    #
    # Timestamp layout:
    #
    # [cycle, rx, tx_slot]
    #
    # tx_slot 0 = physical TX0 / even seq
    # tx_slot 1 = physical TX1 / odd seq
    #
    timestamp_monotonic_ns = np.full(
        (
            len(cycles),
            2,
            2,
        ),
        -1,
        dtype=np.int64,
    )

    timestamp_wall_ns = np.full(
        (
            len(cycles),
            2,
            2,
        ),
        -1,
        dtype=np.int64,
    )


    #
    # RF sample-domain frame-start index.
    #
    # Layout:
    #   [cycle, rx, tx_slot]
    #
    # Generated by sync_short from nitems_read() in the
    # continuous 5 MHz UHD input stream.
    #
    rf_sample_index = np.full(
        (
            len(cycles),
            2,
            2,
        ),
        -1,
        dtype=np.int64,
    )

    for n, (seq_even, seq_odd) in enumerate(cycles):

        #
        # Preserve historical raw 12-bit sequence field.
        #
        seq_table[n] = [
            seq_even % 4096,
            seq_odd % 4096,
        ]

        #
        # Full packet identity across sequence wraps.
        #
        seq_unwrapped_table[n] = [
            seq_even,
            seq_odd,
        ]

        # Physical TX0 = even MAC sequence
        H[n, :, 0, 0] = rx0[seq_even]["H"]
        H[n, :, 1, 0] = rx1[seq_even]["H"]

        # Physical TX1 = odd MAC sequence
        H[n, :, 0, 1] = rx0[seq_odd]["H"]
        H[n, :, 1, 1] = rx1[seq_odd]["H"]

        #
        # Pre-beta spatial CSI uses the exact same packet,
        # RX and physical-TX mapping as H_raw.
        #
        spatial_sources = (
            (0, 0, rx0[seq_even]["H_spatial"]),
            (1, 0, rx1[seq_even]["H_spatial"]),
            (0, 1, rx0[seq_odd]["H_spatial"]),
            (1, 1, rx1[seq_odd]["H_spatial"]),
        )

        for rx_idx, tx_idx, hs in spatial_sources:
            if hs is None:
                continue

            H_spatial[
                n,
                :,
                rx_idx,
                tx_idx,
            ] = hs

            spatial_packet_valid[
                n,
                rx_idx,
                tx_idx,
            ] = True

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

                timestamp_monotonic_ns[
                    n, rx, txslot
                ] = int(
                    m.get(
                        "host_monotonic_ns",
                        -1,
                    )
                )

                timestamp_wall_ns[
                    n, rx, txslot
                ] = int(
                    m.get(
                        "host_wall_time_ns",
                        -1,
                    )
                )


                rf_sample_index[
                    n, rx, txslot
                ] = int(
                    m.get(
                        "rf_sample_index",
                        -1,
                    )
                )

    #
    # --------------------------------------------------------
    # Nominal TX sequence time base
    # --------------------------------------------------------
    #
    # Physical TDM transmission currently uses one MAC packet
    # every 100 ms:
    #
    #   even seq -> physical TX0
    #   odd  seq -> physical TX1
    #
    # We unwrap the 12-bit MAC sequence number and reconstruct
    # the nominal TX scheduling time. Missing packets therefore
    # preserve elapsed nominal time instead of compressing it.
    #
    # IMPORTANT:
    # This is NOT a UHD hardware RF timestamp.
    #
    tx_packet_interval_s = 0.100

    seq_even_raw = (
        seq_table[:, 0]
        .astype(np.int64)
    )

    seq_even_unwrapped = np.empty_like(
        seq_even_raw,
        dtype=np.int64,
    )

    seq_even_unwrapped[0] = (
        seq_even_raw[0]
    )

    for n in range(
        1,
        len(seq_even_raw)
    ):
        prev_raw = (
            seq_even_raw[n - 1]
        )

        cur_raw = (
            seq_even_raw[n]
        )

        delta = (
            cur_raw
            - prev_raw
        )

        if delta < -2048:
            delta += 4096

        elif delta > 2048:
            delta -= 4096

        seq_even_unwrapped[n] = (
            seq_even_unwrapped[n - 1]
            + delta
        )

    cycle_time_nominal_s = (
        (
            seq_even_unwrapped
            - seq_even_unwrapped[0]
        )
        * tx_packet_interval_s
    ).astype(np.float64)

    packet_time_nominal_s = np.stack(
        [
            cycle_time_nominal_s,
            cycle_time_nominal_s
            + tx_packet_interval_s,
        ],
        axis=1,
    )


    #
    # --------------------------------------------------------
    # RF sample-domain time base
    # --------------------------------------------------------
    #
    # sync_short receives the UHD stream directly at 5 MHz.
    # Therefore rf_sample_index is expressed in UHD-stream
    # complex samples.
    #
    # This is a relative sample-domain RF time base.
    # It is NOT an absolute UHD hardware timestamp.
    #
    rf_sample_rate_hz = 5_000_000.0

    rf_sample_time_valid = bool(
        np.all(
            rf_sample_index >= 0
        )
    )

    if rf_sample_time_valid:

        #
        # Global relative origin from the first physical-TX0
        # packet. RX0/RX1 are retained in the same sample
        # reference so their small detection skew remains
        # observable.
        #
        rf_reference_sample_index = int(
            np.rint(
                np.median(
                    rf_sample_index[
                        0,
                        :,
                        0,
                    ]
                )
            )
        )

        rf_time_relative_s = (
            (
                rf_sample_index.astype(
                    np.float64
                )
                - rf_reference_sample_index
            )
            / rf_sample_rate_hz
        )

        #
        # One representative RF time per TDM cycle.
        # Use physical TX0 and median across the two RX chains.
        #
        rf_cycle_time_s = np.median(
            rf_time_relative_s[
                :,
                :,
                0,
            ],
            axis=1,
        )

        #
        # RX1 - RX0 frame-start skew, in input-stream samples.
        #
        rf_rx_sample_skew = (
            rf_sample_index[
                :,
                1,
                :,
            ]
            - rf_sample_index[
                :,
                0,
                :,
            ]
        ).astype(np.int64)

    else:

        rf_reference_sample_index = -1

        rf_time_relative_s = np.full(
            rf_sample_index.shape,
            np.nan,
            dtype=np.float64,
        )

        rf_cycle_time_s = np.full(
            len(cycles),
            np.nan,
            dtype=np.float64,
        )

        rf_rx_sample_skew = np.full(
            (
                len(cycles),
                2,
            ),
            0,
            dtype=np.int64,
        )

    if args.output is None:
        args.output = (
            root
            / "H_raw_tdm_physical_2x2.npz"
        )

    np.savez_compressed(
        args.output,
        H_raw=H,

        H_spatial_raw=H_spatial,

        spatial_packet_valid=
            spatial_packet_valid,

        spatial_csi_available=
            np.bool_(
                np.any(
                    spatial_packet_valid
                )
            ),

        spatial_csi_complete=
            np.bool_(
                np.all(
                    spatial_packet_valid
                )
            ),

        spatial_csi_definition=
            np.array(
                "pre_beta_L_LTF_LS_channel_estimate"
            ),

        spatial_csi_layout=
            np.array(
                "cycle_subcarrier_rx_tx"
            ),

        seq=seq_table,

        seq_unwrapped=
            seq_unwrapped_table,

        mac_sequence_modulus=
            np.int32(4096),

        mac_sequence_unwrapped_valid=
            np.bool_(True),
        snr=snr,
        cfo_hz=cfo,


        rf_sample_index=
            rf_sample_index,

        rf_sample_rate_hz=np.float64(
            rf_sample_rate_hz
        ),

        rf_reference_sample_index=np.int64(
            rf_reference_sample_index
        ),

        rf_time_relative_s=
            rf_time_relative_s,

        rf_cycle_time_s=
            rf_cycle_time_s,

        rf_rx_sample_skew=
            rf_rx_sample_skew,

        rf_sample_time_valid=np.bool_(
            rf_sample_time_valid
        ),

        sample_domain_time_valid=np.bool_(
            rf_sample_time_valid
        ),

        rf_time_source=np.array(
            "sync_short_continuous_sample_index"
        ),

        uhd_absolute_time_valid=np.bool_(
            False
        ),

        timestamp_monotonic_ns=
            timestamp_monotonic_ns,

        timestamp_wall_ns=
            timestamp_wall_ns,

        timestamp_source=np.array(
            "host_decoded_csi_handler"
        ),

        packet_time_nominal_s=
            packet_time_nominal_s,

        cycle_time_nominal_s=
            cycle_time_nominal_s,

        tx_packet_interval_s=np.float64(
            tx_packet_interval_s
        ),

        nominal_time_source=np.array(
            "mac_sequence_plus_tx_packet_interval"
        ),

        hardware_rf_timestamp_valid=np.bool_(
            False
        ),

        fc_hz=np.float64(5.89e9),
        nsub=np.int32(52),
        rx_count=np.int32(2),
        tx_slot_count=np.int32(2),
        tx_mapping=np.array(
            [
                "physical_TX0_even_seq",
                "physical_TX1_odd_seq",
            ]
        ),
        physical_tx_mapping_valid=np.bool_(True),
    )

    print("capture:", root)
    print("common packets:", len(common))
    print("quality-valid packets:", len(valid))
    print("complete TDM cycles:", len(cycles))
    print("H_raw shape:", H.shape)

    print(
        "H_spatial_raw shape:",
        H_spatial.shape,
    )

    print(
        "spatial packets valid:",
        int(
            np.sum(
                spatial_packet_valid
            )
        ),
        "/",
        int(
            spatial_packet_valid.size
        ),
    )

    print(
        "spatial CSI complete:",
        bool(
            np.all(
                spatial_packet_valid
            )
        ),
    )

    print("sequences:")
    print(seq_table)
    print("output:", args.output)

    print()
    print(
        "Physical TX mapping: "
        "tx=0 -> TX0/even-seq, "
        "tx=1 -> TX1/odd-seq"
    )


if __name__ == "__main__":
    main()
