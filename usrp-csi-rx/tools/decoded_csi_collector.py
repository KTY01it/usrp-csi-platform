#!/usr/bin/env python3

import os
import json
import numpy as np
import pmt
from gnuradio import gr


class decoded_csi_collector(gr.basic_block):
    """
    Receive CRC-valid decoded MAC PDUs.

    decode_mac preserves frame_equalizer metadata, including CSI.
    Therefore each record contains CSI and MAC sequence number
    belonging to the SAME packet.
    """

    def __init__(
        self,
        rx_chan=0,
        bin_path=None,
        meta_path=None,
    ):
        gr.basic_block.__init__(
            self,
            name=f"decoded_csi_collector_rx{rx_chan}",
            in_sig=None,
            out_sig=None,
        )

        self.rx_chan = int(rx_chan)
        self.bin_path = bin_path
        self.meta_path = meta_path

        self.key_csi = pmt.intern("csi")
        self.key_snr = pmt.intern("snr")
        self.key_freq = pmt.intern("nominal frequency")
        self.key_foff = pmt.intern("frequency offset")
        self.key_beta = pmt.intern("beta")
        self.key_encoding = pmt.intern("encoding")
        self.key_frame_bytes = pmt.intern("frame bytes")

        self.count = 0

        self.bin_fh = None
        self.meta_fh = None

        if self.bin_path:
            os.makedirs(
                os.path.dirname(self.bin_path),
                exist_ok=True,
            )
            self.bin_fh = open(
                self.bin_path,
                "wb",
            )

        if self.meta_path:
            os.makedirs(
                os.path.dirname(self.meta_path),
                exist_ok=True,
            )
            self.meta_fh = open(
                self.meta_path,
                "w",
                buffering=1,
            )

        self.message_port_register_in(
            pmt.intern("in")
        )

        self.set_msg_handler(
            pmt.intern("in"),
            self._handle,
        )

    @staticmethod
    def _number(v):
        try:
            if pmt.is_integer(v):
                return int(pmt.to_long(v))

            if pmt.is_real(v):
                return float(pmt.to_double(v))
        except Exception:
            pass

        return None

    @staticmethod
    def _payload_bytes(data):
        """
        Support both PMT blob and u8vector forms.
        """
        try:
            if pmt.is_u8vector(data):
                return bytes(
                    bytearray(
                        pmt.u8vector_elements(data)
                    )
                )
        except Exception:
            pass

        try:
            obj = pmt.to_python(data)

            if isinstance(
                obj,
                (bytes, bytearray, memoryview),
            ):
                return bytes(obj)

            # Some PMT versions expose blob as ndarray/buffer-like.
            try:
                return bytes(obj)
            except Exception:
                pass
        except Exception:
            pass

        return None

    @staticmethod
    def _seq_from_mac(raw):
        if raw is None or len(raw) < 24:
            return None

        seq_ctrl = (
            int(raw[22])
            | (int(raw[23]) << 8)
        )

        return (seq_ctrl >> 4) & 0x0FFF

    def _handle(self, msg):
        try:
            if not pmt.is_pair(msg):
                return

            meta = pmt.car(msg)
            data = pmt.cdr(msg)

            if not pmt.is_dict(meta):
                return

            raw = self._payload_bytes(data)
            seq = self._seq_from_mac(raw)

            if seq is None:
                print(
                    f"[CSI-PDU] RX{self.rx_chan} "
                    "cannot parse MAC seq"
                )
                return

            csi_pmt = pmt.dict_ref(
                meta,
                self.key_csi,
                pmt.PMT_NIL,
            )

            if csi_pmt is pmt.PMT_NIL:
                print(
                    f"[CSI-PDU] RX{self.rx_chan} "
                    f"seq={seq} missing CSI metadata"
                )
                return

            H = np.asarray(
                pmt.c32vector_elements(csi_pmt),
                dtype=np.complex64,
            )

            if H.size != 52:
                print(
                    f"[CSI-PDU] RX{self.rx_chan} "
                    f"seq={seq} invalid CSI length={H.size}"
                )
                return

            self.count += 1

            if self.bin_fh:
                H.tofile(self.bin_fh)
                self.bin_fh.flush()

            rec = {
                "idx": int(self.count),
                "seq": int(seq),
                "rx_chan": int(self.rx_chan),
                "csi_len": int(H.size),
                "abs_mean": float(
                    np.mean(np.abs(H))
                ),
            }

            mapping = (
                ("snr", self.key_snr),
                ("nominal_frequency", self.key_freq),
                ("frequency_offset", self.key_foff),
                ("beta", self.key_beta),
                ("encoding", self.key_encoding),
                ("frame_bytes", self.key_frame_bytes),
            )

            for name, key in mapping:
                val = pmt.dict_ref(
                    meta,
                    key,
                    pmt.PMT_NIL,
                )

                if val is not pmt.PMT_NIL:
                    x = self._number(val)

                    if x is not None:
                        rec[name] = x

            if self.meta_fh:
                self.meta_fh.write(
                    json.dumps(rec) + "\n"
                )

            if (
                self.count <= 30
                or self.count % 100 == 0
            ):
                print(
                    f"[CSI-PDU] "
                    f"RX{self.rx_chan} "
                    f"idx={self.count} "
                    f"seq={seq} "
                    f"mean={rec['abs_mean']:.6f} "
                    f"snr={rec.get('snr')}"
                )

        except Exception as e:
            print(
                f"[CSI-PDU] RX{self.rx_chan} "
                f"error={e}"
            )

    def stop(self):
        try:
            if self.bin_fh:
                self.bin_fh.close()
        except Exception:
            pass

        try:
            if self.meta_fh:
                self.meta_fh.close()
        except Exception:
            pass

        return True
