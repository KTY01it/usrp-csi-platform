#!/usr/bin/env python3

import pmt
from gnuradio import gr


class mac_seq_tap(gr.basic_block):
    """
    Read decoded IEEE 802.11 MAC PDU and extract
    the 12-bit Sequence Number from Sequence Control.

    For a normal 24-byte data MAC header:
      bytes 22..23 = Sequence Control, little endian
      seq = bits 4..15
    """

    def __init__(self, rx_chan, callback):
        gr.basic_block.__init__(
            self,
            name=f"mac_seq_tap_rx{rx_chan}",
            in_sig=None,
            out_sig=None,
        )

        self.rx_chan = int(rx_chan)
        self.callback = callback

        self.message_port_register_in(
            pmt.intern("in")
        )

        self.set_msg_handler(
            pmt.intern("in"),
            self._handle,
        )

    def _handle(self, msg):
        try:
            data = pmt.cdr(msg)

            if not pmt.is_u8vector(data):
                return

            raw = bytes(
                bytearray(
                    pmt.u8vector_elements(data)
                )
            )

            if len(raw) < 24:
                return

            seq_ctrl = (
                raw[22]
                | (raw[23] << 8)
            )

            seq = (
                seq_ctrl >> 4
            ) & 0x0FFF

            print(
                f"[MAC-SEQ] "
                f"RX{self.rx_chan} "
                f"seq={seq}"
            )

            self.callback(seq)

        except Exception as e:
            print(
                f"[MAC-SEQ] RX{self.rx_chan} "
                f"error={e}"
            )
