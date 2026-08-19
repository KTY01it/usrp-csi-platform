#!/usr/bin/env python3

import numpy as np
import pmt
from gnuradio import gr


class tdm_packet_router(gr.tagged_stream_block):
    """
    One tagged WiFi packet in.
    Two equal-length tagged streams out.

    packet 0 -> TX0=data, TX1=0
    packet 1 -> TX0=0,    TX1=data
    packet 2 -> TX0=data, TX1=0
    ...

    The block preserves packet timing/length on both UHD ports.
    """

    def __init__(self, length_tag_key="packet_len"):
        gr.tagged_stream_block.__init__(
            self,
            name="tdm_packet_router",
            in_sig=[np.complex64],
            out_sig=[np.complex64, np.complex64],
            length_tag_key=length_tag_key,
        )

        self.packet_index = 0
        self.slot_key = pmt.intern("tx_slot")
        self.cycle_key = pmt.intern("tdm_cycle")

    def calculate_output_stream_length(self, ninput_items):
        return int(ninput_items[0])

    def work(self, input_items, output_items):
        x = input_items[0]
        tx0 = output_items[0]
        tx1 = output_items[1]

        n = len(x)

        tx0[:n] = 0
        tx1[:n] = 0

        slot = self.packet_index & 1
        cycle = self.packet_index // 2

        if slot == 0:
            tx0[:n] = x
        else:
            tx1[:n] = x

        # Metadata tags on BOTH synchronous output streams.
        for port in (0, 1):
            off = self.nitems_written(port)

            self.add_item_tag(
                port,
                off,
                self.slot_key,
                pmt.from_long(slot),
            )

            self.add_item_tag(
                port,
                off,
                self.cycle_key,
                pmt.from_long(cycle),
            )

        self.packet_index += 1

        return n
