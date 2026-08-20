#!/usr/bin/env python3

import os
import numpy as np
import pmt
from gnuradio import gr


class tdm_packet_router(gr.sync_block):
    """
    Tagged WiFi stream -> two synchronous TX streams.

    packet 0: TX0=data, TX1=zero
    packet 1: TX0=zero, TX1=data
    packet 2: TX0=data, TX1=zero
    ...

    packet_len and all other input tags are copied to BOTH outputs.
    This keeps both UHD channels time-aligned and tag-aligned.
    """

    def __init__(self, length_tag_key="packet_len"):
        gr.sync_block.__init__(
            self,
            name="tdm_packet_router",
            in_sig=[np.complex64],
            out_sig=[np.complex64, np.complex64],
        )

        self.length_tag_key = pmt.intern(length_tag_key)
        self.tx_slot_key = pmt.intern("tx_slot")
        self.tdm_cycle_key = pmt.intern("tdm_cycle")

        self.packet_index = 0
        self.current_slot = 0

        # We copy tags ourselves to both outputs.
        self.set_tag_propagation_policy(gr.TPP_DONT)

        # Reduce Python scheduler overhead.
        # At 5 MS/s, small scheduler calls can starve UHD.
        self.set_output_multiple(65536)

        self.debug_limit = 20

        active = os.environ.get(
            "TDM_ACTIVE_TX",
            "both"
        ).strip().lower()

        if active not in ("both", "0", "1"):
            raise ValueError(
                "TDM_ACTIVE_TX must be one of: both, 0, 1"
            )

        self.active_tx = active

        print(
            f"[TDM-ROUTER] active physical TX mode = "
            f"{self.active_tx}"
        )

    def work(self, input_items, output_items):
        x = input_items[0]
        tx0 = output_items[0]
        tx1 = output_items[1]

        n = len(x)

        if n == 0:
            return 0

        abs_start = self.nitems_read(0)
        abs_end = abs_start + n

        # Start with both RF streams zero.
        tx0[:n] = 0
        tx1[:n] = 0

        #
        # Get every input tag in this scheduler window.
        #
        tags = self.get_tags_in_range(
            0,
            abs_start,
            abs_end,
        )

        #
        # packet_len tags define packet starts.
        #
        packet_tags = [
            t for t in tags
            if pmt.eq(t.key, self.length_tag_key)
        ]

        packet_tags.sort(key=lambda t: t.offset)

        #
        # Build segments:
        # current slot is active until a new packet_len tag appears.
        #
        cursor = 0
        slot = self.current_slot

        for tag in packet_tags:
            rel = int(tag.offset - abs_start)

            # Portion before this new packet start belongs
            # to the previous packet/slot.
            if rel > cursor:
                if slot == 0:
                    if self.active_tx in ("both", "0"):
                        tx0[cursor:rel] = x[cursor:rel]
                else:
                    if self.active_tx in ("both", "1"):
                        tx1[cursor:rel] = x[cursor:rel]

            #
            # New packet starts here.
            #
            slot = self.packet_index & 1
            cycle = self.packet_index // 2

            if (
                self.packet_index < self.debug_limit
                or self.packet_index % 100 == 0
            ):
                print(
                    f"[TDM-ROUTER] packet={self.packet_index} "
                    f"cycle={cycle} slot=TX{slot}"
                )

            #
            # Add explicit TDM metadata to BOTH output streams.
            #
            for port in (0, 1):
                out_off = self.nitems_written(port) + rel

                self.add_item_tag(
                    port,
                    out_off,
                    self.tx_slot_key,
                    pmt.from_long(slot),
                )

                self.add_item_tag(
                    port,
                    out_off,
                    self.tdm_cycle_key,
                    pmt.from_long(cycle),
                )

            self.packet_index += 1
            cursor = rel

        #
        # Remaining samples belong to the most recent slot.
        #
        if cursor < n:
            if slot == 0:
                if self.active_tx in ("both", "0"):
                    tx0[cursor:n] = x[cursor:n]
            else:
                if self.active_tx in ("both", "1"):
                    tx1[cursor:n] = x[cursor:n]

        self.current_slot = slot

        #
        # Copy original stream tags to BOTH outputs.
        # This includes packet_len, which UHD needs.
        #
        for tag in tags:
            rel = int(tag.offset - abs_start)

            if rel < 0 or rel >= n:
                continue

            for port in (0, 1):
                out_off = self.nitems_written(port) + rel

                self.add_item_tag(
                    port,
                    out_off,
                    tag.key,
                    tag.value,
                    tag.srcid,
                )

        return n
