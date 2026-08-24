#!/usr/bin/env python3
from gnuradio import gr, blocks, uhd
import numpy as np
import time
import argparse

class rx_power_probe(gr.top_block):
    def __init__(self, seconds=3.0):
        gr.top_block.__init__(self, "rx_power_probe")

        self.seconds = seconds
        self.samp_rate = 5e6
        self.freq = 5890e6
        self.gain = 0.75
        self.nsamp = int(self.samp_rate * self.seconds)

        self.src = uhd.usrp_source(
            ",".join(("", "")),
            uhd.stream_args(
                cpu_format="fc32",
                args="",
                channels=[0, 1],
            ),
            True,
        )
        self.src.set_samp_rate(self.samp_rate)
        self.src.set_time_unknown_pps(uhd.time_spec(0))

        for ch in [0, 1]:
            self.src.set_center_freq(self.freq, ch)
            self.src.set_normalized_gain(self.gain, ch)
            try:
                self.src.set_antenna("RX2", ch)
            except Exception:
                pass

        self.head0 = blocks.head(gr.sizeof_gr_complex, self.nsamp)
        self.head1 = blocks.head(gr.sizeof_gr_complex, self.nsamp)

        self.sink0 = blocks.vector_sink_c()
        self.sink1 = blocks.vector_sink_c()

        self.connect((self.src, 0), self.head0, self.sink0)
        self.connect((self.src, 1), self.head1, self.sink1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()

    tb = rx_power_probe(seconds=args.seconds)
    tb.run()

    x0 = np.array(tb.sink0.data(), dtype=np.complex64)
    x1 = np.array(tb.sink1.data(), dtype=np.complex64)

    def stats(x):
        p = np.abs(x) ** 2
        return {
            "n": x.size,
            "abs_mean": float(np.abs(x).mean()) if x.size else 0.0,
            "power_mean": float(p.mean()) if x.size else 0.0,
            "power_db": float(10 * np.log10(p.mean() + 1e-15)) if x.size else -999.0,
            "abs_max": float(np.abs(x).max()) if x.size else 0.0,
        }

    print("[RX-POWER] RX0:", stats(x0))
    print("[RX-POWER] RX1:", stats(x1))

if __name__ == "__main__":
    main()
