#!/usr/bin/env python3
from gnuradio import gr, blocks, analog, uhd
import time
import signal
import sys

class tx_tone_ch1(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self, "tx_tone_ch1")

        samp_rate = 5e6
        freq = 5890e6
        gain = 0.75
        tone = 100e3

        self.src = analog.sig_source_c(
            samp_rate,
            analog.GR_COS_WAVE,
            tone,
            0.2,
            0
        )

        self.sink = uhd.usrp_sink(
            ",".join(("", "")),
            uhd.stream_args(
                cpu_format="fc32",
                args="",
                channels=[0],
            ),
            "",
        )

        self.sink.set_samp_rate(samp_rate)
        self.sink.set_time_now(uhd.time_spec(time.time()), uhd.ALL_MBOARDS)
        self.sink.set_center_freq(freq, 0)
        self.sink.set_normalized_gain(gain, 0)

        try:
            self.sink.set_antenna("TX/RX", 0)
        except Exception as e:
            print("[WARN] set_antenna failed:", e)

        self.connect(self.src, self.sink)

        print("[TX-TONE-CH0] freq:", freq)
        print("[TX-TONE-CH0] samp_rate:", samp_rate)
        print("[TX-TONE-CH0] tone:", tone)
        print("[TX-TONE-CH0] physical channel: [0], logical channel: 0")

def main():
    tb = tx_tone_ch1()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    print("[TX-TONE-CH0] running")
    signal.pause()

if __name__ == "__main__":
    main()
