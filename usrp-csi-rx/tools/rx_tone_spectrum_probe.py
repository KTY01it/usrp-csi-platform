#!/usr/bin/env python3
from gnuradio import gr, blocks, uhd
import numpy as np
import argparse

class rx_tone_spectrum_probe(gr.top_block):
    def __init__(self, seconds=0.5):
        gr.top_block.__init__(self, "rx_tone_spectrum_probe")

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

def analyze(x, fs):
    x = np.asarray(x, dtype=np.complex64)
    if x.size == 0:
        return {"n": 0}

    # Limit FFT size to keep analysis stable and fast.
    nfft = min(262144, x.size)
    x = x[:nfft]

    w = np.hanning(nfft).astype(np.float32)
    X = np.fft.fftshift(np.fft.fft(x * w))
    p = np.abs(X) ** 2 + 1e-20
    p_db = 10 * np.log10(p)

    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))

    peak_idx = int(np.argmax(p_db))
    peak_freq = float(freqs[peak_idx])
    peak_db = float(p_db[peak_idx])
    median_db = float(np.median(p_db))
    peak_over_median = peak_db - median_db

    # Check local power near expected +100 kHz tone.
    target = 100e3
    bw = 20e3
    mask = np.abs(freqs - target) <= bw
    if np.any(mask):
        tone_bin_db = float(np.max(p_db[mask]))
        tone_over_median = tone_bin_db - median_db
    else:
        tone_bin_db = -999.0
        tone_over_median = -999.0

    return {
        "n": int(x.size),
        "abs_mean": float(np.abs(x).mean()),
        "power_db": float(10 * np.log10(np.mean(np.abs(x) ** 2) + 1e-15)),
        "peak_freq_hz": peak_freq,
        "peak_db": peak_db,
        "median_db": median_db,
        "peak_over_median_db": float(peak_over_median),
        "tone_100k_peak_db": tone_bin_db,
        "tone_100k_over_median_db": float(tone_over_median),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=0.5)
    args = parser.parse_args()

    tb = rx_tone_spectrum_probe(seconds=args.seconds)
    tb.run()

    x0 = np.array(tb.sink0.data(), dtype=np.complex64)
    x1 = np.array(tb.sink1.data(), dtype=np.complex64)

    print("[RX-TONE] RX0:", analyze(x0, tb.samp_rate))
    print("[RX-TONE] RX1:", analyze(x1, tb.samp_rate))

if __name__ == "__main__":
    main()
