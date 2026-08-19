#!/usr/bin/env python3

import os
import sys
import signal

from packaging.version import Version as StrictVersion
from PyQt5 import Qt, QtCore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wifi_tx_dualport_ch1 import wifi_tx as wifi_tx_base
from tdm_packet_router import tdm_packet_router


class wifi_tx_tdm_2x2(wifi_tx_base):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("WiFi 2x2 TDM TX")

        #
        # Remove old fixed-TX1 routing:
        #
        # foo_packet_pad2 -> zero -> UHD TX0
        # foo_packet_pad2 --------> UHD TX1
        #
        self.disconnect(
            (self.foo_packet_pad2_0, 0),
            (self.blocks_multiply_const_zero_tx0, 0),
        )

        self.disconnect(
            (self.blocks_multiply_const_zero_tx0, 0),
            (self.uhd_usrp_sink_0, 0),
        )

        self.disconnect(
            (self.foo_packet_pad2_0, 0),
            (self.uhd_usrp_sink_0, 1),
        )

        #
        # True packet-level alternating TDM router.
        #
        self.tdm_router = tdm_packet_router("packet_len")

        self.connect(
            (self.foo_packet_pad2_0, 0),
            (self.tdm_router, 0),
        )

        self.connect(
            (self.tdm_router, 0),
            (self.uhd_usrp_sink_0, 0),
        )

        self.connect(
            (self.tdm_router, 1),
            (self.uhd_usrp_sink_0, 1),
        )

        #
        # Both physical TX chains enabled with identical gain.
        #
        self.uhd_usrp_sink_0.set_normalized_gain(
            self.tx_gain, 0
        )

        self.uhd_usrp_sink_0.set_normalized_gain(
            self.tx_gain, 1
        )

        print("[TDM-2X2] True alternating packet TDM enabled")
        print("[TDM-2X2] even packet -> physical TX0")
        print("[TDM-2X2] odd  packet -> physical TX1")
        print("[TDM-2X2] UHD channels=[0,1]")
        print("[TDM-2X2] gain TX0 =", self.tx_gain)
        print("[TDM-2X2] gain TX1 =", self.tx_gain)

    def set_tx_gain(self, tx_gain):
        #
        # Override base implementation because the old TX1-only
        # application intentionally forced TX0 gain to zero.
        #
        self.tx_gain = tx_gain

        if hasattr(self, "_tx_gain_callback"):
            self._tx_gain_callback(self.tx_gain)

        self.uhd_usrp_sink_0.set_normalized_gain(
            self.tx_gain, 0
        )

        self.uhd_usrp_sink_0.set_normalized_gain(
            self.tx_gain, 1
        )


def main():

    if (
        StrictVersion("4.5.0")
        <= StrictVersion(Qt.qVersion())
        < StrictVersion("5.0.0")
    ):
        style = "raster"
        Qt.QApplication.setGraphicsSystem(style)

    qapp = Qt.QApplication(sys.argv)

    tb = wifi_tx_tdm_2x2()

    tb.start()
    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()


if __name__ == "__main__":
    main()
