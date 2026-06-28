from packaging.version import Version as StrictVersion
from PyQt5 import Qt
from gnuradio import qtgui
import os
import sys
sys.path.append(os.environ.get('GRC_HIER_PATH', os.path.expanduser('~/.grc_gnuradio')))

from PyQt5.QtCore import QObject, pyqtSlot
from gnuradio import blocks
import pmt
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import network
from gnuradio import uhd
import time
from gnuradio.qtgui import Range, RangeWidget
from PyQt5 import QtCore
from wifi_phy_hier_test import wifi_phy_hier  # grc-generated hier_block
import foo
import ieee802_11
import random


class wifi_tx(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Wifi Tx", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Wifi Tx")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "wifi_tx")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        ##################################################
        # Variables
        ##################################################
        self.tx_gain = tx_gain = 0.75
        self.samp_rate = samp_rate = 5e6
        self.pdu_length = pdu_length = 500
        self.out_buf_size = out_buf_size = 96000
        self.lo_offset = lo_offset = 0
        self.interval = interval = 300
        self.freq = freq = 5180000000
        self.encoding = encoding = 0

        ##################################################
        # Blocks
        ##################################################

        self._tx_gain_range = Range(0, 1, 0.01, 0.75, 200)
        self._tx_gain_win = RangeWidget(self._tx_gain_range, self.set_tx_gain, "'tx_gain'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._tx_gain_win)
        # Create the options list
        self._samp_rate_options = [5000000.0, 10000000.0, 20000000.0]
        # Create the labels list
        self._samp_rate_labels = ['5 MHz', '10 MHz', '20 MHz']
        # Create the combo box
        self._samp_rate_tool_bar = Qt.QToolBar(self)
        self._samp_rate_tool_bar.addWidget(Qt.QLabel("'samp_rate'" + ": "))
        self._samp_rate_combo_box = Qt.QComboBox()
        self._samp_rate_tool_bar.addWidget(self._samp_rate_combo_box)
        for _label in self._samp_rate_labels: self._samp_rate_combo_box.addItem(_label)
        self._samp_rate_callback = lambda i: Qt.QMetaObject.invokeMethod(self._samp_rate_combo_box, "setCurrentIndex", Qt.Q_ARG("int", self._samp_rate_options.index(i)))
        self._samp_rate_callback(self.samp_rate)
        self._samp_rate_combo_box.currentIndexChanged.connect(
            lambda i: self.set_samp_rate(self._samp_rate_options[i]))
        # Create the radio buttons
        self.top_layout.addWidget(self._samp_rate_tool_bar)
        self._pdu_length_range = Range(0, 1500, 1, 500, 200)
        self._pdu_length_win = RangeWidget(self._pdu_length_range, self.set_pdu_length, "'pdu_length'", "counter_slider", int, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._pdu_length_win)
        # Create the options list
        self._lo_offset_options = [0, 6000000.0, 11000000.0]
        # Create the labels list
        self._lo_offset_labels = ['0', '6000000.0', '11000000.0']
        # Create the combo box
        self._lo_offset_tool_bar = Qt.QToolBar(self)
        self._lo_offset_tool_bar.addWidget(Qt.QLabel("'lo_offset'" + ": "))
        self._lo_offset_combo_box = Qt.QComboBox()
        self._lo_offset_tool_bar.addWidget(self._lo_offset_combo_box)
        for _label in self._lo_offset_labels: self._lo_offset_combo_box.addItem(_label)
        self._lo_offset_callback = lambda i: Qt.QMetaObject.invokeMethod(self._lo_offset_combo_box, "setCurrentIndex", Qt.Q_ARG("int", self._lo_offset_options.index(i)))
        self._lo_offset_callback(self.lo_offset)
        self._lo_offset_combo_box.currentIndexChanged.connect(
            lambda i: self.set_lo_offset(self._lo_offset_options[i]))
        # Create the radio buttons
        self.top_layout.addWidget(self._lo_offset_tool_bar)
        
        # self._interval_range = Range(10, 1000, 1, 300, 200)
        self._interval_range = Range(10, 1000, 1, 100, 200)  # trước là 300 ms

        self._interval_win = RangeWidget(self._interval_range, self.set_interval, "'interval'", "counter_slider", int, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._interval_win)
        # Create the options list
        self._freq_options = [2412000000.0, 2417000000.0, 2422000000.0, 2427000000.0, 2432000000.0, 2437000000.0, 2442000000.0, 2447000000.0, 2452000000.0, 2457000000.0, 2462000000.0, 2467000000.0, 2472000000.0, 2484000000.0, 5170000000.0, 5180000000.0, 5190000000.0, 5200000000.0, 5210000000.0, 5220000000.0, 5230000000.0, 5240000000.0, 5250000000.0, 5260000000.0, 5270000000.0, 5280000000.0, 5290000000.0, 5300000000.0, 5310000000.0, 5320000000.0, 5500000000.0, 5510000000.0, 5520000000.0, 5530000000.0, 5540000000.0, 5550000000.0, 5560000000.0, 5570000000.0, 5580000000.0, 5590000000.0, 5600000000.0, 5610000000.0, 5620000000.0, 5630000000.0, 5640000000.0, 5660000000.0, 5670000000.0, 5680000000.0, 5690000000.0, 5700000000.0, 5710000000.0, 5720000000.0, 5745000000.0, 5755000000.0, 5765000000.0, 5775000000.0, 5785000000.0, 5795000000.0, 5805000000.0, 5825000000.0, 5860000000.0, 5870000000.0, 5880000000.0, 5890000000.0, 5900000000.0, 5910000000.0, 5920000000.0]
        # Create the labels list
        self._freq_labels = ['  1 | 2412.0 | 11g', '  2 | 2417.0 | 11g', '  3 | 2422.0 | 11g', '  4 | 2427.0 | 11g', '  5 | 2432.0 | 11g', '  6 | 2437.0 | 11g', '  7 | 2442.0 | 11g', '  8 | 2447.0 | 11g', '  9 | 2452.0 | 11g', ' 10 | 2457.0 | 11g', ' 11 | 2462.0 | 11g', ' 12 | 2467.0 | 11g', ' 13 | 2472.0 | 11g', ' 14 | 2484.0 | 11g', ' 34 | 5170.0 | 11a', ' 36 | 5180.0 | 11a', ' 38 | 5190.0 | 11a', ' 40 | 5200.0 | 11a', ' 42 | 5210.0 | 11a', ' 44 | 5220.0 | 11a', ' 46 | 5230.0 | 11a', ' 48 | 5240.0 | 11a', ' 50 | 5250.0 | 11a', ' 52 | 5260.0 | 11a', ' 54 | 5270.0 | 11a', ' 56 | 5280.0 | 11a', ' 58 | 5290.0 | 11a', ' 60 | 5300.0 | 11a', ' 62 | 5310.0 | 11a', ' 64 | 5320.0 | 11a', '100 | 5500.0 | 11a', '102 | 5510.0 | 11a', '104 | 5520.0 | 11a', '106 | 5530.0 | 11a', '108 | 5540.0 | 11a', '110 | 5550.0 | 11a', '112 | 5560.0 | 11a', '114 | 5570.0 | 11a', '116 | 5580.0 | 11a', '118 | 5590.0 | 11a', '120 | 5600.0 | 11a', '122 | 5610.0 | 11a', '124 | 5620.0 | 11a', '126 | 5630.0 | 11a', '128 | 5640.0 | 11a', '132 | 5660.0 | 11a', '134 | 5670.0 | 11a', '136 | 5680.0 | 11a', '138 | 5690.0 | 11a', '140 | 5700.0 | 11a', '142 | 5710.0 | 11a', '144 | 5720.0 | 11a', '149 | 5745.0 | 11a (SRD)', '151 | 5755.0 | 11a (SRD)', '153 | 5765.0 | 11a (SRD)', '155 | 5775.0 | 11a (SRD)', '157 | 5785.0 | 11a (SRD)', '159 | 5795.0 | 11a (SRD)', '161 | 5805.0 | 11a (SRD)', '165 | 5825.0 | 11a (SRD)', '172 | 5860.0 | 11p', '174 | 5870.0 | 11p', '176 | 5880.0 | 11p', '178 | 5890.0 | 11p', '180 | 5900.0 | 11p', '182 | 5910.0 | 11p', '184 | 5920.0 | 11p']
        # Create the combo box
        self._freq_tool_bar = Qt.QToolBar(self)
        self._freq_tool_bar.addWidget(Qt.QLabel("'freq'" + ": "))
        self._freq_combo_box = Qt.QComboBox()
        self._freq_tool_bar.addWidget(self._freq_combo_box)
        for _label in self._freq_labels: self._freq_combo_box.addItem(_label)
        self._freq_callback = lambda i: Qt.QMetaObject.invokeMethod(self._freq_combo_box, "setCurrentIndex", Qt.Q_ARG("int", self._freq_options.index(i)))
        self._freq_callback(self.freq)
        self._freq_combo_box.currentIndexChanged.connect(
            lambda i: self.set_freq(self._freq_options[i]))
        # Create the radio buttons
        self.top_layout.addWidget(self._freq_tool_bar)
        # Create the options list
        self._encoding_options = [0, 1, 2, 3, 4, 5, 6, 7]
        # Create the labels list
        self._encoding_labels = ['BPSK 1/2', 'BPSK 3/4', 'QPSK 1/2', 'QPSK 3/4', '16QAM 1/2', '16QAM 3/4', '64QAM 2/3', '64QAM 3/4']
        # Create the combo box
        # Create the radio buttons
        self._encoding_group_box = Qt.QGroupBox("'encoding'" + ": ")
        self._encoding_box = Qt.QHBoxLayout()
        class variable_chooser_button_group(Qt.QButtonGroup):
            def __init__(self, parent=None):
                Qt.QButtonGroup.__init__(self, parent)
            @pyqtSlot(int)
            def updateButtonChecked(self, button_id):
                self.button(button_id).setChecked(True)
        self._encoding_button_group = variable_chooser_button_group()
        self._encoding_group_box.setLayout(self._encoding_box)
        for i, _label in enumerate(self._encoding_labels):
            radio_button = Qt.QRadioButton(_label)
            self._encoding_box.addWidget(radio_button)
            self._encoding_button_group.addButton(radio_button, i)
        self._encoding_callback = lambda i: Qt.QMetaObject.invokeMethod(self._encoding_button_group, "updateButtonChecked", Qt.Q_ARG("int", self._encoding_options.index(i)))
        self._encoding_callback(self.encoding)
        self._encoding_button_group.buttonClicked[int].connect(
            lambda i: self.set_encoding(self._encoding_options[i]))
        self.top_layout.addWidget(self._encoding_group_box)
        self.wifi_phy_hier_0 = wifi_phy_hier(
            bandwidth=samp_rate,
            chan_est=ieee802_11.LS,
            encoding=ieee802_11.Encoding(encoding),
            frequency=freq,
            sensitivity=0.56,
        )
        
        
        # --- Toolbar điều khiển TX (start sau 5s) ---
        self._tx_toolbar = Qt.QToolBar(self)
        self._btn_tx_start = Qt.QPushButton("Start TX (5s)")
        self._btn_tx_stop  = Qt.QPushButton("Stop TX")
        self._tx_status    = Qt.QLabel("TX: idle")

        self._tx_toolbar.addWidget(self._btn_tx_start)
        self._tx_toolbar.addWidget(self._btn_tx_stop)
        self._tx_toolbar.addSeparator()
        self._tx_toolbar.addWidget(self._tx_status)
        self.top_layout.addWidget(self._tx_toolbar)


        # --- NEW: chế độ phát antenna ---
        self.TX_ANT_MODE   = 'BOTH'   # 'ALT' | 'TX0' | 'TX1' | 'BOTH'
        self.ANT_SWITCH_MS = max(50, int(self.interval))  # mặc định bám theo interval khung

        self._ant_toolbar = Qt.QToolBar(self)
        self._ant_toolbar.addWidget(Qt.QLabel("TX Ant Mode: "))
        self._ant_combo = Qt.QComboBox()
        for _lab in ["ALT", "TX0", "TX1", "BOTH"]:
            self._ant_combo.addItem(_lab)
        self._ant_combo.setCurrentText(self.TX_ANT_MODE)
        self._ant_combo.currentTextChanged.connect(self._set_tx_ant_mode)
        self._ant_status = Qt.QLabel("Active: TX0")
        self._ant_toolbar.addWidget(self._ant_combo)
        self._ant_toolbar.addSeparator()
        self._ant_toolbar.addWidget(self._ant_status)
        self.top_layout.addWidget(self._ant_toolbar)

        self.active_tx_ant = 0

        # Timer đổi antenna
        self._txa_timer = QtCore.QTimer(self)
        self._txa_timer.setInterval(self.ANT_SWITCH_MS)
        self._txa_timer.timeout.connect(self._toggle_tx_ant)
        
        # NEW: training mode để phát theo CẶP (TX0 rồi TX1), có thể quét nhiều pha
        self.TRAIN_ENABLE = False
        self.PAIR_ID      = 0
        self.PHI_LIST_DEG = [0, 90, 180, 270]   # bạn có thể thay đổi
        self._phi_idx     = 0
        self._in_pair_half = 0  # 0: TX0 lượt 1, 1: TX1 lượt 2

        # Hiển thị nhanh trạng thái training
        self._train_toolbar = Qt.QToolBar(self)
        self._train_toolbar.addWidget(Qt.QLabel("Training: "))
        self._lbl_pair = Qt.QLabel("PAIR=0")
        self._lbl_phi  = Qt.QLabel("PHI=0°")
        self._train_toolbar.addWidget(self._lbl_pair)
        
        
        # === SWEEP / AOD index ===
        self.SWEEP_ID   = 0            # NEW: id của vòng quét AoD hiện tại
        # self._phi_idx đã có (chỉ số trong PHI_LIST_DEG)

        # (tuỳ chọn) hiển thị thêm
        self._lbl_sweep = Qt.QLabel("SWEEP=0")     # NEW
        self._lbl_aod   = Qt.QLabel("AOD=0")       # NEW
        

        self._train_toolbar.addSeparator()
        self._train_toolbar.addWidget(self._lbl_phi)
        self.top_layout.addWidget(self._train_toolbar)
        

        self._btn_tx_start.clicked.connect(self._arm_and_start_countdown)
        self._btn_tx_stop.clicked.connect(self._stop_tx)

        # Trạng thái TX & đếm ngược
        self._fg_running = False
        self._arm_countdown = 0
        self._arm_timer = QtCore.QTimer(self)
        self._arm_timer.setInterval(1000)  # 1 Hz
        self._arm_timer.timeout.connect(self._arm_tick)

        # --- SỬA: dùng 2 kênh TX ---
        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            ",".join(('', "")),
            uhd.stream_args(
                cpu_format="fc32",
                otw_format="sc8",
                args='spp=736,num_send_frames=256,send_frame_size=1472',
                channels=[0, 1],              # NEW: hai kênh
            ),
            'packet_len',
        )
        
        self.uhd_usrp_sink_0.set_samp_rate(samp_rate)
        self.uhd_usrp_sink_0.set_time_now(uhd.time_spec(time.time()), uhd.ALL_MBOARDS)

        self.uhd_usrp_sink_0.set_center_freq(
            uhd.tune_request(freq, rf_freq=freq - lo_offset, rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)
        self.uhd_usrp_sink_0.set_center_freq(
            uhd.tune_request(freq, rf_freq=freq - lo_offset, rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 1)
        self.uhd_usrp_sink_0.set_normalized_gain(tx_gain, 0)
        self.uhd_usrp_sink_0.set_normalized_gain(tx_gain, 1)     # NEW
    
        
        self.network_socket_pdu_0 = network.socket_pdu('TCP_SERVER', '', '52001', 10000, False)
        self.ieee802_11_mac_0 = ieee802_11.mac([0x23, 0x23, 0x23, 0x23, 0x23, 0x23], [0x42, 0x42, 0x42, 0x42, 0x42, 0x42], [0xff, 0xff, 0xff, 0xff, 0xff, 255])

        # self.foo_packet_pad2_0 = foo.packet_pad2(False, False, 0.01, 100, 1000)
        # self.foo_packet_pad2_0 = foo.packet_pad2(True, False, 0.01, 512, 4096)
        self.foo_packet_pad2_0 = foo.packet_pad2(True, True, 0.00, 4096, 16384)

        self.foo_packet_pad2_0.set_min_output_buffer(out_buf_size)
        self.blocks_vector_source_x_0 = blocks.vector_source_c((0,), False, 1, [])
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_cc(0.6)
        self.blocks_multiply_const_vxx_0.set_min_output_buffer(100000)
        self.blocks_message_strobe_0_0 = blocks.message_strobe(pmt.intern("".join("x" for i in range(pdu_length))), interval)
        self.uhd_usrp_sink_0.set_min_output_buffer(1_000_000)

        # --- Gating per-TX (1.0 = bật, 0.0 = tắt) ---
    # --- NEW: gate để bật/tắt từng kênh ---
        self.tx_gate0 = blocks.multiply_const_cc(1.0)  # mặc định TX0 on
        self.tx_gate1 = blocks.multiply_const_cc(1.0)  # mặc định TX1 off
        
        self.tx1_phase = blocks.multiply_const_cc(1.0+0.0j)  # e^{j*phi}, set_k() sẽ cập nhật
        
        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.blocks_message_strobe_0_0, 'strobe'), (self.ieee802_11_mac_0, 'app in'))
        self.msg_connect((self.ieee802_11_mac_0, 'phy out'), (self.wifi_phy_hier_0, 'mac_in'))
        self.msg_connect((self.network_socket_pdu_0, 'pdus'), (self.ieee802_11_mac_0, 'app in'))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.foo_packet_pad2_0, 0))
        self.connect((self.blocks_vector_source_x_0, 0), (self.wifi_phy_hier_0, 0))
        # self.connect((self.foo_packet_pad2_0, 0), (self.uhd_usrp_sink_0, 0))
        self.connect((self.wifi_phy_hier_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        
        # NEW: route qua gate → mỗi kênh riêng
        self.connect((self.foo_packet_pad2_0, 0), (self.tx_gate0, 0))
        self.connect((self.tx_gate0, 0), (self.uhd_usrp_sink_0, 0))

        # self.connect((self.foo_packet_pad2_0, 0), (self.tx_gate1, 0))
        # self.connect((self.tx_gate1, 0), (self.uhd_usrp_sink_0, 1))

        self.connect((self.foo_packet_pad2_0, 0), (self.tx1_phase, 0))            # NEW
        self.connect((self.tx1_phase, 0), (self.tx_gate1, 0))                      # NEW
        self.connect((self.tx_gate1, 0), (self.uhd_usrp_sink_0, 1))   


        # --- HOP cấu hình ---
        self.HOP_ENABLE   = True
        self.HOP_RANDOM   = False           # True: ngẫu nhiên, False: round-robin
        self.HOP_INTERVAL = 20.0            # giây, tuỳ bạn
        self.HOP_FREQS    = [5180e6, 5200e6, 5220e6, 5240e6, 5260e6, 5280e6, 5300e6, 5320e6]
        # self.HOP_FREQS    = [5180e6]
        
        self._hop_idx     = -1
        self._hop_seq     = 0

        # Timer nhảy tần
        self._hop_timer = QtCore.QTimer(self)
        self._hop_timer.setInterval(int(self.HOP_INTERVAL * 1000))
        self._hop_timer.timeout.connect(self._do_hop)
        # if self.HOP_ENABLE:
        #     self._hop_timer.start()

        # Timer đếm theo giây để căn thời gian bên thu
        self._dwell_count_sec = 0
        self._dwell_count_total = int(self.HOP_INTERVAL)  # 15 giây
        self._dwell_count_timer = QtCore.QTimer(self)
        self._dwell_count_timer.setInterval(1000)  # 1 Hz
        self._dwell_count_timer.timeout.connect(self._tick_dwell_counter)
        # self._dwell_count_timer.start()
    
    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "wifi_tx")
        self.settings.setValue("geometry", self.saveGeometry())
        try:
            self._arm_timer.stop()
            self._hop_timer.stop()
            self._dwell_count_timer.stop()
        except Exception:
            pass
        self.stop()
        self.wait()
        event.accept()


    def get_tx_gain(self):
        return self.tx_gain

    def set_tx_gain(self, tx_gain):
        self.tx_gain = tx_gain
        self.uhd_usrp_sink_0.set_normalized_gain(self.tx_gain, 0)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self._samp_rate_callback(self.samp_rate)
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)
        self.wifi_phy_hier_0.set_bandwidth(self.samp_rate)

    def get_pdu_length(self):
        return self.pdu_length

    def set_pdu_length(self, pdu_length):
        self.pdu_length = pdu_length
        self.blocks_message_strobe_0_0.set_msg(pmt.intern("".join("x" for i in range(self.pdu_length))))

    def get_out_buf_size(self):
        return self.out_buf_size

    def set_out_buf_size(self, out_buf_size):
        self.out_buf_size = out_buf_size

    def get_lo_offset(self):
        return self.lo_offset

    def set_lo_offset(self, lo_offset):
        self.lo_offset = lo_offset
        self._lo_offset_callback(self.lo_offset)
        self.uhd_usrp_sink_0.set_center_freq(uhd.tune_request(self.freq, rf_freq = self.freq - self.lo_offset, rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)

    def get_interval(self):
        return self.interval

    def set_interval(self, interval):
        self.interval = interval
        self.blocks_message_strobe_0_0.set_period(self.interval)
        try:
            self.ANT_SWITCH_MS = max(50, int(self.interval))
            self._txa_timer.setInterval(self.ANT_SWITCH_MS)
        except Exception:
            pass


    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        prev = getattr(self, "_prev_freq", None)
        self.freq = freq
        self._freq_callback(self.freq)
        req = uhd.tune_request(self.freq, rf_freq=self.freq - self.lo_offset,
                            rf_freq_policy=uhd.tune_request.POLICY_MANUAL)
        self.uhd_usrp_sink_0.set_center_freq(req, 0)
        self.uhd_usrp_sink_0.set_center_freq(req, 1)     # NEW
        self.wifi_phy_hier_0.set_frequency(self.freq)

        # NEW: nếu tần số đổi → reset vòng sweep
        if prev is None or prev != self.freq:
            self.SWEEP_ID += 1
            self.PAIR_ID = 0
            self._phi_idx = 0
            self._lbl_sweep.setText(f"SWEEP={self.SWEEP_ID}")
            if hasattr(self, "_lbl_pair"): self._lbl_pair.setText(f"PAIR={self.PAIR_ID}")
            if hasattr(self, "_lbl_phi"):  self._lbl_phi.setText(f"PHI={self.PHI_LIST_DEG[self._phi_idx]}°")
            if hasattr(self, "_lbl_aod"):  self._lbl_aod.setText(f"AOD={self._phi_idx}")
            # cập nhật payload ngay
            self._update_strobe_payload(self.freq)
        self._prev_freq = self.freq



    def get_encoding(self):
        return self.encoding

    def set_encoding(self, encoding):
        self.encoding = encoding
        self._encoding_callback(self.encoding)
        self.wifi_phy_hier_0.set_encoding(ieee802_11.Encoding(self.encoding))

    def _set_tx_ant_mode(self, mode):
        self.TX_ANT_MODE = mode
        if mode == 'TX0':
            self.active_tx_ant = 0
            self.tx_gate0.set_k(1.0); self.tx_gate1.set_k(0.0)
            self._txa_timer.stop()
        elif mode == 'TX1':
            self.active_tx_ant = 1
            self.tx_gate0.set_k(0.0); self.tx_gate1.set_k(1.0)
            self._txa_timer.stop()
        elif mode == 'BOTH':
            self.active_tx_ant = 2
            self.tx_gate0.set_k(1.0); self.tx_gate1.set_k(1.0)
            self.tx1_phase.set_k(1+0j)          # pha TX1 = 0°
            self._txa_timer.stop()              # không luân phiên trong BOTH
        else:  # 'ALT'
            # chỉ ALT mới dùng timer
            self._txa_timer.setInterval(self.ANT_SWITCH_MS)
            self._txa_timer.start()
        self._ant_status.setText(f"Active: {'BOTH' if self.active_tx_ant==2 else 'TX'+str(self.active_tx_ant)}")
        self._update_strobe_payload(self.freq)


    def _toggle_tx_ant(self):
        if self.TX_ANT_MODE == 'BOTH':
            self.tx_gate0.set_k(1.0); self.tx_gate1.set_k(1.0)
            self.active_tx_ant = 2
            self._ant_status.setText("Active: BOTH")
            self._update_strobe_payload(self.freq)
            return

        if self.TX_ANT_MODE in ('TX0','TX1'):
            # cố định: không đổi ant, chỉ cập nhật payload để RX biết TXA
            self._ant_status.setText(f"Active: {self.TX_ANT_MODE}")
            self._update_strobe_payload(self.freq)
            return

        # ALT mode
        if not self.TRAIN_ENABLE:
            # như cũ: đổi kênh mỗi interval
            self.active_tx_ant ^= 1
            self.tx_gate0.set_k(1.0 if self.active_tx_ant == 0 else 0.0)
            self.tx_gate1.set_k(1.0 if self.active_tx_ant == 1 else 0.0)
            self._ant_status.setText(f"Active: TX{self.active_tx_ant}")
            self._update_strobe_payload(self.freq)
            return

        # NEW: TRAIN_ENABLE — phát theo CẶP: TX0 → TX1, rồi tăng PAIR; cứ mỗi PAIR có thể đổi PHI
        if self._in_pair_half == 0:
            # nửa đầu cặp: TX0 on, TX1 off
            self.active_tx_ant = 0
            self.tx_gate0.set_k(1.0); self.tx_gate1.set_k(0.0)
            # TX1 phase không ảnh hưởng lần này, nhưng cứ set trước cho nhất quán
            phi_deg = self.PHI_LIST_DEG[self._phi_idx]
            self._set_tx1_phase_deg(phi_deg)
            self._ant_status.setText("Active: TX0")
            self._lbl_pair.setText(f"PAIR={self.PAIR_ID}")
            self._lbl_phi.setText(f"PHI={phi_deg}°")
            self._update_strobe_payload(self.freq)
            self._in_pair_half = 1
        else:
            # nửa sau cặp: TX1 on, TX0 off (dùng cùng PAIR & PHI)
            self.active_tx_ant = 1
            self.tx_gate0.set_k(0.0); self.tx_gate1.set_k(1.0)
            phi_deg = self.PHI_LIST_DEG[self._phi_idx]
            self._set_tx1_phase_deg(phi_deg)
            self._ant_status.setText("Active: TX1")
            self._lbl_pair.setText(f"PAIR={self.PAIR_ID}")
            self._lbl_phi.setText(f"PHI={phi_deg}°")
            self._update_strobe_payload(self.freq)
            # hoàn tất cặp → tăng PAIR, và (tuỳ chọn) đổi PHI
            self.PAIR_ID += 1
            self._phi_idx = (self._phi_idx + 1) % len(self.PHI_LIST_DEG)
            
            # NEW: nếu phi_idx quay về 0 → sang vòng sweep mới
            if self._phi_idx == 0:
                self.SWEEP_ID += 1
                self._lbl_sweep.setText(f"SWEEP={self.SWEEP_ID}")
            self._lbl_aod.setText(f"AOD={self._phi_idx}")  # NEW
            
            self._in_pair_half = 0



    # ===== TX control with 5s countdown =====
    def _arm_and_start_countdown(self):
        if self._fg_running:
            return
        self._arm_countdown = 5
        self._tx_status.setText(f"TX: starting in {self._arm_countdown}s")
        print("[TX] Arming: start in 5s")
        self._arm_timer.start()

    def _arm_tick(self):
        self._arm_countdown -= 1
        if self._arm_countdown > 0:
            self._tx_status.setText(f"TX: starting in {self._arm_countdown}s")
            print(f"[TX] starting in {self._arm_countdown}s")
        else:
            self._arm_timer.stop()
            self._start_tx()
            
    def _start_tx(self):
        if self._fg_running:
            return
        try:
            self.start()
            self._fg_running = True
            self._tx_status.setText("TX: running")
            print("[TX] START")

            # ← thêm 3 dòng này để nhảy tần NGAY khi bắt đầu
            self._hop_idx = -1         # để _do_hop() chọn HOP_FREQS[0]
            self._do_hop()
            if self.HOP_ENABLE:
                self._hop_timer.start()
            self._dwell_count_sec = 0
            self._dwell_count_timer.start()
            # chỉ ALT mới cần timer đổi ant
            if self.TX_ANT_MODE == 'ALT':
                try: self._txa_timer.start()
                except Exception: pass
                    
            try:
                self._txa_timer.start()
            except Exception:
                pass
            
        except Exception as e:
            print("[TX] START failed:", e)


    def _stop_tx(self):
        if not self._fg_running:
            return
        try:
            # Dừng timer phụ trợ
            try: self._hop_timer.stop()
            except Exception: pass
            try: self._dwell_count_timer.stop()
            except Exception: pass
            try: self._txa_timer.stop()
            except Exception: pass
            
            self.stop()
            self.wait()
            self._fg_running = False
            self._tx_status.setText("TX: idle")
            print("[TX] STOP")
        except Exception as e:
            print("[TX] STOP failed:", e)


    def _update_strobe_payload(self, freq_hz: float):
        try:
            self._hop_seq += 1
            freq_mhz = int(round(freq_hz / 1e6))
            txa = self.active_tx_ant if self.active_tx_ant in (0,1) else 2
            phi_deg = self.PHI_LIST_DEG[self._phi_idx] if getattr(self, "TRAIN_ENABLE", False) else 0
            # NEW: thêm SWEEP và AOD (AOD chính là phi_idx)
            s = (f"FREQ={freq_mhz};SEQ={self._hop_seq};TS={int(time.time())};"
                f"TXA={txa};PAIR={self.PAIR_ID};PHI={phi_deg};"
                f"SWEEP={self.SWEEP_ID};AOD={self._phi_idx}")
            b = s.encode("ascii")
            vec = pmt.init_u8vector(len(b), list(b))
            msg = pmt.cons(pmt.PMT_NIL, vec)
            self.blocks_message_strobe_0_0.set_msg(msg)
        except Exception as e:
            print("[WARN] set strobe payload failed:", e)





    def _do_hop(self):
        if not self.HOP_ENABLE or not self.HOP_FREQS:
            return
        if self.HOP_RANDOM:
            f = random.choice(self.HOP_FREQS)
        else:
            self._hop_idx = (self._hop_idx + 1) % len(self.HOP_FREQS)
            f = self.HOP_FREQS[self._hop_idx]
        self.set_freq(f)
        # cập nhật payload để RX đọc được FREQ=...
        self._update_strobe_payload(f)
        print(f"[TX] hop → {f/1e6:.1f} MHz")
        self._dwell_count_sec = 0  # reset đếm giây cho dwell mới


    def _tick_dwell_counter(self):
        # tăng bộ đếm mỗi giây và in ra thông tin để căn thời gian bên thu
        self._dwell_count_sec += 1
        now_local = time.strftime('%H:%M:%S', time.localtime())
        try:
            print(f"[TX] {now_local} | t={self._dwell_count_sec:02d}/{self._dwell_count_total:02d}s "
                  f"| freq={self.freq/1e6:.1f} MHz | seq={self._hop_seq}")
        except Exception as e:
            print("[WARN] dwell-counter print failed:", e)

        # hết 15s thì reset về 0 để bắt đầu chu kỳ kế tiếp
        if self._dwell_count_sec >= self._dwell_count_total:
            self._dwell_count_sec = 0

    def _set_tx1_phase_deg(self, deg):
        # NEW: e^{j*phi}
        import math, cmath
        rad = math.radians(deg)
        self.tx1_phase.set_k(complex(math.cos(rad), math.sin(rad)))



def main(top_block_cls=wifi_tx, options=None):

    try:
        gr.enable_realtime_scheduling()
    except Exception as e:
        print("RT sched failed:", e)

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    # tb.start()

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

if __name__ == '__main__':
    main()
