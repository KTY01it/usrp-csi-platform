from packaging.version import Version as StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")
import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")           # né Wayland
os.environ.setdefault("QT_OPENGL", "software")            # render mềm

from PyQt5 import Qt
from PyQt5.QtCore import QObject, pyqtSlot
from PyQt5.QtGui import QKeySequence
from gnuradio import qtgui
from gnuradio.filter import firdes
import sip
from gnuradio import blocks
from gnuradio import fft
from gnuradio.fft import window
from gnuradio import gr
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import gr, pdu
from gnuradio import uhd
import time, re
from gnuradio.qtgui import Range, RangeWidget
from PyQt5 import QtCore
import ieee802_11
from csi_ltf_estimator import csi_ltf_estimator


from gnuradio import qtgui
import pmt

# ---- Message tap rất nhẹ để gọi callback khi có PDU ----
class activity_tap(gr.basic_block):
    def __init__(self, cb):
        gr.basic_block.__init__(self, name="activity_tap", in_sig=None, out_sig=None)
        self.cb = cb
        self.message_port_register_in(pmt.intern("in"))
        self.set_msg_handler(pmt.intern("in"), self._handler)
    def _handler(self, msg):
        try:
            self.cb()
        except Exception:
            pass

class mac_freq_tap(gr.basic_block):
    def __init__(self, cb):
        gr.basic_block.__init__(self, name="mac_freq_tap", in_sig=None, out_sig=None)
        self.cb = cb
        self.message_port_register_in(pmt.intern("in"))
        self.set_msg_handler(pmt.intern("in"), self._handler)

    def _handler(self, msg):
        try:
            meta = pmt.car(msg)
            data = pmt.cdr(msg)
            if not pmt.is_u8vector(data):
                return
            ba = bytes(bytearray(pmt.u8vector_elements(data)))
            # Tìm mẫu FREQ=xxxx;SEQ=yyy;TS=zzz (ASCII)
            m = re.search(br'FREQ=(\d{4});SEQ=(\d+);TS=(\d+)', ba)
            if m:
                freq_mhz = int(m.group(1))
                seq      = int(m.group(2))
                self.cb(freq_mhz, seq)
        except Exception:
            pass

        
class wifi_rx(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Wifi Rx", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Wifi Rx")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
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

        self.settings = Qt.QSettings("GNU Radio", "wifi_rx")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Variables
        ##################################################
        self.window_size = window_size = 48
        self.sync_length = sync_length = 320
        self.samp_rate = samp_rate = 5e6
        self.lo_offset = lo_offset = 0
        self.gain = gain = 0.75
        self.freq = freq = 5890000000
        self.chan_est = chan_est = 0

        ##################################################
        # Blocks
        ##################################################
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
        self._gain_range = Range(0, 1, 0.01, 0.75, 200)
        self._gain_win = RangeWidget(self._gain_range, self.set_gain, "'gain'", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._gain_win)
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
        self._chan_est_options = [0, 1, 2, 3]
        # Create the labels list
        self._chan_est_labels = ['LS', 'LMS', 'Linear Comb', 'STA']
        # Create the combo box
        # Create the radio buttons
        self._chan_est_group_box = Qt.QGroupBox("'chan_est'" + ": ")
        self._chan_est_box = Qt.QHBoxLayout()
        class variable_chooser_button_group(Qt.QButtonGroup):
            def __init__(self, parent=None):
                Qt.QButtonGroup.__init__(self, parent)
            @pyqtSlot(int)
            def updateButtonChecked(self, button_id):
                self.button(button_id).setChecked(True)
        self._chan_est_button_group = variable_chooser_button_group()
        self._chan_est_group_box.setLayout(self._chan_est_box)
        for i, _label in enumerate(self._chan_est_labels):
            radio_button = Qt.QRadioButton(_label)
            self._chan_est_box.addWidget(radio_button)
            self._chan_est_button_group.addButton(radio_button, i)
        self._chan_est_callback = lambda i: Qt.QMetaObject.invokeMethod(self._chan_est_button_group, "updateButtonChecked", Qt.Q_ARG("int", self._chan_est_options.index(i)))
        self._chan_est_callback(self.chan_est)
        self._chan_est_button_group.buttonClicked[int].connect(
            lambda i: self.set_chan_est(self._chan_est_options[i]))
        self.top_layout.addWidget(self._chan_est_group_box)
        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join(('', "")),
            uhd.stream_args(
                cpu_format="fc32",
                otw_format="sc8",
                # args='',
                args='spp=2040,recv_frame_size=4104,num_recv_frames=256',
                # channels=list(range(0,1)), #original
                channels=[0, 1], #dual channel
                
            ),
        )
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_time_unknown_pps(uhd.time_spec(0))

        # self.uhd_usrp_source_0.set_center_freq(uhd.tune_request(freq, rf_freq = freq - lo_offset, rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)
        # self.uhd_usrp_source_0.set_normalized_gain(gain, 0)
        
        # set center freq & gain cho cả 2 kênh
        self.uhd_usrp_source_0.set_center_freq(
            uhd.tune_request(freq, rf_freq=freq - lo_offset, rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)
        self.uhd_usrp_source_0.set_center_freq(
            uhd.tune_request(freq, rf_freq=freq - lo_offset, rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 1)

        self.uhd_usrp_source_0.set_normalized_gain(gain, 0)
        self.uhd_usrp_source_0.set_normalized_gain(gain, 1)
        
        # --- Toolbar điều khiển FLOWGRAPH ---
        self._fg_toolbar = Qt.QToolBar(self)
        self._btn_fg_start = Qt.QPushButton("Start Flowgraph")
        self._btn_fg_stop  = Qt.QPushButton("Stop Flowgraph")
        self._btn_fg_arm   = Qt.QPushButton("Arm (start @ next 1s)")
        self._fg_status    = Qt.QLabel("FG: idle")

        self._fg_toolbar.addWidget(self._btn_fg_start)
        self._fg_toolbar.addWidget(self._btn_fg_stop)
        self._fg_toolbar.addWidget(self._btn_fg_arm)
        self._fg_toolbar.addSeparator()
        self._fg_toolbar.addWidget(self._fg_status)
        self.top_layout.addWidget(self._fg_toolbar)

        # self._btn_fg_start.clicked.connect(self._fg_start)
        self._btn_fg_start.clicked.connect(lambda: self._fg_start_with_countdown(3))
        self._btn_fg_stop.clicked.connect(self._fg_stop)
        self._btn_fg_arm.clicked.connect(self._fg_start_next_sec)

        Qt.QShortcut(QKeySequence("Space"), self).activated.connect(self._fg_toggle)
        self._fg_is_running = False
    
        self.qtgui_time_sink_x_0 = qtgui.time_sink_f(
            1024, #size
            samp_rate, #samp_rate
            "", #name
            1, #number of inputs
            None # parent
        )
        
        # self.qtgui_time_sink_x_0.set_update_time(0.10)
        # self.qtgui_time_sink_x_0.set_update_time(0.75)
        
        self.qtgui_time_sink_x_0.set_y_axis(-1, 1)

        self.qtgui_time_sink_x_0.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_0.enable_tags(True)
        self.qtgui_time_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_0.enable_autoscale(False)
        self.qtgui_time_sink_x_0.enable_grid(False)
        self.qtgui_time_sink_x_0.enable_axis_labels(True)
        self.qtgui_time_sink_x_0.enable_control_panel(False)
        self.qtgui_time_sink_x_0.enable_stem_plot(False)


        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_time_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_time_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_0_win = sip.wrapinstance(self.qtgui_time_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_0_win)
        self.qtgui_const_sink_x_0 = qtgui.const_sink_c(
            48*10, #size
            "", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_const_sink_x_0.set_update_time(0.25)
        self.qtgui_const_sink_x_0.set_max_output_buffer(8192)
        self.qtgui_const_sink_x_0.set_y_axis(-2, 2)
        self.qtgui_const_sink_x_0.set_x_axis(-2, 2)
        self.qtgui_const_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, "")
        self.qtgui_const_sink_x_0.enable_autoscale(False)
        self.qtgui_const_sink_x_0.enable_grid(False)
        self.qtgui_const_sink_x_0.enable_axis_labels(True)


        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "red", "red", "red",
            "red", "red", "red", "red", "red"]
        styles = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        markers = [0, 0, 0, 0, 0,
            0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_const_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_const_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_const_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_const_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_const_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_const_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_const_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_const_sink_x_0_win = sip.wrapinstance(self.qtgui_const_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_const_sink_x_0_win)
        self.pdu_pdu_to_tagged_stream_0 = pdu.pdu_to_tagged_stream(gr.types.complex_t, 'packet_len')
        self.ieee802_11_sync_short_0 = ieee802_11.sync_short(0.56, 2, False, False)
        self.ieee802_11_sync_long_0 = ieee802_11.sync_long(sync_length, False, False)
        
        # self.ieee802_11_parse_mac_0 = ieee802_11.parse_mac(False, True)
        self.ieee802_11_parse_mac_0 = ieee802_11.parse_mac(False, False)
        
        self.ieee802_11_frame_equalizer_0 = ieee802_11.frame_equalizer(ieee802_11.Equalizer(chan_est), freq, samp_rate, False, False)
        
        # self.ieee802_11_decode_mac_0 = ieee802_11.decode_mac(True, False)
        self.ieee802_11_decode_mac_0 = ieee802_11.decode_mac(False, False)
        
        self.fft_vxx_0 = fft.fft_vcc(64, True, window.rectangular(64), True, 2)
        # self.tag_dbg0 = blocks.tag_debug(gr.sizeof_gr_complex*64, "LTF_tag_ch0", ""); 
        # self.tag_dbg0.set_display(True)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex*1, 64)
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.blocks_moving_average_xx_1 = blocks.moving_average_cc(window_size, 1, 4000, 1)
        self.blocks_moving_average_xx_0 = blocks.moving_average_ff(window_size  + 16, 1, 4000, 1)
        self.blocks_divide_xx_0 = blocks.divide_ff(1)
        self.blocks_delay_0_0 = blocks.delay(gr.sizeof_gr_complex*1, 16)
        self.blocks_delay_0 = blocks.delay(gr.sizeof_gr_complex*1, sync_length)
        self.blocks_conjugate_cc_0 = blocks.conjugate_cc()
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared(1)
        self.blocks_complex_to_mag_0 = blocks.complex_to_mag(1)
        # self.csi_est = csi_ltf_estimator(ltf_tag_key="ofdm_start")
        # self.msg_csi_dump = blocks.message_debug()

        # Lấy 1/64 mẫu symbol để vẽ chòm sao (giảm mạnh áp lực GUI)
        self.keepN_const = blocks.keep_one_in_n(gr.sizeof_gr_complex, 4)
        self.keepN_time = blocks.keep_one_in_n(gr.sizeof_float, 200)  # ~lấy 1/200

        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.ieee802_11_decode_mac_0, 'out'), (self.ieee802_11_parse_mac_0, 'in'))
        self.msg_connect((self.ieee802_11_frame_equalizer_0, 'symbols'), (self.pdu_pdu_to_tagged_stream_0, 'pdus'))
        self.connect((self.blocks_complex_to_mag_0, 0), (self.blocks_divide_xx_0, 0))
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.blocks_moving_average_xx_0, 0))
        self.connect((self.blocks_conjugate_cc_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.blocks_delay_0, 0), (self.ieee802_11_sync_long_0, 1))
        self.connect((self.blocks_delay_0_0, 0), (self.blocks_conjugate_cc_0, 0))
        self.connect((self.blocks_delay_0_0, 0), (self.ieee802_11_sync_short_0, 0))
        self.connect((self.blocks_divide_xx_0, 0), (self.ieee802_11_sync_short_0, 2))
        
        # self.connect((self.blocks_divide_xx_0, 0), (self.qtgui_time_sink_x_0, 0))
        self.connect((self.blocks_divide_xx_0, 0), (self.keepN_time, 0))
        self.connect((self.keepN_time, 0), (self.qtgui_time_sink_x_0, 0))
        self.qtgui_time_sink_x_0.set_update_time(1.0)
        self.qtgui_time_sink_x_0.set_max_output_buffer(8192)
        
        self.connect((self.blocks_moving_average_xx_0, 0), (self.blocks_divide_xx_0, 1))
        self.connect((self.blocks_moving_average_xx_1, 0), (self.blocks_complex_to_mag_0, 0))
        self.connect((self.blocks_moving_average_xx_1, 0), (self.ieee802_11_sync_short_0, 1))
        self.connect((self.blocks_multiply_xx_0, 0), (self.blocks_moving_average_xx_1, 0))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.fft_vxx_0, 0), (self.ieee802_11_frame_equalizer_0, 0))
        # self.connect((self.fft_vxx_0, 0), (self.tag_dbg0, 0))  # Kiểm tra tag sau FFT
        self.connect((self.ieee802_11_frame_equalizer_0, 0), (self.ieee802_11_decode_mac_0, 0))
        self.connect((self.ieee802_11_sync_long_0, 0), (self.blocks_stream_to_vector_0, 0))
        self.connect((self.ieee802_11_sync_short_0, 0), (self.blocks_delay_0, 0))
        self.connect((self.ieee802_11_sync_short_0, 0), (self.ieee802_11_sync_long_0, 0))
        
        # self.connect((self.pdu_pdu_to_tagged_stream_0, 0), (self.qtgui_const_sink_x_0, 0))
        self.connect((self.pdu_pdu_to_tagged_stream_0, 0), (self.keepN_const, 0))
        self.connect((self.keepN_const, 0), (self.qtgui_const_sink_x_0, 0))
        
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_complex_to_mag_squared_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_delay_0_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_multiply_xx_0, 0))

        # Kết nối PDU từ CSI estimator đến message_debug để log thông qua console
        # self.pdu2ts = pdu.pdu_to_tagged_stream(gr.types.complex_t, 'packet_len') 
        # self.csi_sink = blocks.file_sink(gr.sizeof_gr_complex, "/home/ea301b/gr-ieee802-11/examples/csi.bin") 
        # self.msg_connect((self.csi_est, 'csi'), (self.pdu2ts, 'pdus')) 
        # self.connect((self.pdu2ts, 0), (self.csi_sink, 0))
        
        # ===== Nhánh CH0 (blocks) =====
        self.csi_est0 = csi_ltf_estimator(ltf_tag_keys=("ofdm_start", "wifi_start"), 
                                        rx_chan_id=0, 
                                        # meta_path="/tmp/Data/csi_ch0.jsonl",
                                        meta_path=None,
                                        meta_every_n=1)  # nếu block hỗ trợ
        self.pdu2ts0   = pdu.pdu_to_tagged_stream(gr.types.complex_t, 'packet_len')
        # self.csi_sink0 = blocks.file_sink(gr.sizeof_gr_complex, "/tmp/Data/csi_ch0.bin")
        
        # KHỞI TẠO ghi về /dev/null để khỏi tạo file rác
        self.csi_sink0 = blocks.file_sink(gr.sizeof_gr_complex, os.path.join(os.environ.get("RX_CSI_DIR", "csi"), "csi_ch0.bin"))

        # self.msg_csi_dump = blocks.message_debug()  # (tuỳ chọn) in meta ra console

        # kết nối
        self.connect((self.fft_vxx_0, 0), (self.csi_est0, 0))
        # self.msg_connect((self.csi_est0, 'csi'), (self.msg_csi_dump, 'print'))  # xem meta có rx_chan=0
        self.msg_connect((self.csi_est0, 'csi'), (self.pdu2ts0, 'pdus'))
        self.connect((self.pdu2ts0, 0), (self.csi_sink0, 0))
        
        
        # ===== Nhánh CH1 (blocks) =====
        self.ieee802_11_sync_short_1 = ieee802_11.sync_short(0.56, 2, False, False)
        self.ieee802_11_sync_long_1  = ieee802_11.sync_long(self.sync_length, False, False)
        
        self.blocks_delay_1          = blocks.delay(gr.sizeof_gr_complex*1, self.sync_length)
        self.blocks_delay_1_0        = blocks.delay(gr.sizeof_gr_complex*1, 16)
        self.blocks_conjugate_cc_1   = blocks.conjugate_cc()
        self.blocks_multiply_xx_1    = blocks.multiply_vcc(1)
        self.blocks_moving_average_xx_c1 = blocks.moving_average_cc(self.window_size, 1, 4000, 1)
        self.blocks_moving_average_xx_f1 = blocks.moving_average_ff(self.window_size + 16, 1, 4000, 1)
        self.blocks_complex_to_mag_1     = blocks.complex_to_mag(1)
        self.blocks_complex_to_mag_squared_1 = blocks.complex_to_mag_squared(1)
        self.blocks_divide_xx_1          = blocks.divide_ff(1)
        self.blocks_stream_to_vector_1   = blocks.stream_to_vector(gr.sizeof_gr_complex, 64)
        self.fft_vxx_1                   = fft.fft_vcc(64, True, window.rectangular(64), True, 2)

        # CSI cho kênh 1
        self.csi_est1 = csi_ltf_estimator(ltf_tag_keys=("ofdm_start", "wifi_start"), 
                                        rx_chan_id=1, 
                                        # meta_path="/tmp/Data/csi_ch1.jsonl",
                                        meta_path=None,
                                        meta_every_n=1)  # nếu block hỗ trợ
        self.pdu2ts1   = pdu.pdu_to_tagged_stream(gr.types.complex_t, 'packet_len')
        # self.csi_sink1 = blocks.file_sink(gr.sizeof_gr_complex, "/tmp/Data/csi_ch1.bin")
        
        # KHỞI TẠO ghi về /dev/null để khỏi tạo file rác
        self.csi_sink1 = blocks.file_sink(gr.sizeof_gr_complex, os.path.join(os.environ.get("RX_CSI_DIR", "csi"), "csi_ch1.bin"))

        # (tuỳ chọn) xem tag LTF của CH1
        # self.tag_dbg1 = blocks.tag_debug(gr.sizeof_gr_complex*64, "LTF_tag_ch1", ""); self.tag_dbg1.set_display(True)

        sym_samps = 160 if self.samp_rate == 10e6 else 80
        self.csi_est0.set_timebase(self.samp_rate, sym_samps)
        self.csi_est1.set_timebase(self.samp_rate, sym_samps)

        # ——— metric cho sync_short_1 (giống CH0) ———
        self.connect((self.uhd_usrp_source_0, 1), (self.blocks_complex_to_mag_squared_1, 0))
        self.connect((self.blocks_complex_to_mag_squared_1, 0), (self.blocks_moving_average_xx_f1, 0))
        self.connect((self.uhd_usrp_source_0, 1), (self.blocks_delay_1_0, 0))
        self.connect((self.blocks_delay_1_0, 0), (self.blocks_conjugate_cc_1, 0))
        self.connect((self.uhd_usrp_source_0, 1), (self.blocks_multiply_xx_1, 0))
        self.connect((self.blocks_conjugate_cc_1, 0), (self.blocks_multiply_xx_1, 1))
        self.connect((self.blocks_multiply_xx_1, 0), (self.blocks_moving_average_xx_c1, 0))
        self.connect((self.blocks_moving_average_xx_c1, 0), (self.blocks_complex_to_mag_1, 0))
        self.connect((self.blocks_complex_to_mag_1, 0), (self.blocks_divide_xx_1, 0))
        self.connect((self.blocks_moving_average_xx_f1, 0), (self.blocks_divide_xx_1, 1))

        # ——— sync & FFT CH1 ———
        self.connect((self.uhd_usrp_source_0, 1), (self.ieee802_11_sync_short_1, 0))
        self.connect((self.blocks_divide_xx_1, 0), (self.ieee802_11_sync_short_1, 2))
        self.connect((self.blocks_moving_average_xx_c1, 0), (self.ieee802_11_sync_short_1, 1))
        self.connect((self.ieee802_11_sync_short_1, 0), (self.ieee802_11_sync_long_1, 0))
        self.connect((self.ieee802_11_sync_short_1, 0), (self.blocks_delay_1, 0))
        self.connect((self.blocks_delay_1, 0), (self.ieee802_11_sync_long_1, 1))
        self.connect((self.ieee802_11_sync_long_1, 0), (self.blocks_stream_to_vector_1, 0))
        self.connect((self.blocks_stream_to_vector_1, 0), (self.fft_vxx_1, 0))

        # ——— CSI CH1 ———
        self.connect((self.fft_vxx_1, 0), (self.csi_est1, 0))
        # self.connect((self.fft_vxx_1, 0), (self.tag_dbg1, 0))  # nếu muốn xem tag LTF CH1
        # self.msg_connect((self.csi_est1, 'csi'), (self.msg_csi_dump, 'print'))
        self.msg_connect((self.csi_est1, 'csi'), (self.pdu2ts1, 'pdus'))
        self.connect((self.pdu2ts1, 0), (self.csi_sink1, 0))

        # --- Nơi lưu dữ liệu (đổi nếu muốn) ---
        self.save_root = "/home/ea301b/gr-ieee802-11/examples/Data"
        os.makedirs(self.save_root, exist_ok=True)

        # Bắt đầu ở trạng thái SCAN: ghi về /dev/null để tránh rác
        self._record_idle()
        
        # ===== TAP “hoạt động” để biết khi nào có khung (không đổi kiến trúc DSP) =====
        # self.tap0 = activity_tap(self._on_activity)
        # self.tap1 = activity_tap(self._on_activity)
        # self.msg_connect((self.csi_est0, 'csi'), (self.tap0, 'in'))
        # self.msg_connect((self.csi_est1, 'csi'), (self.tap1, 'in'))
        
        # ===== TAP hoạt động: nghe KHUNG MAC đã giải mã (ổn định để neo) =====
        # self.tap = activity_tap(self._on_activity)
        # self.msg_connect((self.ieee802_11_decode_mac_0, 'out'), (self.tap, 'in'))
        # (Tuỳ chọn gắt hơn)
        # self.ieee802_11_parse_mac_0 = ieee802_11.parse_mac(True, False)  # lọc CRC OK
        # self.msg_connect((self.ieee802_11_parse_mac_0, 'out'), (self.tap, 'in'))
        
        # Đếm theo CẶP CSI: mỗi kênh có callback riêng
        self.tap0 = activity_tap(lambda: self._on_activity_ch(0))
        self.tap1 = activity_tap(lambda: self._on_activity_ch(1))
        self.msg_connect((self.csi_est0, 'csi'), (self.tap0, 'in'))
        self.msg_connect((self.csi_est1, 'csi'), (self.tap1, 'in'))


        # ===== TAP “giải mã MAC” để biết khi nào có khung MAC (ổn định để neo) =====
        self.macfreq = mac_freq_tap(self._on_mac_pdu)
        self.msg_connect((self.ieee802_11_decode_mac_0, 'out'), (self.macfreq, 'in'))
        # (Không nối vào _on_activity để tránh đếm trùng)


        # ===== SCAN→LOCK cấu hình =====
        self.SEQ_MODE       = True
        self.SEQ_FREQS      = [5180e6, 5200e6, 5220e6, 5240e6, 5260e6, 5280e6, 5300e6, 5320e6]
        # self.N_REQUIRED     = 5          # cần bắt đủ 5 lần mỗi tần số
        self.SEQ_TIMEOUT_MS = 15000      # tối đa 15s mỗi tần số rồi nhảy tiếp
        self._seq_idx       = 0
        self._seq_hits      = 0

        # --- dwell / idle tracking (PHẢI có trước khi _start_seq_round) ---
        self.MIN_DWELL_MS = 12000   # tối thiểu ở lại 12s
        self.MAX_DWELL_MS = 15000   # tối đa 18s (watchdog)
        self.IDLE_GAP_MS  = 800     # sau min dwell, im lặng MAC >= 0.8s thì nhảy
        self._seq_last_seen_ms = -1
        self._seq_start_ms = 0.0
        self._seq_max_timer = None
        self._seq_idle_timer = None
        self.START_ON_MAC        = True    # <<-- thêm cờ: thấy MAC (đúng freq) là cho ghi luôn
        self.WARMUP_MS           = 200     # chờ 200 ms sau khi tune để đỡ nhiễu settle
        self._arm_time_ms        = 0.0   
        
        
        
        # --- Dwell counter giống TX ---
        self._dwell_count_sec  = 0
        self._dwell_total_sec  = int(self.MAX_DWELL_MS/1000)
        self._dwell_timer      = QtCore.QTimer(self)
        self._dwell_timer.setInterval(1000)
        self._dwell_timer.timeout.connect(self._tick_dwell_counter)

        # --- Countdown khi bấm Start ---
        self._arm_timer     = QtCore.QTimer(self)
        self._arm_timer.setInterval(1000)
        self._arm_timer.timeout.connect(self._fg_arm_tick)
        self._arm_remaining = 0
        

        # Nếu SEQ_MODE bật -> tắt cơ chế SCAN cũ
        if self.SEQ_MODE:
            # Chỉ dùng tuần tự tần số
            self.scanning = False
            self.lock_active = False
            # self._start_seq_round()
            self._seq_active = False
        else:
            # Dùng SCAN/LOCK truyền thống
            self.scan_idx = -1
            self.scanning = True
            self.scan_dwell_ms = 200
            self.lock_dwell_ms = 5000
            self.lock_active = False

            self.scan_timer = Qt.QTimer(self)
            self.scan_timer.setTimerType(QtCore.Qt.PreciseTimer)
            self.scan_timer.setInterval(self.scan_dwell_ms)
            self.scan_timer.timeout.connect(self._scan_step)
            self.scan_timer.start()

            # quét ngay kênh đầu (cần self.scan_freqs đã định nghĩa)
            self._scan_step()

            
        # === Đếm theo “cặp kênh trùng thời điểm” ===
        self.COINC_WINDOW_MS = 10.0   # 2–5 ms là hợp lý cho LTF hai kênh
        self._last_csi_ms = [-1.0, -1.0]  # lưu thời điểm CSI gần nhất của từng kênh
        self._seq_active = False
        # trong __init__
        self.AUTO_EXIT_ON_DONE = True
        # Cửa sổ xác thực MAC payload gần nhất phải trùng FREQ (ms)
        self.MAC_MATCH_WINDOW_MS = 400.0
        # Bật/tắt gating payload
        self.STRICT_FREQ_GATE  = False
        
        # Lưu MAC payload mới nhất (ts theo monotonic ms)
        self._last_mac = {'ts_ms': -1.0, 'freq_mhz': None, 'seq': None}
        # trạng thái ghi thực sự
        self._is_recording = False
 

    def _on_mac_pdu(self, freq_mhz:int, seq:int):
        now_ms = time.monotonic()*1000.0
        self._last_mac = {'ts_ms': now_ms, 'freq_mhz': freq_mhz, 'seq': seq}
        # bơm vào estimator (như bạn đã có)
        try:
            self.csi_est0.set_last_mac(freq_mhz, seq, now_ms)
            self.csi_est1.set_last_mac(freq_mhz, seq, now_ms)
        except Exception:
            pass
        print(f"[MAC] freq={freq_mhz} seq={seq} (ts={now_ms:.1f})")

        # nếu đang đứng đúng tần số & cho phép "start-on-MAC" thì mở ghi ngay
        if (self.START_ON_MAC
            and int(round(self.freq/1e6)) == freq_mhz
            and not self._is_recording
            and now_ms >= self._arm_time_ms):           # sau warmup
            print("[GATE] start-on-MAC → begin recording")
            self._start_recording_current_freq()
            # vẫn giữ pair-hit: khi có pair-hit sẽ log “pair_confirmed” trong JSON (xem mục 2)




   
    # ======== SCAN / LOCK ========
    # def _start_seq_round(self):
    #     self._seq_active = True
    #     self._seq_hits   = 0
    #     self._is_recording = False          # chưa ghi cho tới khi xác thực FREQ
    #     f = self.SEQ_FREQS[self._seq_idx]
    #     self.set_freq(f)

    #     # Khi chưa xác thực, ghi về /dev/null để tránh rác
    #     self._record_idle()

    #     print(f"[SEQ] start f={f/1e6:.1f} MHz; dwell=10s; wait for MAC FREQ match & CSI pair-hit")

    #     try:
    #         self._seq_timer.stop()
    #     except Exception:
    #         pass
    #     self._seq_timer = Qt.QTimer(self)
    #     self._seq_timer.setSingleShot(True)
    #     self._seq_timer.setInterval(self.SEQ_TIMEOUT_MS)
    #     self._seq_timer.timeout.connect(self._seq_timeout)
    #     self._seq_timer.start()

    def _seq_idle_check(self):
        if not self._seq_active:
            return
        now = time.monotonic() * 1000.0
        elapsed = now - getattr(self, "_seq_start_ms", now)
        if elapsed < self.MIN_DWELL_MS:
            return  # chưa đủ tối thiểu

        # Nếu đã đủ tối thiểu mà không thấy MAC khớp thêm IDLE_GAP_MS -> next
        if self._seq_last_seen_ms > 0 and (now - self._seq_last_seen_ms) >= self.IDLE_GAP_MS:
            print(f"[SEQ] idle gap {now - self._seq_last_seen_ms:.0f} ms after min dwell -> next")
            self._seq_idle_timer.stop()
            self._seq_max_timer.stop()
            if self._is_recording:
                self._record_idle()
                self._is_recording = False
            self._advance_seq()


    def _start_seq_round(self):
        self._seq_active = True
        self._seq_hits   = 0
        self._is_recording = False
        self._seq_last_seen_ms = -1
        f = self.SEQ_FREQS[self._seq_idx]
        self.set_freq(f)
        self._arm_time_ms = time.monotonic()*1000.0 + self.WARMUP_MS
        self._record_idle()

        print(f"[SEQ] start f={f/1e6:.1f} MHz; dwell dynamic [{self.MIN_DWELL_MS/1000:.1f}..{self.MAX_DWELL_MS/1000:.1f}] s")

        # Watchdog tối đa
        try: self._seq_max_timer.stop()
        except: pass
        self._seq_max_timer = Qt.QTimer(self)
        self._seq_max_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._seq_max_timer.setSingleShot(True)
        self._seq_max_timer.setInterval(self.MAX_DWELL_MS)
        self._seq_max_timer.timeout.connect(self._seq_timeout)
        self._seq_max_timer.start()

        # IDLE kiểm tra định kỳ sau khi đã qua MIN_DWELL_MS
        try: self._seq_idle_timer.stop()
        except: pass
        self._seq_idle_timer = Qt.QTimer(self)
        self._seq_idle_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._seq_idle_timer.setInterval(100)  # check mỗi 100 ms
        self._seq_idle_timer.timeout.connect(self._seq_idle_check)
        self._seq_idle_timer.start()

        # Mốc bắt đầu
        self._seq_start_ms = time.monotonic() * 1000.0
        # reset & start dwell counter mỗi khi vào kênh mới
        self._dwell_count_sec = 0
        self._dwell_total_sec = int(self.MAX_DWELL_MS/1000)
        try: self._dwell_timer.stop()
        except Exception: pass
        self._dwell_timer.start()

    def _start_recording_current_freq(self):
        """Bắt đầu ghi thực sự cho tần số hiện tại (chỉ gọi sau khi xác thực FREQ)."""
        if self._is_recording:
            return
        self._is_recording = True
        self._switch_record_targets(self.freq)   # mở file theo f_MHz/chX_timestamp
        print(f"[REC] START at {self.freq/1e6:.1f} MHz")


    def _seq_timeout(self):
        f = self.SEQ_FREQS[self._seq_idx]
        print(f"[SEQ] max dwell timeout at {f/1e6:.1f} MHz (recording={self._is_recording}) -> next")
        if self._is_recording:
            self._record_idle()
            self._is_recording = False
        self._advance_seq()



    # def _advance_seq(self):
    #     self._seq_idx += 1
    #     if self._seq_idx >= len(self.SEQ_FREQS):
    #         self._seq_active = False  
    #         print("[SEQ] DONE: đã đi hết danh sách tần số.")
    #         try:
    #             self._seq_timer.stop()
    #         except Exception:
    #             pass
    #         # quay về chế độ idle ghi /dev/null cho an toàn
    #         self._record_idle()
    #         return
    #     self._start_seq_round()
    
    def _advance_seq(self):
        self._seq_idx += 1
        if self._seq_idx >= len(self.SEQ_FREQS):
            self._seq_active = False
            print("[SEQ] DONE: đã đi hết danh sách tần số.")
            try: self._seq_timer.stop()
            except Exception: pass
            self._record_idle()  # mở /dev/null để chắc chắn không ghi nữa
            if getattr(self, "AUTO_EXIT_ON_DONE", False):
                Qt.QTimer.singleShot(0, self._shutdown)   # <- dừng hẳn
            try: self._dwell_timer.stop()
            except Exception: pass
            
            return
        self._start_seq_round()


    def _scan_step(self):
        if not self.scanning:
            return
        self.scan_idx = (self.scan_idx + 1) % len(self.scan_freqs)
        f = self.scan_freqs[self.scan_idx]
        print(f"[SCAN] → {f/1e6:.1f} MHz")
        self.set_freq(f)

    def _on_activity(self):
        # ĐỪNG đụng timer trực tiếp ở đây (đây là GR thread)!
        # Uỷ thác sang GUI thread bằng queued connection:
        Qt.QMetaObject.invokeMethod(self, "_handle_activity", QtCore.Qt.QueuedConnection)

    @QtCore.pyqtSlot()
    def _handle_activity(self):
        if self.SEQ_MODE:
            # Mỗi lần có CSI (LTF) ở kênh 0 → cộng 1
            # self._seq_hits += 1
            # f = self.SEQ_FREQS[self._seq_idx]
            # print(f"[SEQ] hit {self._seq_hits}/{self.N_REQUIRED} @ {f/1e6:.1f} MHz")
            # if self._seq_hits >= self.N_REQUIRED:
            #     self._advance_seq()
            return

        # --- Nhánh cũ SCAN/LOCK (giữ nguyên nếu muốn dùng lại) ---
        if not self.scanning:
            return
        self.scanning = False
        self.lock_active = True
        try:
            if self.scan_timer.isActive():
                self.scan_timer.stop()
        except Exception:
            pass

        # === mở file ghi theo tần số hiện tại ===
        self._switch_record_targets(self.freq)

        print(f"[LOCK] Có khung → neo {self.lock_dwell_ms/1000:.1f}s ở {self.freq/1e6:.1f} MHz")
        self.lock_timer.start(self.lock_dwell_ms)


    def _lock_timeout(self):
        # hết 5s → dừng ghi, chuyển sang kênh kế & tiếp tục quét
        self.lock_active = False
        self.scanning = True

        # quay về chế độ SCAN: ghi /dev/null và tắt JSON
        self._record_idle()

        self._scan_step()
        self.scan_timer.start(self.scan_dwell_ms)
    ################################################################################
    
    # Khi chuyển trạng thái SCAN/LOCK, đổi đích file ghi và bật/tắt ghi JSON meta
    def _safe_makedirs(self, path):
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            pass

    def _record_idle(self):
        """Khi SCAN: ghi về /dev/null và tắt ghi JSON meta."""
        try:
            self.csi_sink0.open("/dev/null")
            self.csi_sink1.open("/dev/null")
        except Exception:
            pass
        # tắt ghi JSONL (nếu estimator đã có set_meta_path)
        try:
            self.csi_est0.set_meta_path(None)
            self.csi_est1.set_meta_path(None)
        except Exception:
            pass

    def _switch_record_targets(self, freq_hz):
        """Khi LOCK: mở file mới theo tần số & kênh."""
        f_mhz = int(round(freq_hz / 1e6))
        ts    = time.strftime("%Y%m%d_%H%M%S")
        base  = os.path.join(self.save_root, f"f{f_mhz}MHz")
        self._safe_makedirs(base)

        bin0  = os.path.join(base, f"ch0_{ts}.bin")
        json0 = os.path.join(base, f"ch0_{ts}.jsonl")
        bin1  = os.path.join(base, f"ch1_{ts}.bin")
        json1 = os.path.join(base, f"ch1_{ts}.jsonl")

        # đổi đích file_sink
        try:
            self.csi_sink0.open(bin0)
            self.csi_sink1.open(bin1)
        except Exception:
            pass

        # báo cho estimator ghi JSON đúng nơi và gắn tần số (nếu hỗ trợ)
        try:
            self.csi_est0.set_meta_path(json0)
            self.csi_est0.center_freq = float(freq_hz)
        except Exception:
            pass
        try:
            self.csi_est1.set_meta_path(json1)
            self.csi_est1.center_freq = float(freq_hz)
        except Exception:
            pass

        print(f"[REC] CH0 → {bin0}")
        print(f"[REC] CH1 → {bin1}")

    
    ##################################################
    # QT sink close method reimplementation
    ##################################################
    
    # def closeEvent(self, event):
    #     self.settings = Qt.QSettings("GNU Radio", "wifi_rx")
    #     self.settings.setValue("geometry", self.saveGeometry())
    #     self.stop()
    #     self.wait()

    #     event.accept()
    
    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "wifi_rx")
        self.settings.setValue("geometry", self.saveGeometry())
        try:
            self.scan_timer.stop()
            self.lock_timer.stop()
        except Exception:
            pass
        self.stop()
        self.wait()
        event.accept()

    def get_window_size(self):
        return self.window_size

    def set_window_size(self, window_size):
        self.window_size = window_size
        self.blocks_moving_average_xx_0.set_length_and_scale(self.window_size  + 16, 1)
        self.blocks_moving_average_xx_1.set_length_and_scale(self.window_size, 1)

    def get_sync_length(self):
        return self.sync_length

    def set_sync_length(self, sync_length):
        self.sync_length = sync_length
        self.blocks_delay_0.set_dly(self.sync_length)

    def _detect_stream_modes(self):
        U = uhd.stream_cmd_t
        # 1) GNURadio thường gặp (enum nằm ngay trong class)
        if hasattr(U, "STREAM_MODE_START_CONTINUOUS") and hasattr(U, "STREAM_MODE_STOP_CONTINUOUS"):
            return U.STREAM_MODE_START_CONTINUOUS, U.STREAM_MODE_STOP_CONTINUOUS
        # 2) Một số bản UHD export enum lồng bên trong
        if hasattr(U, "stream_mode_t"):
            sm = U.stream_mode_t
            if hasattr(sm, "STREAM_MODE_START_CONTINUOUS") and hasattr(sm, "STREAM_MODE_STOP_CONTINUOUS"):
                return sm.STREAM_MODE_START_CONTINUOUS, sm.STREAM_MODE_STOP_CONTINUOUS
        # 3) Fall-back: dùng số (chuẩn UHD: start=0, stop=1)
        return int(0), int(1)


    # def _fg_start(self):
    #     try:
    #         self.start()
    #         self._fg_status.setText("FG: running")
    #         print("[FG] START")
    #     except Exception as e:
    #         print("[FG] START failed:", e)

    # def _fg_stop(self):
    #     try:
    #         self.stop()
    #         self.wait()
    #         self._fg_status.setText("FG: idle")
    #         print("[FG] STOP")
    #     except Exception as e:
    #         print("[FG] STOP failed:", e)

    # def _fg_toggle(self):
    #     if "running" in self._fg_status.text():
    #         self._fg_stop()
    #     else:
    #         self._fg_start()

    # def _fg_start_next_sec(self):
    #     delay_ms = int((1.0 - (time.time() % 1.0)) * 1000)
    #     QtCore.QTimer.singleShot(delay_ms, self._fg_start)
    #     self._fg_status.setText("FG: armed…")
    #     print(f"[FG] armed, will start in ~{delay_ms} ms")
    
    ################################################################################.
    # Flowgraph start/stop with SEQ_MODE support
    ################################################################################
    def _fg_start(self):
        try:
            self.start()
            self._fg_is_running = True
            self._fg_status.setText("FG: running")
            # bắt đầu quét khi flowgraph đã chạy
            if self.SEQ_MODE and not self._seq_active:
                self._start_seq_round()
            print("[FG] START")
        except Exception as e:
            print("[FG] START failed:", e)

    def _fg_stop(self):
        try:
            # dừng timer SEQ nếu đang chạy
            try:
                if getattr(self, "_seq_idle_timer", None): self._seq_idle_timer.stop()
            except Exception: pass
            try:
                if getattr(self, "_seq_max_timer", None): self._seq_max_timer.stop()
            except Exception: pass
            self._seq_active = False

            # đóng ghi nếu đang ghi
            if self._is_recording:
                self._record_idle()
                self._is_recording = False

            self.stop()
            self.wait()
            self._fg_is_running = False
            self._fg_status.setText("FG: idle")
            print("[FG] STOP")
        except Exception as e:
            print("[FG] STOP failed:", e)
            
        try: self._dwell_timer.stop()
        except Exception: pass

    def _fg_toggle(self):
        if "running" in self._fg_status.text():
            self._fg_stop()
        else:
            self._fg_start()

    def _fg_start_next_sec(self):
        delay_ms = int((1.0 - (time.time() % 1.0)) * 1000)
        QtCore.QTimer.singleShot(delay_ms, self._fg_start)
        self._fg_status.setText("FG: armed…")
        print(f"[FG] armed, will start in ~{delay_ms} ms")




    def get_samp_rate(self):
        return self.samp_rate

    # def set_samp_rate(self, samp_rate):
    #     self.samp_rate = samp_rate
    #     self._samp_rate_callback(self.samp_rate)
    #     self.ieee802_11_frame_equalizer_0.set_bandwidth(self.samp_rate)
    #     self.qtgui_time_sink_x_0.set_samp_rate(self.samp_rate)
    #     self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)
    
    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self._samp_rate_callback(self.samp_rate)
        self.ieee802_11_frame_equalizer_0.set_bandwidth(self.samp_rate)
        self.qtgui_time_sink_x_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

    def get_lo_offset(self):
        return self.lo_offset

    # def set_lo_offset(self, lo_offset):
    #     self.lo_offset = lo_offset
    #     self._lo_offset_callback(self.lo_offset)
    #     self.uhd_usrp_source_0.set_center_freq(uhd.tune_request(self.freq, rf_freq = self.freq - self.lo_offset, rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)

    def set_lo_offset(self, lo_offset):
        self.lo_offset = lo_offset
        self._lo_offset_callback(self.lo_offset)
        self.uhd_usrp_source_0.set_center_freq(
            uhd.tune_request(self.freq, rf_freq=self.freq - self.lo_offset,
                            rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)
        self.uhd_usrp_source_0.set_center_freq(
            uhd.tune_request(self.freq, rf_freq=self.freq - self.lo_offset,
                            rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 1)
    
    # def get_gain(self):
    #     return self.gain
    
    def set_gain(self, gain):
        self.gain = gain
        self.uhd_usrp_source_0.set_normalized_gain(self.gain, 0)
        self.uhd_usrp_source_0.set_normalized_gain(self.gain, 1)

    # def set_gain(self, gain):
    #     self.gain = gain
    #     self.uhd_usrp_source_0.set_normalized_gain(self.gain, 0)

    def get_freq(self):
        return self.freq

    # def set_freq(self, freq):
    #     self.freq = freq
    #     self._freq_callback(self.freq)
    #     self.ieee802_11_frame_equalizer_0.set_frequency(self.freq)
    #     self.uhd_usrp_source_0.set_center_freq(uhd.tune_request(self.freq, rf_freq = self.freq - self.lo_offset, rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)

    def set_freq(self, freq):
        self.freq = freq
        self._freq_callback(self.freq)
        self.ieee802_11_frame_equalizer_0.set_frequency(self.freq)
        self.uhd_usrp_source_0.set_center_freq(
            uhd.tune_request(self.freq, rf_freq=self.freq - self.lo_offset,
                            rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 0)
        self.uhd_usrp_source_0.set_center_freq(
            uhd.tune_request(self.freq, rf_freq=self.freq - self.lo_offset,
                            rf_freq_policy=uhd.tune_request.POLICY_MANUAL), 1)
    
    def get_chan_est(self):
        return self.chan_est

    def set_chan_est(self, chan_est):
        self.chan_est = chan_est
        self._chan_est_callback(self.chan_est)
        self.ieee802_11_frame_equalizer_0.set_algorithm(ieee802_11.Equalizer(self.chan_est))

    
    def _on_activity_ch(self, ch: int):
        """
        Được gọi mỗi khi một kênh (0 hoặc 1) bắn ra 1 CSI/LTF hợp lệ.
        Chỉ khi CẢ HAI kênh có CSI trong cửa sổ COINC_WINDOW_MS → tính 1 lần.
        """
        now_ms = time.monotonic() * 1000.0
        self._last_csi_ms[ch] = now_ms
        other = 1 - ch
        t_other = self._last_csi_ms[other]

        # Chỉ đếm khi đang ở SEQ_MODE (quét tuần tự theo tần số)
        if not getattr(self, "SEQ_MODE", False) or not self._seq_active:
                return

        # Kiểm tra “trùng thời điểm” 2 kênh
        if t_other > 0 and abs(now_ms - t_other) <= self.COINC_WINDOW_MS:
            self._last_csi_ms = [-1.0, -1.0]
            Qt.QMetaObject.invokeMethod(self, "_on_csi_pair_hit_slot", QtCore.Qt.QueuedConnection)


    # @QtCore.pyqtSlot()
    # def _on_csi_pair_hit_slot(self):
    #     if (not self._seq_active) or (self._seq_idx >= len(self.SEQ_FREQS)):
    #         return
    #     self._seq_hits += 1
    #     f = self.SEQ_FREQS[self._seq_idx]
    #     print(f"[SEQ] pair-hit {self._seq_hits}/{self.N_REQUIRED} @ {f/1e6:.1f} MHz")
    #     if self._seq_hits >= self.N_REQUIRED:
    #         self._advance_seq()
    
    @QtCore.pyqtSlot()
    def _on_csi_pair_hit_slot(self):
        now_ms = time.monotonic()*1000.0
        if now_ms < self._arm_time_ms:
            return  # chưa hết warmup

        # Guard
        if (not self._seq_active) or (self._seq_idx >= len(self.SEQ_FREQS)):
            return

        # STRICT gate: yêu cầu MAC gần nhất khớp FREQ đang tune
        if self.STRICT_FREQ_GATE:
            now_ms = time.monotonic() * 1000.0
            f_rx   = int(round(self.freq / 1e6))
            ok = (
                self._last_mac['ts_ms'] > 0 and
                (now_ms - self._last_mac['ts_ms']) <= self.MAC_MATCH_WINDOW_MS and
                self._last_mac['freq_mhz'] == f_rx
            )
            if not ok:
                # Debug nhẹ (bật khi cần)
                # print(f"[GATE] reject pair-hit: last_mac={self._last_mac}, f_rx={f_rx}")
                return

        if not self._is_recording:
            print("[GATE] OK (pair-hit + MAC match) → start recording")
            self._start_recording_current_freq()
        else:
            print("[GATE] pair-hit confirmed while recording")

        # Không tăng _seq_hits và KHÔNG advance ở đây.
        # Ta ở lại kênh này đến hết dwell 10s rồi mới chuyển.


    @QtCore.pyqtSlot()
    def _shutdown(self):
        try:
            self.scan_timer.stop()
        except Exception:
            pass
        try:
            self._seq_timer.stop()
        except Exception:
            pass
        self.stop()
        self.wait()
        Qt.QApplication.quit()

    def _tick_dwell_counter(self):
        if not getattr(self, "_seq_active", False):
            return
        self._dwell_count_sec += 1
        now_local = time.strftime('%H:%M:%S', time.localtime())
        rec = "on" if self._is_recording else "off"
        try:
            print(f"[RX] {now_local} | t={self._dwell_count_sec:02d}/{self._dwell_total_sec:02d}s "
                f"| freq={self.freq/1e6:.1f} MHz | rec={rec}")
        except Exception:
            pass

    def _fg_start_with_countdown(self, secs=3):
        if self._fg_is_running:
            return
        self._arm_remaining = int(secs)
        print(f"[RX] Arming: start in {self._arm_remaining}s")
        self._fg_status.setText(f"FG: starting in {self._arm_remaining}s")
        try: self._arm_timer.stop()
        except Exception: pass
        self._arm_timer.start()

    def _fg_arm_tick(self):
        self._arm_remaining -= 1
        if self._arm_remaining > 0:
            print(f"[RX] starting in {self._arm_remaining}s")
            self._fg_status.setText(f"FG: starting in {self._arm_remaining}s")
        else:
            self._arm_timer.stop()
            self._fg_start()


def main(top_block_cls=wifi_rx, options=None):

    # dùng OpenGL mềm để tránh treo do driver
    Qt.QApplication.setAttribute(QtCore.Qt.AA_UseSoftwareOpenGL)

    # bật lịch real-time để giảm jitter
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
