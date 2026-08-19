# csi_ltf_estimator.py
import os, json, time
import numpy as np
from gnuradio import gr
import pmt

# ===== Active subcarriers (802.11a/g 20 MHz): -26..-1 and +1..+26 =====
ACTIVE_K = np.r_[np.arange(-26, 0), np.arange(1, 27)]
def k_to_bin(k, N=64):  # map subcarrier index k -> FFT bin (0..63)
    return (k + N) % N
ACTIVE_BINS = np.array([k_to_bin(k) for k in ACTIVE_K], dtype=np.int32)

# ===== L-LTF freq-domain reference =====
# Đơn giản hoá: đặt X_LTF = 1 ở mọi bin active (đủ tốt cho AoA pha tương đối)
X_LTF_64 = np.zeros(64, np.complex64)
X_LTF_64[ACTIVE_BINS] = 1.0 + 0.0j

class csi_ltf_estimator(gr.sync_block):
    """
    Nhận: stream vector complex64 (vlen=64) sau FFT.
    Khi phát hiện 2 LTF liên tiếp (tag / tương quan), xuất PDU 'csi' (52 complex)
    và (nếu cấu hình) ghi meta thưa thớt sang JSONL.

    API bổ sung:
      - set_meta_path(path): đổi file JSONL trong lúc chạy (None = tắt).
      - set_rate_limit(hz): giới hạn tối đa số lần publish CSI theo Hz.
      - set_cooldown_syms(n): số symbol “nghỉ” sau khi đã publish 1 CSI.
      - set_corr_thresh(x): chỉnh ngưỡng tương quan nếu muốn.
      - Thuộc tính tuỳ chọn: self.center_freq (Hz) → ghi kèm vào JSON.
    """
    def __init__(self,
                 ltf_tag_keys=("ofdm_start", "wifi_start"),
                 rx_chan_id=-1,
                 ltf_tag_key=None,
                 corr_thresh=0.995,          # ngưỡng tương quan chặt
                 meta_path=None,
                 meta_every_n=100,
                 sample_rate=None,          # Hz, để suy ra microseconds từ rx_time
                 sym_samps=80,               # số sample/OFDM symbol (20 MHz -> 80)
                 require_tag=False,
                 ):           # ghi JSON thưa
        gr.sync_block.__init__(self,
            name="csi_ltf_estimator",
            in_sig=[(np.complex64, 64)],
            out_sig=None)

        self.emit_id = 0  # đếm CSI
        
        # ===== Ports =====
        self.message_port_register_out(pmt.intern("csi"))

        # ===== Tag keys =====
        if ltf_tag_key is not None:   # tương thích kiểu cũ (1 key)
            ltf_tag_keys = (ltf_tag_key,)
        self.tag_keys = [pmt.intern(k) for k in ltf_tag_keys]
        # >>> key cho rx_time
        self._rx_time_key = pmt.intern("rx_time")

        # ===== Runtime state =====
        self.rx_chan_id = int(rx_chan_id)
        self.prev_vec = None
        self.sym_idx = 0
        self.corr_thresh = float(corr_thresh)
        self.require_tag = bool(require_tag)

        # Chống “mưa CSI”
        self.cooldown_syms = 160      # ~ mỗi khung 1 CSI
        self.cooldown = 0
        self.seen_first_ltf = False
        self.min_emit_interval = 0.05 # ≥50 ms → tối đa ~20 Hz
        self.last_emit_ts = 0.0

        # >>> timebase để tính tsf_us
        self.sample_rate = float(sample_rate) if sample_rate else None
        self.sym_samps   = int(sym_samps)
        self._last_rx_time_tuple = None   # (secs, frac)
        self._last_rx_time_off   = None   # offset (items) tại thời điểm gắn tag

        # >>> (tuỳ chọn) lưu MAC gần nhất để chèn seq khi phù hợp
        self._last_mac = {'ts_ms': -1.0, 'freq_mhz': None, 'seq': None}
        self._mac_window_ms = 50.0  # chấp nhận MAC trong 50ms gần nhất
        
        # ===== Sidecar JSONL =====
        self.meta_path = None
        self.meta_fh = None
        self.meta_every_n = int(meta_every_n)
        self.meta_count = 0
        # mở nếu có đường dẫn ngay từ đầu
        self.set_meta_path(meta_path)

    # ---------- Public setters ----------
    def set_meta_path(self, path):
        """Đổi file JSONL khi đang chạy. None → tắt ghi."""
        # đóng file cũ
        try:
            if self.meta_fh:
                self.meta_fh.close()
        except Exception:
            pass
        self.meta_fh = None
        self.meta_path = path
        # mở file mới (nếu có)
        if self.meta_path:
            try:
                os.makedirs(os.path.dirname(self.meta_path), exist_ok=True)
                self.meta_fh = open(self.meta_path, "a", buffering=1, encoding="utf-8")
            except Exception:
                self.meta_fh = None  # nếu lỗi, coi như tắt ghi

    def set_rate_limit(self, hz):
        """Giới hạn tối đa số lần publish CSI (Hz)."""
        if hz is None or hz <= 0:
            self.min_emit_interval = 0.0
        else:
            self.min_emit_interval = 1.0 / float(hz)

    def set_cooldown_syms(self, n):
        """Số symbol nghỉ sau khi phát 1 CSI."""
        self.cooldown_syms = max(0, int(n))

    def set_corr_thresh(self, x):
        """Chỉnh ngưỡng tương quan (0..1)."""
        self.corr_thresh = float(x)

    # >>> đặt timebase từ ngoài (wifi_rx gọi sau khi tạo block)
    def set_timebase(self, sample_rate, sym_samps=80):
        self.sample_rate = float(sample_rate) if sample_rate else None
        self.sym_samps   = int(sym_samps)

    # >>> cập nhật MAC gần nhất (wifi_rx gọi mỗi khi bắt được MAC payload FREQ=...;SEQ=...)
    def set_last_mac(self, freq_mhz, seq, ts_ms=None):
        if ts_ms is None:
            ts_ms = time.monotonic() * 1000.0
        self._last_mac = {'ts_ms': float(ts_ms), 'freq_mhz': int(freq_mhz), 'seq': int(seq)}
        
    # ---------- Helpers ----------
    def _has_ltf_tag(self, rel_idx):
        abs_start = self.nitems_read(0)
        start = abs_start + rel_idx
        end   = start + 1
        for t in self.get_tags_in_range(0, start, end):
            try:
                key_str = pmt.symbol_to_string(t.key)
            except Exception:
                key_str = str(t.key)
            if (t.key in self.tag_keys) or (key_str in ("ltf","LLTF","ofdm_sync_long","wifi_start","frame_start")):
                return True, t
        return False, None

    def _looks_like_ltf_pair(self, a, b):
        num = np.vdot(a, b)
        den = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
        rho = np.abs(num / den)
        return rho >= self.corr_thresh, float(rho)

    def _ls_csi(self, Y):
        H64 = np.zeros(64, np.complex64)
        nz = X_LTF_64 != 0
        H64[nz] = Y[nz] / X_LTF_64[nz]
        return H64

    def _write_meta_jsonl(self, meta_dict):
        if self.meta_fh is None:
            return
        try:
            self.meta_fh.write(json.dumps(meta_dict, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # >>> tìm tag rx_time tại vị trí hiện tại (nếu có)
    def _scan_rx_time_tag(self, rel_idx):
        abs_start = self.nitems_read(0)
        start = abs_start + rel_idx
        end   = start + 1
        for t in self.get_tags_in_range(0, start, end):
            if t.key == self._rx_time_key:
                try:
                    secs = pmt.to_uint64(pmt.tuple_ref(t.value, 0))
                    frac = pmt.to_double(pmt.tuple_ref(t.value, 1))
                    self._last_rx_time_tuple = (float(secs), float(frac))
                    self._last_rx_time_off   = int(t.offset)
                except Exception:
                    pass

    # >>> tính tsf_us (microseconds) nếu biết rx_time + timebase; ngược lại trả None
    def _compute_tsf_us(self, abs_item_idx):
        if (self._last_rx_time_tuple is None or
            self._last_rx_time_off is None or
            self.sample_rate is None or self.sample_rate <= 0.0 or
            self.sym_samps <= 0):
            return None
        secs, frac = self._last_rx_time_tuple
        d_items = int(abs_item_idx) - int(self._last_rx_time_off)
        dt_s    = (d_items * self.sym_samps) / self.sample_rate
        tsf     = (secs + frac + dt_s) * 1e6
        return int(tsf)
    
    # ---------- GNU Radio ----------
    def work(self, input_items, output_items):
        vecs = input_items[0]  # shape: [N, 64]

        for rel_idx, v in enumerate(vecs):
            # >>> cố gắng bắt rx_time tag (nếu đi kèm ở domain hiện tại)
            self._scan_rx_time_tag(rel_idx)
            self._scan_rx_time_tag(rel_idx)  # cố đọc tag rx_time
            
            # 1) Cooldown — bỏ qua cho tới khi hết thời gian “nghỉ”
            if self.cooldown > 0:
                self.cooldown -= 1
                self.prev_vec = v.copy()
                self.sym_idx += 1
                continue

            # 2) Phát hiện LTF (ưu tiên tag; tương quan chỉ dự phòng)
            by_tag, tag = self._has_ltf_tag(rel_idx)
            by_corr, rho = (False, None)

            if (
                (not self.require_tag)
                and (not by_tag)
                and (self.prev_vec is not None)
            ):
                by_corr, rho = self._looks_like_ltf_pair(
                    self.prev_vec, v
                )

            is_ltf_here = by_tag or by_corr

            # 3) FSM: SEARCH → LTF1 → LTF2 → emit CSI → COOLDOWN
            if not self.seen_first_ltf:
                if is_ltf_here:
                    self.seen_first_ltf = True
            else:
                if is_ltf_here and (self.prev_vec is not None):
                    # Trung bình 2 LTF để giảm nhiễu
                    Y = 0.5 * (self.prev_vec + v)
                    H64 = self._ls_csi(Y)
                    Hact = H64[ACTIVE_BINS]  # (52,)

                    now = time.time()
                    # Rate limit (nếu đặt > 0)
                    if (self.min_emit_interval <= 0.0) or ((now - self.last_emit_ts) >= self.min_emit_interval):
                        # --- Build meta PMT với timestamp/freq/seq ---
                        abs_item_idx = self.nitems_read(0) + rel_idx
                        tsf_us = self._compute_tsf_us(abs_item_idx)
                        now_ms = int(now * 1000.0)

                        self.emit_id += 1

                        # meta PMT
                        meta_pmt = pmt.make_dict()
                        meta_pmt = pmt.dict_add(meta_pmt, pmt.intern("csi_idx"), pmt.from_long(self.emit_id))
                        
                        if self.rx_chan_id >= 0:
                            meta_pmt = pmt.dict_add(meta_pmt, pmt.intern("rx_chan"), pmt.from_long(self.rx_chan_id))
                        meta_pmt = pmt.dict_add(meta_pmt, pmt.intern("fft_len"), pmt.from_long(64))
                        meta_pmt = pmt.dict_add(meta_pmt, pmt.intern("active_bins"),
                                                pmt.init_u64vector(len(ACTIVE_BINS), ACTIVE_BINS.astype(np.uint64)))
                        meta_pmt = pmt.dict_add(meta_pmt, pmt.intern("sym_idx"), pmt.from_long(self.sym_idx))
                        meta_pmt = pmt.dict_add(meta_pmt, pmt.intern("ts_wall_ms"), pmt.from_long(now_ms))
                        if tsf_us is not None:
                            meta_pmt = pmt.dict_add(meta_pmt, pmt.intern("tsf_us"), pmt.from_long(int(tsf_us)))

                        # freq (nếu đã set từ ngoài)
                        if hasattr(self, "center_freq"):
                            try:
                                cf = int(float(self.center_freq))
                                meta_pmt = pmt.dict_add(meta_pmt, pmt.intern("freq"), pmt.from_long(cf))
                                freq_mhz = int(round(cf / 1e6))
                            except Exception:
                                pass
                        #         freq_mhz = None
                        # else:
                        #     freq_mhz = None

                        # (tuỳ chọn) chèn seq nếu có MAC gần đây & đúng tần số
                        seq_to_add = None
                        try:
                            lm = self._last_mac
                            if lm['ts_ms'] > 0:
                                if (time.monotonic()*1000.0 - lm['ts_ms']) <= self._mac_window_ms:
                                    if (freq_mhz is None) or (lm['freq_mhz'] == freq_mhz):
                                        seq_to_add = int(lm['seq'])
                        except Exception:
                            pass
                        if seq_to_add is not None:
                            meta_pmt = pmt.dict_add(meta_pmt, pmt.intern("seq"), pmt.from_long(seq_to_add))

                        


                        # --- Publish CSI (52 complex) ---
                        self.message_port_pub(
                            pmt.intern("csi"),
                            pmt.cons(meta_pmt, pmt.init_c32vector(len(Hact), np.array(Hact, dtype=np.complex64)))
                        )
                        self.last_emit_ts = now
                        self.meta_count += 1

                        # --- JSONL (mỗi meta_every_n lần) ---
                        if (self.meta_fh is not None) and (self.meta_count % self.meta_every_n == 0):
                            meta_py = {
                                "csi_idx": int(self.emit_id),
                                "ts_wall_ms": now_ms,
                                "rx_chan": self.rx_chan_id,
                                "fft_len": 64,
                                "csi_len": int(len(Hact)),
                                "sym_idx": int(self.sym_idx),
                                "detected_by": "tag" if by_tag else "corr",
                            }
                            if tsf_us is not None:
                                meta_py["tsf_us"] = int(tsf_us)
                            if hasattr(self, "center_freq"):
                                try: meta_py["freq"] = int(float(self.center_freq))
                                except Exception: pass
                            if seq_to_add is not None:
                                meta_py["seq"] = int(seq_to_add)
                            if rho is not None:
                                meta_py["corr_rho"] = float(rho)
                            if tag is not None:
                                try:  meta_py["tag_key"] = pmt.symbol_to_string(tag.key)
                                except: meta_py["tag_key"] = str(tag.key)
                                try:  meta_py["tag_offset"] = int(tag.offset)
                                except: pass
                            self._write_meta_jsonl(meta_py)
                        # --- Kết thúc ---

                        # Cooldown để tránh bơm CSI dồn dập
                        self.cooldown = self.cooldown_syms

                    # reset về SEARCH cho khung kế tiếp
                    self.seen_first_ltf = False

            # 4) Cập nhật trạng thái
            self.prev_vec = v.copy()
            self.sym_idx += 1
            

        return len(vecs)

    def __del__(self):
        try:
            if self.meta_fh:
                self.meta_fh.close()
        except Exception:
            pass
