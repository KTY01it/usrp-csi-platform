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
                 meta_every_n=100):           # ghi JSON thưa
        gr.sync_block.__init__(self,
            name="csi_ltf_estimator",
            in_sig=[(np.complex64, 64)],
            out_sig=None)

        # ===== Ports =====
        self.message_port_register_out(pmt.intern("csi"))

        # ===== Tag keys =====
        if ltf_tag_key is not None:   # tương thích kiểu cũ (1 key)
            ltf_tag_keys = (ltf_tag_key,)
        self.tag_keys = [pmt.intern(k) for k in ltf_tag_keys]

        # ===== Runtime state =====
        self.rx_chan_id = int(rx_chan_id)
        self.prev_vec = None
        self.sym_idx = 0
        self.corr_thresh = float(corr_thresh)

        # Chống “mưa CSI”
        self.cooldown_syms = 160      # ~ mỗi khung 1 CSI
        self.cooldown = 0
        self.seen_first_ltf = False
        self.min_emit_interval = 0.05 # ≥50 ms → tối đa ~20 Hz
        self.last_emit_ts = 0.0

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

    # ---------- GNU Radio ----------
    def work(self, input_items, output_items):
        vecs = input_items[0]  # shape: [N, 64]

        for rel_idx, v in enumerate(vecs):
            # 1) Cooldown — bỏ qua cho tới khi hết thời gian “nghỉ”
            if self.cooldown > 0:
                self.cooldown -= 1
                self.prev_vec = v.copy()
                self.sym_idx += 1
                continue

            # 2) Phát hiện LTF (ưu tiên tag; tương quan chỉ dự phòng)
            by_tag, tag = self._has_ltf_tag(rel_idx)
            by_corr, rho = (False, None)
            if (not by_tag) and (self.prev_vec is not None):
                by_corr, rho = self._looks_like_ltf_pair(self.prev_vec, v)
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
                        # --- Publish PDU CSI (52 cpx) ---
                        meta_pmt = pmt.make_dict()
                        if self.rx_chan_id >= 0:
                            meta_pmt = pmt.dict_add(meta_pmt, pmt.intern("rx_chan"), pmt.from_long(self.rx_chan_id))
                        meta_pmt = pmt.dict_add(meta_pmt, pmt.intern("fft_len"), pmt.from_long(64))
                        meta_pmt = pmt.dict_add(
                            meta_pmt, pmt.intern("active_bins"),
                            pmt.init_u64vector(len(ACTIVE_BINS), ACTIVE_BINS.astype(np.uint64))
                        )
                        meta_pmt = pmt.dict_add(meta_pmt, pmt.intern("sym_idx"), pmt.from_long(self.sym_idx))

                        self.message_port_pub(
                            pmt.intern("csi"),
                            pmt.cons(meta_pmt, pmt.init_c32vector(len(Hact), np.array(Hact, dtype=np.complex64)))
                        )
                        self.last_emit_ts = now
                        self.meta_count += 1

                        # --- JSONL thưa (nếu bật) ---
                        if (self.meta_fh is not None) and (self.meta_count % self.meta_every_n == 0):
                            meta_py = {
                                "ts_wall": now,
                                "rx_chan": self.rx_chan_id,
                                "fft_len": 64,
                                "csi_len": int(len(Hact)),
                                "sym_idx": int(self.sym_idx),
                                "detected_by": "tag" if by_tag else "corr",
                            }
                            if hasattr(self, "center_freq"):
                                try:
                                    meta_py["center_freq"] = float(self.center_freq)
                                except Exception:
                                    pass
                            if rho is not None:
                                meta_py["corr_rho"] = float(rho)
                            if tag is not None:
                                try:
                                    meta_py["tag_key"] = pmt.symbol_to_string(tag.key)
                                except Exception:
                                    meta_py["tag_key"] = str(tag.key)
                                try:
                                    meta_py["tag_offset"] = int(tag.offset)
                                except Exception:
                                    pass
                            self._write_meta_jsonl(meta_py)

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
