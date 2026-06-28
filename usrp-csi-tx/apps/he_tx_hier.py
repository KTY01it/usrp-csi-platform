# he_tx_hier.py
# GNU Radio 3.10 — Phát HE-SU NDP (802.11ax 20 MHz, FFT=256)
# RU=242 tone (234 data + 8 pilot), GI=1.6 us, NSS=1
# NOTE: Đây là NDP (không có HE-Data). Nếu cần chuẩn hóa tuyệt đối,
# bạn có thể thay phần Legacy=64-FFT chính xác + mã bit HE-SIG-A theo chuẩn.

from gnuradio import gr, blocks
import numpy as np

# =========================
# === BẢNG THAM SỐ 11ax ===
# =========================

FFT_LEN = 256

# RU-242 (full 20 MHz): active subcarriers ở hệ trục tần số k∈[-128..+127], bỏ DC.
# Chọn dải hoạt động: [-121..-1] ∪ [1..121]  → 242 tone (đúng tổng 242).
RU242_ACTIVE_K = np.array(list(range(-121, 0)) + list(range(1, 122)), dtype=int)

# Pilot 256-FFT (8 vị trí) dùng bộ như VHT-80 (chuẩn công nghiệp; 11ax HE-20 cùng 256-FFT):
# ±11, ±39, ±75, ±103
PILOT_K = np.array([-103, -75, -39, -11, 11, 39, 75, 103], dtype=int)

def k_to_fft_idx(k: np.ndarray, nfft: int = FFT_LEN) -> np.ndarray:
    """Đổi chỉ số tần số k∈[-N/2..N/2-1] sang chỉ số mảng [0..N-1] (ifftshift)."""
    return (k + nfft) % nfft

# Pilot polarity theo OFDM symbol: với NDP/preamble ta có ít symbol, cho đơn giản dùng +1.
# (Nếu bạn muốn "đúng bài" hơn, có thể thay bằng chuỗi polarity theo chuẩn.)
def pilot_polarity(sym_idx: int, n_pilots: int) -> np.ndarray:
    return np.ones(n_pilots, dtype=np.float32)

# HE-LTF sequence cho NSS=1: chuỗi Hadamard-based → với 1 stream có thể lấy toàn +1 (chuẩn vẫn hợp lệ).
def he_ltf_sequence_nss1(n_active: int) -> np.ndarray:
    return np.ones(n_active, dtype=np.complex64)

# ===== Legacy phần (xấp xỉ) =====
# Để đơn giản, tạo Legacy (L-STF/L-LTF/L-SIG/RL-SIG) trên 256-FFT bằng core 52-tone mở rộng.
# Muốn nghiêm chỉnh tuyệt đối → tự build 64-FFT rồi nội suy/ghép về Fs=20 Msps.
LEGACY_CORE = np.r_[range(-26, 0), range(1, 27)]

def legacy_symbol_approx(kind: str, nfft: int = FFT_LEN) -> np.ndarray:
    X = np.zeros(nfft, dtype=np.complex64)
    idx = k_to_fft_idx(LEGACY_CORE, nfft)
    amp = 1.0
    if kind == "L-STF":
        amp = 1.2
    X[idx] = amp + 0j
    return X

def ifft_cp(Xk: np.ndarray, cp_len: int) -> np.ndarray:
    x = np.fft.ifft(np.fft.ifftshift(Xk)).astype(np.complex64)
    if cp_len > 0:
        return np.concatenate([x[-cp_len:], x])
    return x

class he_tx_hier(gr.hier_block2):
    def __init__(self,
                 bw=20e6,
                 gi_us=1.6,
                 nss=1,
                 n_he_ltf=1,
                 ru='242',
                 ndp_interval_ms=300,
                 scale=1.0):
        gr.hier_block2.__init__(
            self, "he_tx_hier",
            gr.io_signature(0, 0, 0),
            gr.io_signature(1, 1, np.complex64().nbytes)
        )

        # Numerology
        self.fs = float(bw)            # 20e6
        self.nfft = FFT_LEN            # 256
        self.delta_f = self.fs / self.nfft  # 78.125 kHz
        self.t_sym = 1.0 / self.delta_f     # 12.8 us
        self.gi = gi_us * 1e-6
        self.cp_len_he = int(round(self.fs * self.gi))     # GI cho HE (1.6us -> 32 mẫu @20Msps)
        self.cp_len_leg = int(round(self.fs * 0.8e-6))     # GI legacy ~0.8us

        # Tiền biên dịch các vị trí
        self.active_bins = k_to_fft_idx(RU242_ACTIVE_K, self.nfft)
        self.pilot_bins  = k_to_fft_idx(PILOT_K, self.nfft)

        # Xây 1 khung HE-SU NDP (Legacy + HE-SIG-A + HE-STF + HE-LTFs)
        iq_ndp = self._build_he_su_ndp(nss=nss, n_he_ltf=n_he_ltf)

        # Chèn khoảng nghỉ để PicoScenes dễ bám — mặc định 300 ms
        total_len = len(iq_ndp)
        burst_period = int(round(self.fs * (ndp_interval_ms / 1000.0)))
        gap_len = max( int(0.02*self.fs), burst_period - total_len )   # tối thiểu 20 ms idle
        gap = np.zeros(gap_len, dtype=np.complex64)

        burst = (iq_ndp * scale).astype(np.complex64)
        full_vec = np.concatenate([burst, gap])

        self.src = blocks.vector_source_c(full_vec.tolist(), True, 1, [])
        self.connect(self.src, self)

    def _build_he_su_ndp(self, nss=1, n_he_ltf=1) -> np.ndarray:
        fields = []

        # ===== 1) Legacy portion (xấp xỉ) =====
        for kind in ["L-STF", "L-LTF", "L-SIG", "RL-SIG"]:
            X = legacy_symbol_approx(kind, self.nfft)
            fields.append(ifft_cp(X, self.cp_len_leg))

        # ===== 2) HE-SIG-A (SU, LENGTH=0 vì NDP) — placeholder: BPSK trên RU, pilot đúng vị trí =====
        X_siga = np.zeros(self.nfft, dtype=np.complex64)
        X_siga[self.active_bins] = 1 + 0j  # Data BPSK giả định
        pol = pilot_polarity(sym_idx=0, n_pilots=len(self.pilot_bins))
        X_siga[self.pilot_bins] = pol.astype(np.float32) + 0j
        fields.append(ifft_cp(X_siga, self.cp_len_he))

        # ===== 3) HE-STF — placeholder: pattern đơn giản trên toàn RU =====
        X_hestf = np.zeros(self.nfft, dtype=np.complex64)
        X_hestf[self.active_bins] = 1 + 0j
        fields.append(ifft_cp(X_hestf, self.cp_len_he))

        # ===== 4) HE-LTF (>=1) — BẮT BUỘC ĐỂ LẤY HE-CSI =====
        for s in range(n_he_ltf):
            X_heltf = np.zeros(self.nfft, dtype=np.complex64)
            seq = he_ltf_sequence_nss1(len(self.active_bins))
            X_heltf[self.active_bins] = seq
            pol = pilot_polarity(sym_idx=s, n_pilots=len(self.pilot_bins))
            X_heltf[self.pilot_bins] = pol.astype(np.float32) + 0j
            fields.append(ifft_cp(X_heltf, self.cp_len_he))

        return np.concatenate(fields).astype(np.complex64)
