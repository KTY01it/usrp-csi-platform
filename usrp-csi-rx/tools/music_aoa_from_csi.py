import os, json
import numpy as np

# ========= THAM SỐ =========
# ĐƯỜNG DẪN (đổi theo thư mục bạn lưu)
DIR = "/home/ea301b/gr-ieee802-11/examples/Data/1/f5180MHz"
CH0_BIN = f"{DIR}/ch0_20250903_222609.bin"
CH1_BIN = f"{DIR}/ch1_20250903_222609.bin"
CH0_JSON = f"{DIR}/ch0_20250903_222609.jsonl"
CH1_JSON = f"{DIR}/ch1_20250903_222609.jsonl"

# THÔNG SỐ VẬT LÝ
fc_hz = 5.18e9        # tần số trung tâm (Hz) — ví dụ 5180 MHz
chan_bw_hz = 20e6     # băng thông 802.11a=20e6, 802.11p=10e6 → đặt 10e6 nếu bạn thu 10 MHz
d_m = None            # khoảng cách anten (m). Nếu None → dùng d = λ/2
c = 3e8               # tốc độ ánh sáng
lam = c / fc_hz
if d_m is None:
    d_m = lam / 2

# CỬA SỔ/HIỆU CHUẨN
calib_frames = 100    # số khung đầu để hiệu chuẩn lệch pha phần cứng theo subcarrier
win_frames   = 20     # số khung dùng để tính 1 ước lượng MUSIC (trung bình theo thời gian)
hop_frames   = 5      # bước trượt

# LƯỚI QUÉT MUSIC
angle_grid_deg = np.linspace(-90, 90, 721)  # bước 0.25°

# ========= HÀM PHỤ =========
ACTIVE_K = np.r_[np.arange(-26,0), np.arange(1,27)]  # -26..-1, +1..+26
delta_f = chan_bw_hz / 64.0
# tần số từng subcarrier active:
f_k = fc_hz + ACTIVE_K * delta_f
lam_k = c / f_k  # bước sóng từng subcarrier

def load_csi_matrix(path_bin):
    x = np.fromfile(path_bin, dtype=np.complex64)
    if len(x) % 52 != 0:
        raise ValueError(f"{path_bin}: kích thước không chia hết cho 52 (complex64). len={len(x)}")
    return x.reshape(-1, 52)  # (N, 52)

def load_idx_jsonl(path_json):
    idx = []
    ts = []
    if not os.path.exists(path_json):
        return None, None
    with open(path_json, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                idx.append(int(obj["csi_idx"]))
                # dùng ms nếu có, ngược lại bỏ qua
                ts.append(float(obj.get("ts_wall_ms", np.nan)))
            except Exception:
                pass
    idx = np.array(idx, dtype=int)
    ts  = np.array(ts, dtype=float)
    return idx, ts

def align_by_idx(H0, idx0, H1, idx1):
    """Căn theo giao idx (csi_idx) để chắc chắn H0/H1 cùng gói."""
    if idx0 is None or idx1 is None:  # không có sidecar -> fallback: cắt theo min length
        N = min(len(H0), len(H1))
        return H0[:N], H1[:N], np.arange(N), None
    common = np.intersect1d(idx0, idx1)
    if len(common) == 0:
        raise RuntimeError("Không có csi_idx chung giữa ch0 và ch1")
    # map idx -> vị trí dòng
    map0 = {int(i): p for p, i in enumerate(idx0)}
    map1 = {int(i): p for p, i in enumerate(idx1)}
    i0 = np.array([map0[int(i)] for i in common], dtype=int)
    i1 = np.array([map1[int(i)] for i in common], dtype=int)
    return H0[i0], H1[i1], common, None

def circ_mean(ph, w=None, axis=0):
    if w is None:
        v = np.exp(1j*ph).mean(axis=axis)
    else:
        w = np.asarray(w)
        v = (w*np.exp(1j*ph)).sum(axis=axis) / (w.sum(axis=axis) + 1e-12)
    return np.angle(v)

def circ_sub(a, b):
    return np.angle(np.exp(1j*(a - b)))

def music_pseudospectrum(R, d_m, lam_k, angle_grid_deg):
    """
    R: ma trận hiệp phương sai 2x2 (đã trung bình theo thời gian/subcarrier)
    Trả về P_MUSIC(θ) theo θ∈angle_grid_deg. Với 2 anten, số nguồn giả định = 1.
    Dùng steering theo từng λ_k và lấy trung bình mẫu phổ theo subcarrier.
    """
    # phân rã riêng trị
    vals, vecs = np.linalg.eigh(R)            # tăng dần
    En = vecs[:, [0]]                         # eigenvector ứng với eigenvalue nhỏ hơn (noise)
    EnEnH = En @ En.conj().T                  # 2x2

    ang_rad = np.deg2rad(angle_grid_deg)
    P = np.zeros_like(ang_rad, dtype=float)
    # Trung bình theo subcarrier: với mỗi θ, tính a_k rồi cộng nghịch đảo khoảng cách tới noise-subspace
    for n, th in enumerate(ang_rad):
        sinth = np.sin(th)
        denom = 0.0
        for lk in lam_k:
            a = np.array([1.0, np.exp(-1j * 2*np.pi * d_m * sinth / lk)], dtype=np.complex64).reshape(2,1)
            denom += np.real((a.conj().T @ EnEnH @ a)[0,0])
        P[n] = 1.0 / (denom / len(lam_k) + 1e-15)
    return P

# ========= MAIN =========
# 1) Đọc & căn chỉnh theo csi_idx
H0 = load_csi_matrix(CH0_BIN)
H1 = load_csi_matrix(CH1_BIN)
idx0, _ = load_idx_jsonl(CH0_JSON)
idx1, _ = load_idx_jsonl(CH1_JSON)
H0, H1, idx_common, _ = align_by_idx(H0, idx0, H1, idx1)
N = len(H0)
print(f"[INFO] số khung khớp theo csi_idx: {N}")

# 2) Hiệu chuẩn lệch pha phần cứng theo subcarrier (dựa trên calib_frames đầu)
M = min(calib_frames, N)
# chênh pha từng subcarrier, từng khung
dphi = np.angle(H1[:M] * np.conj(H0[:M]))     # (M,52)
w    = np.abs(H0[:M]) * np.abs(H1[:M])        # trọng số theo biên độ
phi_bias = circ_mean(dphi, w=w, axis=0)       # (52,)
# áp dụng hiệu chuẩn cho toàn bộ chuỗi
H1_cal = H1 * np.exp(-1j*phi_bias)            # (N,52)
# (tùy chọn) cân biên độ giữa hai kênh để đỡ méo R
amp_scale = (np.abs(H0[:M]).mean() + 1e-12) / (np.abs(H1[:M]).mean() + 1e-12)
H1_cal = H1_cal * amp_scale

# 3) MUSIC theo cửa sổ trượt
angle_peaks = []
for start in range(0, N - win_frames + 1, hop_frames):
    sl = slice(start, start + win_frames)

    # tạo các snapshot 2xK từ T khung và 52 subcarrier
    # cách 1 (đơn giản & hiệu quả): tính R = E[x x^H] với x=[H0;H1] gộp qua thời gian & subcarrier
    x0 = H0[sl].reshape(-1, 52)
    x1 = H1_cal[sl].reshape(-1, 52)
    # loại subcarrier biên băng (tùy chọn): bỏ 2 rìa mỗi phía
    keep = np.r_[np.arange(2,26), np.arange(26+1, 52-2)]  # giữ 48 subcarriers
    x0 = x0[:, keep]
    x1 = x1[:, keep]
    lam_keep = lam_k[keep]

    X = np.vstack([x0.flatten(), x1.flatten()])  # (2, T*K')
    R = (X @ X.conj().T) / X.shape[1]           # (2x2)

    # MUSIC
    P = music_pseudospectrum(R, d_m, lam_keep, angle_grid_deg)
    kmax = int(np.argmax(P))
    angle_peaks.append(angle_grid_deg[kmax])

angle_peaks = np.array(angle_peaks)
print("[INFO] AoA MUSIC (deg) - 10 giá trị đầu:", np.round(angle_peaks[:10], 2))

# 4) Lưu CSV
out_csv = os.path.join(DIR, "aoa_music_timeseries.csv")
with open(out_csv, "w") as f:
    f.write("win_center_index,aoa_deg\n")
    centers = np.arange(0, N - win_frames + 1, hop_frames) + win_frames//2
    for i, ang in zip(centers, angle_peaks):
        f.write(f"{int(i)},{float(ang)}\n")
print("[INFO] Đã lưu:", out_csv)
