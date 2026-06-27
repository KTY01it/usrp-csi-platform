import os, json
import numpy as np

# ======= THAM SỐ CHỈNH =======
PATH0 = "/home/ea301b/gr-ieee802-11/examples/csi_ch0.bin"   # đổi đúng đường dẫn bạn ghi file
PATH1 = "/home/ea301b/gr-ieee802-11/examples/csi_ch1.bin"
META0 = "/home/ea301b/gr-ieee802-11/examples/csi_ch0.jsonl" # nếu có sidecar; không có thì OK
META1 = "/home/ea301b/gr-ieee802-11/examples/csi_ch1.jsonl"

fc = 5.89e9         # Hz: tần số trung tâm thu thực tế của bạn (ví dụ 5.89 GHz)
c  = 3e8
lam = c / fc
d = lam/2           # m: khoảng cách 2 anten (nếu ≈ λ/2 là ổn cho AoA thô)

calib_frames = 100  # số khung dùng để hiệu chuẩn pha lệch phần cứng
win_frames   = 20   # độ dài cửa sổ trượt (số khung) để lấy AoA ổn định
hop_frames   = 5    # bước trượt (số khung)

# ======= HÀM PHỤ =======
def load_csi(path):
    x = np.fromfile(path, dtype=np.complex64)
    if len(x) % 52 != 0:
        raise ValueError(f"File {path} không chia hết cho 52 complex: len={len(x)}")
    return x.reshape(-1, 52)  # (num_frames, 52)

def circular_mean(ph, w=None, axis=0):
    """
    ph: góc (rad), w: trọng số >=0
    trả về: góc trung bình 'tròn' theo trục axis
    """
    if w is None:
        v = np.exp(1j*ph).mean(axis=axis)
    else:
        w = np.asarray(w)
        v = (w*np.exp(1j*ph)).sum(axis=axis) / (w.sum(axis=axis) + 1e-12)
    return np.angle(v)

def circular_subtract(a, b):
    """
    trả về (a - b) theo nghĩa góc, đã wrap về [-pi, pi]
    """
    return np.angle(np.exp(1j*(a - b)))

def try_load_meta_ts(meta_path):
    """Đọc ts_wall (epoch) từ JSONL nếu có để gán trục thời gian (tùy chọn)."""
    if not os.path.exists(meta_path): return None
    ts = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                ts.append(float(obj.get("ts_wall", np.nan)))
            except Exception:
                pass
    ts = np.array(ts, dtype=float)
    if len(ts) == 0 or np.all(~np.isfinite(ts)):
        return None
    return ts

# ======= CHẠY =======
H0 = load_csi(PATH0)   # (N0, 52)
H1 = load_csi(PATH1)   # (N1, 52)
N  = min(len(H0), len(H1))
H0 = H0[:N]
H1 = H1[:N]

# Chênh pha từng subcarrier, từng khung
# dphi[n,k] = angle( H1[n,k] * conj(H0[n,k]) )
dphi = np.angle(H1 * np.conj(H0))   # (N, 52)

# Trọng số theo biên độ để giảm ảnh hưởng subcarrier SNR thấp (tùy chọn)
w = np.abs(H0) * np.abs(H1)  # (N, 52)

# Hiệu chuẩn pha lệch phần cứng theo subcarrier (dùng calib_frames đầu)
M = min(calib_frames, N)
phi_bias = circular_mean(dphi[:M, :], w=w[:M, :], axis=0)  # (52,)

# Trừ bias theo nghĩa tròn
dphi_corr = circular_subtract(dphi, phi_bias)  # (N, 52)

# Lấy trung bình theo subcarrier (circular) -> 1 góc pha/khung
dphi_bar_per_frame = circular_mean(dphi_corr, w=w, axis=1)  # (N,)

# Cửa sổ trượt theo thời gian để mượt hơn
aoa_deg = []
centers = []
for start in range(0, N - win_frames + 1, hop_frames):
    sl = slice(start, start + win_frames)
    dphi_win = dphi_bar_per_frame[sl]

    # trung bình tròn qua cửa sổ (không cần trọng số thêm vì đã trung bình qua subcarrier)
    dphi_win_bar = circular_mean(dphi_win, axis=0)

    # Đổi pha -> góc đến: Δφ ≈ 2π d sinθ / λ
    arg = (lam / (2*np.pi*d)) * dphi_win_bar
    arg = np.clip(arg, -1.0, 1.0)  # tránh NaN do nhiễu
    theta = np.arcsin(arg)         # rad
    aoa_deg.append(np.degrees(theta))
    centers.append(start + win_frames//2)

aoa_deg = np.array(aoa_deg)
centers = np.array(centers)

# (Tùy chọn) trục thời gian từ meta JSONL nếu có
t0 = try_load_meta_ts(META0)
t1 = try_load_meta_ts(META1)
if t0 is not None and t1 is not None:
    # căn độ dài và dùng trung bình của 2 kênh làm thời gian
    T = min(len(t0), len(t1), N)
    t = 0.5*(t0[:T] + t1[:T])
    # nội suy t theo chỉ số frame và lấy theo centers
    # (đơn giản: lấy trực tiếp nếu 1-1)
    if len(t) == N:
        t_centers = t[centers]
    else:
        # fallback: nội suy tuyến tính theo index
        idx = np.arange(len(t))
        t_centers = np.interp(centers, idx, t)
else:
    # nếu không có meta, dùng chỉ số frame làm “thời gian”
    t_centers = centers.astype(float)

# In vài giá trị kiểm tra
print(f"N={N}, số điểm AoA={len(aoa_deg)}")
print("AoA (deg) 10 giá trị đầu:", np.round(aoa_deg[:10], 2))

# (Tùy chọn) lưu CSV
OUT = "/home/ea301b/gr-ieee802-11/examples/aoa_timeseries.csv"
with open(OUT, "w") as f:
    f.write("index_or_time,aoa_deg\n")
    for tt, ang in zip(t_centers, aoa_deg):
        f.write(f"{tt},{ang}\n")
print("Đã lưu:", OUT)
