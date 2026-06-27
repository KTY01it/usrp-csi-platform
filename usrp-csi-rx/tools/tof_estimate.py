#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tof_estimate.py
- Đọc dữ liệu CSI ghi bởi wifi_rx_* (nhiều band), lọc theo seq (nếu có)
- Stitch wideband: gom 52 subcarrier của từng band với khoảng cách Δf = samp_rate/64
- Áp dụng calib pha per-band (phi_i) nếu cung cấp
- Tính ToF bằng NUDFT/IFFT rải tần (scan τ), in ra đỉnh mạnh nhất & vài đỉnh tiếp theo
"""

import os, json, argparse, glob, math
import numpy as np

ACTIVE_K = np.r_[np.arange(-26, 0), np.arange(1, 27)]  # 52 SC

def _read_bin(path):
    x = np.fromfile(path, dtype=np.complex64)
    if x.size % 52 != 0:
        n = x.size // 52
        x = x[:n*52]
    return x.reshape(-1, 52)

def _read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

def _pair_bin_json(bin_path, json_path):
    X = _read_bin(bin_path)
    J = _read_jsonl(json_path)
    n = min(len(J), len(X))
    if n == 0:
        return []
    X = X[:n]
    J = J[:n]
    recs = []
    for i in range(n):
        d = J[i]
        d["_vec"] = X[i]     # 52 complex
        recs.append(d)
    return recs

def load_all_records(root, rx_chan):
    """
    Tải tất cả record cho 1 kênh RX, gộp từ mọi band/thư mục.
    Trả về list record (mỗi record có freq, seq?, ts, _vec(52)).
    """
    recs_all = []
    for freq_dir in glob.glob(os.path.join(root, "f*MHz")):
        base = os.path.basename(freq_dir)
        try:
            f_mhz = int(base[1:-3])
        except Exception:
            continue
        for b in sorted(glob.glob(os.path.join(freq_dir, f"ch{rx_chan}_*.bin"))):
            j = b[:-4] + ".jsonl"
            if not os.path.exists(j):
                continue
            recs = _pair_bin_json(b, j)
            for r in recs:
                if "freq" not in r or r["freq"] is None:
                    r["freq"] = int(f_mhz*1e6)
            recs_all.extend(recs)
    return recs_all

def choose_seq(recs):
    """Chọn SEQ phổ biến nhất (nếu tồn tại); nếu không có 'seq' thì trả None."""
    seqs = [r.get("seq") for r in recs if "seq" in r]
    if not seqs:
        return None
    vals, cnts = np.unique(seqs, return_counts=True)
    return int(vals[np.argmax(cnts)])

def aggregate_per_band(records, agg="median"):
    """
    Gom theo center_freq; với mỗi band lấy vector đại diện theo thời gian (median/mean).
    Trả về dict: freq_Hz -> complex[52]
    """
    by_f = {}
    for r in records:
        f = int(r["freq"])
        by_f.setdefault(f, []).append(r["_vec"])
    band = {}
    for f, vecs in by_f.items():
        M = np.stack(vecs, axis=0)  # [T, 52]
        v = np.median(M, axis=0) if agg=="median" else np.mean(M, axis=0)
        band[f] = v
    return band

def apply_phase_cal(band_vecs, cal_dict):
    """
    band_vecs: dict freq_Hz -> v(52)
    cal_dict: {"5180": phi_rad, ...} (freq MHz string/int)
    Trả về dict mới đã trừ φᵢ.
    """
    if not cal_dict:
        return band_vecs
    out = {}
    for fHz, v in band_vecs.items():
        kMHz = str(int(round(fHz/1e6)))
        phi = float(cal_dict.get(kMHz, 0.0))
        out[fHz] = v * np.exp(-1j*phi)
    return out

def stitch_frequency_grid(band_vecs, samp_rate):
    """
    Dựng lưới (f_list, H_list) từ nhiều band.
    Δf = samp_rate / 64 (Hz) (tương thích với 802.11 FFT-64)
    """
    df = float(samp_rate) / 64.0
    f_all = []
    H_all = []
    for f0, v in sorted(band_vecs.items()):
        # active k: [-26..-1, 1..26]
        ks = ACTIVE_K.astype(np.float64)
        fks = f0 + ks * df
        f_all.append(fks)
        H_all.append(v.astype(np.complex128))
    f_all = np.concatenate(f_all)      # [52*nband]
    H_all = np.concatenate(H_all)
    # Lọc NaN/zero nếu có
    m = np.isfinite(f_all) & np.isfinite(H_all.real) & np.isfinite(H_all.imag)
    return f_all[m], H_all[m]

def estimate_tof(f_Hz, H, tau_max_ns=300.0, tau_step_ns=0.1, normalize=True):
    """
    NUDFT trên lưới τ: h(τ) = Σ H(f) * exp(+j 2π f τ)
    - f_Hz: (N,) Hz
    - H: (N,) complex
    - τ scan: 0..tau_max_ns, bước tau_step_ns
    Trả về (taus_ns, |h|, idx_peaks_sorted)
    """
    taus = np.arange(0.0, tau_max_ns + 1e-12, tau_step_ns) * 1e-9  # s
    # Vector hoá: exp(j 2π f τ) với shape [N, T]
    # Cẩn thận memory: làm batch nếu N*T quá lớn.
    N = f_Hz.size
    T = taus.size
    # Normalization
    W = 1.0 / max(1, N) if normalize else 1.0
    # chunk để tránh tốn RAM
    chunk = 20000  # tuỳ
    h_tau = np.zeros(T, dtype=np.complex128)
    for i0 in range(0, N, chunk):
        ii = slice(i0, min(N, i0+chunk))
        ex = np.exp(1j * 2*np.pi * np.outer(f_Hz[ii], taus))  # [nchunk, T]
        h_tau += (H[ii][:,None] * ex).sum(axis=0)
    h_abs = np.abs(W * h_tau)
    # Tìm đỉnh (đơn giản): lấy top-5
    order = np.argsort(h_abs)[::-1]
    return (taus*1e9, h_abs, order[:5])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Thư mục Data (chứa fXXXXMHz/...)")
    ap.add_argument("--rx-chan", type=int, default=0, help="Kênh RX dùng để tính ToF (0..3)")
    ap.add_argument("--seq", type=int, default=None, help="Chỉ lấy bản ghi có SEQ này; nếu bỏ trống sẽ tự chọn SEQ phổ biến")
    ap.add_argument("--samp-rate", type=float, default=10e6, help="Sample rate mỗi band (Hz): 10e6 hoặc 20e6…")
    ap.add_argument("--phase-cal", default=None, help="File JSON phi_i per-band (tạo bởi phase_calibrate.py)")
    ap.add_argument("--agg", choices=["mean","median"], default="median", help="Tổng hợp theo thời gian trong mỗi band")
    ap.add_argument("--tau-max-ns", type=float, default=300.0)
    ap.add_argument("--tau-step-ns", type=float, default=0.1)
    args = ap.parse_args()

    recs = load_all_records(args.root, args.rx_chan)
    if not recs:
        raise SystemExit("Không tìm thấy dữ liệu.")

    # Lọc theo SEQ
    seq = args.seq
    if seq is None:
        seq = choose_seq(recs)
        if seq is not None:
            print(f"[INFO] Chọn tự động SEQ phổ biến: {seq}")
    if seq is not None:
        recs = [r for r in recs if r.get("seq")==seq]
        if not recs:
            raise SystemExit(f"Không có bản ghi với seq={seq}")

    # Gom theo band & tổng hợp theo thời gian
    band_vecs = aggregate_per_band(recs, agg=args.agg)

    # Áp dụng calib pha (nếu có)
    cal = None
    if args.phase_cal and os.path.exists(args.phase_cal):
        with open(args.phase_cal, "r", encoding="utf-8") as f:
            cal = json.load(f)
        print(f"[INFO] Áp dụng phase calib từ {args.phase_cal}")
    band_vecs = apply_phase_cal(band_vecs, cal)

    # Stitch thành lưới f, H
    f_Hz, H = stitch_frequency_grid(band_vecs, args.samp_rate)
    # Khử chuẩn biên độ tổng (tuỳ chọn)
    # H = H / (np.max(np.abs(H)) + 1e-12)

    taus_ns, h_abs, top_idx = estimate_tof(
        f_Hz, H, tau_max_ns=args.tau_max_ns, tau_step_ns=args.tau_step_ns, normalize=True
    )

    # In kết quả
    print("\n=== Kết quả ToF (top-5 đỉnh) ===")
    for rank, i in enumerate(top_idx, 1):
        print(f"{rank:>2}. τ ≈ {taus_ns[i]:8.3f} ns | |h| = {h_abs[i]:.4f}")
    if len(top_idx):
        print(f"\n=> Gợi ý ToF (đỉnh 1): ~{taus_ns[top_idx[0]]:.3f} ns")

if __name__ == "__main__":
    main()
