#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phase_calibrate.py
- Quét thư mục dữ liệu (đã ghi bởi wifi_rx_*), gom theo từng center_freq (Hz)
- Với một kênh RX chọn (rx_chan), tính pha trung bình band và lấy band đầu làm mốc
- Xuất JSON { "<freq_MHz>": phi_rad } để trừ đi khi stitch wideband
Lưu ý: dữ liệu calib nên là loopback/cáp (kênh gần như 1 tap) hoặc LOS tĩnh rất sạch.
"""

import os, json, argparse, glob
import numpy as np

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

def load_dataset(root, rx_chan):
    """
    Đọc tất cả cặp file ch{rx_chan}_*.{bin,jsonl} trong mọi thư mục fXXXXMHz dưới root.
    Trả về dict: freq_Hz -> list of records (mỗi record chứa '_vec': 52 complex)
    """
    freq_map = {}
    # mẫu đường dẫn: <root>/f5180MHz/ch0_*.bin & .jsonl
    for freq_dir in glob.glob(os.path.join(root, "f*MHz")):
        # tách tần số
        base = os.path.basename(freq_dir)
        try:
            f_mhz = int(base[1:-3])  # "f5180MHz" -> 5180
        except Exception:
            continue
        bin_files = sorted(glob.glob(os.path.join(freq_dir, f"ch{rx_chan}_*.bin")))
        for b in bin_files:
            j = b[:-4] + ".jsonl"
            if not os.path.exists(j):
                continue
            recs = _pair_bin_json(b, j)
            if not recs:
                continue
            # nếu meta không có 'freq', đặt từ thư mục
            for r in recs:
                if "freq" not in r or r["freq"] is None:
                    r["freq"] = int(f_mhz * 1e6)
            freq_map.setdefault(int(f_mhz*1e6), []).extend(recs)
    return freq_map

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Thư mục Data (chứa fXXXXMHz/...)")
    ap.add_argument("--rx-chan", type=int, default=0, help="Kênh RX dùng để calib (0..3)")
    ap.add_argument("--out", default="phase_cal.json", help="File JSON lưu phi_i per band")
    ap.add_argument("--agg", choices=["mean","median"], default="median", help="Cách tổng hợp theo thời gian trong mỗi band")
    args = ap.parse_args()

    freq_map = load_dataset(args.root, args.rx_chan)
    if not freq_map:
        raise SystemExit("Không tìm thấy dữ liệu phù hợp.")

    # Pha trung bình/median toàn band (gộp 52 SC và toàn thời gian mỗi band)
    band_phase = {}
    for fHz, recs in sorted(freq_map.items()):
        # ghép các vector theo thời gian
        M = np.stack([r["_vec"] for r in recs], axis=0)  # [T, 52]
        if args.agg == "median":
            v = np.median(M, axis=0)
        else:
            v = np.mean(M, axis=0)
        # pha trung bình toàn 52 SC
        phi = np.angle(np.mean(v))
        band_phase[int(round(fHz/1e6))] = float(phi)

    # chuẩn hoá: lấy band nhỏ nhất làm mốc (trừ đi)
    if band_phase:
        ref_key = sorted(band_phase.keys())[0]
        ref_phi = band_phase[ref_key]
        for k in band_phase:
            band_phase[k] = float(np.angle(np.exp(1j*(band_phase[k]-ref_phi))))  # wrap về (-pi,pi]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(band_phase, f, indent=2)
    print(f"[OK] Lưu calib pha vào {args.out}")
    print(json.dumps(band_phase, indent=2))

if __name__ == "__main__":
    main()
