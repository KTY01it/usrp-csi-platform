# # convert_csi_bin_with_meta.py
# import os, sys, json, numpy as np, pandas as pd
# CAND_KEYS = ["packet_len","csi_len","num_complex","ncplx","len","Nsc","n_subcarriers"]

# def read_complex_bin(bin_path):
#     a = np.fromfile(bin_path, dtype=np.complex64)
#     if a.size % 1 != 0:
#         print("Warning: size not multiple of complex64? bytes:", os.path.getsize(bin_path))
#     return a

# def read_jsonl_lengths(jsonl_path):
#     lens = []
#     metas = []
#     with open(jsonl_path, "r", encoding="utf-8") as f:
#         for ln in f:
#             if not ln.strip(): continue
#             m = json.loads(ln)
#             metas.append(m)
#             L = None
#             # thử các khóa phổ biến
#             for k in CAND_KEYS:
#                 if k in m and isinstance(m[k], int) and m[k] > 0:
#                     L = m[k]; break
#             # một số block có thể ghi "shape": [N] hoặc "csi_shape"
#             if L is None:
#                 for k in ["shape","csi_shape"]:
#                     if k in m and isinstance(m[k], (list,tuple)) and len(m[k])>0:
#                         maybe = int(m[k][-1])
#                         if maybe>0: L = maybe; break
#             if L is None:
#                 raise ValueError(f"Không tìm thấy độ dài PDU trong meta: {m.keys()}")
#             lens.append(int(L))
#     return lens, metas

# def slice_frames(flat, lengths):
#     idx = 0; frames = []
#     for L in lengths:
#         frames.append(flat[idx: idx+L])
#         idx += L
#     if idx != flat.size:
#         print(f"WARNING: tổng length meta = {idx}, nhưng bin có {flat.size}. Số mẫu thừa sẽ bị bỏ qua." )
#     return frames

# def save_frames_to_csv(frames, metas, out_dir, separate=True):
#     os.makedirs(out_dir, exist_ok=True)
#     if separate:
#         for i,(frm,meta) in enumerate(zip(frames, metas), 1):
#             df = pd.DataFrame({
#                 "subc":  np.arange(frm.size, dtype=np.int32),
#                 "real":  frm.real.astype(np.float32),
#                 "imag":  frm.imag.astype(np.float32),
#                 "mag":   np.abs(frm).astype(np.float32),
#                 "phase": np.angle(frm).astype(np.float32),
#             })
#             # thêm một vài trường meta hữu ích nếu có
#             for k in ["seq","center_freq","rx_chan","ts_ms","freq_mhz"]:
#                 if k in meta: df[k] = meta[k]
#             out_csv = os.path.join(out_dir, f"frame_{i:05d}.csv")
#             df.to_csv(out_csv, index=False)
#         print(f"Saved {len(frames)} frame CSVs to: {out_dir}")
#     else:
#         # gộp một file
#         rows = []
#         for i,(frm,meta) in enumerate(zip(frames, metas), 1):
#             rows.append(pd.DataFrame({
#                 "frame_id": i,
#                 "subc":     np.arange(frm.size, dtype=np.int32),
#                 "real":     frm.real.astype(np.float32),
#                 "imag":     frm.imag.astype(np.float32),
#                 "mag":      np.abs(frm).astype(np.float32),
#                 "phase":    np.angle(frm).astype(np.float32),
#             }))
#         big = pd.concat(rows, ignore_index=True)
#         out_csv = os.path.join(out_dir, "all_frames.csv")
#         big.to_csv(out_csv, index=False)
#         print(f"Saved: {out_csv}  (rows={len(big):,})")

# if __name__ == "__main__":
#     # Usage:
#     #   python3 convert_csi_bin_with_meta.py /path/to/ch0_*.bin /path/to/ch0_*.jsonl out_dir [separate=1]
#     bin_path   = sys.argv[1]
#     jsonl_path = sys.argv[2]
#     out_dir    = sys.argv[3]
#     separate   = bool(int(sys.argv[4])) if len(sys.argv)>4 else True

#     flat   = read_complex_bin(bin_path)
#     lens, metas = read_jsonl_lengths(jsonl_path)
#     frames = slice_frames(flat, lens)
#     save_frames_to_csv(frames, metas, out_dir, separate=separate)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, json, argparse, math, time
import numpy as np
import pandas as pd

# ===== mapping mặc định 802.11a/g (20 MHz) cho 64-FFT =====
ACTIVE_K = np.r_[np.arange(-26, 0), np.arange(1, 27)]           # [-26..-1, 1..26] (52 SC)
def k_to_bin(k, N=64): return (k + N) % N
DEFAULT_ACTIVE_BINS = np.array([k_to_bin(k) for k in ACTIVE_K], dtype=np.int32)

CAND_LEN_KEYS = ["csi_len","packet_len","num_complex","ncplx","len","Nsc"]
CAND_FREQ_HZ_KEYS = ["freq","center_freq_hz","center_freq","frequency_hz"]
CAND_SEQ_KEYS = ["seq","mac_seq","seqno"]

def read_complex_bin(path):
    arr = np.fromfile(path, dtype=np.complex64)
    if arr.size == 0:
        raise RuntimeError(f"Empty or unreadable bin: {path}")
    return arr

def parse_jsonl(path):
    metas = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            metas.append(json.loads(line))
    if not metas:
        raise RuntimeError(f"No meta lines in: {path}")
    return metas

def extract_len(meta):
    # ưu tiên csi_len (đúng với estimator bạn dùng)
    for k in CAND_LEN_KEYS:
        if k in meta and isinstance(meta[k], int) and meta[k] > 0:
            return int(meta[k])
    # fallback từ shape
    for k in ["shape","csi_shape"]:
        if k in meta and isinstance(meta[k], (list,tuple)) and len(meta[k]) > 0:
            val = int(meta[k][-1])
            if val > 0: return val
    raise KeyError("Không tìm thấy chiều dài frame (csi_len / packet_len / shape...) trong meta")

def extract_active_bins(meta, L):
    # nếu meta có active_bins → dùng luôn
    ab = meta.get("active_bins", None)
    if isinstance(ab, (list, tuple)) and len(ab) == L:
        return np.array(ab, dtype=np.int32)
    # nếu không có: với L=52, dùng mặc định 11a/g
    if L == 52:
        return DEFAULT_ACTIVE_BINS.copy()
    # ngược lại, không rõ mapping → trả None, sẽ gán 0..L-1
    return None

def extract_freq_mhz(meta):
    hz = None
    for k in CAND_FREQ_HZ_KEYS:
        if k in meta and meta[k] is not None:
            try:
                hz = int(meta[k])
                break
            except Exception:
                # đôi khi meta["freq"] là string → thử ép float
                try: hz = int(float(meta[k])); break
                except: pass
    if hz is None:
        return None
    return int(round(hz / 1e6))

def extract_seq(meta):
    for k in CAND_SEQ_KEYS:
        if k in meta:
            try: return int(meta[k])
            except: pass
    return None

def ts_wall_iso(ts_ms):
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts_ms/1000.0))
    except Exception:
        return None

def slice_frames(flat, lengths):
    frames = []
    idx = 0
    for L in lengths:
        end = idx + L
        if end > flat.size: break
        frames.append(flat[idx:end])
        idx = end
    leftover = flat.size - idx
    if leftover != 0:
        print(f"[WARN] Bin has {flat.size} samples, but sum(lengths) used {idx}. Leftover={leftover} (ignored).")
    return frames

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def per_frame_df(frame, meta, Nfft=64):
    L = frame.size
    abins = extract_active_bins(meta, L)

    if abins is None:
        # không có mapping → dùng index 0..L-1, k = NaN
        bin_idx = np.arange(L, dtype=np.int32)
        k = np.full(L, np.nan, dtype=np.float32)
    else:
        bin_idx = abins.astype(np.int32)
        # map bin (0..N-1) → k∈[-N/2..N/2-1]
        k = (bin_idx.copy()).astype(np.int32)
        k[k >= Nfft//2] -= Nfft
        k = k.astype(np.int32)

    freq_mhz = extract_freq_mhz(meta)
    seq = extract_seq(meta)
    csi_idx = meta.get("csi_idx", None)
    rx_chan = meta.get("rx_chan", None)
    ts_wall_ms = meta.get("ts_wall_ms", None)
    tsf_us = meta.get("tsf_us", None)
    detected_by = meta.get("detected_by", None)
    corr_rho = meta.get("corr_rho", None)

    df = pd.DataFrame({
        "bin":   bin_idx,
        "k":     k,
        "real":  frame.real.astype(np.float32),
        "imag":  frame.imag.astype(np.float32),
        "mag":   np.abs(frame).astype(np.float32),
        "phase": np.angle(frame).astype(np.float32),
    })
    # gắn nhãn meta (cùng giá trị cho mọi hàng của frame)
    if csi_idx is not None:   df["csi_idx"]   = int(csi_idx)
    if rx_chan is not None:   df["rx_chan"]   = int(rx_chan)
    if freq_mhz is not None:  df["freq_mhz"]  = int(freq_mhz)
    if seq is not None:       df["seq"]       = int(seq)
    if tsf_us is not None:    df["tsf_us"]    = int(tsf_us)
    if ts_wall_ms is not None:
        df["ts_wall_ms"] = int(ts_wall_ms)
        iso = ts_wall_iso(ts_wall_ms)
        if iso: df["ts_wall_iso"] = iso
    if detected_by is not None: df["detected_by"] = str(detected_by)
    if corr_rho is not None:    df["corr_rho"]    = float(corr_rho)

    return df

def main():
    ap = argparse.ArgumentParser(
        description="Convert CSI .bin (+ JSONL meta) → CSV (per frame or combined)."
    )
    ap.add_argument("/home/ea301b/gr-ieee802-11/examples/Data/1/f5180MHz/ch0_20250903_222609.bin")
    ap.add_argument("/home/ea301b/gr-ieee802-11/examples/Data/1/f5180MHz/ch0_20250903_222609.jsonl")
    ap.add_argument("/home/ea301b/gr-ieee802-11/examples/Data/1/f5180MHz/ch0_csv")
    ap.add_argument("--combine", action="store_true",
                    help="Gộp tất cả frame vào 1 file all_frames.csv (mặc định: tách từng frame).")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Chỉ xuất tối đa N frame đầu.")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    flat = read_complex_bin(args.bin_path)
    metas = parse_jsonl(args.jsonl_path)

    # Lấy độ dài từng frame theo meta
    lengths = [extract_len(m) for m in metas]
    frames = slice_frames(flat, lengths)

    if args.max_frames is not None:
        frames = frames[:args.max_frames]
        metas  = metas[:args.max_frames]

    if args.combine:
        rows = []
        for frm, meta in zip(frames, metas):
            rows.append(per_frame_df(frm, meta))
        big = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        out_csv = os.path.join(args.out_dir, "all_frames.csv")
        big.to_csv(out_csv, index=False)
        print(f"[OK] Saved {out_csv} (rows={len(big):,})")
    else:
        for i, (frm, meta) in enumerate(zip(frames, metas), 1):
            df = per_frame_df(frm, meta)
            out_csv = os.path.join(args.out_dir, f"frame_{i:05d}.csv")
            df.to_csv(out_csv, index=False)
        print(f"[OK] Saved {len(frames)} CSV files to {args.out_dir}")

if __name__ == "__main__":
    main()
