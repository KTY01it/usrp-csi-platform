# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import os, json, glob, time, csv
# import numpy as np
# import pandas as pd

# # ================== CẤU HÌNH ==================
# ROOT = r"D:/Project/WifiSenssing/Data/DataOFDM"  # thư mục chứa 1..50
# SITES = list(range(1, 51))
# FREQS_MHZ = [5180, 5200, 5220, 5240, 5260, 5280, 5300, 5320]
# CHANS = ["ch0", "ch1"]
# NFFT = 64
# REPORT_PATH = os.path.join(ROOT, "consistency_report.csv")
# # ==============================================

# ACTIVE_K = np.r_[np.arange(-26, 0), np.arange(1, 27)]
# def k_to_bin(k, N=64): return (k + N) % N
# DEFAULT_ACTIVE_BINS = np.array([k_to_bin(k) for k in ACTIVE_K], dtype=np.int32)

# CAND_LEN_KEYS = ["csi_len","packet_len","num_complex","ncplx","len","Nsc"]
# CAND_FREQ_HZ_KEYS = ["freq","center_freq_hz","center_freq","frequency_hz"]

# def read_complex_bin(path):
#     arr = np.fromfile(path, dtype=np.complex64)
#     if arr.size == 0:
#         raise RuntimeError(f"Empty or unreadable bin: {path}")
#     return arr

# def parse_json_any(path):
#     with open(path, "r", encoding="utf-8") as f:
#         text = f.read()
#     name = os.path.basename(path).lower()
#     if name.endswith(".jsonl"):
#         lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
#         metas = [json.loads(ln) for ln in lines]
#     else:
#         try:
#             obj = json.loads(text)
#             if isinstance(obj, list): metas = obj
#             elif isinstance(obj, dict): metas = [obj]
#             else:
#                 # fallback: đôi khi .json nhưng thực chất là jsonl
#                 metas = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
#         except:
#             metas = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
#     if not metas:
#         raise RuntimeError(f"No meta objects in {path}")
#     return metas

# def extract_len(meta):
#     for k in CAND_LEN_KEYS:
#         if k in meta and isinstance(meta[k], int) and meta[k] > 0:
#             return int(meta[k])
#     # fallback: shape[-1]
#     for k in ["shape","csi_shape"]:
#         if k in meta and isinstance(meta[k], (list,tuple)) and len(meta[k])>0:
#             v = int(meta[k][-1])
#             if v>0: return v
#     return None

# def extract_active_bins(meta, L):
#     ab = meta.get("active_bins", None)
#     if isinstance(ab, (list,tuple)):
#         return np.array(ab, dtype=np.int32)
#     if L == 52:
#         return DEFAULT_ACTIVE_BINS.copy()
#     return None

# def extract_freq_mhz(meta):
#     hz = None
#     for k in CAND_FREQ_HZ_KEYS:
#         if k in meta and meta[k] is not None:
#             try: hz = int(meta[k]); break
#             except:
#                 try: hz = int(float(meta[k])); break
#                 except: pass
#     if hz is None: return None
#     return int(round(hz/1e6))

# def find_pairs(freq_dir, chan):
#     bins = sorted(glob.glob(os.path.join(freq_dir, f"*{chan}*.bin")))
#     metas = sorted(glob.glob(os.path.join(freq_dir, f"*{chan}*.json"))) + \
#             sorted(glob.glob(os.path.join(freq_dir, f"*{chan}*.jsonl")))
#     pairs = []
#     for b in bins:
#         # chọn meta cùng kênh, ưu tiên độ dài tên gần nhau
#         cand = [m for m in metas if chan in os.path.basename(m)]
#         if not cand:
#             pairs.append((b, None)); continue
#         base = os.path.basename(b)
#         m = min(cand, key=lambda x: abs(len(os.path.basename(x))-len(base)))
#         pairs.append((b, m))
#     return pairs

# def check_pair(bin_path, meta_path, expect_mhz=None, expect_chan=None):
#     info = dict(
#         bin_path=bin_path, meta_path=meta_path or "",
#         ok=True, errors=[], warnings=[],
#         n_meta_frames=0, n_cut_frames=0,
#         bin_samples=0, sum_lengths=0, leftover=0,
#         truncated_frames=0, zero_len_frames=0,
#         bad_active_bins=0, csiidx_dups=0, csiidx_missing=0,
#         freq_mhz_mismatch=0, rx_chan_mismatch=0
#     )
#     try:
#         flat = read_complex_bin(bin_path)
#         info["bin_samples"] = int(flat.size)
#     except Exception as e:
#         info["ok"] = False
#         info["errors"].append(f"Read BIN failed: {e}")
#         return info

#     if not meta_path or not os.path.isfile(meta_path):
#         info["ok"] = False
#         info["errors"].append("Missing META file for this BIN")
#         return info

#     try:
#         metas = parse_json_any(meta_path)
#     except Exception as e:
#         info["ok"] = False
#         info["errors"].append(f"Parse META failed: {e}")
#         return info

#     # lengths
#     lengths = []
#     for idx, m in enumerate(metas):
#         L = extract_len(m)
#         if L is None or L <= 0:
#             info["zero_len_frames"] += 1
#             lengths.append(0)
#         else:
#             lengths.append(L)
#     info["n_meta_frames"] = len(metas)
#     info["sum_lengths"] = int(sum(lengths))

#     # cắt thử từ bin
#     cut_frames = []
#     off = 0
#     for L in lengths:
#         if L <= 0:
#             cut_frames.append(None)
#             continue
#         if off + L <= flat.size:
#             cut_frames.append((off, off+L))
#             off += L
#         else:
#             info["truncated_frames"] += 1
#             cut_frames.append(None)
#             break  # các frame sau chắc chắn thiếu
#     info["n_cut_frames"] = sum(1 for c in cut_frames if c is not None)
#     info["leftover"] = int(flat.size - off)

#     # active_bins & freq/rx_chan/csi_idx checks
#     csi_indices = []
#     for i, m in enumerate(metas):
#         L = lengths[i]
#         if L and L>0:
#             ab = extract_active_bins(m, L)
#             if ab is not None and len(ab) != L:
#                 info["bad_active_bins"] += 1

#         fmhz = extract_freq_mhz(m)
#         if expect_mhz is not None and fmhz is not None and int(fmhz) != int(expect_mhz):
#             info["freq_mhz_mismatch"] += 1

#         if expect_chan is not None and "rx_chan" in m:
#             try:
#                 if int(m["rx_chan"]) != (0 if expect_chan=="ch0" else 1):
#                     info["rx_chan_mismatch"] += 1
#             except:
#                 info["rx_chan_mismatch"] += 1

#         csi_idx = m.get("csi_idx", None)
#         if not isinstance(csi_idx, (int, np.integer)) or csi_idx < 0:
#             info["csiidx_missing"] += 1
#         else:
#             csi_indices.append(int(csi_idx))

#     if csi_indices:
#         # đếm trùng lặp
#         uniq = set()
#         dups = 0
#         for x in csi_indices:
#             if x in uniq: dups += 1
#             else: uniq.add(x)
#         info["csiidx_dups"] = dups

#     # đánh giá tổng quát
#     if info["n_cut_frames"] < info["n_meta_frames"]:
#         info["ok"] = False
#         info["errors"].append(
#             f"BIN lacks samples for {info['n_meta_frames'] - info['n_cut_frames']} frame(s) declared in META"
#         )
#     if info["leftover"] != 0:
#         # Không lỗi nghiêm trọng, nhưng cảnh báo: BIN dư/thiếu mẫu so với sum(lengths)
#         info["warnings"].append(f"Leftover (BIN - used) = {info['leftover']} samples")

#     if info["zero_len_frames"]>0: info["warnings"].append(f"{info['zero_len_frames']} zero/invalid length frame(s) in META")
#     if info["bad_active_bins"]>0: info["warnings"].append(f"{info['bad_active_bins']} frame(s) have active_bins length mismatch")
#     if info["csiidx_missing"]>0: info["warnings"].append(f"{info['csiidx_missing']} frame(s) missing/invalid csi_idx")
#     if info["csiidx_dups"]>0: info["warnings"].append(f"{info['csiidx_dups']} duplicated csi_idx value(s)")
#     if info["freq_mhz_mismatch"]>0: info["warnings"].append(f"{info['freq_mhz_mismatch']} frame(s) with freq_mhz mismatch")
#     if info["rx_chan_mismatch"]>0: info["warnings"].append(f"{info['rx_chan_mismatch']} frame(s) with rx_chan mismatch")

#     return info

# def main():
#     rows = []
#     for site in SITES:
#         site_dir = os.path.join(ROOT, str(site))
#         if not os.path.isdir(site_dir):
#             print(f"[WARN] Missing site dir: {site_dir}")
#             continue

#         for fmhz in FREQS_MHZ:
#             freq_dir = os.path.join(site_dir, f"f{fmhz}MHz")
#             if not os.path.isdir(freq_dir):
#                 print(f"[WARN] Missing freq dir: {freq_dir}")
#                 continue

#             for chan in CHANS:
#                 pairs = find_pairs(freq_dir, chan)
#                 if not pairs:
#                     print(f"[WARN] No BIN found in {freq_dir} ({chan})")
#                     continue

#                 for (b, m) in pairs:
#                     info = check_pair(b, m, expect_mhz=fmhz, expect_chan=chan)
#                     okflag = "OK" if info["ok"] else "FAIL"
#                     print(f"[{okflag}] {os.path.basename(b)}  |  meta={os.path.basename(m) if m else 'None'}")
#                     if info["errors"]:
#                         for e in info["errors"]:
#                             print("   ERR:", e)
#                     if info["warnings"]:
#                         for w in info["warnings"]:
#                             print("   WARN:", w)

#                     rows.append({
#                         "site": site, "freq_mhz": fmhz, "chan": chan,
#                         "bin_file": os.path.basename(b),
#                         "meta_file": os.path.basename(m) if m else "",
#                         "ok": info["ok"],
#                         "bin_samples": info["bin_samples"],
#                         "n_meta_frames": info["n_meta_frames"],
#                         "n_cut_frames": info["n_cut_frames"],
#                         "sum_lengths": info["sum_lengths"],
#                         "leftover": info["leftover"],
#                         "truncated_frames": info["truncated_frames"],
#                         "zero_len_frames": info["zero_len_frames"],
#                         "bad_active_bins": info["bad_active_bins"],
#                         "csiidx_missing": info["csiidx_missing"],
#                         "csiidx_dups": info["csiidx_dups"],
#                         "freq_mhz_mismatch": info["freq_mhz_mismatch"],
#                         "rx_chan_mismatch": info["rx_chan_mismatch"],
#                         "errors": " | ".join(info["errors"]),
#                         "warnings": " | ".join(info["warnings"]),
#                         "bin_path": b,
#                         "meta_path": m or ""
#                     })

#     if rows:
#         df = pd.DataFrame(rows)
#         df.to_csv(REPORT_PATH, index=False, quoting=csv.QUOTE_MINIMAL)
#         print("\n===== SUMMARY =====")
#         print("Total pairs checked:", len(rows))
#         print("OK:", int(df["ok"].sum()), "| FAIL:", int((~df["ok"]).sum()))
#         print(f"Saved report to: {REPORT_PATH}")
#     else:
#         print("No pairs scanned — please check ROOT/SITES/FREQS/CHANS.")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, json
import numpy as np
import pandas as pd

# ================== CẤU HÌNH ==================
ROOT = r"/home/ea301b/gr-ieee802-11/examples/Data"  # thư mục chứa 1..50
SITES = list(range(1, 51))
FREQS_MHZ = [5180, 5200, 5220, 5240, 5260, 5280, 5300, 5320]
CHANS = ["ch0", "ch1"]
SUMMARY_PATH = os.path.join(ROOT, "per_site_freq_summary.csv")
# ==============================================

CAND_LEN_KEYS = ["csi_len","packet_len","num_complex","ncplx","len","Nsc"]
ID_PRIORITY = ["csi_idx", "seq", "tsf_us", "ts_wall_ms"]  # thứ tự ưu tiên để khớp 2 kênh

def read_complex_bin(path):
    arr = np.fromfile(path, dtype=np.complex64)
    if arr.size == 0:
        raise RuntimeError(f"Empty or unreadable bin: {path}")
    return arr

def parse_json_any(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    name = os.path.basename(path).lower()
    if name.endswith(".jsonl"):
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        metas = [json.loads(ln) for ln in lines]
    else:
        try:
            obj = json.loads(text)
            if isinstance(obj, list): metas = obj
            elif isinstance(obj, dict): metas = [obj]
            else:
                metas = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
        except:
            metas = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    if not metas:
        raise RuntimeError(f"No meta objects in {path}")
    return metas

def extract_len(meta):
    for k in CAND_LEN_KEYS:
        if k in meta and isinstance(meta[k], int) and meta[k] > 0:
            return int(meta[k])
    for k in ["shape","csi_shape"]:
        if k in meta and isinstance(meta[k], (list,tuple)) and len(meta[k])>0:
            v = int(meta[k][-1])
            if v>0: return v
    return None

def find_pairs(freq_dir, chan):
    bins = sorted(glob.glob(os.path.join(freq_dir, f"*{chan}*.bin")))
    metas = sorted(glob.glob(os.path.join(freq_dir, f"*{chan}*.json"))) + \
            sorted(glob.glob(os.path.join(freq_dir, f"*{chan}*.jsonl")))
    pairs = []
    for b in bins:
        cand = [m for m in metas if chan in os.path.basename(m)]
        m = None
        if cand:
            base = os.path.basename(b)
            m = min(cand, key=lambda x: abs(len(os.path.basename(x))-len(base)))
        pairs.append((b, m))
    return pairs

def cuttable_frames(flat, metas):
    """
    Trả về:
      - ids: list ID cho mỗi frame (đã chuẩn hóa thành tuple (key, value)) hoặc None nếu không có
      - n_meta_frames: số frame trong meta
      - n_cut_frames: số frame thực sự cắt được từ bin
      - truncated: số frame bị thiếu mẫu (không cắt được)
      - leftover: phần dư mẫu trong bin sau khi cắt
    """
    lengths = []
    ids = []
    for i, m in enumerate(metas):
        L = extract_len(m)
        if L is None or L <= 0:
            lengths.append(0)
            ids.append(None)
            continue
        lengths.append(L)
        id_val = None
        for k in ID_PRIORITY:
            if k in m and m[k] is not None:
                try:
                    id_val = (k, int(m[k]))
                except:
                    try:
                        id_val = (k, int(float(m[k])))
                    except:
                        id_val = (k, str(m[k]))
                break
        if id_val is None:
            id_val = ("index", i)  # fallback
        ids.append(id_val)

    n_meta = len(metas)
    off = 0
    n_cut = 0
    truncated = 0
    valid_ids = []
    for L, idv in zip(lengths, ids):
        if L is None or L <= 0:
            continue
        if off + L <= flat.size:
            n_cut += 1
            valid_ids.append(idv)
            off += L
        else:
            truncated += 1
            break
    leftover = int(flat.size - off)
    return valid_ids, n_meta, n_cut, truncated, leftover

def summarize_one(freq_dir):
    """
    Gộp tất cả file của từng kênh trong freq_dir, coi như một stream nối tiếp.
    Đếm tổng frame cắt được trên ch0/ch1 và tính giao/khác biệt theo ID.
    """
    totals = {}
    id_sets = {}
    meta_totals = {}
    truncated_totals = {}
    leftover_totals = {}

    for chan in CHANS:
        pairs = find_pairs(freq_dir, chan)
        all_ids = []
        n_meta_total = 0
        n_cut_total = 0
        truncated_total = 0
        leftover_total = 0

        for (b, m) in pairs:
            if m is None:
                continue
            try:
                flat = read_complex_bin(b)
                metas = parse_json_any(m)
                ids, n_meta, n_cut, trunc, leftover = cuttable_frames(flat, metas)
            except Exception as e:
                # Nếu lỗi, bỏ qua file này
                continue
            all_ids.extend(ids)
            n_meta_total += n_meta
            n_cut_total  += n_cut
            truncated_total += trunc
            leftover_total  += leftover

        totals[chan] = n_cut_total
        meta_totals[chan] = n_meta_total
        truncated_totals[chan] = truncated_total
        leftover_totals[chan] = leftover_total
        id_sets[chan] = set(all_ids)

    ch0_ids = id_sets.get("ch0", set())
    ch1_ids = id_sets.get("ch1", set())
    matched = len(ch0_ids & ch1_ids)
    missing_on_ch0 = len(ch1_ids - ch0_ids)
    missing_on_ch1 = len(ch0_ids - ch1_ids)

    return {
        "n_frames_ch0": totals.get("ch0", 0),
        "n_frames_ch1": totals.get("ch1", 0),
        "n_meta_frames_ch0": meta_totals.get("ch0", 0),
        "n_meta_frames_ch1": meta_totals.get("ch1", 0),
        "truncated_ch0": truncated_totals.get("ch0", 0),
        "truncated_ch1": truncated_totals.get("ch1", 0),
        "leftover_ch0": leftover_totals.get("ch0", 0),
        "leftover_ch1": leftover_totals.get("ch1", 0),
        "matched_frames": matched,
        "missing_on_ch0": missing_on_ch0,
        "missing_on_ch1": missing_on_ch1,
    }

def main():
    rows = []
    for site in SITES:
        site_dir = os.path.join(ROOT, str(site))
        if not os.path.isdir(site_dir):
            print(f"[WARN] Missing site dir: {site_dir}")
            continue
        for fmhz in FREQS_MHZ:
            freq_dir = os.path.join(site_dir, f"f{fmhz}MHz")
            if not os.path.isdir(freq_dir):
                print(f"[WARN] Missing freq dir: {freq_dir}")
                continue
            s = summarize_one(freq_dir)
            rows.append({
                "site": site,
                "freq_mhz": fmhz,
                **s
            })
            print(f"[{site} | {fmhz} MHz] ch0={s['n_frames_ch0']}  ch1={s['n_frames_ch1']}  matched={s['matched_frames']}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(SUMMARY_PATH, index=False)
        print("\nSaved per-site/freq summary to:", SUMMARY_PATH)
    else:
        print("No data summarized. Check ROOT/SITES/FREQS paths.")

if __name__ == "__main__":
    main()
