# convert_bin_flat.py
import numpy as np, pandas as pd, os, sys

def bin_to_csv_flat(bin_path, out_csv=None, max_rows=None):
    if out_csv is None:
        out_csv = os.path.splitext(bin_path)[0] + "_flat.csv"
    # gr_complex = complex64: 2*float32 -> 8 bytes/mẫu
    raw = np.fromfile(bin_path, dtype=np.complex64)
    if raw.size == 0:
        print("Empty or unreadable:", bin_path); return
    df = pd.DataFrame({
        "index": np.arange(raw.size, dtype=np.int64),
        "real":  raw.real.astype(np.float32),
        "imag":  raw.imag.astype(np.float32),
        "mag":   np.abs(raw).astype(np.float32),
        "phase": np.angle(raw).astype(np.float32),
    })
    if max_rows:
        df = df.head(int(max_rows))
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}  (rows={len(df):,})")

if __name__ == "__main__":
    # Usage: python3 convert_bin_flat.py /path/to/ch0_xxx.bin [max_rows]
    bin_path = sys.argv[1]
    max_rows = int(sys.argv[2]) if len(sys.argv) > 2 else None
    bin_to_csv_flat(bin_path, max_rows=max_rows)
