import json, math, csv
from collections import defaultdict

JSON_PATH = "csi_log.json"   # file .json của bạn
PAIR_CSV  = "paired_indices.csv"

# tham số ghép
TSF_TOL_US   = 100      # chênh TSF tối đa giữa 2 ant để coi là đồng thời (us)
USE_BUCKET   = False    # nếu True, dùng bucket TSF thay vì so |Δ|
TSF_BUCKET_US= 2000

# map tên trường phổ biến -> lấy giá trị
def pick(d, *cands, default=None):
    for k in cands:
        if k in d: 
            return d[k]
    return default

def get_ant_id(d):
    # Ưu tiên trường rõ ràng
    v = pick(d, "ant", "antenna", "rx_port", "rx_chain", "chain")
    if v is not None:
        return int(v)
    # Bitmask -> nếu 2 chain cùng lúc (0b11) thì bạn có thể tạo 2 bản ghi ảo (ant0 & ant1)
    mask = pick(d, "rx_chain_mask", "antmask", "chainmask")
    if mask is not None:
        m = int(mask)
        # trả về list các ant có mặt
        return [i for i in (0,1) if (m >> i) & 1]
    # Không có thông tin ăng-ten
    return None

def norm_ant_id(v):
    # trả về list các ant hợp lệ trong {0,1}
    if v is None:
        return []
    if isinstance(v, list):
        out = []
        for x in v:
            x = int(x)
            if x in (0,1): out.append(x)
            elif x in (1,2): out.append(x-1)
        return sorted(set(out))
    x = int(v)
    if x in (0,1): return [x]
    if x in (1,2): return [x-1]
    return []

def tsf_us(d):
    # cố gắng lấy TSF/timestamp ở microseconds
    return int(pick(d, "tsf", "tsft", "timestamp_us", "ts_us", "ts", default=0))

def seq_no(d):
    # 12-bit seq nếu cần
    s = int(pick(d, "seq", "seqno", "seq_num", default=-1))
    if s >= 0:
        return s & 0xFFF
    return s

def freq_hz(d):
    return int(pick(d, "freq", "frequency", "fc", default=0))

def mac_src(d):
    return pick(d, "addr2", "sa", "src", default="")

# 1) load json (giả sử JSON lines; nếu là mảng, đổi cách đọc)
records = []
with open(JSON_PATH, "r") as f:
    for line in f:
        line=line.strip()
        if not line: continue
        try:
            rec = json.loads(line)
            records.append(rec)
        except:
            pass

# 2) group theo key
groups = defaultdict(lambda: {0:[], 1:[]})

for i, r in enumerate(records):
    s   = seq_no(r)
    fc  = freq_hz(r)
    ts  = tsf_us(r)
    mac = mac_src(r) or ""  # tuỳ bạn có dùng vào key không

    if USE_BUCKET:
        ts_bucket = ts // TSF_BUCKET_US
        key = (s, fc, ts_bucket)   # + (mac,) nếu nhiều nguồn phát
        # so Δ sẽ làm sau khi đủ 2 ant
    else:
        key = (s, fc)              # + (mac,)

    ants = norm_ant_id(get_ant_id(r))
    if not ants:
        continue

    for a in ants:
        entry = {"idx": i, "tsf": ts, "rec": r}
        groups[key][a].append(entry)

# 3) ghép cặp tốt nhất theo |Δtsf| (mỗi key có thể nhiều bản ghi/ant -> chọn cặp gần nhau nhất)
pairs = []   # (idx0, idx1, seq, freq, tsf0, tsf1, dtsf)
for key, ch in groups.items():
    list0, list1 = ch[0], ch[1]
    if not list0 or not list1:
        continue

    used1 = set()
    # greedy: với mỗi ant0, chọn ant1 gần tsf nhất và chưa dùng
    for e0 in sorted(list0, key=lambda x: x["tsf"]):
        best = None
        best_dt = None
        for j,e1 in enumerate(list1):
            if j in used1: continue
            dt = abs(e0["tsf"] - e1["tsf"])
            if USE_BUCKET:
                # nếu dùng bucket thì thường dt nhỏ; vẫn kiểm tra ngưỡng
                pass
            if best is None or dt < best_dt:
                best, best_dt = (j,e1), dt
        if best is None: 
            continue
        j,e1 = best
        if best_dt is not None and best_dt <= TSF_TOL_US:
            used1.add(j)
            seq, fc = key[0], key[1]
            pairs.append((e0["idx"], e1["idx"], seq, fc, e0["tsf"], e1["tsf"], best_dt))

# 4) lưu CSV index cặp (để bạn load CSI từ .bin theo index tương ứng)
with open(PAIR_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["idx_ant0","idx_ant1","seq","freq_hz","tsf0_us","tsf1_us","dtsf_us"])
    for row in pairs:
        w.writerow(row)

print(f"Total records: {len(records)}")
print(f"Valid AoA pairs: {len(pairs)}")
print(f"Saved to {PAIR_CSV}")