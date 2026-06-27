#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil
from pathlib import Path

# === CẤU HÌNH ===
BASE = Path(r"D:\Project\WifiSenssing\Data\Data\Front")  # thư mục chứa các folder 1..50
FOLDER_RANGE = range(1, 51)  # 1..50
FREQ_DIR_NAMES = [  # 8 tần số
    "f5180MHz", "f5200MHz", "f5220MHz", "f5240MHz",
    "f5260MHz", "f5280MHz", "f5300MHz", "f5320MHz",
]
CSV_DIRNAME = "csv"   # thư mục cần xoá bên trong mỗi fXXXXMHz
DRY_RUN = False        # True = chỉ liệt kê; False = XOÁ THẬT

# === CHƯƠNG TRÌNH ===
found, deleted, missing = 0, 0, 0
errors = []

for i in FOLDER_RANGE:
    root_i = BASE / str(i)
    if not root_i.exists():
        print(f"[WARN] Bỏ qua {root_i} (không tồn tại)")
        continue

    for freq in FREQ_DIR_NAMES:
        csv_dir = root_i / freq / CSV_DIRNAME
        if csv_dir.is_dir():
            found += 1
            if DRY_RUN:
                print(f"[FOUND] {csv_dir}")
            else:
                try:
                    shutil.rmtree(csv_dir)
                    print(f"[DELETED] {csv_dir}")
                    deleted += 1
                except Exception as e:
                    errors.append((str(csv_dir), str(e)))
                    print(f"[ERROR] {csv_dir} -> {e}")
        else:
            missing += 1
            # uncomment nếu muốn log chi tiết:
            print(f"[MISS ] {csv_dir}")

print("\n=== TÓM TẮT ===")
print(f"Đã quét thư mục gốc: {BASE}")
print(f"Số csv/ TÌM THẤY : {found}")
print(f"Số csv/ ĐÃ XOÁ   : {deleted} (DRY_RUN={DRY_RUN})")
print(f"Số csv/ KHÔNG CÓ : {missing}")
if errors:
    print("\n[ERRORS]")
    for path, msg in errors:
        print(f"- {path} -> {msg}")
