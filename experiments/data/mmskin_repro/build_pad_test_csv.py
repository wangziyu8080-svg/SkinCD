#!/usr/bin/env python3
"""PAD_test_isic_local.csv -> SkinCD 评测格式（question,image,categories）。

- 源：PAD-UFES-20 测试集 2298 张（ISIC 化本地图片，全部在盘）
- 标签空间与 SkinCD 内置 `pad` preset 完全一致（6 类）
- image 保持相对路径（相对 classification 目录），--image-root 指向该目录
"""
import csv
import os

SRC = "/path/to/Dataframe/test/classification/PAD_test_isic_local.csv"  # adjust to your local dataset root
IMG_ROOT = os.path.dirname(SRC)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PAD_test_isic_local.csv")

question = "This is a clinical skin image. Which disease is shown?"

rows_out = []
with open(SRC, newline="") as handle:
    for row in csv.DictReader(handle):
        rel = row["image"]
        if not os.path.exists(os.path.join(IMG_ROOT, rel)):
            continue
        rows_out.append(
            {"question": question, "image": rel, "categories": row["categories"]}
        )

with open(OUT, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["question", "image", "categories"])
    writer.writeheader()
    writer.writerows(rows_out)

from collections import Counter

print(f"wrote {len(rows_out)} rows -> {OUT}")
print(Counter(r["categories"] for r in rows_out).most_common())
