#!/usr/bin/env python3
"""用户提供的 Patch16 2类评测集（10% seed42 子采样，2804张）-> SkinCD 格式。

- 来源：内部工作区 Patch16_2class_test_10pct_seed42.csv
- 标签空间：tumor / nontumor（2类，自然分布 31.6% 肿瘤）
- 路径映射：Patch16/<cls>/<f>.jpg -> tiles/<cls>/<f>.jpg（相对 tiles 根）
"""
import csv
import os

SRC = "/root/.cursor/projects/root-experiment-models-skin-MM-Skin/attachments/f3430e9c-349a-4b62-865c-04c877154cfb/Patch16_2class_test_10pct_seed42.csv"
TILES = "/data/phase7/Patch16/extracted/data/tiles"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Patch16_2class_test_10pct.csv")

question = "This is a histopathology slide image. Which diagnosis is shown?"

rows_out = []
missing = 0
with open(SRC, newline="") as handle:
    for row in csv.DictReader(handle):
        rel = row["image"]
        if rel.startswith("Patch16/"):
            rel = rel[len("Patch16/"):]
        if not os.path.exists(os.path.join(TILES, rel)):
            missing += 1
            continue
        rows_out.append(
            {"question": question, "image": rel, "categories": row["categories"]}
        )

with open(OUT, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["question", "image", "categories"])
    writer.writeheader()
    writer.writerows(rows_out)

from collections import Counter

print(f"wrote {len(rows_out)} rows (missing={missing}) -> {OUT}")
print(Counter(r["categories"] for r in rows_out).most_common())
