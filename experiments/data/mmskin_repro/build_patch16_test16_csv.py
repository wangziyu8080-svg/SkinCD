#!/usr/bin/env python3
"""Patch16 官方 Test split -> SkinCD 16类评测CSV（分层子集）。

- 来源：/data/phase7/Patch16/extracted/data/tiles-v2.csv（官方划分，Test=28039）
- 每 16 类抽取约 125 张（分层，总数 ~2000），随机种子 42
- 类别映射为可读英文组织名（对齐 HAM10K 协议风格）
- 输出格式：question,image,categories（image 相对 tiles 根目录）
"""
import csv
import os
import random
from collections import Counter, defaultdict

SRC = "/data/phase7/Patch16/extracted/data/tiles-v2.csv"
TILES_ROOT = "/data/phase7/Patch16/extracted/data/tiles"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Patch16_test16_sub2000.csv")
PER_CLASS = 125
SEED = 42

CLASS_MAP = {
    "nontumor_skin_chondraltissue_chondraltissue": "Cartilage Tissue",
    "nontumor_skin_dermis_dermis": "Dermis",
    "nontumor_skin_elastosis_elastosis": "Elastosis",
    "nontumor_skin_epidermis_epidermis": "Epidermis",
    "nontumor_skin_hairfollicle_hairfollicle": "Hair Follicle",
    "nontumor_skin_muscle_skeletal": "Skeletal Muscle",
    "nontumor_skin_necrosis_necrosis": "Necrosis",
    "nontumor_skin_nerves_nerves": "Nerves",
    "nontumor_skin_sebaceousglands_sebaceousglands": "Sebaceous Glands",
    "nontumor_skin_subcutis_subcutis": "Subcutis",
    "nontumor_skin_sweatglands_sweatglands": "Sweat Glands",
    "nontumor_skin_vessel_vessel": "Blood Vessel",
    "tumor_skin_epithelial_bcc": "Basal Cell Carcinoma",
    "tumor_skin_epithelial_sqcc": "Squamous Cell Carcinoma",
    "tumor_skin_melanoma_melanoma": "Melanoma",
    "tumor_skin_naevus_naevus": "Nevus",
}

question = "This is a histopathology slide image. Which diagnosis is shown?"

by_class = defaultdict(list)
with open(SRC, newline="") as handle:
    for row in csv.DictReader(handle):
        if row["set"] != "Test":
            continue
        rel = row["file"]
        if not rel.startswith("data/tiles/"):
            continue
        rel = rel[len("data/tiles/"):]
        if not os.path.exists(os.path.join(TILES_ROOT, rel)):
            continue
        by_class[row["class"]].append(rel)

rng = random.Random(SEED)
rows_out = []
for raw_class, files in sorted(by_class.items()):
    label = CLASS_MAP[raw_class]
    take = min(PER_CLASS, len(files))
    for rel in sorted(rng.sample(files, take)):
        rows_out.append(
            {"question": question, "image": rel, "categories": str([label])}
        )

rng.shuffle(rows_out)

with open(OUT, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["question", "image", "categories"])
    writer.writeheader()
    writer.writerows(rows_out)

print(f"wrote {len(rows_out)} rows -> {OUT}")
print(Counter(r["categories"] for r in rows_out).most_common())
