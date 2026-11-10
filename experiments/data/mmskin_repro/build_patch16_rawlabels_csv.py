#!/usr/bin/env python3
"""Patch16 16类子集 -> 原始类名标签版（对齐 内部 dataset registry 协议）。

与 Patch16_test16_sub2000.csv 完全相同的 1941 张图（同序），仅把标签从
自造可读名换回协议原始名（下划线转空格、去重复实体段）：
  nontumor_skin_dermis_dermis        -> nontumor skin dermis
  nontumor_skin_muscle_skeletal      -> nontumor skin muscle skeletal
  tumor_skin_epithelial_bcc          -> tumor skin epithelial bcc
标签空间 = 内部评测脚本 ZS_classification.py Patch16 targets 的 16 个键。
"""
import ast
import csv
import os

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Patch16_test16_sub2000.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Patch16_test16_rawlabels.csv")

READABLE2RAW = {
    "Cartilage Tissue": "nontumor skin chondraltissue",
    "Dermis": "nontumor skin dermis",
    "Elastosis": "nontumor skin elastosis",
    "Epidermis": "nontumor skin epidermis",
    "Hair Follicle": "nontumor skin hairfollicle",
    "Skeletal Muscle": "nontumor skin muscle skeletal",
    "Necrosis": "nontumor skin necrosis",
    "Nerves": "nontumor skin nerves",
    "Sebaceous Glands": "nontumor skin sebaceousglands",
    "Subcutis": "nontumor skin subcutis",
    "Sweat Glands": "nontumor skin sweatglands",
    "Blood Vessel": "nontumor skin vessel",
    "Basal Cell Carcinoma": "tumor skin epithelial bcc",
    "Squamous Cell Carcinoma": "tumor skin epithelial sqcc",
    "Melanoma": "tumor skin melanoma",
    "Nevus": "tumor skin naevus",
}

rows_out = []
with open(SRC, newline="") as handle:
    for row in csv.DictReader(handle):
        label = ast.literal_eval(row["categories"])[0]
        raw = READABLE2RAW[label]
        rows_out.append(
            {"question": row["question"], "image": row["image"], "categories": str([raw])}
        )

with open(OUT, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["question", "image", "categories"])
    writer.writeheader()
    writer.writerows(rows_out)

from collections import Counter

print(f"wrote {len(rows_out)} rows -> {OUT}")
print(Counter(r["categories"] for r in rows_out).most_common())
