#!/usr/bin/env python3
"""Build HAM10000 test CSV in SkinCD eval format (question,image,categories)."""
import csv
import json
import os

SPLIT = "/data/phase7/HAM10000/splits_group/splits_seed42.json"
META = "/data/phase7/HAM10000/HAM10000_metadata.csv"
IMG_ROOT = "/data/phase7/HAM10000/images"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HAM10K_ISIC2018_test.csv")

splits = json.load(open(SPLIT))
label_map = splits["meta"]["label_map"]
test_images = {img for img, split in splits["image_assign"].items() if split == "test"}

question = "This is a skin lesion image. Which disease is shown?"

rows_out = []
with open(META, newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        row = {k: v.strip('"') for k, v in row.items()}
        if row["image_id"] in test_images:
            image_rel = f"{row['image_id']}.jpg"
            img = os.path.join(IMG_ROOT, image_rel)
            if not os.path.exists(img):
                continue
            rows_out.append(
                {
                    "question": question,
                    "image": image_rel,
                    "categories": str([label_map[row["dx"]]]),
                }
            )

with open(OUT, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["question", "image", "categories"])
    writer.writeheader()
    writer.writerows(rows_out)

from collections import Counter

print(f"wrote {len(rows_out)} rows -> {OUT}")
print(Counter(r["categories"] for r in rows_out).most_common())
