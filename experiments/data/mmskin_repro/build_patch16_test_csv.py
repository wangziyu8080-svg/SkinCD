#!/usr/bin/env python3
"""Convert 内部 Patch16_test_sub580.csv (640 rows) to SkinCD eval format.

Labels are mapped to paper-style Patch16 categories; a pathology-appropriate
question replaces the dermoscopy question. Image column is made relative to
the tiles root so --image-root works.
"""
import csv
import os

SRC = "/path/to/Dataframe/test/classification/Patch16_test_sub580.csv"  # adjust to your local dataset root
TILES_ROOT = "/data/phase7/Patch16/extracted/data/tiles"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Patch16_test_sub640.csv")

LABEL_MAP = {
    "tumor_skin_epithelial_bcc": "Basal Cell Carcinoma",
    "tumor_skin_epithelial_sqcc": "Squamous Cell Carcinoma",
    "tumor_skin_melanoma_melanoma": "Melanoma",
    "tumor_skin_naevus_naevus": "Nevus",
}

question = "This is a histopathology slide image. Which disease is shown?"

rows_out = []
skipped_tumor = 0
skipped_missing = 0
with open(SRC, newline="") as handle:
    for row in csv.DictReader(handle):
        categories = row["categories"].strip()
        assert categories.startswith("['") and categories.endswith("']"), categories
        raw_label = categories[2:-2]
        if raw_label in LABEL_MAP:
            label = LABEL_MAP[raw_label]
        else:
            skipped_tumor += 1
            label = None
        img_abs = row["image"]
        img_rel = os.path.relpath(img_abs, TILES_ROOT)
        if not os.path.exists(os.path.join(TILES_ROOT, img_rel)):
            skipped_missing += 1
            continue
        if label is None:
            continue
        rows_out.append(
            {
                "question": question,
                "image": img_rel,
                "categories": str([label]),
            }
        )

with open(OUT, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["question", "image", "categories"])
    writer.writeheader()
    writer.writerows(rows_out)

from collections import Counter

print(f"wrote {len(rows_out)} rows -> {OUT}")
print(f"skipped non-tumor={skipped_tumor}, missing={skipped_missing}")
print(Counter(r["categories"] for r in rows_out).most_common())
