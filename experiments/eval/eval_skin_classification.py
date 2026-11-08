import argparse
import csv
import json
import os
import re

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


DATASET_LABELS = {
    "ham10k": [
        "Actinic Keratoses",
        "Basal Cell Carcinoma",
        "Benign Keratosis",
        "Dermatofibroma",
        "Melanoma",
        "Nevus",
        "Vascular lesions",
    ],
    "pad": [
        "Actinic Keratosis",
        "Basal Cell Carcinoma",
        "Melanoma",
        "Nevus",
        "Seborrheic Keratosis",
        "Squamous Cell Carcinoma",
    ],
}

DATASET_ALIASES = {
    "ham10k": {
        "actinic keratosis": "Actinic Keratoses",
        "actinic keratoses": "Actinic Keratoses",
        "act": "Actinic Keratoses",
        "basal cell carcinoma": "Basal Cell Carcinoma",
        "bcc": "Basal Cell Carcinoma",
        "bas": "Basal Cell Carcinoma",
        "benign keratosis": "Benign Keratosis",
        "ben": "Benign Keratosis",
        "dermatofibroma": "Dermatofibroma",
        "derm": "Dermatofibroma",
        "d": "Dermatofibroma",
        "melanoma": "Melanoma",
        "mel": "Melanoma",
        "nevus": "Nevus",
        "naevus": "Nevus",
        "nev": "Nevus",
        "n": "Nevus",
        "vascular lesion": "Vascular lesions",
        "vascular lesions": "Vascular lesions",
        "vas": "Vascular lesions",
        "v": "Vascular lesions",
    },
    "pad": {
        "actinic keratosis": "Actinic Keratosis",
        "actinic keratoses": "Actinic Keratosis",
        "ak": "Actinic Keratosis",
        "act": "Actinic Keratosis",
        "basal cell carcinoma": "Basal Cell Carcinoma",
        "bcc": "Basal Cell Carcinoma",
        "bas": "Basal Cell Carcinoma",
        "melanoma": "Melanoma",
        "mel": "Melanoma",
        "nevus": "Nevus",
        "naevus": "Nevus",
        "nev": "Nevus",
        "n": "Nevus",
        "seborrheic keratosis": "Seborrheic Keratosis",
        "seborrhoeic keratosis": "Seborrheic Keratosis",
        "seb": "Seborrheic Keratosis",
        "se": "Seborrheic Keratosis",
        "squamous cell carcinoma": "Squamous Cell Carcinoma",
        "scc": "Squamous Cell Carcinoma",
        "sq": "Squamous Cell Carcinoma",
    },
}


def _normalize_label_text(text: str) -> str:
    low = text.strip().lower()
    low = re.sub(r"[^a-z0-9\s]", " ", low)
    low = re.sub(r"\s+", " ", low).strip()
    return low


def infer_dataset_spec(rows):
    labels = []
    seen = set()
    for row in rows:
        label = row["ground_truth"].strip()
        if label and label not in seen:
            seen.add(label)
            labels.append(label)

    aliases = {}
    for label in labels:
        normalized = _normalize_label_text(label)
        if normalized:
            aliases[normalized] = label
        for token in normalized.split():
            if len(token) >= 3 and token not in aliases:
                aliases[token] = label

    return labels, aliases


def load_predictions(pred_file: str):
    with open(os.path.expanduser(pred_file), "r", newline="") as handle:
        return list(csv.DictReader(handle))


def canonicalize_label(text: str, aliases) -> str:
    low = text.strip().lower()
    low = re.sub(r"[^a-z\s]", " ", low)
    low = re.sub(r"\s+", " ", low).strip()
    if not low:
        return ""
    if low in aliases:
        return aliases[low]
    first_token = low.split()[0]
    if first_token in aliases:
        return aliases[first_token]
    for alias, label in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        if len(alias) < 3:
            continue
        if alias in low:
            return label
    return ""


def main(args):
    rows = load_predictions(args.pred_file)
    if args.dataset_preset.strip().lower() == "auto":
        labels, aliases = infer_dataset_spec(rows)
    else:
        labels = DATASET_LABELS[args.dataset_preset]
        aliases = DATASET_ALIASES[args.dataset_preset]
    y_true = [row["ground_truth"] for row in rows]
    y_pred = [row["predicted_answer"] or canonicalize_label(row.get("raw_text", ""), aliases) for row in rows]

    acc = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    metrics = {
        "n": len(rows),
        "num_classes": len(labels),
        "acc": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "unmapped_predictions": sum(1 for value in y_pred if not value),
    }

    print(json.dumps(metrics, indent=2))

    if args.metrics_file:
        metrics_path = os.path.expanduser(args.metrics_file)
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--dataset-preset", type=str, default="ham10k")
    parser.add_argument("--metrics_file", type=str, default="")
    main(parser.parse_args())