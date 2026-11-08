#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
IMAGE_ROOT="${IMAGE_ROOT:?Set IMAGE_ROOT to the dataset root}"
SKINREFOCUS_ROOT="${SKINREFOCUS_ROOT:?Set SKINREFOCUS_ROOT to the dataset-registry project root}"
RESULT_ROOT="${RESULT_ROOT:-./output/multidataset_dola_vola}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3,4,5,6,7}"

# Ready classification datasets from SkinRefocus registry.
DATASETS=(
  bcn20000
  ddi
  dermnet
  ham10k
  hiba
  hiba_2class
  isic2019
  pad
  patch16
  patch16_2class
  scin
)
METHODS=(dola vola)

IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_CSV"
NUM_GPUS="${#GPU_IDS[@]}"

mkdir -p "$RESULT_ROOT/logs" "$RESULT_ROOT/results"

resolve_dataset() {
  local dataset_key="$1"
  "$PYTHON_BIN" "$SKINREFOCUS_ROOT/utils/dataset_registry.py" resolve \
    --dataset-key "$dataset_key" \
    --project-root "$SKINREFOCUS_ROOT" \
    --image-folder "$IMAGE_ROOT"
}

launch_job() {
  local gpu_id="$1"
  local dataset_key="$2"
  local method="$3"
  local data_file="$4"
  local answers_file="$RESULT_ROOT/results/${dataset_key}_${method}.csv"
  local metrics_file="$RESULT_ROOT/results/${dataset_key}_${method}_metrics.csv"

  local extra_args=()
  if [[ "$method" == "dola" ]]; then
    extra_args+=(--cd-method dola --dola-relative-top 0.1)
  else
    extra_args+=(
      --cd-method vola
      --dola-layers 12,28
      --dola-relative-top 0.2
      --vola-perturb-method color
      --vola-perturb-strength 80
      --vola-gamma 0.5
    )
  fi

  echo "[launch] gpu=${gpu_id} dataset=${dataset_key} method=${method}"
  CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" ./eval/classification.py \
    --dataset-preset auto \
    --model-path "$MODEL_PATH" \
    --data-file "$data_file" \
    --image-root "$IMAGE_ROOT" \
    --answers-file "$answers_file" \
    --device-map auto \
    --num-shards 1 \
    --shard-index 0 \
    --seed 952 \
    --constrain-labels \
    --use_cd \
    --do-sample \
    "${extra_args[@]}"

  "$PYTHON_BIN" ./eval/eval_skin_classification.py \
    --dataset-preset auto \
    --pred_file "$answers_file" \
    --metrics_file "$metrics_file"
}

task_index=0
for dataset_key in "${DATASETS[@]}"; do
  eval "$(resolve_dataset "$dataset_key")"
  if [[ "$DATASET_READY" != "1" ]]; then
    echo "[skip] dataset=${dataset_key} not ready: ${DATASET_MISSING}"
    continue
  fi

  for method in "${METHODS[@]}"; do
    gpu_id="${GPU_IDS[$((task_index % NUM_GPUS))]}"
    log_file="$RESULT_ROOT/logs/${dataset_key}_${method}.log"
    {
      echo "[start] $(date -Iseconds) dataset=${dataset_key} method=${method} gpu=${gpu_id}"
      echo "[dataframe] ${DATAFRAME_OVERRIDE}"
      launch_job "$gpu_id" "$dataset_key" "$method" "$DATAFRAME_OVERRIDE"
      echo "[done] $(date -Iseconds) dataset=${dataset_key} method=${method}"
    } &> "$log_file" &
    task_index=$((task_index + 1))

    while (( $(jobs -rp | wc -l) >= NUM_GPUS )); do
      wait -n
    done
  done
done

wait

RESULT_ROOT_ENV="$RESULT_ROOT" "$PYTHON_BIN" - <<'PY'
import csv
import os

result_root = os.path.join(os.path.expanduser(os.environ["RESULT_ROOT_ENV"]), "results")
rows = []
for name in sorted(os.listdir(result_root)):
  if not name.endswith("_metrics.csv"):
    continue
  dataset_key, method, _ = name.rsplit("_", 2)
  with open(os.path.join(result_root, name), newline="") as handle:
    row = next(csv.DictReader(handle))
  rows.append(
    {
      "dataset": dataset_key,
      "method": method,
      "acc": row["acc"],
      "precision_macro": row["precision_macro"],
      "precision_weighted": row["precision_weighted"],
      "f1_macro": row["f1_macro"],
      "f1_weighted": row["f1_weighted"],
      "n": row["n"],
      "num_classes": row["num_classes"],
    }
  )

summary_path = os.path.join(result_root, "summary.csv")
with open(summary_path, "w", newline="") as handle:
  writer = csv.DictWriter(
    handle,
    fieldnames=["dataset", "method", "acc", "precision_macro", "precision_weighted", "f1_macro", "f1_weighted", "n", "num_classes"],
  )
  writer.writeheader()
  writer.writerows(rows)

print(f"summary written to {summary_path}")
PY

echo "All jobs finished at $(date -Iseconds)"
