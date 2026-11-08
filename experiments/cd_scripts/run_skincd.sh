#!/usr/bin/env bash
# SkinCD paper-table runner: one command per row of the results table in README §1.
# Verified environment: transformers 5.8.0 / torch 2.7.1 (see requirements.txt & environment.json).
# All four arms below are bit-exact on RTX 4090 D (driver 580.95.05); on other hardware expect
# small distributional differences, not row-identical outputs.
#
# Usage:
#   bash run_skincd.sh <arm> <dataset> <model_path> <image_root> <out_dir>
#   arm     := base | skincd
#   dataset := patch16 | dermnet | ham10k_official | ham10k_repro
set -euo pipefail

arm="$1"; dataset="$2"; model_path="$3"; image_root="$4"; out_dir="$5"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-python}"

case "$dataset" in
  patch16)
    data_csv="$REPO_ROOT/data/csv/Patch16_2class_test_10pct_seed42.csv"
    # image_root must contain Patch16/<tissue>/<file>.jpg (see DATASET_PREP.md)
    ;;
  dermnet)
    data_csv="$REPO_ROOT/data/csv/Dermnet_test.csv"
    ;;
  ham10k_official)
    data_csv="$REPO_ROOT/data/csv/HAM10K_ISIC2018_test.csv"
    ;;
  ham10k_repro)
    data_csv="$REPO_ROOT/data/csv/ham10k_repro_1511.csv"
    ;;
  *) echo "unknown dataset: $dataset" >&2; exit 2 ;;
esac

mkdir -p "$out_dir"

common_args=(
  "$PY" -u "$REPO_ROOT/experiments/eval/classification.py"
  --model-path "$model_path"
  --data-file "$data_csv"
  --image-root "$image_root"
  --answers-file "$out_dir/answers.csv"
  --dataset-preset auto
  --device-map cuda:0 --torch-dtype bfloat16
  --num-shards 1 --shard-index 0
  --seed 55 --constrain-labels
  --do-sample --temperature 0.5 --top_p 0.9 --top_k 0
  --max_new_tokens 32
)

case "$dataset" in
  patch16)  layers="12,28"; gamma="0.5" ;;
  dermnet)  layers="28,28"; gamma="0.1" ;;
  ham10k_*) layers="12,28"; gamma="0.5" ;;  # paper-default VOLA config, see README §5
esac

if [[ "$arm" == "skincd" ]]; then
  "${common_args[@]}" \
    --use_cd --cd-method vola \
    --dola-layers "$layers" --dola-relative-top 0.001 \
    --vola-gamma "$gamma" \
    --vola-perturb-method skin_morph --vola-perturb-strength 1.0 \
    --skin-morph-color 0.0 --skin-morph-boundary 0.7 \
    --skin-morph-texture 0.45 --skin-morph-occlusion 0.4
else
  "${common_args[@]}"
fi

"$PY" "$REPO_ROOT/experiments/eval/eval_skin_classification.py" \
  --pred_file "$out_dir/answers.csv" --dataset-preset auto \
  --metrics_file "$out_dir/metrics.csv"

echo "== $dataset / $arm =="
cat "$out_dir/metrics.csv"
