#!/usr/bin/env bash
set -euo pipefail

gpu_devices=${CUDA_DEVICES:-0}
export CUDA_VISIBLE_DEVICES="${gpu_devices}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exp_dir="$(cd "${script_dir}/.." && pwd)"
cd "${exp_dir}"

seed=${1:-952}
model_path=${2:-"Qwen/Qwen2.5-VL-7B-Instruct"}
cd_beta=${3:-0.1}
noise_step=${4:-500}
alphas=${ALPHAS:-"10 20 40 80"}
output_dir=${OUTPUT_DIR:-"${exp_dir}/output/strong_color_sweep"}

mkdir -p "${output_dir}"

for alpha in ${alphas}; do
  echo "===== alpha=${alpha} color ====="
  CUDA_DEVICES="${CUDA_VISIBLE_DEVICES}" USE_CD=1 CONSTRAIN_LABELS=1 CD_METHOD=color \
    bash "${script_dir}/qwen2_5_ham10k.sh" "${seed}" "${model_path}" "${alpha}" "${cd_beta}" "${noise_step}" color
  cp "${exp_dir}/output/qwen2_5_vl_ham10k_color_constrained_seed${seed}_shard0of1.csv" "${output_dir}/color_alpha${alpha}.csv"
  cp "${exp_dir}/output/qwen2_5_vl_ham10k_color_constrained_seed${seed}_shard0of1_metrics.csv" "${output_dir}/color_alpha${alpha}_metrics.csv"

  echo "===== alpha=${alpha} vcd ====="
  CUDA_DEVICES="${CUDA_VISIBLE_DEVICES}" USE_CD=1 CONSTRAIN_LABELS=1 CD_METHOD=vcd \
    bash "${script_dir}/qwen2_5_ham10k.sh" "${seed}" "${model_path}" "${alpha}" "${cd_beta}" "${noise_step}" vcd
  cp "${exp_dir}/output/qwen2_5_vl_ham10k_vcd_constrained_seed${seed}_shard0of1.csv" "${output_dir}/vcd_alpha${alpha}.csv"
  cp "${exp_dir}/output/qwen2_5_vl_ham10k_vcd_constrained_seed${seed}_shard0of1_metrics.csv" "${output_dir}/vcd_alpha${alpha}_metrics.csv"
done

python - <<'PY'
import csv
import os

alphas = [int(value) for value in os.environ.get("ALPHAS", "10 20 40 80").split()]
output_dir = os.environ.get("OUTPUT_DIR")

summary_rows = []
for alpha in alphas:
    for method in ("color", "vcd"):
        metrics_path = os.path.join(output_dir, f"{method}_alpha{alpha}_metrics.csv")
        with open(metrics_path, newline="") as handle:
            row = next(csv.DictReader(handle))
        row["method"] = method
        row["alpha"] = alpha
        row["pred_file"] = f"{method}_alpha{alpha}.csv"
        summary_rows.append(row)

summary_path = os.path.join(output_dir, "summary.csv")
with open(summary_path, "w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "method",
            "alpha",
            "n",
            "acc",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
            "unmapped_predictions",
            "pred_file",
        ],
    )
    writer.writeheader()
    writer.writerows(summary_rows)

compare_rows = []
for alpha in alphas:
    with open(os.path.join(output_dir, f"color_alpha{alpha}.csv"), newline="") as handle:
        color_rows = list(csv.DictReader(handle))
    with open(os.path.join(output_dir, f"vcd_alpha{alpha}.csv"), newline="") as handle:
        vcd_rows = list(csv.DictReader(handle))

    compare_rows.append(
        {
            "alpha": alpha,
            "num_predicted_answer_diff": sum(
                color_row["predicted_answer"] != vcd_row["predicted_answer"]
                for color_row, vcd_row in zip(color_rows, vcd_rows)
            ),
            "num_raw_text_diff": sum(
                color_row["raw_text"] != vcd_row["raw_text"]
                for color_row, vcd_row in zip(color_rows, vcd_rows)
            ),
            "num_is_correct_diff": sum(
                color_row["is_correct"] != vcd_row["is_correct"]
                for color_row, vcd_row in zip(color_rows, vcd_rows)
            ),
        }
    )

compare_path = os.path.join(output_dir, "color_vs_vcd_diff.csv")
with open(compare_path, "w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["alpha", "num_predicted_answer_diff", "num_raw_text_diff", "num_is_correct_diff"],
    )
    writer.writeheader()
    writer.writerows(compare_rows)

print(summary_path)
print(compare_path)
PY
