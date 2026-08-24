#!/usr/bin/env bash
set -euo pipefail

gpu_devices=${CUDA_DEVICES:-0}
export CUDA_VISIBLE_DEVICES="${gpu_devices}"
echo "[VCD-HAM10K] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exp_dir="$(cd "${script_dir}/.." && pwd)"
cd "${exp_dir}"

seed=${1:-55}
model_path=${2:-"Qwen/Qwen2.5-VL-7B-Instruct"}
cd_alpha=${3:-1}
cd_beta=${4:-0.1}
noise_step=${5:-500}
cd_method=${6:-${CD_METHOD:-vcd}}
device_map=${DEVICE_MAP:-auto}
data_file=${HAM10K_DATA_FILE:?Set HAM10K_DATA_FILE to the evaluation CSV path}
image_root=${HAM10K_IMAGE_ROOT:?Set HAM10K_IMAGE_ROOT to the dataset root}
num_shards=${NUM_SHARDS:-1}
shard_index=${SHARD_INDEX:-0}
max_samples=${MAX_SAMPLES:-}
use_cd=${USE_CD:-1}
constrain_labels=${CONSTRAIN_LABELS:-0}

mode_tag="${cd_method}"
gen_extra_args=""
label_tag=""
if [[ "${constrain_labels}" == "1" ]]; then
  gen_extra_args="${gen_extra_args} --constrain-labels"
  label_tag="_constrained"
  echo "[VCD-HAM10K] Label-constrained decoding enabled"
fi
if [[ "${use_cd}" == "1" ]]; then
  gen_extra_args="${gen_extra_args} --use_cd --do-sample --cd-method ${cd_method} --cd_alpha ${cd_alpha} --cd_beta ${cd_beta} --noise_step ${noise_step}"
  if [[ "${cd_method}" == "dola" ]]; then
    dola_layers=${DOLA_LAYERS:-}
    dola_relative_top=${DOLA_RELATIVE_TOP:-0.1}
    if [[ -n "${dola_layers}" ]]; then
      gen_extra_args="${gen_extra_args} --dola-layers ${dola_layers}"
    fi
    gen_extra_args="${gen_extra_args} --dola-relative-top ${dola_relative_top}"
  elif [[ "${cd_method}" == "vola" ]]; then
    dola_layers=${DOLA_LAYERS:-}
    dola_relative_top=${DOLA_RELATIVE_TOP:-0.1}
    vola_perturb_method=${VOLA_PERTURB_METHOD:-color}
    vola_perturb_strength=${VOLA_PERTURB_STRENGTH:-80}
    if [[ -n "${dola_layers}" ]]; then
      gen_extra_args="${gen_extra_args} --dola-layers ${dola_layers}"
    fi
    gen_extra_args="${gen_extra_args} --dola-relative-top ${dola_relative_top} --vola-perturb-method ${vola_perturb_method} --vola-perturb-strength ${vola_perturb_strength}"
  fi
  echo "[VCD-HAM10K] Contrastive decoding enabled: method=${cd_method}, alpha=${cd_alpha}, beta=${cd_beta}, noise_step=${noise_step}"
else
  mode_tag="base"
  echo "[VCD-HAM10K] Baseline decoding enabled (no contrastive branch)"
fi

answers_file="./output/qwen2_5_vl_ham10k_${mode_tag}${label_tag}_seed${seed}_shard${shard_index}of${num_shards}.csv"
metrics_file="./output/qwen2_5_vl_ham10k_${mode_tag}${label_tag}_seed${seed}_shard${shard_index}of${num_shards}_metrics.csv"

max_sample_args=""
if [[ -n "${max_samples}" ]]; then
  max_sample_args="--max-samples ${max_samples}"
fi

python ./eval/classification.py \
  --model-path "${model_path}" \
  --data-file "${data_file}" \
  --image-root "${image_root}" \
  --answers-file "${answers_file}" \
  --device-map "${device_map}" \
  --num-shards "${num_shards}" \
  --shard-index "${shard_index}" \
  --seed "${seed}" \
  ${gen_extra_args} \
  ${max_sample_args}

python ./eval/eval_skin_classification.py \
  --pred_file "${answers_file}" \
  --metrics_file "${metrics_file}"
