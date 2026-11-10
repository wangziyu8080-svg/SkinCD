# SkinCD Patch16 / DermNet 复现记录的真实验证报告（2026-11-06）

**版本：2026-11-06；状态：本文所有数字均为本次真实全量 GPU 重跑结果，非读取历史文件。**

## 0. 一眼看结论

对 `SKINCD_PATCH16_DERMNET_REPRO_SUMMARY.md` 中两套最佳配置进行了**从零全量重放**（同一台 RTX 4090 D，冻结源码 + 恢复的冻结环境），结果：

| 配置 | n | 记录 Accuracy | 本次重跑 Accuracy | 逐图差异 |
|---|---:|---:|---:|---:|
| **Patch16 最佳**（`vola`，层 `12,28`，gamma 0.5，r=0.001，去色0） | 2,804 | 0.8384450784593438 | **0.8384450784593438** | **0 / 2,804** |
| **DermNet 最佳**（`vola`，层 `28,28`，gamma 0.1，r=0.001，去色0） | 4,002 | 0.27211394302848574 | **0.27211394302848574** | **0 / 4,002** |

Macro-F1、Recall 等全部指标到小数点后 16 位完全一致；两套 `answers.csv` 与历史证据逐图（image / ground_truth / prompt / predicted_answer / use_cd 五字段）**零差异**。

**结论：复现记录可信，可以精确复现。** 此前"无法复现"的根因是**环境漂移**（见第 2 节），不是记录或证据的问题。

## 1. 本次验证了什么

1. 冻结包完整性：`patch16_repro_20261010` 176 个文件哈希全部通过（`reproduce.py` 的 `verify_bundle()`）。
2. 数据指纹：Patch16 2,804 张、DermNet 4,002 张逐图 SHA-256 全部匹配；两份测试 CSV 哈希匹配。
3. 模型指纹：Qwen2.5-VL-7B-Instruct 的 14 个文件（5 片权重 + tokenizer/processor）大小与 SHA-256 全部匹配。
4. **真实全量推理**：Patch16 约 21 分钟，DermNet 约 75 分钟；随后独立重算指标并与历史 `answers.csv` 逐图比对。

## 2. 根因：环境漂移（此前无法复现的真正原因）

冻结记录要求 `/data/miniconda3` Python 3.12.9 + **transformers 5.8.0**。但该主环境现已变为 **transformers 4.48.3**（被后续其他实验覆盖），`reproduce.py` 的 `check_environment()` 直接报错：

```
ValueError: Package mismatch: transformers expected=5.8.0 actual=4.48.3
```

用错误版本的 transformers 跑 VOLA 对比解码，行为可能与历史不一致。**本文档所在工作区没有修改主环境**，而是按 `patch16_repro_20261010/environment.json` 逐版本恢复了独立 venv：

**复现必须使用的 Python：`/data/venvs/skincd_tf580/bin/python`**

| 包 | 版本（与冻结记录一致） |
|---|---|
| python | 3.12.9（基于 `/data/miniconda3/bin/python -m venv`） |
| transformers / tokenizers | 5.8.0 / 0.22.2 |
| torch / torchvision | 2.7.1 / 0.22.1（cu126） |
| accelerate | 1.13.0 |
| numpy / pillow / scikit-learn | 2.2.6 / 11.3.0 / 1.7.1 |
| safetensors / huggingface-hub | 0.8.0 / 1.14.0 |
| sentencepiece / einops / peft / timm | 0.2.2 / 0.8.2 / 0.20.0 / 1.0.19 |

无 flash-attn 依赖（冻结源码与历史环境均未使用）。

> **注意**：`SKINCD_PATCH16_DERMNET_REPRO_SUMMARY.md` 3.2 节"优先使用 `/data/miniconda3/bin/python`"的指引**已失效**，应以本节 venv 为准。

## 3. 逐条可执行的复现步骤

> 下述命令中的绝对路径为本机记录原样（`$S` 指向本仓库检出目录；`/data/phase7/...` 等为本地数据根），首次使用请按 `DATASET_PREP.md` 与 README §4 替换为自己的模型/数据路径。

### 3.1 环境变量（同一 shell）

```bash
set -euo pipefail
export R=/root/experiment-models/skin/MM-Skin/third_party/anon_repos/SkinCD/experiments/output/mmskin_repro
export B="$R/patch16_repro_20261010"
export S="$B/source/SkinCD"
export PY=/data/venvs/skincd_tf580/bin/python
export MODEL=/root/experiment-models/skin/models/Qwen2.5-VL-7B-Instruct
export PATCH_CSV="$S/experiments/data/mmskin_repro/Patch16_2class_test_10pct_seed42.csv"
export PATCH_TILES=/data/phase7/Patch16/extracted/data/tiles
export D_RUN="$R/dermnet_compare_20261011_123354"
export DERM_CSV="$D_RUN/data/Dermnet_test.csv"
export DERM_IMAGES="$D_RUN/data/images"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2
export PYTHONPATH="$S:$S/experiments"
```

### 3.2 指纹校验（CPU，不占 GPU）

```bash
"$PY" - <<'PY'
import ast, csv, hashlib, json, os, runpy
from pathlib import Path
b = Path(os.environ['B'])
h = runpy.run_path(str(b / 'reproduce.py'))
h['verify_bundle']()
# 注：h['check_environment']() 校验的是 /data/miniconda3 主环境，已漂移为 4.48.3，会报错；
# venv skincd_tf580 已按 environment.json 恢复同版本，可独立验证：
import transformers, torch, accelerate
assert transformers.__version__ == '5.8.0' and torch.__version__.startswith('2.7.1')
model = Path(os.environ['MODEL'])
meta = json.loads((b / 'repro_manifest.json').read_text())
def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
for name, item in meta['model_files'].items():
    path = model / name
    assert path.is_file() and path.stat().st_size == item['bytes'], path
    assert sha(path) == item['sha256'], path
specs = [
    ('Patch16', Path(os.environ['PATCH_CSV']), 2804, 2,
     '4afa7dbacd7bee577f544089e9634ea849ee713ec32f6b2bfa6961770a604cda',
     b / 'images_manifest.json', Path(os.environ['PATCH_TILES']), True),
    ('DermNet', Path(os.environ['DERM_CSV']), 4002, 23,
     'f3ee4b27d8c2f4b205731b165032b4397eec9aab3d9c3aa61c0334e8803730fe',
     Path(os.environ['D_RUN']) / 'data/images_manifest.json', Path(os.environ['DERM_IMAGES']), False),
]
for name, csv_path, n, k, checksum, records_path, image_root, strip_prefix in specs:
    assert sha(csv_path) == checksum, csv_path
    with csv_path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    pairs = [(x['image'], ast.literal_eval(x['categories'])[0]) for x in rows]
    labels = list(dict.fromkeys(y for _, y in pairs))
    assert len(rows) == n and len(set(x for x, _ in pairs)) == n and len(labels) == k
    records = json.loads(records_path.read_text())
    assert [(x['image'], x['label']) for x in records] == pairs
    for item in records:
        rel = item['image'].removeprefix('Patch16/') if strip_prefix else item['image']
        path = image_root / rel
        assert path.is_file() and sha(path) == item['sha256'], path
    print(f'PASS {name}: {n} images / {k} classes')
print('PASS bundle, model, data')
PY
```

### 3.3 Patch16 最佳全量重放（本次实测约 21 分钟，需 ~16GB 显存）

```bash
export REPLAY="$R/replay_patch16_best_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPLAY/patch16_best"
mkdir -p "$REPLAY/patch_images"
ln -s "$PATCH_TILES" "$REPLAY/patch_images/Patch16"
CUDA_VISIBLE_DEVICES=0 "$PY" -u "$S/experiments/eval/classification.py" \
  --model-path "$MODEL" --data-file "$PATCH_CSV" --image-root "$REPLAY/patch_images" \
  --answers-file "$REPLAY/patch16_best/answers.csv" --dataset-preset auto \
  --device-map cuda:0 --torch-dtype bfloat16 \
  --num-shards 1 --shard-index 0 \
  --do-sample --temperature 0.5 --top_p 0.9 --top_k 0 \
  --max_new_tokens 32 --seed 55 --constrain-labels \
  --use_cd --cd-method vola --dola-layers 12,28 \
  --dola-relative-top 0.001 --vola-gamma 0.5 \
  --vola-perturb-method skin_morph --vola-perturb-strength 1.0 \
  --skin-morph-color 0.0 --skin-morph-boundary 0.7 \
  --skin-morph-texture 0.45 --skin-morph-occlusion 0.4 \
  > "$REPLAY/patch16_best/run.log" 2>&1
"$PY" "$S/experiments/eval/eval_skin_classification.py" \
  --pred_file "$REPLAY/patch16_best/answers.csv" --dataset-preset auto \
  --metrics_file "$REPLAY/patch16_best/metrics.csv"
```

### 3.4 DermNet 最佳全量重放（本次实测约 75 分钟）

```bash
export REPLAY="$R/replay_dermnet_best_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPLAY/dermnet_best"
CUDA_VISIBLE_DEVICES=0 "$PY" -u "$S/experiments/eval/classification.py" \
  --model-path "$MODEL" --data-file "$DERM_CSV" --image-root "$DERM_IMAGES" \
  --answers-file "$REPLAY/dermnet_best/answers.csv" --dataset-preset auto \
  --device-map cuda:0 --torch-dtype bfloat16 \
  --num-shards 1 --shard-index 0 \
  --do-sample --temperature 0.5 --top_p 0.9 --top_k 0 \
  --max_new_tokens 32 --seed 55 --constrain-labels \
  --use_cd --cd-method vola --dola-layers 28,28 \
  --dola-relative-top 0.001 --vola-gamma 0.1 \
  --vola-perturb-method skin_morph --vola-perturb-strength 1.0 \
  --skin-morph-color 0.0 --skin-morph-boundary 0.7 \
  --skin-morph-texture 0.45 --skin-morph-occlusion 0.4 \
  > "$REPLAY/dermnet_best/run.log" 2>&1
"$PY" "$S/experiments/eval/eval_skin_classification.py" \
  --pred_file "$REPLAY/dermnet_best/answers.csv" --dataset-preset auto \
  --metrics_file "$REPLAY/dermnet_best/metrics.csv"
```

### 3.5 逐图验收（与历史证据比对）

```bash
# REPLAY_P / REPLAY_D 指向各自重放的输出目录，按实际路径调整：
export REPLAY_P="$R/replay_patch16_best_YYYYMMDD_HHMMSS/patch16_best"
export REPLAY_D="$R/replay_dermnet_best_YYYYMMDD_HHMMSS/dermnet_best"
"$PY" - <<'PY'
import csv, os
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
r = Path(os.environ['R'])
for tag, new_dir, old_dir, count, classes in [
    ('patch16_best', Path(os.environ['REPLAY_P']), r / 'patch16_test_improve_20261010_150600/original_test/no_color', 2804, 2),
    ('dermnet_best', Path(os.environ['REPLAY_D']), r / 'dermnet_refine_20261012_153212/full_test/layer28_gamma0.1', 4002, 23),
]:
    def read(p):
        with p.open(newline='') as f:
            return list(csv.DictReader(f))
    actual, expected = read(new_dir / 'answers.csv'), read(old_dir / 'answers.csv')
    assert len(actual) == len(expected) == count
    fields = ('image', 'ground_truth', 'prompt', 'predicted_answer', 'use_cd')
    mism = [i for i, (x, y) in enumerate(zip(actual, expected)) if any(x[k] != y[k] for k in fields)]
    truth, pred = [x['ground_truth'] for x in actual], [x['predicted_answer'] for x in actual]
    labels = list(dict.fromkeys(truth))
    assert len(labels) == classes and all(p in labels for p in pred)
    print(tag, 'Accuracy=', accuracy_score(truth, pred), 'different_rows=', len(mism))
PY
```

## 4. 本次运行的全部证据位置（相对 `R`）

| 内容 | 路径 |
|---|---|
| Patch16 重跑 | `replay_patch16_best_20261106_155019/`（`answers.csv`、`metrics.csv`、`run.log`、`evaluation.log`、`verification.log`、`status.txt`） |
| DermNet 重跑 | `replay_dermnet_best_20261106_161433/`（同上结构） |
| 历史对照（Patch16 最佳） | `patch16_test_improve_20261010_150600/original_test/no_color/` |
| 历史对照（DermNet 最佳） | `dermnet_refine_20261012_153212/full_test/layer28_gamma0.1/` |
| 恢复的环境 | `/data/venvs/skincd_tf580/` |

## 5. 本次实测结果明细

### Patch16 最佳（`replay_patch16_best_20261106_155019`）

```
n=2804, num_classes=2, acc=0.8384450784593438
precision_macro=0.822016562075435, recall_macro=0.7930793168522918, f1_macro=0.804738707902658
precision_weighted=0.8351238814743253, recall_weighted=0.8384450784593438, f1_weighted=0.8346549200533989
unmapped_predictions=0
```

verification.log：`Accuracy= 0.8384450784593438 Macro-F1= 0.804738707902658 different_rows= 0` → `PASS`

### DermNet 最佳（`replay_dermnet_best_20261106_161433`）

```
n=4002, num_classes=23, acc=0.27211394302848574
precision_macro=0.23834797923809206, recall_macro=0.2220268048151461, f1_macro=0.19539187216874104
precision_weighted=0.29840489908794626, recall_weighted=0.27211394302848574, f1_weighted=0.23691773930143267
unmapped_predictions=0
```

verification.log：`Accuracy= 0.27211394302848574 Macro-F1= 0.19539187216874104 different_rows= 0` → `PASS`

## 6. 边界与注意事项

1. 本验证是**记录口径**的复现（Patch16 Base 80.24%→83.84%，DermNet 25.46%→27.21%），**不等于论文口径**（论文 48.85%→66.42% 的 Patch16 协议未在仓库中锁定，详见 `docs/PROTOCOL_INVESTIGATION.md` 与 `docs/PATCH16_PROTOCOL_NOTE.md`）。
2. 逐位一致的前提：同机同卡（RTX 4090 D，驱动 580.95.05）、冻结源码、venv 精确版本、同 seed=55。异机/异驱动可能因浮点差异出现少量预测漂移，属预期，不代表失败。
3. 运行显存约 16GB（模型加载 warmup 即需 ~15.5GB）；与其他训练任务共卡会 OOM。
4. 本次没有重跑 Base（80.24% / 25.46%）；历史 Base 证据在 `patch16_repro_20261010/evidence/reference/base` 与 `dermnet_compare_20261011_123354/full_test`，如需可用 3.3/3.4 的命令去掉 `--use_cd` 段后重放。
