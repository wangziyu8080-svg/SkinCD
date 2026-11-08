#!/usr/bin/env python
"""OPERA (CVPR 2024) 移植实现：Qwen2.5-VL 闭集皮肤分类适配版
官方实现绑定 transformers-4.29.2 的 LLaVA/MiniGPT4；此处按论文算法独立实现：
  1. Partial autoregressive 解码（候选级前向取注意力）
  2. Over-trust 惩罚：前10步对图像 token 的聚合注意力过高 → 惩罚候选分数
  3. Rollback：局部注意力峰值位置历史重复 → 回滚重解码
参数（论文默认）: scale_factor=50, threshold=15, num_attn_candidates=5, penalty_weights=1.0
v2 修复：候选打分用 base 前向末位 logits（v1 误用候选后位置分布导致空输出）；
        base 前向不请求注意力（省 28 层显存）；候选前向注意力即时聚合释放。
"""
import torch


@torch.inference_mode()
def opera_generate_qwen(
    model,
    processor,
    inputs,
    max_new_tokens=32,
    scale_factor=50.0,
    threshold=15,
    num_attn_candidates=5,
    penalty_weights=1.0,
    rollback_budget=3,
):
    device = inputs["input_ids"].device
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]
    img_token_id = _img_token_id(model)

    def build_kwargs(ids, need_attn):
        kwargs = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "use_cache": False}
        for k in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw",
                  "mm_token_type_ids"):
            if inputs.get(k) is not None:
                kwargs[k] = inputs[k]
        # M-RoPE: 混合序列需显式 3D 位置；追加 token 时 mm_token_type_ids 同步扩 0
        try:
            mm = kwargs.get("mm_token_type_ids")
            if mm is not None and mm.shape[1] < ids.shape[1]:
                pad = torch.zeros(
                    (mm.shape[0], ids.shape[1] - mm.shape[1]),
                    dtype=mm.dtype, device=mm.device)
                mm = torch.cat([mm, pad], dim=1)
                kwargs["mm_token_type_ids"] = mm
            position_ids, _ = model.model.get_rope_index(
                ids,
                mm_token_type_ids=mm,
                image_grid_thw=kwargs.get("image_grid_thw"),
                video_grid_thw=kwargs.get("video_grid_thw"),
                attention_mask=kwargs["attention_mask"],
            )
            kwargs["position_ids"] = position_ids
        except Exception as _e:
            print(f"[opera] rope fallback: {type(_e).__name__}: {str(_e)[:120]}", flush=True)
        kwargs["output_attentions"] = need_attn
        return kwargs

    def forward_attn(ids, need_attn=True):
        return model(**build_kwargs(ids, need_attn))

    generated = []
    rollback_count = 0
    history_locs = []
    eos = _eos_ids(processor)

    step = 0
    while step < max_new_tokens:
        full_ids = (
            torch.cat([input_ids, torch.tensor([generated], device=device)], dim=1)
            if generated else input_ids
        )

        # base 前向：不取注意力（省显存），末位 logits 出候选
        base_out = forward_attn(full_ids, need_attn=False)
        if not getattr(model, "_opera_attn_ready", False):
            # 首步用一次注意力前向自检 eager
            _chk = forward_attn(full_ids, need_attn=True)
            if not _chk.attentions:
                raise RuntimeError("OPERA 需要 eager 注意力权重：请加载时加 attn_implementation='eager'")
            model._opera_attn_ready = True
            del _chk
        base_logits = base_out.logits[0, -1].float()
        cand_tokens = base_logits.topk(num_attn_candidates).indices.tolist()
        cand_logprobs = torch.log_softmax(base_logits, -1)[cand_tokens]  # 候选正确打分
        del base_out

        # --- 每候选一次前向：仅取末层注意力（partial autoregressive）---
        attn_rows = []
        for tok in cand_tokens:
            ids = torch.cat([full_ids, torch.tensor([[tok]], device=device)], dim=1)
            out = forward_attn(ids, need_attn=True)
            # 即时头聚合并释放整栈注意力 [1,1,q,kv]
            attn_rows.append(out.attentions[-1].max(1, keepdim=True).values)
            del out
        attn_last = torch.cat(attn_rows, 1)  # [1, C, q, kv]
        del attn_rows

        attn_last = attn_last / attn_last.sum(-1, keepdim=True).clamp_min(1e-9)
        q_len = attn_last.shape[2]

        resp_start = prompt_len
        img_mask = (full_ids[0] == img_token_id)
        img_pos = img_mask.nonzero(as_tuple=True)[0]

        n_resp = q_len - resp_start
        if n_resp <= 0:
            generated.append(cand_tokens[0]); step += 1
            if cand_tokens[0] in eos: break
            continue

        attn_local = scale_factor * attn_last[:, :, resp_start:, resp_start:]
        # over-trust 惩罚项：末位 query 对图像 token 的聚合注意力（须在 attn_last 释放前计算）
        if len(img_pos):
            attn_i = attn_last[:, :, -1, img_pos.min():img_pos.max() + 1].sum(-1)  # [1, C]
        else:
            attn_i = torch.zeros(1, len(cand_tokens), device=device)

        w = attn_local.shape[-1]
        local_scores = torch.zeros(1, len(cand_tokens), w, device=device)
        for j in range(w):
            local_scores[:, :, j] = (1e-7 * attn_local[..., j:, j].prod(-1)).float()
        del attn_local, attn_last

        penalty = -attn_i if n_resp <= 10 else local_scores.max(-1).values
        cand_scores = cand_logprobs - penalty_weights * penalty[0]
        best = int(cand_scores.argmax())

        # 回滚判定：局部峰值位置众数进入历史（论文 threshold 次重复）
        locs = local_scores[0].argmax(-1).flatten().tolist()
        mode_loc = max(set(locs), key=locs.count)
        history_locs.append(mode_loc)
        if n_resp > 10 and history_locs[-threshold:].count(mode_loc) >= threshold:
            if rollback_count < rollback_budget:
                cut = max(1, len(generated) - max(1, mode_loc))
                generated = generated[:cut]
                rollback_count += 1
                history_locs = []
                continue

        generated.append(cand_tokens[best])
        step += 1
        if cand_tokens[best] in eos:
            break

    out_ids = torch.tensor([generated], device=device)
    return torch.cat([input_ids, out_ids], dim=1)


def _eos_ids(processor):
    tok = getattr(processor, "tokenizer", processor)
    ids = set()
    v = getattr(tok, "eos_token_id", None)
    if v is not None:
        ids.add(int(v))
    return ids or {151645}


def _img_token_id(model):
    cfg = getattr(model, "config", None)
    v = getattr(cfg, "image_token_id", None)
    if v is not None:
        return int(v)
    return 151655  # <|image_pad|>
