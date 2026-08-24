import copy
import inspect
import json
import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput
from transformers.generation.configuration_utils import GenerationConfig

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer

try:
    from transformers.generation.utils import SampleOutput
except ImportError:
    SampleOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]

try:
    from transformers.generation.utils import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput
except ImportError:
    SampleDecoderOnlyOutput = GenerateDecoderOnlyOutput
    SampleEncoderDecoderOutput = GenerateEncoderDecoderOutput


_ORIGINAL_VALIDATE_MODEL_KWARGS = None
_ORIGINAL_SAMPLE = None


def _append_cd_debug_record(debug_path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
    with open(debug_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _relative_top_filter(
    scores: torch.FloatTensor,
    relative_top: float = 0.1,
    min_tokens_to_keep: int = 1,
    filter_value: float = -1e3,
) -> torch.FloatTensor:
    scores_normalized = scores.log_softmax(dim=-1)
    sorted_logits, _ = torch.sort(scores_normalized, descending=True)
    min_thresh = sorted_logits[..., min_tokens_to_keep - 1]
    probs_max = torch.max(scores_normalized, dim=-1).values
    probs_thresh = probs_max + torch.log(torch.tensor(relative_top, device=scores.device, dtype=scores.dtype))
    probs_thresh = torch.min(min_thresh, probs_thresh)
    probs_thresh = probs_thresh.unsqueeze(-1)
    return scores_normalized.masked_fill(scores_normalized < probs_thresh, filter_value)


def _project_hidden_to_logits(model, hidden_state: torch.FloatTensor) -> torch.FloatTensor:
    lm_head_device = model.lm_head.weight.device
    return model.lm_head(hidden_state.to(lm_head_device))


def _compute_dola_logits(
    model,
    outputs,
    mature_layer: int,
    premature_layer: Optional[int],
    candidate_premature_layers: Optional[List[int]],
    relative_top: float,
) -> Tuple[torch.FloatTensor, Dict[str, Any]]:
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise ValueError("DoLa requires output_hidden_states=True, but hidden states were not returned")

    if mature_layer >= len(hidden_states):
        raise ValueError(f"mature_layer={mature_layer} out of range for hidden_states len={len(hidden_states)}")

    mature_logits = _project_hidden_to_logits(model, hidden_states[mature_layer][:, -1:, :])[:, -1, :]
    debug_info: Dict[str, Any] = {
        "dola_mature_layer": mature_layer,
        "dola_selected_premature_layer": premature_layer,
    }

    if candidate_premature_layers:
        stacked_premature_logits = torch.stack(
            [_project_hidden_to_logits(model, hidden_states[layer][:, -1:, :])[:, -1, :] for layer in candidate_premature_layers],
            dim=0,
        )
        softmax_mature_layer = F.softmax(mature_logits, dim=-1)
        softmax_premature_layers = F.softmax(stacked_premature_logits, dim=-1)
        mixture = 0.5 * (softmax_mature_layer.unsqueeze(0) + softmax_premature_layers)
        log_softmax_mature_layer = F.log_softmax(mature_logits, dim=-1)
        log_softmax_premature_layers = F.log_softmax(stacked_premature_logits, dim=-1)
        kl1 = F.kl_div(log_softmax_mature_layer.unsqueeze(0), mixture, reduction="none").mean(-1)
        kl2 = F.kl_div(log_softmax_premature_layers, mixture, reduction="none").mean(-1)
        js_divs = 0.5 * (kl1 + kl2)
        js_divs = js_divs.mean(-1)
        selected_index = int(js_divs.argmax().cpu().item())
        premature_layer = candidate_premature_layers[selected_index]
        base_logits = stacked_premature_logits[selected_index]
        debug_info.update(
            {
                "dola_selected_premature_layer": premature_layer,
                "dola_js_divergences": {
                    str(layer): float(js_divs[idx].item()) for idx, layer in enumerate(candidate_premature_layers)
                },
            }
        )
    else:
        if premature_layer is None:
            raise ValueError("DoLa requires either premature_layer or candidate_premature_layers")
        if premature_layer >= len(hidden_states):
            raise ValueError(
                f"premature_layer={premature_layer} out of range for hidden_states len={len(hidden_states)}"
            )
        base_logits = _project_hidden_to_logits(model, hidden_states[premature_layer][:, -1:, :])[:, -1, :]

    final_logits = mature_logits
    if relative_top > 0.0:
        final_logits = _relative_top_filter(final_logits, relative_top)
        base_logits = base_logits.log_softmax(dim=-1)
        mask = final_logits < -1e3
        base_logits = base_logits.masked_fill(mask, -1e3)

    return final_logits - base_logits, debug_info


def _compute_cross_view_dola_logits(
    model,
    mature_outputs,
    premature_outputs,
    mature_layer: int,
    premature_layer: Optional[int],
    candidate_premature_layers: Optional[List[int]],
    relative_top: float,
    gamma: float,
) -> Tuple[torch.FloatTensor, Dict[str, Any]]:
    mature_hidden_states = mature_outputs.hidden_states
    premature_hidden_states = premature_outputs.hidden_states
    if mature_hidden_states is None or premature_hidden_states is None:
        raise ValueError("Vola requires output_hidden_states=True on both clean and perturbed branches")

    if mature_layer >= len(mature_hidden_states):
        raise ValueError(
            f"mature_layer={mature_layer} out of range for clean hidden_states len={len(mature_hidden_states)}"
        )

    mature_logits = _project_hidden_to_logits(model, mature_hidden_states[mature_layer][:, -1:, :])[:, -1, :]
    debug_info: Dict[str, Any] = {
        "vola_mature_layer": mature_layer,
        "vola_selected_premature_layer": premature_layer,
        "vola_gamma": gamma,
    }

    if candidate_premature_layers:
        stacked_premature_logits = torch.stack(
            [
                _project_hidden_to_logits(model, premature_hidden_states[layer][:, -1:, :])[:, -1, :]
                for layer in candidate_premature_layers
            ],
            dim=0,
        )
        softmax_mature_layer = F.softmax(mature_logits, dim=-1)
        softmax_premature_layers = F.softmax(stacked_premature_logits, dim=-1)
        mixture = 0.5 * (softmax_mature_layer.unsqueeze(0) + softmax_premature_layers)
        log_softmax_mature_layer = F.log_softmax(mature_logits, dim=-1)
        log_softmax_premature_layers = F.log_softmax(stacked_premature_logits, dim=-1)
        kl1 = F.kl_div(log_softmax_mature_layer.unsqueeze(0), mixture, reduction="none").mean(-1)
        kl2 = F.kl_div(log_softmax_premature_layers, mixture, reduction="none").mean(-1)
        js_divs = 0.5 * (kl1 + kl2)
        js_divs = js_divs.mean(-1)
        selected_index = int(js_divs.argmax().cpu().item())
        premature_layer = candidate_premature_layers[selected_index]
        base_logits = stacked_premature_logits[selected_index]
        debug_info.update(
            {
                "vola_selected_premature_layer": premature_layer,
                "vola_js_divergences": {
                    str(layer): float(js_divs[idx].item()) for idx, layer in enumerate(candidate_premature_layers)
                },
            }
        )
    else:
        if premature_layer is None:
            raise ValueError("Vola requires either premature_layer or candidate_premature_layers")
        if premature_layer >= len(premature_hidden_states):
            raise ValueError(
                f"premature_layer={premature_layer} out of range for perturbed hidden_states len={len(premature_hidden_states)}"
            )
        base_logits = _project_hidden_to_logits(model, premature_hidden_states[premature_layer][:, -1:, :])[:, -1, :]

    final_logits = mature_logits
    if relative_top > 0.0:
        final_logits = _relative_top_filter(final_logits, relative_top)
        base_logits = base_logits.log_softmax(dim=-1)
        mask = final_logits < -1e3
        base_logits = base_logits.masked_fill(mask, -1e3)

    return final_logits - gamma * base_logits, debug_info


def _validate_model_kwargs_with_vcd(self, model_kwargs):
    """Allow VCD-specific kwargs to pass HF generate() validation."""
    vcd_kwargs = {}
    for key in (
        "images_cd",
        "cd_alpha",
        "cd_beta",
        "cd_strategy",
        "cd_dola_mature_layer",
        "cd_dola_premature_layer",
        "cd_dola_candidate_premature_layers",
        "cd_dola_relative_top",
        "cd_vola_perturb_method",
        "cd_vola_perturb_strength",
        "cd_vola_gamma",
        "vcd_greedy",
        "cd_use_apc",
        "cd_method",
        "cd_keep_images_steps",
        "cd_debug_path",
        "cd_debug_sample_id",
        "cd_debug_topk",
    ):
        if key in model_kwargs:
            vcd_kwargs[key] = model_kwargs.pop(key)

    try:
        return _ORIGINAL_VALIDATE_MODEL_KWARGS(self, model_kwargs)
    finally:
        # Restore VCD kwargs so downstream sampling logic can still read them.
        model_kwargs.update(vcd_kwargs)



def sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[Any, torch.LongTensor]:
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    batch_size = input_ids.shape[0]
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    this_peer_finished = False

    model_kwargs_clean = model_kwargs.copy()
    vcd_only_kwargs = {}
    for key in (
        "images_cd",
        "cd_alpha",
        "cd_beta",
        "cd_strategy",
        "cd_dola_mature_layer",
        "cd_dola_premature_layer",
        "cd_dola_candidate_premature_layers",
        "cd_dola_relative_top",
        "cd_vola_perturb_method",
        "cd_vola_perturb_strength",
        "cd_vola_gamma",
        "vcd_greedy",
        "cd_use_apc",
        "cd_method",
        "cd_keep_images_steps",
        "cd_debug_path",
        "cd_debug_sample_id",
        "cd_debug_topk",
    ):
        if key in model_kwargs_clean:
            vcd_only_kwargs[key] = model_kwargs_clean.pop(key)
    model_kwargs_cd = model_kwargs_clean.copy()
    model_kwargs_cd.update(vcd_only_kwargs)

    cd_method = model_kwargs_cd.get("cd_method")
    use_dola = cd_method == "dola"
    use_vola = cd_method == "vola"
    use_cd_branch = model_kwargs_cd.get("images_cd") is not None
    use_cd = use_cd_branch or use_dola
    vcd_greedy = bool(model_kwargs_cd.get("vcd_greedy", False)) if use_cd else False
    cd_alpha = float(model_kwargs_cd.get("cd_alpha", 0.5)) if use_cd else 0.0
    cd_beta = float(model_kwargs_cd.get("cd_beta", 0.1)) if use_cd else 0.1
    cd_strategy = str(model_kwargs_cd.get("cd_strategy", "linear")) if use_cd else "linear"
    cd_dola_mature_layer = model_kwargs_cd.get("cd_dola_mature_layer") if use_dola else None
    cd_dola_premature_layer = model_kwargs_cd.get("cd_dola_premature_layer") if use_dola else None
    cd_dola_candidate_premature_layers = (
        list(model_kwargs_cd.get("cd_dola_candidate_premature_layers") or []) if use_dola else None
    )
    cd_dola_relative_top = float(model_kwargs_cd.get("cd_dola_relative_top", 0.1)) if use_dola else 0.0
    if use_vola:
        cd_dola_mature_layer = model_kwargs_cd.get("cd_dola_mature_layer")
        cd_dola_premature_layer = model_kwargs_cd.get("cd_dola_premature_layer")
        cd_dola_candidate_premature_layers = list(model_kwargs_cd.get("cd_dola_candidate_premature_layers") or [])
        cd_dola_relative_top = float(model_kwargs_cd.get("cd_dola_relative_top", 0.1))
    cd_vola_gamma = float(model_kwargs_cd.get("cd_vola_gamma", 1.0)) if use_vola else 1.0
    use_apc = bool(model_kwargs_cd.get("cd_use_apc", False)) if use_cd else False
    cd_keep_images_steps = int(model_kwargs_cd.get("cd_keep_images_steps", 0)) if use_cd else 0
    cd_debug_path = model_kwargs_cd.get("cd_debug_path") if use_cd else None
    cd_debug_sample_id = model_kwargs_cd.get("cd_debug_sample_id") if use_cd else None
    cd_debug_topk = int(model_kwargs_cd.get("cd_debug_topk", 8)) if use_cd else 8
    debug_written = False
    cd_decode_steps = 0
    generation_config.output_hidden_states = output_hidden_states or use_dola
    if use_dola or use_vola:
        model_kwargs_clean["output_hidden_states"] = True
    if use_vola:
        model_kwargs_cd["output_hidden_states"] = True

    model_forward = (
        self.get_compiled_call(generation_config.compile_config)
        if self._valid_auto_compile_criteria(model_kwargs_clean, generation_config)
        else self.__call__
    )

    prefill_consumed_clean = False
    outputs = self._prefill(
        input_ids,
        generation_config,
        model_kwargs_clean,
        is_first_iteration=not generation_config.is_assistant,
    )

    prefill_consumed_cd = False
    outputs_cd = None
    if use_cd_branch:
        model_kwargs_cd_prefill = model_kwargs_cd.copy()
        images_cd_prefill = model_kwargs_cd_prefill.pop("images_cd", None)
        model_kwargs_cd_prefill.pop("cd_alpha", None)
        model_kwargs_cd_prefill.pop("cd_beta", None)
        model_kwargs_cd_prefill.pop("cd_strategy", None)
        model_kwargs_cd_prefill.pop("cd_dola_mature_layer", None)
        model_kwargs_cd_prefill.pop("cd_dola_premature_layer", None)
        model_kwargs_cd_prefill.pop("cd_dola_candidate_premature_layers", None)
        model_kwargs_cd_prefill.pop("cd_dola_relative_top", None)
        model_kwargs_cd_prefill.pop("cd_vola_perturb_method", None)
        model_kwargs_cd_prefill.pop("cd_vola_perturb_strength", None)
        model_kwargs_cd_prefill.pop("cd_vola_gamma", None)
        model_kwargs_cd_prefill.pop("vcd_greedy", None)
        model_kwargs_cd_prefill.pop("cd_use_apc", None)
        model_kwargs_cd_prefill.pop("cd_method", None)
        model_kwargs_cd_prefill.pop("cd_keep_images_steps", None)
        model_kwargs_cd_prefill.pop("cd_debug_path", None)
        model_kwargs_cd_prefill.pop("cd_debug_sample_id", None)
        model_kwargs_cd_prefill.pop("cd_debug_topk", None)
        if images_cd_prefill is not None:
            model_kwargs_cd_prefill["pixel_values"] = images_cd_prefill
        outputs_cd = self._prefill(
            input_ids,
            generation_config,
            model_kwargs_cd_prefill,
            is_first_iteration=not generation_config.is_assistant,
        )
        # Keep cache/state produced from the CD-prefill forward pass.
        model_kwargs_cd = model_kwargs_cd_prefill

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        if prefill_consumed_clean:
            next_sequence_length = 1 if model_kwargs_clean.get("use_cache", False) else None
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, next_sequence_length=next_sequence_length, **model_kwargs_clean
            )
            with self._optimize_model_for_decode():
                outputs = model_forward(**model_inputs, return_dict=True)
        prefill_consumed_clean = True
        model_kwargs_clean = self._update_model_kwargs_for_generation(
            outputs, model_kwargs_clean, is_encoder_decoder=self.config.is_encoder_decoder
        )

        if use_cd_branch:
            if prefill_consumed_cd:
                next_sequence_length_cd = 1 if model_kwargs_cd.get("use_cache", False) else None
                if cd_keep_images_steps > 0 and model_kwargs_cd.get("pixel_values") is None and "images_cd" in vcd_only_kwargs:
                    if cd_decode_steps < cd_keep_images_steps:
                        model_kwargs_cd["images_cd"] = vcd_only_kwargs["images_cd"]
                    else:
                        model_kwargs_cd.pop("images_cd", None)
                model_inputs_cd = self.prepare_inputs_for_generation_cd(
                    input_ids, next_sequence_length=next_sequence_length_cd, **model_kwargs_cd
                )
                with self._optimize_model_for_decode():
                    outputs_cd = model_forward(**model_inputs_cd, return_dict=True)
                cd_decode_steps += 1
            prefill_consumed_cd = True
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd, model_kwargs_cd, is_encoder_decoder=self.config.is_encoder_decoder
            )

        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

        if use_dola:
            next_token_logits, dola_debug = _compute_dola_logits(
                self,
                outputs,
                mature_layer=int(cd_dola_mature_layer),
                premature_layer=int(cd_dola_premature_layer) if cd_dola_premature_layer is not None else None,
                candidate_premature_layers=cd_dola_candidate_premature_layers,
                relative_top=cd_dola_relative_top,
            )
            next_token_logits = next_token_logits.to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token_scores = logits_processor(input_ids, next_token_logits)

            if cd_debug_path and not debug_written:
                topk = min(cd_debug_topk, next_token_scores.shape[-1])
                combined_vals, combined_ids = torch.topk(next_token_scores[0], k=topk)
                _append_cd_debug_record(
                    cd_debug_path,
                    {
                        "sample_id": cd_debug_sample_id,
                        "cd_method": cd_method,
                        "topk": topk,
                        "combined_top": [
                            {"token_id": int(token_id), "score": float(score)}
                            for score, token_id in zip(combined_vals.tolist(), combined_ids.tolist())
                        ],
                        **dola_debug,
                    },
                )
                debug_written = True
        elif use_vola:
            next_token_logits, vola_debug = _compute_cross_view_dola_logits(
                self,
                outputs,
                outputs_cd,
                mature_layer=int(cd_dola_mature_layer),
                premature_layer=int(cd_dola_premature_layer) if cd_dola_premature_layer is not None else None,
                candidate_premature_layers=cd_dola_candidate_premature_layers,
                relative_top=cd_dola_relative_top,
                gamma=cd_vola_gamma,
            )
            next_token_logits = next_token_logits.to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token_scores = logits_processor(input_ids, next_token_logits)

            if cd_debug_path and not debug_written:
                topk = min(cd_debug_topk, next_token_scores.shape[-1])
                combined_vals, combined_ids = torch.topk(next_token_scores[0], k=topk)
                _append_cd_debug_record(
                    cd_debug_path,
                    {
                        "sample_id": cd_debug_sample_id,
                        "cd_method": cd_method,
                        "topk": topk,
                        "combined_top": [
                            {"token_id": int(token_id), "score": float(score)}
                            for score, token_id in zip(combined_vals.tolist(), combined_ids.tolist())
                        ],
                        **vola_debug,
                    },
                )
                debug_written = True
        elif use_cd_branch:
            next_token_logits_cd = outputs_cd.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            clean_scores = logits_processor(input_ids, next_token_logits)
            cd_scores = logits_processor(input_ids, next_token_logits_cd)

            finite_mask = torch.isfinite(clean_scores) & torch.isfinite(cd_scores)
            if cd_strategy == "candidate_logprob":
                next_token_scores = clean_scores.clone()
                for row_idx in range(clean_scores.shape[0]):
                    row_mask = finite_mask[row_idx]
                    if not row_mask.any():
                        continue
                    clean_row = clean_scores[row_idx, row_mask]
                    cd_row = cd_scores[row_idx, row_mask]
                    clean_row_logprob = torch.log_softmax(clean_row, dim=-1)
                    cd_row_logprob = torch.log_softmax(cd_row, dim=-1)
                    row_delta = clean_row_logprob - cd_row_logprob
                    next_token_scores[row_idx, row_mask] = clean_row + cd_alpha * row_delta
            else:
                # True continuous perturbation on top of base scores.
                # alpha=0 -> exactly base; small alpha -> naturally near base.
                delta = torch.zeros_like(clean_scores)
                delta[finite_mask] = clean_scores[finite_mask] - cd_scores[finite_mask]
                next_token_scores = clean_scores + cd_alpha * delta

            if use_apc:
                cutoff = torch.log(torch.tensor(cd_beta, device=clean_scores.device)) + clean_scores.max(
                    dim=-1, keepdim=True
                ).values
                next_token_scores = next_token_scores.masked_fill(clean_scores < cutoff, -float("inf"))

            invalid_rows = ~torch.isfinite(next_token_scores).any(dim=-1)
            if invalid_rows.any():
                next_token_scores[invalid_rows] = clean_scores[invalid_rows]

            if cd_debug_path and not debug_written:
                topk = min(cd_debug_topk, next_token_scores.shape[-1])
                raw_clean_vals, raw_clean_ids = torch.topk(next_token_logits[0], k=topk)
                raw_cd_vals, raw_cd_ids = torch.topk(next_token_logits_cd[0], k=topk)
                clean_vals, clean_ids = torch.topk(clean_scores[0], k=topk)
                cd_vals, cd_ids = torch.topk(cd_scores[0], k=topk)
                combined_vals, combined_ids = torch.topk(next_token_scores[0], k=topk)
                finite_raw_mask = torch.isfinite(next_token_logits[0]) & torch.isfinite(next_token_logits_cd[0])
                finite_debug_mask = torch.isfinite(clean_scores[0]) & torch.isfinite(cd_scores[0])
                finite_combined_mask = torch.isfinite(clean_scores[0]) & torch.isfinite(next_token_scores[0])
                raw_clean_cd_abs = (next_token_logits[0][finite_raw_mask] - next_token_logits_cd[0][finite_raw_mask]).abs()
                clean_cd_abs = (clean_scores[0][finite_debug_mask] - cd_scores[0][finite_debug_mask]).abs()
                combined_clean_abs = (next_token_scores[0][finite_combined_mask] - clean_scores[0][finite_combined_mask]).abs()
                _append_cd_debug_record(
                    cd_debug_path,
                    {
                        "sample_id": cd_debug_sample_id,
                        "cd_alpha": cd_alpha,
                        "cd_beta": cd_beta,
                        "cd_strategy": cd_strategy,
                        "cd_method": model_kwargs.get("cd_method") or model_kwargs_cd.get("cd_method"),
                        "topk": topk,
                        "raw_clean_top": [
                            {"token_id": int(token_id), "score": float(score)}
                            for score, token_id in zip(raw_clean_vals.tolist(), raw_clean_ids.tolist())
                        ],
                        "raw_cd_top": [
                            {"token_id": int(token_id), "score": float(score)}
                            for score, token_id in zip(raw_cd_vals.tolist(), raw_cd_ids.tolist())
                        ],
                        "clean_top": [
                            {"token_id": int(token_id), "score": float(score)}
                            for score, token_id in zip(clean_vals.tolist(), clean_ids.tolist())
                        ],
                        "cd_top": [
                            {"token_id": int(token_id), "score": float(score)}
                            for score, token_id in zip(cd_vals.tolist(), cd_ids.tolist())
                        ],
                        "combined_top": [
                            {"token_id": int(token_id), "score": float(score)}
                            for score, token_id in zip(combined_vals.tolist(), combined_ids.tolist())
                        ],
                        "raw_clean_cd_finite_count": int(finite_raw_mask.sum().item()),
                        "raw_clean_cd_max_abs_diff": float(raw_clean_cd_abs.max().item()) if raw_clean_cd_abs.numel() else 0.0,
                        "raw_clean_cd_mean_abs_diff": float(raw_clean_cd_abs.mean().item()) if raw_clean_cd_abs.numel() else 0.0,
                        "clean_cd_finite_count": int(finite_debug_mask.sum().item()),
                        "combined_clean_finite_count": int(finite_combined_mask.sum().item()),
                        "clean_cd_max_abs_diff": float(clean_cd_abs.max().item()) if clean_cd_abs.numel() else 0.0,
                        "clean_cd_mean_abs_diff": float(clean_cd_abs.mean().item()) if clean_cd_abs.numel() else 0.0,
                        "combined_clean_max_abs_diff": float(combined_clean_abs.max().item()) if combined_clean_abs.numel() else 0.0,
                        "combined_clean_mean_abs_diff": float(combined_clean_abs.mean().item()) if combined_clean_abs.numel() else 0.0,
                    },
                )
                debug_written = True
        else:
            next_token_scores = logits_processor(input_ids, next_token_logits)

        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        if do_sample and not vcd_greedy:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0

        del outputs
        if use_cd and outputs_cd is not None:
            del outputs_cd

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids

def evolve_vcd_sampling():
    global _ORIGINAL_SAMPLE
    if _ORIGINAL_SAMPLE is None:
        _ORIGINAL_SAMPLE = transformers.generation.utils.GenerationMixin._sample

    transformers.generation.utils.GenerationMixin.sample = sample
    # sample is now a protected function in the latest Transformers library
    transformers.generation.utils.GenerationMixin._sample = sample

    global _ORIGINAL_VALIDATE_MODEL_KWARGS
    if _ORIGINAL_VALIDATE_MODEL_KWARGS is None:
        _ORIGINAL_VALIDATE_MODEL_KWARGS = transformers.generation.utils.GenerationMixin._validate_model_kwargs
        transformers.generation.utils.GenerationMixin._validate_model_kwargs = _validate_model_kwargs_with_vcd
