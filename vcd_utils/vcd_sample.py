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
    premature_hidden_states = premature_