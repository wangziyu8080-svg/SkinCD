import argparse
import ast
import csv
import difflib
import os
import re
import sys
from typing import List

import torch
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoConfig, AutoProcessor, set_seed
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from vcd_utils.perturbation_factory import build_contrast_image
from vcd_utils.vcd_sample import evolve_vcd_sampling

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model


DATASET_CONFIGS = {
    "ham10k": {
        "labels": [
            "Actinic Keratoses",
            "Basal Cell Carcinoma",
            "Benign Keratosis",
            "Dermatofibroma",
            "Melanoma",
            "Nevus",
            "Vascular lesions",
        ],
        "aliases": {
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
    },
    "pad": {
        "labels": [
            "Actinic Keratosis",
            "Basal Cell Carcinoma",
            "Melanoma",
            "Nevus",
            "Seborrheic Keratosis",
            "Squamous Cell Carcinoma",
        ],
        "aliases": {
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
    },
}


def _normalize_label_text(text: str) -> str:
    low = text.strip().lower()
    low = re.sub(r"[^a-z0-9\s]", " ", low)
    low = re.sub(r"\s+", " ", low).strip()
    return low


def infer_dataset_config(rows):
    labels = []
    seen = set()
    for row in rows:
        label = parse_ground_truth(row["categories"])
        if label not in seen:
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

    return {"labels": labels, "aliases": aliases}


def _patch_qwen25_prepare_inputs_for_generation_cd():
    if hasattr(Qwen2_5_VLForConditionalGeneration, "prepare_inputs_for_generation_cd"):
        return

    original_prepare = Qwen2_5_VLForConditionalGeneration.prepare_inputs_for_generation

    def prepare_inputs_for_generation_cd(self, input_ids, **kwargs):
        kwargs = dict(kwargs)
        images_cd = kwargs.pop("images_cd", None)
        kwargs.pop("cd_alpha", None)
        kwargs.pop("cd_beta", None)
        kwargs.pop("cd_strategy", None)
        kwargs.pop("vcd_greedy", None)
        kwargs.pop("cd_use_apc", None)
        kwargs.pop("cd_method", None)
        kwargs.pop("cd_keep_images_steps", None)
        kwargs.pop("cd_dola_mature_layer", None)
        kwargs.pop("cd_dola_premature_layer", None)
        kwargs.pop("cd_dola_candidate_premature_layers", None)
        kwargs.pop("cd_dola_relative_top", None)
        kwargs.pop("cd_vola_perturb_method", None)
        kwargs.pop("cd_vola_perturb_strength", None)
        kwargs.pop("cd_vola_gamma", None)
        kwargs.pop("cd_debug_path", None)
        kwargs.pop("cd_debug_sample_id", None)
        kwargs.pop("cd_debug_topk", None)
        keep_steps = int(kwargs.pop("cd_keep_images_steps", 0) or 0)
        if images_cd is not None and (
            kwargs.get("past_key_values", None) is None or keep_steps > 0
        ):
            kwargs["pixel_values"] = images_cd
            if keep_steps > 0 and kwargs.get("past_key_values", None) is not None:
                kwargs["cd_keep_images_steps"] = keep_steps - 1
        return original_prepare(self, input_ids, **kwargs)

    Qwen2_5_VLForConditionalGeneration.prepare_inputs_for_generation_cd = prepare_inputs_for_generation_cd


_patch_qwen25_prepare_inputs_for_generation_cd()


def build_default_dola_layers(model_config) -> List[int]:
    text_config = getattr(model_config, "text_config", model_config)
    num_layers = getattr(text_config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError("Could not determine text model hidden layer count for DoLa")
    start = max(2, num_layers // 2)
    candidate_layers = list(range(start, num_layers, 2))
    if not candidate_layers or candidate_layers[-1] != num_layers:
        candidate_layers.append(num_layers)
    return candidate_layers


def build_default_vola_layers(model_config) -> List[int]:
    text_config = getattr(model_config, "text_config", model_config)
    num_layers = getattr(text_config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError("Could not determine text model hidden layer count for Vola")
    mature_layer = num_layers
    shallow_limit = max(6, num_layers // 2)
    candidate_layers = list(range(2, shallow_limit + 1, 2))
    if not candidate_layers:
        candidate_layers = [max(1, num_layers - 1)]
    if candidate_layers[-1] == mature_layer:
        candidate_layers = candidate_layers[:-1]
    return candidate_layers + [mature_layer]


def resolve_dola_layer_config(args, model_config):
    if args.cd_method not in ("dola", "vola"):
        return None, None, None
    if args.dola_layers.strip():
        layers = [int(value.strip()) for value in args.dola_layers.split(",") if value.strip()]
    elif args.cd_method == "vola":
        layers = build_default_vola_layers(model_config)
    else:
        layers = build_default_dola_layers(model_config)
    if len(layers) < 2:
        raise ValueError("DoLa requires at least two layers in --dola-layers")
    if len(layers) == 2:
        return layers[-1], layers[0], None
    return layers[-1], None, layers[:-1]


def get_dataset_config(dataset_name: str):
    key = dataset_name.strip().lower()
    if key not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset preset: {dataset_name}")
    return DATASET_CONFIGS[key]


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

    match = difflib.get_close_matches(low, list(aliases.keys()), n=1, cutoff=0.7)
    if match:
        return aliases[match[0]]

    return ""


def parse_ground_truth(raw_categories: str) -> str:
    categories = ast.literal_eval(raw_categories)
    if not categories:
        raise ValueError("Empty categories field")
    return str(categories[0])


def build_prompt(labels) -> str:
    joined = ", ".join(labels)
    return (
        "This is a skin lesion image. "
        f"From the following categories: {joined}, which one is the diagnosis? "
        "Respond using exactly one category name only."
    )


def build_label_token_sequences(tokenizer, labels):
    sequences = []
    seen = set()
    for label in labels:
        for variant in (label, f" {label}"):
            token_ids = tuple(tokenizer.encode(variant, add_special_tokens=False))
            if token_ids and token_ids not in seen:
                seen.add(token_ids)
                sequences.append(list(token_ids))
    return sequences


def build_prefix_allowed_tokens_fn(prompt_len, label_token_sequences, eos_token_id):
    def prefix_allowed_tokens_fn(batch_id, input_ids):
        del batch_id
        generated = input_ids[prompt_len:].tolist()
        allowed = set()
        for sequence in label_token_sequences:
            if len(generated) > len(sequence):
                continue
            if sequence[: len(generated)] != generated:
                continue
            if len(generated) == len(sequence):
                allowed.add(eos_token_id)
            else:
                allowed.add(sequence[len(generated)])
        if allowed:
            return sorted(allowed)

        fallback = {sequence[0] for sequence in label_token_sequences if sequence}
        return sorted(fallback)

    return prefix_allowed_tokens_fn


def apply_qwen_chat_template(processor, messages):
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


def load_rows(data_file: str):
    with open(os.path.expanduser(data_file), "r", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def shard_rows(rows, num_shards: int, shard_index: int):
    if num_shards <= 1:
        return rows
    return [row for idx, row in enumerate(rows) if idx % num_shards == shard_index]


def detect_model_family(model_path: str) -> str:
    model_type = AutoConfig.from_pretrained(model_path, trust_remote_code=True).model_type
    if model_type == "qwen2_5_vl":
        return "qwen2_5_vl"
    if model_type.startswith("llava"):
        return "llava"
    raise ValueError(f"Unsupported model type for classification evaluation: {model_type}")


def load_qwen_backend(args):
    model_path = os.path.expanduser(args.model_path)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=True,
    ).eval()
    return processor, model


def load_llava_backend(args):
    model_path = os.path.expanduser(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        args.model_base,
        os.path.basename(model_path.rstrip("/")),
        device_map=args.device_map,
    )
    model.eval()
    return tokenizer, model, image_processor


def build_llava_prompt(prompt_text: str, model_config, conv_mode: str):
    if getattr(model_config, "mm_use_im_start_end", False):
        text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt_text
    else:
        text = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)
    return conv, conv.get_prompt()


def strip_stop_text(text: str, conv) -> str:
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    text = text.strip()
    if text.endswith(stop_str):
        text = text[: -len(stop_str)]
    return text.strip()


def eval_model(args):
    if args.use_cd:
        evolve_vcd_sampling()

    rows = load_rows(args.data_file)
    if args.dataset_preset.strip().lower() == "auto":
        dataset_config = infer_dataset_config(rows)
    else:
        dataset_config = get_dataset_config(args.dataset_preset)
    labels = dataset_config["labels"]
    aliases = dataset_config["aliases"]

    rows = shard_rows(rows, args.num_shards, args.shard_index)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    model_path = os.path.expanduser(args.model_path)
    family = detect_model_family(model_path)
    prompt = build_prompt(labels)
    do_sample = bool(args.do_sample)

    if family == "qwen2_5_vl":
        processor, model = load_qwen_backend(args)
        tokenizer = processor.tokenizer
        image_processor = processor
    else:
        tokenizer, model, image_processor = load_llava_backend(args)

    dola_mature_layer, dola_premature_layer, dola_candidate_premature_layers = resolve_dola_layer_config(args, model.config)
    eos_token_id = tokenizer.eos_token_id
    label_token_sequences = build_label_token_sequences(tokenizer, labels) if args.constrain_labels else []

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    with open(answers_file, "w", newline="") as handle:
        fieldnames = [
            "question_id",
            "image",
            "prompt",
            "ground_truth",
            "predicted_answer",
            "raw_text",
            "is_correct",
            "use_cd",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(tqdm(rows), 1):
            image_rel = row["image"]
            ground_truth = parse_ground_truth(row["categories"])
            image_path = os.path.join(args.image_root, image_rel)

            if family == "qwen2_5_vl":
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "path": image_path},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                inputs = apply_qwen_chat_template(processor, messages).to(model.device)
                prompt_len = inputs["input_ids"].shape[1]
                prefix_allowed_tokens_fn = None
                if args.constrain_labels:
                    prefix_allowed_tokens_fn = build_prefix_allowed_tokens_fn(
                        prompt_len=prompt_len,
                        label_token_sequences=label_token_sequences,
                        eos_token_id=eos_token_id,
                    )
                images_cd = None
                if args.use_cd and args.cd_method not in ("dola",):
                    perturb_method = args.cd_method
                    perturb_strength = args.cd_alpha
                    if args.cd_method == "vola":
                        perturb_method = args.vola_perturb_method
                        perturb_strength = args.vola_perturb_strength
                    images_cd = build_contrast_image(
                        image_processor,
                        image_path,
                        inputs["pixel_values"],
                        method=perturb_method,
                        noise_step=args.noise_step,
                        strength=perturb_strength,
                    )

                with torch.inference_mode():
                    generate_kwargs = {
                        "do_sample": do_sample,
                        "temperature": args.temperature if do_sample else None,
                        "top_p": args.top_p if do_sample else None,
                        "top_k": args.top_k if do_sample else None,
                        "max_new_tokens": args.max_new_tokens,
                        "pad_token_id": eos_token_id,
                        "use_cache": True,
                    }
                    if args.use_cd:
                        generate_kwargs.update(
                            {
                                "images_cd": images_cd,
                                "cd_alpha": args.cd_alpha,
                                "cd_beta": args.cd_beta,
                                "cd_strategy": args.cd_strategy,
                                "cd_method": args.cd_method,
                                "cd_dola_mature_layer": dola_mature_layer,
                                "cd_dola_premature_layer": dola_premature_layer,
                                "cd_dola_candidate_premature_layers": dola_candidate_premature_layers,
                                "cd_dola_relative_top": args.dola_relative_top,
                                "cd_vola_perturb_method": args.vola_perturb_method,
                                "cd_vola_perturb_strength": args.vola_perturb_strength,
                                "cd_vola_gamma": args.vola_gamma,
                                "cd_keep_images_steps": args.cd_keep_images_steps,
                                "vcd_greedy": True,
                            }
                        )
                    if prefix_allowed_tokens_fn is not None:
                        generate_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
                    output_ids = model.generate(**inputs, **generate_kwargs)

                generated = output_ids[:, prompt_len:]
                raw_text = processor.batch_decode(
                    generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()
            else:
                conv, formatted_prompt = build_llava_prompt(prompt, model.config, args.conv_mode)
                input_ids = tokenizer_image_token(
                    formatted_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                ).unsqueeze(0).to(model.device)
                prompt_len = input_ids.shape[1]
                prefix_allowed_tokens_fn = None
                if args.constrain_labels:
                    prefix_allowed_tokens_fn = build_prefix_allowed_tokens_fn(
                        prompt_len=prompt_len,
                        label_token_sequences=label_token_sequences,
                        eos_token_id=eos_token_id,
                    )
                image = Image.open(image_path).convert("RGB")
                image_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=model.dtype)
                images_cd = None
                if args.use_cd and args.cd_method not in ("dola",):
                    perturb_method = args.cd_method
                    perturb_strength = args.cd_alpha
                    if args.cd_method == "vola":
                        perturb_method = args.vola_perturb_method
                        perturb_strength = args.vola_perturb_strength
                    images_cd = build_contrast_image(
                        image_processor,
                        image_path,
                        image_tensor,
                        method=perturb_method,
                        noise_step=args.noise_step,
                        strength=perturb_strength,
                    )

                with torch.inference_mode():
                    generate_kwargs = {
                        "input_ids": input_ids,
                        "images": image_tensor,
                        "do_sample": do_sample,
                        "temperature": args.temperature if do_sample else None,
                        "top_p": args.top_p if do_sample else None,
                        "top_k": args.top_k if do_sample else None,
                        "max_new_tokens": args.max_new_tokens,
                        "pad_token_id": eos_token_id,
                        "use_cache": True,
                    }
                    if args.use_cd:
                        generate_kwargs.update(
                            {
                                "images_cd": images_cd,
                                "cd_alpha": args.cd_alpha,
                                "cd_beta": args.cd_beta,
                                "cd_strategy": args.cd_strategy,
                                "cd_method": args.cd_method,
                                "cd_dola_mature_layer": dola_mature_layer,
                                "cd_dola_premature_layer": dola_premature_layer,
                                "cd_dola_candidate_premature_layers": dola_candidate_premature_layers,
                                "cd_dola_relative_top": args.dola_relative_top,
                                "cd_vola_perturb_method": args.vola_perturb_method,
                                "cd_vola_perturb_strength": args.vola_perturb_strength,
                                "cd_vola_gamma": args.vola_gamma,
                                "cd_keep_images_steps": args.cd_keep_images_steps,
                                "vcd_greedy": True,
                            }
                        )
                    if prefix_allowed_tokens_fn is not None:
                        generate_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
                    output_ids = model.generate(**generate_kwargs)

                raw_text = tokenizer.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)[0]
                raw_text = strip_stop_text(raw_text, conv)

            predicted_answer = canonicalize_label(raw_text, aliases)
            is_correct = predicted_answer == ground_truth
            writer.writerow(
                {
                    "question_id": idx,
                    "image": image_rel,
                    "prompt": prompt,
                    "ground_truth": ground_truth,
                    "predicted_answer": predicted_answer,
                    "raw_text": raw_text,
                    "is_correct": is_correct,
                    "use_cd": args.use_cd,
                }
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--dataset-preset", type=str, default="ham10k")
    parser.add_argument("--data-file", type=str, required=True)
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--use_cd", action="store_true", default=False)
    parser.add_argument("--constrain-labels", action="store_true", default=False)
    parser.add_argument(
        "--cd-method",
        type=str,
        default="vcd",
        choices=["vcd", "color", "boundary_blur", "texture_blur", "occlusion", "skin_morph", "dola", "vola"],
    )
    parser.add_argument("--cd_alpha", type=float, default=1.0)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--cd_strategy", type=str, default="linear", choices=["linear", "candidate_logprob"])
    parser.add_argument("--dola-layers", type=str, default="")
    parser.add_argument("--dola-relative-top", type=float, default=0.1)
    parser.add_argument(
        "--vola-perturb-method",
        type=str,
        default="color",
        choices=["vcd", "color", "boundary_blur", "texture_blur", "occlusion", "skin_morph"],
    )
    parser.add_argument("--vola-perturb-strength", type=float, default=80.0)
    parser.add_argument("--vola-gamma", type=float, default=1.0)
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--torch-dtype", type=str, default="auto")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--cd-keep-images-steps", type=int, default=0)
    parser.add_argument("--cd-debug-path", type=str, default="")
    parser.add_argument("--cd-debug-topk", type=int, default=8)
    args = parser.parse_args()

    if args.torch_dtype == "auto":
        args.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else None
    set_seed(args.seed)
    eval_model(args)