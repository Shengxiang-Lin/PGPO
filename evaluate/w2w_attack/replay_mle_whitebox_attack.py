import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def detect_dataset_from_path(path: str) -> str:
    path_lower = str(path).lower()
    if "amazon-book" in path_lower or "books" in path_lower:
        return "amazon-book"
    if "ml-1m" in path_lower:
        return "ml-1m"
    if "ml-100k" in path_lower or "movielens" in path_lower or "movie" in path_lower:
        return "ml-1m"
    return "ml-1m"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay-MLE / No-history white-box inversion attack for PGPO-generated variants."
    )
    parser.add_argument("pairs_json_path_positional", nargs="?", default="")
    parser.add_argument("--pairs_json_path", type=str, default="")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "base_models", "Qwen3-4B-Base"),
        help="Base causal LM path used when `lora_path` is not provided, or as the base model under a LoRA adapter.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Optional PGPO LoRA checkpoint path. If omitted, the attack uses the base model only.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=os.path.join(PROJECT_ROOT, "models", "prompt", "prompt.txt"),
        help="Prompt template used by PGPO generation.",
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        choices=["pgpo", "emoji"],
        default="pgpo",
        help="Prompt construction mode. `pgpo` uses the original AM-LLM template; `emoji` uses EmojiPrompt-compatible prompts.",
    )
    parser.add_argument(
        "--prompt_dataset",
        type=str,
        default="",
        help="Dataset label used by prompt builders that depend on dataset/domain semantics, e.g. EmojiPrompt.",
    )
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--candidate_batch_size", type=int, default=8)
    parser.add_argument("--prompt_batch_size", type=int, default=128)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--load_in_4bit", type=int, choices=[0, 1], default=1)
    parser.add_argument(
        "--loader_backend",
        type=str,
        choices=["auto", "unsloth", "transformers"],
        default="auto",
        help="Model loading backend. `auto` tries unsloth first and falls back to transformers+peft on import/runtime incompatibility.",
    )
    parser.add_argument(
        "--candidate_mode",
        type=str,
        choices=["all", "same_category", "sampled_same_category"],
        default="all",
        help="Whether each target scores against the full original vocabulary or only candidates in the same category.",
    )
    parser.add_argument(
        "--num_random_negatives",
        type=int,
        default=20,
        help="Number of same-category random negatives per target when candidate_mode=sampled_same_category.",
    )
    parser.add_argument(
        "--sampling_trials",
        type=int,
        default=1,
        help="Number of repeated random-negative trials when candidate_mode=sampled_same_category.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--max_candidates_per_target",
        type=int,
        default=0,
        help="Optional hard cap on candidate count per target for smoke tests; truth is forced into the pool if needed.",
    )
    parser.add_argument(
        "--rank_eval_ks",
        type=str,
        default="1,5,10,50",
        help="Comma-separated K values used for ReID@K reporting.",
    )
    parser.add_argument("--detail_top_k", type=int, default=10)
    parser.add_argument("--save_details", action="store_true")
    parser.add_argument(
        "--posterior_temperature",
        type=float,
        default=1.0,
        help="Temperature used to convert replay scores into the attack-induced posterior.",
    )
    parser.add_argument(
        "--append_closing_answer_tag",
        type=int,
        choices=[0, 1],
        default=1,
        help="Score variant completions as `variant</answer>` to match generation prompt format.",
    )
    return parser.parse_args()


def normalize_text(value) -> str:
    return str(value).strip()


def parse_int_list(raw: str) -> List[int]:
    values = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return sorted(set(v for v in values if v > 0))


def default_output_path(input_path: str) -> str:
    base_dir = os.path.dirname(input_path)
    return os.path.join(base_dir, "w2w_replay_mle_whitebox_report.json")


def load_prompt_template(prompt_file: str) -> str:
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()


def create_no_history_prompt(word: str, category: str, prompt_template: str) -> str:
    return prompt_template.format(
        word=word,
        category=category,
        similarity_note="",
        context_str="",
    )


def create_emoji_prompt(word: str, prompt_dataset: str) -> str:
    embedding_dir = os.path.join(PROJECT_ROOT, "evaluate", "embedding")
    if embedding_dir not in sys.path:
        sys.path.insert(0, embedding_dir)
    from EmojiPrompt.prompt import build_emoji_prompt

    return build_emoji_prompt(original_word=word, dataset=prompt_dataset)


def load_pairs(path: str, max_rows: int) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    if max_rows > 0:
        rows = rows[:max_rows]

    samples = []
    for idx, row in enumerate(rows):
        original_word = normalize_text(row.get("original_word", ""))
        variant = normalize_text(row.get("variant", ""))
        category = normalize_text(row.get("category", ""))
        if not original_word or not variant or not category:
            continue
        samples.append(
            {
                "row_index": idx,
                "word_id": row.get("word_id", idx + 1),
                "original_word": original_word,
                "variant": variant,
                "category": category,
                "source": normalize_text(row.get("source", "")),
                "truth_index": len(samples),
            }
        )
    return samples


def resolve_device(requested_device: str) -> str:
    if requested_device:
        return requested_device
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def resolve_base_model_path(lora_path: Optional[str], base_model_path: str) -> str:
    if not lora_path:
        if os.path.isabs(base_model_path):
            return base_model_path
        return os.path.abspath(os.path.join(PROJECT_ROOT, base_model_path))

    adapter_config_path = os.path.join(lora_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(f"adapter_config.json not found under {lora_path}")

    with open(adapter_config_path, "r", encoding="utf-8") as f:
        adapter_config = json.load(f)

    base_model_path = adapter_config.get("base_model_name_or_path", "")
    if not base_model_path:
        raise ValueError(f"`base_model_name_or_path` missing in {adapter_config_path}")

    if os.path.isabs(base_model_path):
        return base_model_path
    return os.path.abspath(os.path.join(PROJECT_ROOT, base_model_path))


def load_model_and_tokenizer_with_transformers(
    lora_path: Optional[str],
    base_model_path: str,
    device: str,
):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_base_model_path = resolve_base_model_path(lora_path, base_model_path)
    if lora_path:
        print(f"Loading replay model with Transformers+PEFT from base model {resolved_base_model_path}")
    else:
        print(f"Loading replay model with Transformers from base model {resolved_base_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_base_model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = None
    if device != "cpu":
        if torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        resolved_base_model_path,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch_dtype,
        device_map=device if device != "cpu" else None,
    )
    if lora_path:
        model = PeftModel.from_pretrained(
            model,
            lora_path,
            local_files_only=True,
        )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    loader_name = "transformers+peft" if lora_path else "transformers"
    return model, tokenizer, loader_name


def load_model_and_tokenizer_with_unsloth(
    lora_path: Optional[str],
    base_model_path: str,
    max_seq_length: int,
    load_in_4bit: bool,
    device: str,
):
    try:
        import unsloth  # noqa: F401
        from unsloth import FastLanguageModel
    except Exception as exc:
        raise RuntimeError(f"Unsloth import failed: {exc}") from exc

    model_device = device if device.startswith("cuda") and torch.cuda.is_available() else "auto"
    effective_load_in_4bit = load_in_4bit and device != "cpu"
    if load_in_4bit and not effective_load_in_4bit:
        print("CPU mode detected: forcing load_in_4bit=0.")
    model_path = lora_path if lora_path else resolve_base_model_path(None, base_model_path)
    print(f"Loading replay model from {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=effective_load_in_4bit,
        fast_inference=False,
        gpu_memory_utilization=0.8,
        device_map=model_device,
        local_files_only=True,
    )
    model.eval()
    loader_name = "unsloth+lora" if lora_path else "unsloth"
    return model, tokenizer, loader_name


def load_model_and_tokenizer(
    lora_path: Optional[str],
    base_model_path: str,
    max_seq_length: int,
    load_in_4bit: bool,
    device: str,
    loader_backend: str,
):
    if loader_backend == "transformers":
        return load_model_and_tokenizer_with_transformers(
            lora_path=lora_path,
            base_model_path=base_model_path,
            device=device,
        )

    if loader_backend == "unsloth":
        return load_model_and_tokenizer_with_unsloth(
            lora_path=lora_path,
            base_model_path=base_model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            device=device,
        )

    try:
        return load_model_and_tokenizer_with_unsloth(
            lora_path=lora_path,
            base_model_path=base_model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            device=device,
        )
    except Exception as exc:
        print(f"Unsloth backend failed, fallback to Transformers+PEFT: {exc}")
        return load_model_and_tokenizer_with_transformers(
            lora_path=lora_path,
            base_model_path=base_model_path,
            device=device,
        )


def build_prompt_texts(
    samples: Sequence[Dict],
    prompt_mode: str,
    prompt_template: Optional[str],
    prompt_dataset: str,
) -> List[str]:
    prompts = []
    for sample in samples:
        if prompt_mode == "pgpo":
            prompts.append(
                create_no_history_prompt(
                    word=sample["original_word"],
                    category=sample["category"],
                    prompt_template=prompt_template,
                )
            )
        elif prompt_mode == "emoji":
            prompts.append(
                create_emoji_prompt(
                    word=sample["original_word"],
                    prompt_dataset=prompt_dataset,
                )
            )
        else:
            raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")
    return prompts


def compute_prompt_token_lengths(
    prompt_texts: Sequence[str],
    tokenizer,
    batch_size: int,
) -> List[int]:
    lengths: List[int] = []
    for start in tqdm(range(0, len(prompt_texts), batch_size), desc="Tokenize prompts", unit="batch"):
        batch_texts = list(prompt_texts[start : start + batch_size])
        encoded = tokenizer(
            batch_texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return lengths


def compute_suffix_token_length(suffix_text: str, tokenizer) -> int:
    encoded = tokenizer(
        suffix_text,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    return len(encoded["input_ids"])


def maybe_limit_candidate_indices(
    candidate_indices: Sequence[int],
    truth_index: int,
    max_candidates: int,
) -> Tuple[List[int], bool]:
    candidate_list = list(candidate_indices)
    if max_candidates <= 0 or len(candidate_list) <= max_candidates:
        return candidate_list, False
    limited = candidate_list[:max_candidates]
    truth_forced = False
    if truth_index not in limited and limited:
        limited[-1] = truth_index
        truth_forced = True
    return limited, truth_forced


def build_candidate_lookup(samples: Sequence[Dict], candidate_mode: str) -> Dict[str, List[int]]:
    if candidate_mode == "all":
        return {"__all__": list(range(len(samples)))}

    by_category: Dict[str, List[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        by_category[sample["category"]].append(idx)
    return dict(by_category)


def resolve_candidate_indices(
    sample: Dict,
    candidate_lookup: Dict[str, List[int]],
    candidate_mode: str,
    num_random_negatives: int,
    rng: Optional[random.Random],
) -> List[int]:
    if candidate_mode == "all":
        return list(candidate_lookup["__all__"])

    category_candidates = list(candidate_lookup.get(sample["category"], []))
    if candidate_mode == "same_category":
        return category_candidates

    if candidate_mode != "sampled_same_category":
        raise ValueError(f"Unsupported candidate_mode: {candidate_mode}")

    truth_index = sample["truth_index"]
    negatives = [idx for idx in category_candidates if idx != truth_index]
    sample_size = min(max(0, num_random_negatives), len(negatives))
    picked_negatives = rng.sample(negatives, sample_size) if sample_size > 0 else []
    sampled_candidates = [truth_index] + picked_negatives
    rng.shuffle(sampled_candidates)
    return sampled_candidates


def estimate_candidate_count(
    sample: Dict,
    candidate_lookup: Dict[str, List[int]],
    candidate_mode: str,
    num_random_negatives: int,
    max_candidates_per_target: int,
) -> int:
    if candidate_mode == "all":
        count = len(candidate_lookup["__all__"])
    else:
        same_category = candidate_lookup.get(sample["category"], [])
        if candidate_mode == "same_category":
            count = len(same_category)
        elif candidate_mode == "sampled_same_category":
            count = 1 + min(max(0, num_random_negatives), max(0, len(same_category) - 1))
        else:
            raise ValueError(f"Unsupported candidate_mode: {candidate_mode}")

    if max_candidates_per_target > 0:
        count = min(count, max_candidates_per_target)
    return count


@torch.inference_mode()
def score_target_against_candidates(
    target_suffix: str,
    candidate_indices: Sequence[int],
    prompt_texts: Sequence[str],
    prompt_token_lengths: Sequence[int],
    tokenizer,
    model,
    device: str,
    candidate_batch_size: int,
    max_seq_length: int,
) -> torch.Tensor:
    suffix_token_len = compute_suffix_token_length(target_suffix, tokenizer)
    if suffix_token_len <= 0:
        raise ValueError("Target suffix token length is zero; cannot score empty target.")

    all_scores = []
    for start in range(0, len(candidate_indices), candidate_batch_size):
        batch_indices = list(candidate_indices[start : start + candidate_batch_size])
        batch_prompt_lengths = [prompt_token_lengths[idx] for idx in batch_indices]
        max_required = max(batch_prompt_lengths) + suffix_token_len
        if max_required > max_seq_length:
            raise ValueError(
                f"max_seq_length={max_seq_length} is too small for replay scoring "
                f"(need at least {max_required} tokens for some prompt+target pair)."
            )

        combined_texts = [prompt_texts[idx] + target_suffix for idx in batch_indices]
        inputs = tokenizer(
            combined_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        ).to(device)

        outputs = model(**inputs)
        shift_logits = outputs.logits[:, :-1, :]
        shift_labels = inputs["input_ids"][:, 1:]
        shift_mask = inputs["attention_mask"][:, 1:].bool()
        token_logprobs = torch.log_softmax(shift_logits, dim=-1)
        token_logprobs = token_logprobs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        seq_positions = torch.arange(token_logprobs.size(1), device=token_logprobs.device).unsqueeze(0)
        prompt_lens_tensor = torch.tensor(batch_prompt_lengths, device=token_logprobs.device).unsqueeze(1)
        target_mask = seq_positions >= (prompt_lens_tensor - 1)
        seq_scores = (token_logprobs * (shift_mask & target_mask)).sum(dim=1)
        all_scores.append(seq_scores.to(dtype=torch.float32).cpu())

    return torch.cat(all_scores, dim=0)


def init_metric_state(rank_eval_ks: Sequence[int]) -> Dict:
    return {
        "total_rows": 0,
        "mrr_sum": 0.0,
        "rank_sum": 0.0,
        "rank_values": [],
        "hit_at_k": {int(k): 0 for k in rank_eval_ks},
        "entropy_bits_sum": 0.0,
        "prior_entropy_bits_sum": 0.0,
        "mi_proxy_bits_sum": 0.0,
        "normalized_leakage_sum": 0.0,
        "true_source_nll_bits_sum": 0.0,
        "top1_posterior_sum": 0.0,
    }


def compute_attack_posterior_metrics(
    scores: Sequence[float],
    truth_pos: int,
    candidate_count: int,
    posterior_temperature: float,
) -> Dict[str, float]:
    if posterior_temperature <= 0:
        raise ValueError("posterior_temperature must be > 0.")

    score_tensor = torch.tensor(scores, dtype=torch.float64)
    log_q = torch.log_softmax(score_tensor / posterior_temperature, dim=0)
    q = torch.exp(log_q)

    entropy_bits = float(-(q * log_q).sum().item() / math.log(2))
    prior_entropy_bits = math.log2(candidate_count) if candidate_count > 0 else 0.0
    mi_proxy_bits = prior_entropy_bits - entropy_bits
    normalized_leakage = (
        mi_proxy_bits / prior_entropy_bits if prior_entropy_bits > 0 else 0.0
    )
    true_source_nll_bits = float(-log_q[truth_pos].item() / math.log(2))
    top1_posterior = float(q.max().item())

    return {
        "entropy_bits": entropy_bits,
        "prior_entropy_bits": prior_entropy_bits,
        "mi_proxy_bits": mi_proxy_bits,
        "normalized_leakage": normalized_leakage,
        "true_source_nll_bits": true_source_nll_bits,
        "top1_posterior": top1_posterior,
    }


def update_metric_state(
    state: Dict,
    truth_rank: Optional[int],
    posterior_metrics: Dict[str, float],
    rank_eval_ks: Sequence[int],
):
    state["total_rows"] += 1
    state["entropy_bits_sum"] += float(posterior_metrics["entropy_bits"])
    state["prior_entropy_bits_sum"] += float(posterior_metrics["prior_entropy_bits"])
    state["mi_proxy_bits_sum"] += float(posterior_metrics["mi_proxy_bits"])
    state["normalized_leakage_sum"] += float(posterior_metrics["normalized_leakage"])
    state["true_source_nll_bits_sum"] += float(posterior_metrics["true_source_nll_bits"])
    state["top1_posterior_sum"] += float(posterior_metrics["top1_posterior"])

    if truth_rank is None:
        return
    state["mrr_sum"] += 1.0 / float(truth_rank)
    state["rank_sum"] += float(truth_rank)
    state["rank_values"].append(int(truth_rank))
    for k in rank_eval_ks:
        if truth_rank <= k:
            state["hit_at_k"][int(k)] += 1


def finalize_metric_state(state: Dict, rank_eval_ks: Sequence[int]) -> Dict:
    total = state["total_rows"]
    rank_values = state["rank_values"]
    result = {
        "total_rows": int(total),
        "mrr": float(state["mrr_sum"] / total) if total else 0.0,
        "mean_truth_rank": float(state["rank_sum"] / total) if total else None,
        "median_truth_rank": float(median(rank_values)) if rank_values else None,
        "posterior_entropy_bits": float(state["entropy_bits_sum"] / total) if total else 0.0,
        "prior_entropy_bits": float(state["prior_entropy_bits_sum"] / total) if total else 0.0,
        "mi_proxy_bits": float(state["mi_proxy_bits_sum"] / total) if total else 0.0,
        "normalized_leakage": float(state["normalized_leakage_sum"] / total) if total else 0.0,
        "true_source_nll_bits": float(state["true_source_nll_bits_sum"] / total) if total else 0.0,
        "mean_top1_posterior": float(state["top1_posterior_sum"] / total) if total else 0.0,
    }
    result["effective_anonymity_set"] = float(2 ** result["posterior_entropy_bits"])
    for k in rank_eval_ks:
        result[f"reid_at_{int(k)}"] = float(state["hit_at_k"][int(k)] / total) if total else 0.0
    return result


def average_summary_dicts(summary_dicts: Sequence[Dict]) -> Dict:
    if not summary_dicts:
        return {}

    keys = summary_dicts[0].keys()
    averaged = {}
    for key in keys:
        values = [item.get(key) for item in summary_dicts]
        valid_values = [value for value in values if value is not None]
        if key == "total_rows":
            averaged[key] = int(values[0]) if values else 0
        elif not valid_values:
            averaged[key] = None
        else:
            averaged[key] = float(sum(valid_values) / len(valid_values))
    return averaged


def average_category_summaries(category_summaries: Sequence[Dict[str, Dict]]) -> Dict[str, Dict]:
    all_categories = sorted({category for summary in category_summaries for category in summary.keys()})
    averaged = {}
    for category in all_categories:
        per_trial = [summary[category] for summary in category_summaries if category in summary]
        averaged[category] = average_summary_dicts(per_trial)
    return averaged


def main():
    args = parse_args()
    pairs_json_path = args.pairs_json_path or args.pairs_json_path_positional
    if not pairs_json_path:
        raise ValueError("pairs_json_path is required.")

    os.chdir(PROJECT_ROOT)

    output_path = args.output_path or default_output_path(pairs_json_path)
    device = resolve_device(args.device)
    rank_eval_ks = parse_int_list(args.rank_eval_ks)
    if not rank_eval_ks:
        raise ValueError("rank_eval_ks must contain at least one positive integer.")
    if args.num_random_negatives < 0:
        raise ValueError("num_random_negatives must be >= 0.")
    if args.sampling_trials <= 0:
        raise ValueError("sampling_trials must be >= 1.")
    if args.posterior_temperature <= 0:
        raise ValueError("posterior_temperature must be > 0.")

    print(f"Loading pairs from {pairs_json_path}")
    samples = load_pairs(pairs_json_path, args.max_rows)
    if not samples:
        raise ValueError("No valid pairs were loaded from input json.")

    prompt_dataset = args.prompt_dataset if args.prompt_dataset else detect_dataset_from_path(pairs_json_path)
    prompt_template = load_prompt_template(args.prompt_file) if args.prompt_mode == "pgpo" else None
    prompt_texts = build_prompt_texts(
        samples=samples,
        prompt_mode=args.prompt_mode,
        prompt_template=prompt_template,
        prompt_dataset=prompt_dataset,
    )

    model, tokenizer, model_loader_name = load_model_and_tokenizer(
        lora_path=args.lora_path,
        base_model_path=args.base_model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=bool(args.load_in_4bit),
        device=device,
        loader_backend=args.loader_backend,
    )
    prompt_token_lengths = compute_prompt_token_lengths(
        prompt_texts=prompt_texts,
        tokenizer=tokenizer,
        batch_size=args.prompt_batch_size,
    )

    candidate_lookup = build_candidate_lookup(samples, args.candidate_mode)
    detail_top_k = max(1, args.detail_top_k)
    effective_trials = args.sampling_trials if args.candidate_mode == "sampled_same_category" else 1

    pair_count_estimate_per_trial = 0
    for sample in samples:
        pair_count_estimate_per_trial += estimate_candidate_count(
            sample=sample,
            candidate_lookup=candidate_lookup,
            candidate_mode=args.candidate_mode,
            num_random_negatives=args.num_random_negatives,
            max_candidates_per_target=args.max_candidates_per_target,
        )

    print(
        f"Replay-MLE scoring starts: rows={len(samples)}, "
        f"candidate_mode={args.candidate_mode}, trials={effective_trials}, "
        f"approx_pairs_per_trial={pair_count_estimate_per_trial}"
    )

    trial_summaries = []
    trial_category_summaries = []
    trial_reports = []
    total_forced_truth_count = 0

    for trial_idx in range(effective_trials):
        rng = random.Random(args.seed + trial_idx)
        metrics_all = init_metric_state(rank_eval_ks)
        metrics_by_category: Dict[str, Dict] = defaultdict(lambda: init_metric_state(rank_eval_ks))
        forced_truth_count = 0
        results = []

        iterator = tqdm(
            samples,
            desc=f"Replay-MLE trial {trial_idx + 1}/{effective_trials}",
            unit="row",
        )
        for sample in iterator:
            base_candidates = resolve_candidate_indices(
                sample=sample,
                candidate_lookup=candidate_lookup,
                candidate_mode=args.candidate_mode,
                num_random_negatives=args.num_random_negatives,
                rng=rng,
            )
            candidate_indices, truth_forced = maybe_limit_candidate_indices(
                candidate_indices=base_candidates,
                truth_index=sample["truth_index"],
                max_candidates=args.max_candidates_per_target,
            )
            if truth_forced:
                forced_truth_count += 1

            target_suffix = sample["variant"]
            if args.prompt_mode == "pgpo" and args.append_closing_answer_tag:
                target_suffix = f"{target_suffix}</answer>"

            scores = score_target_against_candidates(
                target_suffix=target_suffix,
                candidate_indices=candidate_indices,
                prompt_texts=prompt_texts,
                prompt_token_lengths=prompt_token_lengths,
                tokenizer=tokenizer,
                model=model,
                device=device,
                candidate_batch_size=args.candidate_batch_size,
                max_seq_length=args.max_seq_length,
            ).numpy()

            order = scores.argsort(kind="mergesort")[::-1]
            ranked_candidate_indices = [int(candidate_indices[pos]) for pos in order]
            truth_rank = None
            truth_pos = candidate_indices.index(sample["truth_index"])
            for rank_pos, cand_idx in enumerate(ranked_candidate_indices, start=1):
                if cand_idx == sample["truth_index"]:
                    truth_rank = rank_pos
                    break

            posterior_metrics = compute_attack_posterior_metrics(
                scores=scores,
                truth_pos=truth_pos,
                candidate_count=len(candidate_indices),
                posterior_temperature=args.posterior_temperature,
            )

            update_metric_state(metrics_all, truth_rank, posterior_metrics, rank_eval_ks)
            update_metric_state(
                metrics_by_category[sample["category"]],
                truth_rank,
                posterior_metrics,
                rank_eval_ks,
            )

            if args.save_details and trial_idx == 0:
                top_entries = []
                for pos in order[:detail_top_k]:
                    cand_idx = int(candidate_indices[pos])
                    cand = samples[cand_idx]
                    raw_score = float(scores[pos])
                    top_entries.append(
                        {
                            "candidate_index": cand_idx,
                            "candidate_word_id": cand["word_id"],
                            "candidate_original_word": cand["original_word"],
                            "candidate_category": cand["category"],
                            "logp": raw_score,
                        }
                    )
            else:
                top_entries = None

            if args.save_details and trial_idx == 0:
                result_row = {
                    "row_index": int(sample["row_index"]),
                    "word_id": sample["word_id"],
                    "original_word": sample["original_word"],
                    "category": sample["category"],
                    "variant": sample["variant"],
                    "candidate_count": int(len(candidate_indices)),
                    "truth_rank": int(truth_rank) if truth_rank is not None else None,
                    "predicted_word_id": samples[ranked_candidate_indices[0]]["word_id"],
                    "predicted_original_word": samples[ranked_candidate_indices[0]]["original_word"],
                    "predicted_category": samples[ranked_candidate_indices[0]]["category"],
                    "predicted_logp": float(scores[order[0]]),
                    "posterior_entropy_bits": float(posterior_metrics["entropy_bits"]),
                    "prior_entropy_bits": float(posterior_metrics["prior_entropy_bits"]),
                    "mi_proxy_bits": float(posterior_metrics["mi_proxy_bits"]),
                    "normalized_leakage": float(posterior_metrics["normalized_leakage"]),
                    "true_source_nll_bits": float(posterior_metrics["true_source_nll_bits"]),
                    "top1_posterior": float(posterior_metrics["top1_posterior"]),
                }
                if top_entries is not None:
                    result_row["top_candidates"] = top_entries
                results.append(result_row)

            iterator.set_postfix(
                {
                    "trial": f"{trial_idx + 1}/{effective_trials}",
                    "rank1": metrics_all["hit_at_k"].get(1, 0),
                    "rows": metrics_all["total_rows"],
                }
            )

        summary = finalize_metric_state(metrics_all, rank_eval_ks)
        category_summary = {
            category: finalize_metric_state(state, rank_eval_ks)
            for category, state in sorted(metrics_by_category.items())
        }
        trial_summaries.append(summary)
        trial_category_summaries.append(category_summary)
        total_forced_truth_count += forced_truth_count
        trial_reports.append(
            {
                "trial_index": int(trial_idx),
                "seed": int(args.seed + trial_idx),
                "forced_truth_into_candidate_pool": int(forced_truth_count),
                "summary": summary,
                "summary_by_category": category_summary,
            }
        )

    summary = average_summary_dicts(trial_summaries)
    category_summary = average_category_summaries(trial_category_summaries)
    posterior_entropy_label = (
        "H_sample-att(X|Z,O,C)"
        if args.candidate_mode == "sampled_same_category"
        else "H_att(X|Z,O)"
    )
    posterior_entropy_display_name = (
        "sampled_posterior_entropy"
        if args.candidate_mode == "sampled_same_category"
        else "attack_induced_posterior_entropy"
    )
    mi_proxy_label = (
        "sampled MI proxy based on log2|C_j| - H_sample-att(X|Z,O,C)"
        if args.candidate_mode == "sampled_same_category"
        else "attack-induced MI proxy based on log2|C_j| - H_att(X|Z,O)"
    )

    report = {
        "attack_name": "replay_mle_whitebox_no_history",
        "config": {
            "pairs_json_path": os.path.abspath(pairs_json_path),
            "base_model_path": os.path.abspath(resolve_base_model_path(None, args.base_model_path)),
            "lora_path": os.path.abspath(args.lora_path) if args.lora_path else None,
            "prompt_mode": args.prompt_mode,
            "prompt_dataset": prompt_dataset,
            "prompt_file": os.path.abspath(args.prompt_file) if args.prompt_mode == "pgpo" else None,
            "output_path": os.path.abspath(output_path),
            "device": device,
            "max_rows": int(args.max_rows),
            "candidate_batch_size": int(args.candidate_batch_size),
            "prompt_batch_size": int(args.prompt_batch_size),
            "max_seq_length": int(args.max_seq_length),
            "load_in_4bit": bool(args.load_in_4bit),
            "loader_backend": args.loader_backend,
            "candidate_mode": args.candidate_mode,
            "num_random_negatives": int(args.num_random_negatives),
            "sampling_trials": int(effective_trials),
            "seed": int(args.seed),
            "posterior_temperature": float(args.posterior_temperature),
            "max_candidates_per_target": int(args.max_candidates_per_target),
            "rank_eval_ks": [int(k) for k in rank_eval_ks],
            "append_closing_answer_tag": bool(args.append_closing_answer_tag),
            "save_details": bool(args.save_details),
        },
        "notes": [
            "No-history replay attack: context_str is kept empty for every prompt when using the PGPO template.",
            "Each target is scored with teacher forcing on log p(target | Prompt(original_word, ...)).",
            "The candidate pool is built from original words in the provided pairs json.",
            "When candidate_mode=sampled_same_category, each target uses truth + same-category random negatives.",
            "Posterior entropy is reported as an attack-induced posterior entropy, not as an exact information-theoretic H(X|Z,O).",
        ],
        "metric_semantics": {
            "posterior_entropy_display_name": posterior_entropy_display_name,
            "posterior_entropy_label": posterior_entropy_label,
            "mi_proxy_label": mi_proxy_label,
        },
        "dataset_stats": {
            "num_rows": int(len(samples)),
            "approx_scored_pairs_per_trial": int(pair_count_estimate_per_trial),
            "approx_scored_pairs_total": int(pair_count_estimate_per_trial * effective_trials),
            "forced_truth_into_candidate_pool_total": int(total_forced_truth_count),
        },
        "summary": summary,
        "summary_by_category": category_summary,
        "trial_summaries": trial_reports,
        "runtime": {
            "model_loader": model_loader_name,
        },
    }
    if args.save_details:
        report["rows_trial_0"] = results

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Replay-MLE report saved to {output_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
