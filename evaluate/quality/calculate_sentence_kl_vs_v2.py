import argparse
import json
import math
import os
import statistics
import gc

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

KL_CLIP_VALUE = 1e9


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "variant_names_positional",
        nargs="*",
        default=[],
        help="Variant directory names under dataset root",
    )
    parser.add_argument("--data_root", type=str, default="../embedding/data")
    parser.add_argument("--dataset", type=str, default="ml-1m")
    parser.add_argument("--baseline_name", type=str, default="v2")
    parser.add_argument(
        "--variant_names",
        nargs="*",
        default=[],
        help="Variant directory names under dataset root",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="../../base_models/Qwen3-4B-Base",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_ids", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--save_texts", action="store_true")
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


def percentile(values, p):
    if not values:
        return None
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    idx = (n - 1) * p
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_vals[lo])
    w = idx - lo
    return float(sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w)


def load_item_content(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object at {json_path}, got {type(data)}")
    out = {}
    for k, v in data.items():
        key = str(k).strip()
        text = str(v).strip() if v is not None else ""
        if key and text:
            out[key] = text
    return out


def tokenize_texts(tokenizer, texts, max_length, device):
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in encoded.items()}


def compute_batch_kl(model, tokenizer, originals, variants, max_length, device):
    def forward(inputs):
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        attn = inputs["attention_mask"]
        logits = logits[:, :-1, :]
        attn = attn[:, 1:]
        return logits, attn

    ori_inputs = tokenize_texts(tokenizer, originals, max_length, device)
    var_inputs = tokenize_texts(tokenizer, variants, max_length, device)
    
    ori_logits, ori_attn = forward(ori_inputs)
    var_logits, var_attn = forward(var_inputs)

    min_seq_len = min(ori_logits.size(1), var_logits.size(1))
    ori_logits = ori_logits[:, :min_seq_len, :]
    var_logits = var_logits[:, :min_seq_len, :]
    ori_attn = ori_attn[:, :min_seq_len]
    var_attn = var_attn[:, :min_seq_len]

    ori_log_probs = torch.log_softmax(ori_logits.float(), dim=-1)
    var_log_probs = torch.log_softmax(var_logits.float(), dim=-1)

    kl_token = F.kl_div(
        var_log_probs,
        ori_log_probs,
        reduction="none",
        log_target=True,
    ).sum(dim=-1)

    joint_attn = (ori_attn & var_attn).float()
    batch_kl = []
    for i in range(kl_token.size(0)):
        mask = joint_attn[i] == 1
        if mask.sum().item() <= 0:
            batch_kl.append(None)
            continue
        batch_kl.append(kl_token[i][mask].mean().item())
    return batch_kl


def default_output_path(data_root, variant_name):
    return os.path.join(data_root, variant_name, "kl_vs_v2_report.json")


def resolve_dataset_root(data_root, dataset, baseline_name):
    data_root = str(data_root).strip()
    dataset = str(dataset).strip()
    if not data_root:
        data_root = "./data"
    if not dataset:
        dataset = "ml-1m"

    candidate = data_root
    baseline_file = os.path.join(candidate, baseline_name, "item_content.json")
    if os.path.exists(baseline_file):
        return candidate
    return os.path.join(data_root, dataset)


def load_model(base_model_path):
    """简洁稳定版：仅使用 8-bit 量化"""
    print("Loading model with 8-bit quantization...")

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=torch.float16,          # 推荐写法（避免 deprecated warning）
    )

    model.eval()
    print("Model loaded successfully (8-bit).")
    return model


def compute_variant_report(
    model,
    tokenizer,
    data_root,
    baseline_name,
    variant_name,
    batch_size,
    max_length,
    device,
    max_ids,
    save_texts,
):
    baseline_path = os.path.join(data_root, baseline_name, "item_content.json")
    variant_path = os.path.join(data_root, variant_name, "item_content.json")
    
    baseline = load_item_content(baseline_path)
    variant = load_item_content(variant_path)

    def sort_key(item_id: str):
        if item_id.isdigit():
            return 0, int(item_id)
        return 1, item_id

    common_ids = sorted(set(baseline.keys()) & set(variant.keys()), key=sort_key)
    if max_ids > 0:
        common_ids = common_ids[:max_ids]

    results = []
    valid_kls = []
    pbar = tqdm(total=len(common_ids), desc=f"KL vs v2 ({variant_name})", unit="id")

    for start in range(0, len(common_ids), batch_size):
        batch_ids = common_ids[start : start + batch_size]
        originals = [baseline[i] for i in batch_ids]
        variants = [variant[i] for i in batch_ids]

        # 轻量内存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        batch_kl = compute_batch_kl(
            model=model,
            tokenizer=tokenizer,
            originals=originals,
            variants=variants,
            max_length=max_length,
            device=device,
        )

        for item_id, kl_value, base_text, var_text in zip(batch_ids, batch_kl, originals, variants):
            normalized_kl = kl_value
            if isinstance(kl_value, float) and (math.isnan(kl_value) or math.isinf(kl_value)):
                normalized_kl = KL_CLIP_VALUE
            
            row = {"id": item_id, "kl_divergence": normalized_kl}
            if save_texts:
                row["v2_text"] = base_text
                row["variant_text"] = var_text
            results.append(row)
            if normalized_kl is not None:
                valid_kls.append(normalized_kl)

        pbar.update(len(batch_ids))

    pbar.close()

    mean_kl = float(sum(valid_kls) / len(valid_kls)) if valid_kls else None
    median_kl = float(statistics.median(valid_kls)) if valid_kls else None
    p10_kl = percentile(valid_kls, 0.10)
    p90_kl = percentile(valid_kls, 0.90)

    report = {
        "data_root": data_root,
        "baseline_name": baseline_name,
        "variant_name": variant_name,
        "baseline_item_content_path": baseline_path,
        "variant_item_content_path": variant_path,
        "total_common_ids": len(common_ids),
        "valid_ids": len(valid_kls),
        "mean_kl_divergence": mean_kl,
        "median_kl_divergence": median_kl,
        "p10_kl_divergence": p10_kl,
        "p90_kl_divergence": p90_kl,
        "max_length": max_length,
        "results": results,
    }
    return report


def main():
    args = parse_args()
    variant_names = list(args.variant_names_positional) + list(args.variant_names)
    variant_names = [v.strip().strip("/").strip() for v in variant_names if str(v).strip()]
    
    if not variant_names:
        raise SystemExit("no variant_names provided")

    dataset_root = resolve_dataset_root(args.data_root, args.dataset, args.baseline_name)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.base_model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for idx, variant_name in enumerate(variant_names):
        report = compute_variant_report(
            model=model,
            tokenizer=tokenizer,
            data_root=dataset_root,
            baseline_name=args.baseline_name,
            variant_name=variant_name,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
            max_ids=args.max_ids,
            save_texts=args.save_texts,
        )

        if args.output_path and len(variant_names) == 1:
            output_path = args.output_path
        else:
            output_path = default_output_path(dataset_root, variant_name)
            if args.output_path and len(variant_names) > 1:
                base, ext = os.path.splitext(args.output_path)
                ext = ext or ".json"
                output_path = f"{base}.{variant_name}{ext}"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(
            f"[{idx + 1}/{len(variant_names)}] {variant_name}: "
            f"mean={report['mean_kl_divergence']:.4f} | "
            f"median={report['median_kl_divergence']:.4f} | "
            f"p10={report['p10_kl_divergence']:.4f} | "
            f"p90={report['p90_kl_divergence']:.4f} | "
            f"valid={report['valid_ids']}/{report['total_common_ids']} "
            f"→ {output_path}"
        )


if __name__ == "__main__":
    main()
