import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))


def load_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm <= 1e-12:
        return vec
    return vec / norm


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


def split_base_categories(category) -> List[str]:
    return [c.strip() for c in str(category).split(",") if c and c.strip()]


def prepare_valid_samples(data: List[dict]) -> List[dict]:
    valid = []
    for item in data:
        original_word = item.get("original_word")
        variant = item.get("variant")
        category = item.get("category")
        if not original_word or not variant or not category:
            continue
        cats = split_base_categories(category)
        if not cats:
            continue
        valid.append({
            "word_id": item.get("word_id"),
            "original_word": str(original_word),
            "variant": str(variant),
            "categories": cats,
        })
    return valid

def compute_distinct_1(texts: List[str]) -> Dict[str, float]:
    all_unigrams = []
    for text in texts:
        all_unigrams.extend(simple_tokenize(text))
    distinct_1 = len(set(all_unigrams)) / len(all_unigrams) if all_unigrams else 0.0
    return {"distinct_1": distinct_1, "total_unigrams": len(all_unigrams)}

def compute_variant_original_similarity(
    valid_samples: List[dict],
    embedding_model_path: str,
) -> Tuple[List[float], Dict]:
    print(f"Loading embedding model from: {embedding_model_path}")
    model = SentenceTransformer(embedding_model_path)

    variants = [s["variant"] for s in valid_samples]
    originals = [s["original_word"] for s in valid_samples]

    all_texts = sorted(set(variants + originals))
    embeddings = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=True)
    emb_cache = {t: safe_l2_normalize(e.astype(np.float64)) for t, e in zip(all_texts, embeddings)}

    similarities = []
    for var, orig in zip(variants, originals):
        sim = float(np.dot(emb_cache[var], emb_cache[orig]))
        similarities.append(sim)

    stats = {"num_pairs": len(similarities), "unique_texts": len(all_texts)}
    return similarities, stats

def compute_length_ratios(valid_samples: List[dict]) -> List[float]:
    ratios = []
    for s in valid_samples:
        orig_len = len(s["original_word"])
        var_len = len(s["variant"])
        ratio = var_len / orig_len if orig_len > 0 else 1.0
        ratios.append(ratio)
    return ratios

def evaluate(args):
    data = load_json(args.generated_file)

    if args.only_original:
        original_texts = [item["original_word"] for item in data if "original_word" in item]
        if not original_texts:
            raise ValueError("No 'original_word' found in the input file.")

        print(f"Only evaluating original_word: {len(original_texts)} samples")

        d1_report = compute_distinct_1(original_texts)

        report = {
            "num_samples": len(original_texts),
            "mode": "original_only",
            "metrics": {
                "distinct_1": d1_report["distinct_1"],
            },
            "details": {
                "distinct_stats": d1_report,
            },
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        return

    valid_samples = prepare_valid_samples(data)
    print(f"Evaluating variant: {len(valid_samples)} valid samples")
    if not valid_samples:
        raise ValueError("No valid samples found with 'original_word', 'variant', and 'category'.")

    sim_scores, sim_stats = compute_variant_original_similarity(
        valid_samples, args.embedding_model_path
    )

    var_texts = [s["variant"] for s in valid_samples]
    d1_report = compute_distinct_1(var_texts)

    len_ratios = compute_length_ratios(valid_samples)

    report = {
        "num_samples": len(valid_samples),
        "mode": "variant_evaluation",
        "metrics": {
            "variant_original_sim_mean": float(np.mean(sim_scores)) if sim_scores else None,
            "distinct_1": d1_report["distinct_1"],
            "length_ratio_mean": float(np.mean(len_ratios)) if len_ratios else None,
        },
        "details": {
            "sim_stats": sim_stats,
            "distinct_stats": d1_report,
        },
        "config": {
            "generated_file": args.generated_file,
            "embedding_model_path": args.embedding_model_path,
        },
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated word quality")
    parser.add_argument("--generated_file", required=True, help="JSON array with original_word/variant/category")
    parser.add_argument(
        "--embedding_model_path",
        default=os.path.join(PROJECT_ROOT, "base_models/bert-base-uncased"),
        help="Path to sentence embedding model (not used in --only_original mode)"
    )
    parser.add_argument("--output_file", default=None, help="Optional output JSON report")
    parser.add_argument(
        "--only_original",
        action="store_true",
        help="Only evaluate original_word's Distinct-1 (no variants)"
    )
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
