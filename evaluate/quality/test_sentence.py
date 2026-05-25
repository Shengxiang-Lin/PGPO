import argparse
import json
import math
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))


def load_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


def phrase_word_tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text.lower())


def safe_l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm <= 1e-12:
        return vec
    return vec / norm


def compute_distinct_metrics(texts: List[str]) -> Dict[str, float]:
    EXCLUDE_TOKENS = {
        "actors", "plotandtheme", "production", "quality", "soundtrack",
        #"actors", "publicationcontext", "toneandstyle", "plotandtheme",
        ":", "|"
    }
    unigrams = []
    bigrams = []
    for text in texts:
        tokens = simple_tokenize(text)
        tokens = [t for t in tokens if t not in EXCLUDE_TOKENS]
        unigrams.extend(tokens)
        bigrams.extend(list(zip(tokens[:-1], tokens[1:])))

    distinct_1 = len(set(unigrams)) / len(unigrams) if unigrams else 0.0
    distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0
    return {
        "distinct_1": float(distinct_1),
        "distinct_2": float(distinct_2),
        "total_unigrams": len(unigrams),
        "total_bigrams": len(bigrams),
    }


def compute_ppl_metrics(
    texts: List[str],
    lm_model_path: str,
    max_length: int = 128,
) -> Tuple[List[dict], Dict[str, int]]:
    print(f"Loading LM from: {lm_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(lm_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        lm_model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    ppl_records = []
    skipped_short = 0
    with torch.no_grad():
        for text in tqdm(texts, desc="PPL", unit="item"):
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)
            seq_len = int(input_ids.shape[1])
            token_count = max(seq_len - 1, 0) 
            if token_count <= 0:
                skipped_short += 1
                continue
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            mean_nll = float(outputs.loss.item())
            ppl = math.exp(mean_nll)
            ppl_div_token = ppl / token_count
            ppl_records.append({
                "token_count": token_count,
                "mean_nll": mean_nll,
                "ppl": ppl,
                "normalized_ppl": ppl_div_token,
            })
    stats = {"evaluated_count": len(ppl_records), "skipped_short_text": skipped_short}
    return ppl_records, stats


def compute_word_vector_coherence(
    texts: List[str],
    embedding_model_path: str,
) -> Tuple[List[float], Dict[str, int]]:
    print(f"Loading embedding model from: {embedding_model_path}")
    emb_model = SentenceTransformer(embedding_model_path)

    tokenized_texts = [phrase_word_tokenize(t) for t in texts]
    unique_tokens = sorted({tok for toks in tokenized_texts for tok in toks})

    token_emb_cache: Dict[str, np.ndarray] = {}
    if unique_tokens:
        token_embeddings = emb_model.encode(unique_tokens, convert_to_numpy=True, show_progress_bar=True)
        for token, emb in zip(unique_tokens, token_embeddings):
            token_emb_cache[token] = safe_l2_normalize(emb.astype(np.float64))

    coherence_scores = []
    skipped_empty = 0
    for tokens in tqdm(tokenized_texts, desc="Word vector coherence", unit="item"):
        if not tokens:
            skipped_empty += 1
            continue
        vecs = [token_emb_cache[t] for t in tokens if t in token_emb_cache]
        if not vecs:
            skipped_empty += 1
            continue
        if len(vecs) == 1:
            coherence_scores.append(1.0)
            continue
        mean_vec = safe_l2_normalize(np.mean(np.stack(vecs, axis=0), axis=0))
        sims = [float(np.dot(v, mean_vec)) for v in vecs]
        coherence_scores.append(float(np.mean(sims)))
    stats = {
        "evaluated_count": len(coherence_scores),
        "skipped_empty_or_oov": skipped_empty,
        "unique_token_count": len(unique_tokens),
    }
    return coherence_scores, stats


def evaluate(args):
    print("Loading input data...")
    data = load_json(args.input_file)

    if isinstance(data, dict):
        texts = list(data.values())
    elif isinstance(data, list):
        texts = data
    else:
        raise ValueError("Input JSON must be a dict or list of strings.")

    print(f"Loaded {len(texts)} texts.")
    if len(texts) == 0:
        raise ValueError("No texts found.")

    distinct_report = compute_distinct_metrics(texts)

    ppl_records, ppl_stats = compute_ppl_metrics(
        texts, args.lm_model_path, args.max_length
    )

    coherence_scores, coherence_stats = compute_word_vector_coherence(
        texts, args.embedding_model_path
    )

    ppl_values = [r["ppl"] for r in ppl_records]
    norm_ppl_values = [r["normalized_ppl"] for r in ppl_records]

    report = {
        "num_texts": len(texts),
        "metrics": {
            "ppl_mean": float(np.mean(ppl_values)) if ppl_values else None,
            "ppl_median": float(np.median(ppl_values)) if ppl_values else None,
            "normalized_ppl_mean": float(np.mean(norm_ppl_values)) if norm_ppl_values else None,
            "normalized_ppl_median": float(np.median(norm_ppl_values)) if norm_ppl_values else None,
            "distinct_1": distinct_report["distinct_1"],
            "distinct_2": distinct_report["distinct_2"],
            "word_vector_coherence_mean": float(np.mean(coherence_scores)) if coherence_scores else None,
        },
        "details": {
            "distinct": distinct_report,
            "ppl_stats": ppl_stats,
            "coherence_stats": coherence_stats,
        },
        "config": {
            "input_file": args.input_file,
            "lm_model_path": args.lm_model_path,
            "embedding_model_path": args.embedding_model_path,
            "max_length": args.max_length,
        },
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated sentences with automatic metrics.")
    parser.add_argument("--input_file", required=True, help="Path to JSON file containing texts (dict or list).")
    parser.add_argument("--output_file", default=None, help="Path to output JSON report.")
    parser.add_argument(
        "--lm_model_path",
        default=os.path.join(PROJECT_ROOT, "base_models/Qwen2.5-3B-Instruct"),
        help="Path to causal LM for PPL scoring.",
    )
    parser.add_argument(
        "--embedding_model_path",
        default=os.path.join(PROJECT_ROOT, "base_models/bert-base-uncased"),
        help="Path to sentence embedding model for coherence.",
    )
    parser.add_argument("--max_length", type=int, default=1024, help="Max tokens for LM evaluation.")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()