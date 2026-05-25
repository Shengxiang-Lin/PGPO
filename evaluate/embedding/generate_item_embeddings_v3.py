import argparse
import json
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="../../base_models/bert-base-uncased")
    parser.add_argument("--encoder_backend", type=str, default="hf", choices=["hf", "sentence_transformers"])
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--data_root", type=str, default="../../data")
    parser.add_argument("--dataset", type=str, default="all", choices=["ml-1m", "amazon-book", "all"])
    parser.add_argument("--max_terms_per_field", type=int, default=6)
    parser.add_argument("--version_name", type=str, default="v3")
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Path to custom variants json file. If set, it overrides dataset default variants file.",
    )
    return parser.parse_args()


class HFTextEncoder:
    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

    def encode(self, texts, batch_size, max_length, normalize_embeddings):
        vectors = []
        with torch.no_grad():
            for start in tqdm(range(0, len(texts), batch_size)):
                batch_text = texts[start : start + batch_size]
                inputs = self.tokenizer(
                    batch_text,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(**inputs, return_dict=True)
                hidden = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                if normalize_embeddings:
                    pooled = torch.nn.functional.normalize(pooled, dim=-1)
                vectors.append(pooled.cpu().numpy())
        return np.concatenate(vectors, axis=0)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_variants_file(handled_dir, dataset, input_path=""):
    if input_path:
        custom_path = os.path.expanduser(input_path)
        if os.path.exists(custom_path):
            return custom_path
        raise FileNotFoundError(f"--input file not found: {input_path}")

    candidates = []
    if dataset == "ml-1m":
        candidates = [
            os.path.join(handled_dir, "generate", "generated_variants_movielens_5000_3000_2_2_3_1.2.json"),
        ]
    elif dataset == "amazon-book":
        candidates = [
            os.path.join(handled_dir, "generate", "generated_variants_amazon-book_5000_3000_2_3_2_1.2.json"),
        ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Cannot find variants file for {dataset}. Tried: {candidates}")


def normalize_text(value):
    return str(value).strip()


def build_variant_mapping(variants):
    strict_map = {}
    lower_map = {}
    score = {}
    lower_score = {}
    for row in variants:
        original = normalize_text(row.get("original_word", ""))
        category = normalize_text(row.get("category", ""))
        variant = normalize_text(row.get("variant", ""))
        reward = float(row.get("reward", 0.0))
        if not original or not category or not variant:
            continue
        strict_key = (category, original)
        lower_key = (category.lower(), original.lower())
        if strict_key not in strict_map or reward > score[strict_key]:
            strict_map[strict_key] = variant
            score[strict_key] = reward
        if lower_key not in lower_map or reward > lower_score[lower_key]:
            lower_map[lower_key] = variant
            lower_score[lower_key] = reward
    return strict_map, lower_map


def sorted_item_ids(id_item):
    keys = list(id_item.keys())
    if all(str(k).isdigit() for k in keys):
        return sorted(keys, key=lambda x: int(x))
    return sorted(keys)


def build_item_texts_v3(data_root, dataset, max_terms_per_field, input_variants_path=""):
    handled_dir = os.path.join(data_root, dataset, "handled")
    id_item_path = os.path.join(handled_dir, "id_item.json")
    item_edges_path = os.path.join(handled_dir, "cleaned_item_edges.json")
    variants_path = choose_variants_file(handled_dir, dataset, input_variants_path)

    id_item = load_json(id_item_path)
    item_edges = load_json(item_edges_path)
    variants = load_json(variants_path)
    strict_variant_map, lower_variant_map = build_variant_mapping(variants)

    item_ids = sorted_item_ids(id_item)
    v3 = {}
    for item_id in item_ids:
        title = normalize_text(id_item.get(item_id, f"Item {item_id}"))
        edge = item_edges.get(item_id, {})
        categories = [k for k in edge.keys() if k != "item_name"]
        segments_v3 = []
        for category in categories:
            values = edge.get(category, [])
            if not isinstance(values, list):
                continue
            values = [normalize_text(v) for v in values if normalize_text(v)]
            values = values[:max_terms_per_field]
            if not values:
                continue
            replaced_values = []
            for value in values:
                strict_key = (category, value)
                lower_key = (category.lower(), value.lower())
                replaced_values.append(
                    strict_variant_map.get(strict_key, lower_variant_map.get(lower_key, value))
                )
            segments_v3.append(f"{category}: {', '.join(replaced_values)}")
        v3[item_id] = " | ".join(segments_v3) if segments_v3 else title
    return item_ids, v3


def encode_and_save(
    encoder,
    variant_name,
    item_ids,
    item_text_map,
    output_dir,
    batch_size,
    encoder_backend,
    max_length,
):
    texts = [item_text_map[item_id] for item_id in item_ids]
    id_map = {item_id: idx for idx, item_id in enumerate(item_ids)}
    if encoder_backend == "hf":
        embeddings = encoder.encode(
            texts=texts,
            batch_size=batch_size,
            max_length=max_length,
            normalize_embeddings=True,
        )
    else:
        embeddings = encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    variant_dir = os.path.join(output_dir, variant_name)
    os.makedirs(variant_dir, exist_ok=True)
    np.save(os.path.join(variant_dir, "item_embeddings.npy"), embeddings)
    with open(os.path.join(variant_dir, "item_id_map.json"), "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)
    with open(os.path.join(variant_dir, "item_content.json"), "w", encoding="utf-8") as f:
        json.dump(item_text_map, f, ensure_ascii=False, indent=2)
    print(f"{variant_name}: saved {len(texts)} embeddings to {variant_dir}")


def generate_for_dataset(encoder, args, dataset):
    item_ids, v3 = build_item_texts_v3(
        data_root=args.data_root,
        dataset=dataset,
        max_terms_per_field=args.max_terms_per_field,
        input_variants_path=args.input,
    )
    dataset_output_dir = os.path.join(args.output_dir, dataset)
    os.makedirs(dataset_output_dir, exist_ok=True)
    encode_and_save(
        encoder,
        args.version_name,
        item_ids,
        v3,
        dataset_output_dir,
        args.batch_size,
        args.encoder_backend,
        args.max_length,
    )


def main():
    args = parse_args()
    if not args.version_name.strip():
        raise ValueError("--version_name must be a non-empty string")
    os.makedirs(args.output_dir, exist_ok=True)
    if args.encoder_backend == "hf":
        encoder = HFTextEncoder(args.model_name)
    else:
        encoder = SentenceTransformer(args.model_name)
    if args.dataset == "all":
        datasets = ["ml-1m", "amazon-book"]
    else:
        datasets = [args.dataset]
    for dataset in datasets:
        generate_for_dataset(encoder, args, dataset)


if __name__ == "__main__":
    main()
