import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="amazon-book", choices=["ml-1m", "amazon-book"])
    parser.add_argument("--data_root", type=str, default="../../../data")
    parser.add_argument("--embedding_root", type=str, default="../../embedding/data")
    parser.add_argument("--embedding_variant", type=str, default="v2", choices=["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"])
    parser.add_argument("--split_reference_variant", type=str, default="v1", choices=["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"])
    parser.add_argument("--output_root", type=str, default="./data")
    parser.add_argument("--max_seq_length", type=int, default=20)
    parser.add_argument("--min_seq_length", type=int, default=10)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_interactions", type=int, default=30000)
    parser.add_argument("--min_rating", type=float, default=4.0)
    return parser.parse_args()


def load_embedding_item_map(embedding_root, dataset, embedding_variant):
    map_path = os.path.join(embedding_root, dataset, embedding_variant, "item_id_map.json")
    emb_path = os.path.join(embedding_root, dataset, embedding_variant, "item_embeddings.npy")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"item_id_map not found: {map_path}")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"item_embeddings not found: {emb_path}")
    with open(map_path, "r", encoding="utf-8") as f:
        raw_map = json.load(f)
    item_to_idx = {str(k): int(v) for k, v in raw_map.items()}
    return item_to_idx, map_path, emb_path


def build_v0_item_map(interactions):
    item_ids = sorted({str(item_id) for _, item_id, _, _ in interactions})
    all_numeric = all(item_id.isdigit() for item_id in item_ids)
    if all_numeric:
        return {item_id: int(item_id) for item_id in item_ids}
    return {item_id: idx for idx, item_id in enumerate(item_ids)}


def read_ml1m_interactions(raw_dir, min_rating):
    ratings_path = os.path.join(raw_dir, "ratings.dat")
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"ratings file not found: {ratings_path}")
    interactions = []
    with open(ratings_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) != 4:
                continue
            user_id, item_id, rating, timestamp = parts
            rating = float(rating)
            if rating < min_rating:
                continue
            interactions.append((str(user_id), str(item_id), rating, int(float(timestamp))))
    return interactions


def read_amazon_book_interactions(raw_dir, min_rating):
    ratings_path = os.path.join(raw_dir, "Books.csv")
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"ratings file not found: {ratings_path}")
    interactions = []
    with open(ratings_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            item_id, user_id, rating, timestamp = row[:4]
            rating = float(rating)
            if rating < min_rating:
                continue
            interactions.append((str(user_id), str(item_id), rating, int(float(timestamp))))
    return interactions


def read_interactions(dataset, data_root, min_rating):
    raw_dir = os.path.join(data_root, dataset, "raw-0")
    if dataset == "ml-1m":
        return read_ml1m_interactions(raw_dir, min_rating)
    return read_amazon_book_interactions(raw_dir, min_rating)


def build_reference_user_sequences(interactions, reference_item_map, max_interactions):
    filtered = []
    for user_id, item_id, rating, timestamp in interactions:
        if item_id not in reference_item_map:
            continue
        filtered.append((user_id, item_id, rating, timestamp))
    if max_interactions is not None and max_interactions > 0:
        filtered.sort(key=lambda x: x[3])
        filtered = filtered[-max_interactions:]
    user_items = defaultdict(list)
    for user_id, item_id, rating, timestamp in filtered:
        user_items[user_id].append((item_id, rating, timestamp))
    for user_id in user_items:
        user_items[user_id].sort(key=lambda x: x[2])
    return user_items


def generate_raw_sequences(user_items, max_seq_length, min_seq_length):
    sequences = []
    for user_id in sorted(user_items.keys()):
        interactions = user_items[user_id]
        if len(interactions) <= min_seq_length:
            continue
        for target_idx in range(min_seq_length, len(interactions)):
            history = interactions[:target_idx]
            if len(history) < min_seq_length:
                continue
            target_item_id, target_rating, _ = interactions[target_idx]
            history = history[-max_seq_length:]
            seq_item_ids = [str(item_id) for item_id, _, _ in history]
            seq_ratings = [float(rating) for _, rating, _ in history]
            len_seq = len(seq_item_ids)
            sequences.append(
                {
                    "seq_item_ids": seq_item_ids,
                    "seq_ratings": seq_ratings,
                    "next_item_id": str(target_item_id),
                    "next_rating": float(target_rating),
                    "len_seq": int(len_seq),
                }
            )
    return sequences


def build_split_index_path(args):
    split_dir = os.path.join(args.output_root, args.dataset, "_split_index")
    os.makedirs(split_dir, exist_ok=True)
    split_name = (
        f"ref_{args.split_reference_variant}_"
        f"minr{args.min_rating}_"
        f"maxseq{args.max_seq_length}_"
        f"minseq{args.min_seq_length}_"
        f"train{args.train_ratio}_"
        f"val{args.val_ratio}_"
        f"maxint{args.max_interactions}_"
        f"seed{args.seed}.json"
    )
    split_name = split_name.replace("/", "_")
    return os.path.join(split_dir, split_name)


def load_or_create_split_indices(num_sequences, train_ratio, val_ratio, seed, split_index_path):
    if os.path.exists(split_index_path):
        with open(split_index_path, "r", encoding="utf-8") as f:
            split_obj = json.load(f)
        indices = split_obj.get("indices", [])
        if len(indices) != num_sequences:
            raise ValueError(
                f"split index length mismatch for {split_index_path}: "
                f"expected {num_sequences}, got {len(indices)}"
            )
        return [int(i) for i in indices]
    rng = np.random.default_rng(seed)
    indices = np.arange(num_sequences)
    rng.shuffle(indices)
    split_obj = {
        "num_sequences": int(num_sequences),
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "indices": [int(i) for i in indices.tolist()],
    }
    with open(split_index_path, "w", encoding="utf-8") as f:
        json.dump(split_obj, f, ensure_ascii=False, indent=2)
    return split_obj["indices"]


def map_raw_sequences_to_variant(raw_sequences, item_to_idx, max_seq_length, pad_item_id):
    mapped = []
    for row in raw_sequences:
        seq_item_ids = row["seq_item_ids"]
        seq_ratings = row["seq_ratings"]
        next_item_id = row["next_item_id"]
        next_rating = row["next_rating"]
        if next_item_id not in item_to_idx:
            raise KeyError(f"target item id is not in current variant map: {next_item_id}")
        seq = []
        for item_id, rating in zip(seq_item_ids, seq_ratings):
            if item_id not in item_to_idx:
                raise KeyError(f"history item id is not in current variant map: {item_id}")
            seq.append((int(item_to_idx[item_id]), float(rating)))
        len_seq = int(row["len_seq"])
        if len_seq < max_seq_length:
            seq = seq + [(int(pad_item_id), 0.0)] * (max_seq_length - len_seq)
        mapped.append(
            {
                "seq": seq,
                "next": (int(item_to_idx[next_item_id]), float(next_rating)),
                "len_seq": len_seq,
            }
        )
    return mapped


def split_and_save(mapped_sequences, split_indices, output_dir, train_ratio, val_ratio):
    os.makedirs(output_dir, exist_ok=True)
    shuffled = [mapped_sequences[i] for i in split_indices]
    total = len(shuffled)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    train_data = shuffled[:train_size]
    val_data = shuffled[train_size: train_size + val_size]
    test_data = shuffled[train_size + val_size:]
    pd.DataFrame(train_data).to_pickle(os.path.join(output_dir, "train_data.df"))
    pd.DataFrame(val_data).to_pickle(os.path.join(output_dir, "Val_data.df"))
    pd.DataFrame(test_data).to_pickle(os.path.join(output_dir, "Test_data.df"))
    return len(train_data), len(val_data), len(test_data)


def main():
    args = parse_args()
    interactions = read_interactions(
        dataset=args.dataset,
        data_root=args.data_root,
        min_rating=args.min_rating,
    )
    reference_item_to_idx, reference_map_path, _ = load_embedding_item_map(
        embedding_root=args.embedding_root,
        dataset=args.dataset,
        embedding_variant=args.split_reference_variant,
    )
    item_to_idx, map_path, emb_path = load_embedding_item_map(
        embedding_root=args.embedding_root,
        dataset=args.dataset,
        embedding_variant=args.embedding_variant,
    )
    user_items = build_reference_user_sequences(
        interactions=interactions,
        reference_item_map=reference_item_to_idx,
        max_interactions=args.max_interactions,
    )
    num_items = max(item_to_idx.values()) + 1
    pad_item_id = num_items
    raw_sequences = generate_raw_sequences(
        user_items=user_items,
        max_seq_length=args.max_seq_length,
        min_seq_length=args.min_seq_length,
    )
    split_index_path = build_split_index_path(args)
    split_indices = load_or_create_split_indices(
        num_sequences=len(raw_sequences),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        split_index_path=split_index_path,
    )
    mapped_sequences = map_raw_sequences_to_variant(
        raw_sequences=raw_sequences,
        item_to_idx=item_to_idx,
        max_seq_length=args.max_seq_length,
        pad_item_id=pad_item_id,
    )
    output_dir = os.path.join(args.output_root, args.dataset, args.embedding_variant)
    train_size, val_size, test_size = split_and_save(
        mapped_sequences=mapped_sequences,
        split_indices=split_indices,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    meta = {
        "dataset": args.dataset,
        "embedding_variant": args.embedding_variant,
        "split_reference_variant": args.split_reference_variant,
        "padding_item_id": pad_item_id,
        "num_items": num_items,
        "reference_embedding_item_id_map_path": reference_map_path,
        "embedding_item_id_map_path": map_path,
        "embedding_file_path": emb_path,
        "split_index_path": split_index_path,
        "max_seq_length": args.max_seq_length,
        "min_seq_length": args.min_seq_length,
        "min_rating": args.min_rating,
        "sizes": {"train": train_size, "val": val_size, "test": test_size},
    }
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
