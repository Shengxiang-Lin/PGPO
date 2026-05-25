import argparse
import json
import os

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="amazon-book", choices=["ml-1m", "amazon-book"])
    parser.add_argument("--data_root", type=str, default="../../../data")
    parser.add_argument("--output_dir", type=str, default="./data/recbole")
    parser.add_argument("--min_rating", type=float, default=4.0)
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument(
        "--ml1m_sparsify_keep_prob",
        type=float,
        default=1.0,
        help="Only for ml-1m. Keep probability for non-anchor interactions before splitting.",
    )
    parser.add_argument(
        "--ml1m_sparsify_seed",
        type=int,
        default=2026,
        help="Only for ml-1m. Random seed for pre-split sparsification.",
    )
    return parser.parse_args()


def resolve_dataset_name(dataset):
    if dataset == "ml-1m":
        return "ml1m_recbole"
    return "amazon_book_recbole"


def load_interactions(dataset, data_root):
    raw_dir = os.path.join(data_root, dataset, "raw-0")
    if dataset == "ml-1m":
        ratings_path = os.path.join(raw_dir, "ratings.dat")
        if not os.path.exists(ratings_path):
            raise FileNotFoundError(f"ratings file not found: {ratings_path}")
        df = pd.read_csv(
            ratings_path,
            sep="::",
            header=None,
            names=["user_id", "item_id", "rating", "timestamp"],
            engine="python",
        )
        return df
    ratings_path = os.path.join(raw_dir, "Books.csv")
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"ratings file not found: {ratings_path}")
    df = pd.read_csv(
        ratings_path,
        header=None,
        names=["item_id", "user_id", "rating", "timestamp"],
    )
    return df[["user_id", "item_id", "rating", "timestamp"]]


def build_seq_row(row, history_items, max_seq_len):
    return {
        "user_id": row["user_id"],
        "item_id": row["item_id"],
        "item_id_list": list(history_items[-max_seq_len:]),
        "rating": row["rating"],
        "timestamp": row["timestamp"],
    }


def sort_rows(rows):
    rows.sort(key=lambda row: (row["user_id"], row["timestamp"], row["item_id"]))


def progress(iterable, total=None, desc=""):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def random_sparsify_ml1m(df, keep_prob, seed):
    if not 0 < keep_prob <= 1:
        raise ValueError(f"ml1m_sparsify_keep_prob must be in (0, 1], got {keep_prob}")
    stats = {
        "applied": False,
        "keep_prob": float(keep_prob),
        "seed": int(seed),
        "rows_before": int(len(df)),
        "rows_after": int(len(df)),
        "num_item_anchors": 0,
    }
    if keep_prob >= 1.0:
        return df, stats

    rng = np.random.default_rng(seed)
    keep_mask = np.zeros(len(df), dtype=bool)
    item_indices = df.groupby("item_id", sort=False).indices

    for indices in progress(item_indices.values(), total=len(item_indices), desc="Sparsify items"):
        indices = np.asarray(indices)
        anchor_idx = indices[rng.integers(len(indices))]
        keep_mask[anchor_idx] = True

    remaining = np.flatnonzero(~keep_mask)
    if len(remaining) > 0:
        keep_mask[remaining] = rng.random(len(remaining)) < keep_prob

    sparse_df = df.loc[keep_mask].copy()
    sparse_df = sparse_df.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    stats.update(
        {
            "applied": True,
            "rows_after": int(len(sparse_df)),
            "num_item_anchors": int(len(item_indices)),
        }
    )
    return sparse_df, stats


def split_interactions_temporal(df):
    train_rows, val_rows, test_rows = [], [], []
    stats = {
        "num_train_only_users": 0,
        "num_split_users": 0,
    }
    grouped_users = df.groupby("user_id", sort=True)
    num_users = df["user_id"].nunique()

    for _, user_df in progress(grouped_users, total=num_users, desc="Split users"):
        user_df = user_df.sort_values(["timestamp", "item_id"]).reset_index(drop=True)
        user_rows = user_df[["user_id", "item_id", "rating", "timestamp"]].to_dict("records")
        num_interactions = len(user_rows)

        # Keep sparse users in train only to avoid cold-start evaluation users.
        if num_interactions < 3:
            train_rows.extend(user_rows)
            stats["num_train_only_users"] += 1
            continue

        val_n = max(1, int(num_interactions * 0.1))
        test_n = max(1, int(num_interactions * 0.1))
        train_n = num_interactions - val_n - test_n
        train_user_rows = user_rows[:train_n]
        val_user_rows = user_rows[train_n : train_n + val_n]
        test_user_rows = user_rows[train_n + val_n :]

        train_rows.extend(train_user_rows)
        val_rows.extend(val_user_rows)
        test_rows.extend(test_user_rows)
        stats["num_split_users"] += 1

    return train_rows, val_rows, test_rows, stats


def ensure_train_item_coverage(train_rows, val_rows, test_rows):
    train_rows = list(train_rows)
    val_rows = list(val_rows)
    test_rows = list(test_rows)
    train_items = {row["item_id"] for row in train_rows}
    all_items = train_items | {row["item_id"] for row in val_rows} | {row["item_id"] for row in test_rows}
    missing_items = all_items - train_items
    promoted = 0

    for item_id in sorted(missing_items):
        candidates = [
            ("valid", row) for row in val_rows if row["item_id"] == item_id
        ] + [
            ("test", row) for row in test_rows if row["item_id"] == item_id
        ]
        if not candidates:
            continue
        source_name, chosen_row = min(
            candidates,
            key=lambda pair: (pair[1]["timestamp"], pair[1]["user_id"], pair[1]["item_id"]),
        )
        if source_name == "valid":
            val_rows.remove(chosen_row)
        else:
            test_rows.remove(chosen_row)
        train_rows.append(chosen_row)
        promoted += 1

    sort_rows(train_rows)
    sort_rows(val_rows)
    sort_rows(test_rows)
    return train_rows, val_rows, test_rows, promoted


def build_seq_splits(train_rows, val_rows, test_rows, max_seq_len):
    train_seq_rows, val_seq_rows, test_seq_rows = [], [], []
    per_user_rows = {}

    for split_name, rows in [("train", train_rows), ("valid", val_rows), ("test", test_rows)]:
        for row in rows:
            per_user_rows.setdefault(row["user_id"], []).append((row["timestamp"], row["item_id"], split_name, row))

    sorted_user_ids = sorted(per_user_rows)
    for user_id in progress(sorted_user_ids, total=len(sorted_user_ids), desc="Build sequences"):
        history_items = []
        entries = sorted(per_user_rows[user_id], key=lambda entry: (entry[0], entry[1]))
        for _, _, split_name, row in entries:
            if history_items:
                seq_row = build_seq_row(row, history_items, max_seq_len)
                if split_name == "train":
                    train_seq_rows.append(seq_row)
                elif split_name == "valid":
                    val_seq_rows.append(seq_row)
                else:
                    test_seq_rows.append(seq_row)
            history_items.append(row["item_id"])

    return train_seq_rows, val_seq_rows, test_seq_rows


def write_inter_file(file_path, rows):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        for row in rows:
            f.write(
                f"{row['user_id']}\t{row['item_id']}\t"
                f"{float(row['rating']):.1f}\t{float(row['timestamp']):.1f}\n"
            )


def write_seq_inter_file(file_path, rows):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("user_id:token\titem_id:token\titem_id_list:token_seq\trating:float\ttimestamp:float\n")
        for row in rows:
            item_id_list = " ".join(row["item_id_list"])
            f.write(
                f"{row['user_id']}\t{row['item_id']}\t{item_id_list}\t"
                f"{float(row['rating']):.1f}\t{float(row['timestamp']):.1f}\n"
            )


def count_users(rows):
    return len({row["user_id"] for row in rows})


def save_outputs(
    dataset_name,
    dataset_dir,
    full_rows,
    train_rows,
    val_rows,
    test_rows,
    train_seq_rows,
    val_seq_rows,
    test_seq_rows,
    split_stats,
    sparsify_stats,
):
    full_path = os.path.join(dataset_dir, f"{dataset_name}.inter")
    train_path = os.path.join(dataset_dir, f"{dataset_name}.train.inter")
    valid_path = os.path.join(dataset_dir, f"{dataset_name}.valid.inter")
    test_path = os.path.join(dataset_dir, f"{dataset_name}.test.inter")
    train_seq_path = os.path.join(dataset_dir, f"{dataset_name}.train_seq.inter")
    valid_seq_path = os.path.join(dataset_dir, f"{dataset_name}.valid_seq.inter")
    test_seq_path = os.path.join(dataset_dir, f"{dataset_name}.test_seq.inter")

    write_inter_file(full_path, full_rows)
    write_inter_file(train_path, train_rows)
    write_inter_file(valid_path, val_rows)
    write_inter_file(test_path, test_rows)
    write_seq_inter_file(train_seq_path, train_seq_rows)
    write_seq_inter_file(valid_seq_path, val_seq_rows)
    write_seq_inter_file(test_seq_path, test_seq_rows)

    meta = {
        "split_mode": "per_user_temporal_8_1_1",
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "max_seq_len": int(max(len(row["item_id_list"]) for row in train_seq_rows + val_seq_rows + test_seq_rows) if (train_seq_rows or val_seq_rows or test_seq_rows) else 0),
        "num_rows": int(len(full_rows)),
        "num_train_rows": int(len(train_rows)),
        "num_valid_rows": int(len(val_rows)),
        "num_test_rows": int(len(test_rows)),
        "num_train_seq_rows": int(len(train_seq_rows)),
        "num_valid_seq_rows": int(len(val_seq_rows)),
        "num_test_seq_rows": int(len(test_seq_rows)),
        "num_users": int(count_users(full_rows)),
        "num_items": int(len({row["item_id"] for row in full_rows})),
        "num_users_in_valid": int(count_users(val_rows)),
        "num_users_in_test": int(count_users(test_rows)),
        "num_train_only_users": int(split_stats["num_train_only_users"]),
        "num_temporally_split_users": int(split_stats["num_split_users"]),
        "num_items_promoted_to_train": int(split_stats["num_items_promoted_to_train"]),
        "sparsify_applied": bool(sparsify_stats["applied"]),
        "sparsify_keep_prob": float(sparsify_stats["keep_prob"]),
        "sparsify_seed": int(sparsify_stats["seed"]),
        "sparsify_rows_before": int(sparsify_stats["rows_before"]),
        "sparsify_rows_after": int(sparsify_stats["rows_after"]),
        "sparsify_num_item_anchors": int(sparsify_stats["num_item_anchors"]),
    }
    meta_path = os.path.join(dataset_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "full_path": full_path,
        "train_path": train_path,
        "valid_path": valid_path,
        "test_path": test_path,
        "train_seq_path": train_seq_path,
        "valid_seq_path": valid_seq_path,
        "test_seq_path": test_seq_path,
        "meta_path": meta_path,
        "meta": meta,
    }


def main():
    args = parse_args()
    dataset_name = resolve_dataset_name(args.dataset)
    df = load_interactions(dataset=args.dataset, data_root=args.data_root)
    original_rows = len(df)
    df = df[df["rating"] >= args.min_rating].copy()
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["rating"] = df["rating"].astype(float)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df["timestamp"] = df["timestamp"].astype(np.int64)

    # De-duplicate repeated user-item events to avoid split leakage.
    df = df.sort_values(["timestamp"]).drop_duplicates(subset=["user_id", "item_id"], keep="last")
    df = df.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    rows_after_filter_dedup = len(df)

    sparsify_stats = {
        "applied": False,
        "keep_prob": 1.0,
        "seed": int(args.ml1m_sparsify_seed),
        "rows_before": int(len(df)),
        "rows_after": int(len(df)),
        "num_item_anchors": 0,
    }
    if args.dataset == "ml-1m":
        df, sparsify_stats = random_sparsify_ml1m(
            df,
            keep_prob=args.ml1m_sparsify_keep_prob,
            seed=args.ml1m_sparsify_seed,
        )

    (
        train_rows,
        val_rows,
        test_rows,
        split_stats,
    ) = split_interactions_temporal(df)
    train_rows, val_rows, test_rows, promoted = ensure_train_item_coverage(train_rows, val_rows, test_rows)
    split_stats["num_items_promoted_to_train"] = int(promoted)
    train_seq_rows, val_seq_rows, test_seq_rows = build_seq_splits(
        train_rows,
        val_rows,
        test_rows,
        max_seq_len=args.max_seq_len,
    )
    full_rows = df[["user_id", "item_id", "rating", "timestamp"]].to_dict("records")
    dataset_dir = os.path.join(args.output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    saved = save_outputs(
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        full_rows=full_rows,
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        train_seq_rows=train_seq_rows,
        val_seq_rows=val_seq_rows,
        test_seq_rows=test_seq_rows,
        split_stats=split_stats,
        sparsify_stats=sparsify_stats,
    )
    print(
        f"dataset={args.dataset}, recbole_dataset={dataset_name}, "
        f"rows_before={original_rows}, rows_after_filter_dedup={rows_after_filter_dedup}, "
        f"rows_after_sparsify={len(df)}, "
        f"min_rating={args.min_rating}, max_seq_len={args.max_seq_len}, "
        f"split_mode=per_user_temporal_8_1_1, "
        f"ml1m_sparsify_keep_prob={args.ml1m_sparsify_keep_prob}"
    )
    print(json.dumps(saved["meta"], ensure_ascii=False, indent=2))
    print(f"saved interactions to {saved['full_path']}")
    print(f"saved train split to {saved['train_path']}")
    print(f"saved valid split to {saved['valid_path']}")
    print(f"saved test split to {saved['test_path']}")
    print(f"saved train seq split to {saved['train_seq_path']}")
    print(f"saved valid seq split to {saved['valid_seq_path']}")
    print(f"saved test seq split to {saved['test_seq_path']}")


if __name__ == "__main__":
    main()
