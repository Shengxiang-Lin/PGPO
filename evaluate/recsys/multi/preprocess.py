import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess interactions for multi-recsys baselines.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="amazon-book",
        choices=["ml-1m", "amazon-book"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="../../../data",
        help="Root directory of raw datasets.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output root. Final files will be saved to <output_dir>/<dataset>/.",
    )
    parser.add_argument(
        "--min_rating",
        type=float,
        default=4.0,
        help="Keep interactions with rating >= min_rating.",
    )
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


def remap_ids(df):
    user_codes, user_uniques = pd.factorize(df["user_id"], sort=True)
    item_codes, item_uniques = pd.factorize(df["item_id"], sort=True)
    df = df.copy()
    df["uid"] = user_codes.astype(np.int64)
    df["iid"] = item_codes.astype(np.int64)
    user2id = {str(u): int(i) for i, u in enumerate(user_uniques.tolist())}
    item2id = {str(it): int(i) for i, it in enumerate(item_uniques.tolist())}
    return df, user2id, item2id


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

    for indices in item_indices.values():
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
    train, val, test = [], [], []
    stats = {
        "num_train_only_users": 0,
        "num_split_users": 0,
    }

    for _, user_df in df.groupby("uid", sort=True):
        user_df = user_df.sort_values(["timestamp", "iid"]).reset_index(drop=True)
        interactions = user_df[["uid", "iid", "timestamp"]].to_dict("records")
        num_interactions = len(interactions)

        # Keep sparse users in train only to avoid cold-start evaluation users.
        if num_interactions < 3:
            train.extend(interactions)
            stats["num_train_only_users"] += 1
            continue

        val_n = max(1, int(num_interactions * 0.1))
        test_n = max(1, int(num_interactions * 0.1))
        train_n = num_interactions - val_n - test_n

        train.extend(interactions[:train_n])
        val.extend(interactions[train_n:train_n + val_n])
        test.extend(interactions[train_n + val_n:])
        stats["num_split_users"] += 1

    return train, val, test, stats


def ensure_train_item_coverage(train_rows, val_rows, test_rows):
    train_rows = list(train_rows)
    val_rows = list(val_rows)
    test_rows = list(test_rows)
    train_items = {int(row["iid"]) for row in train_rows}
    all_items = train_items | {int(row["iid"]) for row in val_rows} | {int(row["iid"]) for row in test_rows}
    missing_items = all_items - train_items
    promoted = 0

    for item_id in sorted(missing_items):
        candidates = [
            ("val", row) for row in val_rows if int(row["iid"]) == item_id
        ] + [
            ("test", row) for row in test_rows if int(row["iid"]) == item_id
        ]
        if not candidates:
            continue
        source_name, chosen_row = min(
            candidates,
            key=lambda pair: (pair[1]["timestamp"], pair[1]["uid"], pair[1]["iid"]),
        )
        if source_name == "val":
            val_rows.remove(chosen_row)
        else:
            test_rows.remove(chosen_row)
        train_rows.append(chosen_row)
        promoted += 1

    train_rows.sort(key=lambda row: (row["uid"], row["timestamp"], row["iid"]))
    val_rows.sort(key=lambda row: (row["uid"], row["timestamp"], row["iid"]))
    test_rows.sort(key=lambda row: (row["uid"], row["timestamp"], row["iid"]))
    return train_rows, val_rows, test_rows, promoted


def pair_interactions(rows):
    return [(int(row["uid"]), int(row["iid"])) for row in rows]


def build_grouped_rows(interactions):
    grouped = defaultdict(list)
    for u, i in interactions:
        grouped[int(u)].append(int(i))

    rows = []
    for user in sorted(grouped.keys()):
        rows.append([int(user), *grouped[user]])
    return rows


def build_train_user_item_dict(train_rows, num_user):
    grouped = {u: [] for u in range(num_user)}
    for row in train_rows:
        grouped[int(row["uid"])].append(int(row["iid"]))
    return grouped


def save_outputs(
    dataset,
    output_dir,
    train_interactions,
    val_interactions,
    test_interactions,
    user_item_dict,
    user2id,
    item2id,
    split_stats,
    sparsify_stats,
):
    out_dir = os.path.join(output_dir, dataset)
    os.makedirs(out_dir, exist_ok=True)

    train_arr = np.asarray(train_interactions, dtype=np.int64)
    val_rows = np.asarray(build_grouped_rows(val_interactions), dtype=object)
    test_rows = np.asarray(build_grouped_rows(test_interactions), dtype=object)

    np.save(os.path.join(out_dir, "train.npy"), train_arr)
    np.save(os.path.join(out_dir, "val.npy"), val_rows)
    np.save(os.path.join(out_dir, "test.npy"), test_rows)
    np.save(os.path.join(out_dir, "val_full.npy"), val_rows)
    np.save(os.path.join(out_dir, "test_full.npy"), test_rows)
    np.save(os.path.join(out_dir, "user_item_dict.npy"), user_item_dict)

    with open(os.path.join(out_dir, "user2id.json"), "w", encoding="utf-8") as f:
        json.dump(user2id, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "item2id.json"), "w", encoding="utf-8") as f:
        json.dump(item2id, f, ensure_ascii=False, indent=2)

    meta = {
        "split_mode": "per_user_temporal_8_1_1",
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "num_user": len(user2id),
        "num_item": len(item2id),
        "num_train_interactions": int(len(train_interactions)),
        "num_val_interactions": int(len(val_interactions)),
        "num_test_interactions": int(len(test_interactions)),
        "num_users_in_val": int(len(val_rows)),
        "num_users_in_test": int(len(test_rows)),
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
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved processed files to: {out_dir}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_interactions(args.dataset, args.data_root)
    rows_before = len(df)

    df = df[df["rating"] >= args.min_rating].copy()
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
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

    df, user2id, item2id = remap_ids(df)
    train_rows, val_rows, test_rows, split_stats = split_interactions_temporal(df)
    train_rows, val_rows, test_rows, promoted = ensure_train_item_coverage(train_rows, val_rows, test_rows)
    split_stats["num_items_promoted_to_train"] = int(promoted)
    train = pair_interactions(train_rows)
    val = pair_interactions(val_rows)
    test = pair_interactions(test_rows)
    user_item_dict = build_train_user_item_dict(train_rows, num_user=len(user2id))

    print(
        f"dataset={args.dataset}, rows_before={rows_before}, rows_after_filter_dedup={rows_after_filter_dedup}, "
        f"rows_after_sparsify={len(df)}, min_rating={args.min_rating}, "
        f"split_mode=per_user_temporal_8_1_1, ml1m_sparsify_keep_prob={args.ml1m_sparsify_keep_prob}"
    )
    save_outputs(
        dataset=args.dataset,
        output_dir=args.output_dir,
        train_interactions=train,
        val_interactions=val,
        test_interactions=test,
        user_item_dict=user_item_dict,
        user2id=user2id,
        item2id=item2id,
        split_stats=split_stats,
        sparsify_stats=sparsify_stats,
    )


if __name__ == "__main__":
    main()
