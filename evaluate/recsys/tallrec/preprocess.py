import csv
import json
import os
import random
import math
from typing import Dict, List, Tuple, Set

import argparse
from tqdm import tqdm


def load_id_item_map(data_root: str, dataset: str) -> Dict[str, str]:
    handled_path = os.path.join(data_root, dataset, "handled", "id_item.json")
    if os.path.exists(handled_path):
        with open(handled_path, "r", encoding="utf-8") as f:
            return {str(k): str(v) for k, v in json.load(f).items()}
    return {}


def load_ml1m(
    raw_dir: str,
    positive_threshold: float = 3.0,
    show_progress: bool = True,
) -> Tuple[List[Tuple[str, str, int, int]], Dict[str, str]]:
    ratings_path = os.path.join(raw_dir, "ratings.dat")
    movies_path = os.path.join(raw_dir, "movies.dat")

    title_map: Dict[str, str] = {}
    with open(movies_path, "r", encoding="ISO-8859-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("::")
            if len(parts) >= 2:
                title_map[str(parts[0])] = parts[1]

    interactions: List[Tuple[str, str, int, int]] = []
    with open(ratings_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading ML-1M ratings", disable=not show_progress):
            parts = line.rstrip("\n").split("::")
            if len(parts) < 4:
                continue
            user_id, item_id, rating_str, timestamp_str = parts[:4]
            label = 1 if float(rating_str) > positive_threshold else 0
            interactions.append((str(user_id), str(item_id), label, int(timestamp_str)))
    return interactions, title_map


def load_amazon_book(
    raw_dir: str,
    data_root: str,
    dataset: str,
    positive_threshold: float = 4.0,
    show_progress: bool = True,
) -> Tuple[List[Tuple[str, str, int, int]], Dict[str, str]]:
    ratings_path = os.path.join(raw_dir, "Books.csv")
    title_map = load_id_item_map(data_root=data_root, dataset=dataset)

    interactions: List[Tuple[str, str, int, int]] = []
    with open(ratings_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in tqdm(reader, desc="Loading Amazon ratings", disable=not show_progress):
            if len(row) < 4:
                continue
            item_id, user_id, rating_str, timestamp_str = row[:4]
            label = 1 if float(rating_str) > positive_threshold else 0
            interactions.append((str(user_id), str(item_id), label, int(timestamp_str)))
    return interactions, title_map


def _iterative_core_filter_interactions(
    interactions: List[Tuple[str, str, int, int]],
    min_user_inter: int,
    min_item_inter: int,
) -> List[Tuple[str, str, int, int]]:
    if min_user_inter <= 1 and min_item_inter <= 1:
        return interactions

    current = interactions
    while True:
        before_len = len(current)

        user_count = {}
        item_count = {}
        for user_id, item_id, _, _ in current:
            user_count[user_id] = user_count.get(user_id, 0) + 1
            item_count[item_id] = item_count.get(item_id, 0) + 1

        current = [
            x for x in current
            if user_count.get(x[0], 0) >= min_user_inter and item_count.get(x[1], 0) >= min_item_inter
        ]
        if len(current) == before_len:
            break
    return current


def sparsify_interactions(
    interactions: List[Tuple[str, str, int, int]],
    sparsify_ratio: float = 1.0,
    min_user_inter: int = 3,
    min_item_inter: int = 1,
    seed: int = 42,
) -> List[Tuple[str, str, int, int]]:
    if not (0 < sparsify_ratio <= 1.0):
        raise ValueError(f"sparsify_ratio must be in (0, 1], got {sparsify_ratio}")
    if min_user_inter < 1 or min_item_inter < 1:
        raise ValueError("min_user_inter and min_item_inter must be >= 1")
    if sparsify_ratio >= 1.0:
        return _iterative_core_filter_interactions(interactions, min_user_inter, min_item_inter)

    base = _iterative_core_filter_interactions(interactions, min_user_inter, min_item_inter)
    rng = random.Random(seed)

    by_user: Dict[str, List[Tuple[str, str, int, int]]] = {}
    for x in base:
        by_user.setdefault(x[0], []).append(x)

    selected: List[Tuple[str, str, int, int]] = []
    for user_id, rows in by_user.items():
        n = len(rows)
        keep_n = min(n, max(min_user_inter, int(math.ceil(n * sparsify_ratio))))
        if keep_n >= n:
            chosen = rows
        else:
            chosen = rng.sample(rows, keep_n)
        selected.extend(chosen)

    selected = _iterative_core_filter_interactions(selected, min_user_inter, min_item_inter)
    return selected


def print_interaction_stats(name: str, interactions: List[Tuple[str, str, int, int]]):
    users = len(set(x[0] for x in interactions))
    items = len(set(x[1] for x in interactions))
    pos = sum(1 for x in interactions if int(x[2]) == 1)
    total = len(interactions)
    print(
        f"{name}: interactions={total}, users={users}, items={items}, "
        f"pos={pos}, neg={total - pos}, pos_ratio={(pos / total if total else 0):.4f}"
    )


def build_samples(
    dataset: str,
    interactions: List[Tuple[str, str, int, int]],
    title_map: Dict[str, str],
    seq_len: int,
    show_progress: bool = True,
    progress_desc: str = "Building samples",
) -> List[Dict]:
    if dataset == "amazon-book":
        target_noun = "book"
        instruction = "Given user preference and unpreference items, identify whether the user will like the target item/book by answering \"Yes.\" or \"No.\"."
    else:
        target_noun = "movie"
        instruction = "Given user preference and unpreference items, identify whether the user will like the target item/movie by answering \"Yes.\" or \"No.\"."

    by_user: Dict[str, List[Tuple[str, int, int]]] = {}
    for user_id, item_id, label, timestamp in interactions:
        by_user.setdefault(user_id, []).append((item_id, label, timestamp))

    samples: List[Dict] = []
    user_iter = sorted(by_user.keys())
    for user_id in tqdm(user_iter, desc=progress_desc, disable=not show_progress):
        seq = sorted(by_user[user_id], key=lambda x: x[2])
        if len(seq) <= seq_len:
            continue
        for i in range(seq_len, len(seq)):
            history = seq[i - seq_len:i]
            target_item_id, target_label, _ = seq[i]
            target_item_id = str(target_item_id)

            preference_ids = []
            unpreference_ids = []
            for item_id, label, _ in history:
                item_id = str(item_id)
                if label == 1:
                    preference_ids.append(item_id)
                else:
                    unpreference_ids.append(item_id)

            preference_str = ", ".join([f"\"{rid} [MOVIE]\"" for rid in preference_ids])
            unpreference_str = ", ".join([f"\"{rid} [MOVIE]\"" for rid in unpreference_ids])

            samples.append(
                {
                    "instruction": instruction,
                    "input": (
                        f"User Preference: {preference_str}\n"
                        f"User Unpreference: {unpreference_str}\n"
                        f"Whether the user will like the target {target_noun} \"{target_item_id} [MOVIE]\"?"
                    ),
                    "output": "Yes." if int(target_label) == 1 else "No.",
                    "preference_ids": preference_ids,
                    "unpreference_ids": unpreference_ids,
                    "target_movie_id": target_item_id,
                    # Keep metadata for temporal split and debugging.
                    "user_id": str(user_id),
                    "target_timestamp": int(seq[i][2]),
                }
            )
    return samples


def split_samples_811(
    samples: List[Dict],
    split_mode: str = "random",
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    if split_mode not in {"random", "temporal"}:
        raise ValueError(f"unsupported split_mode={split_mode}, expect one of: random, temporal")

    ordered = samples[:]
    if split_mode == "random":
        rng = random.Random(seed)
        rng.shuffle(ordered)
    else:
        ordered.sort(
            key=lambda s: (
                int(s.get("target_timestamp", 0)),
                str(s.get("user_id", "")),
                str(s.get("target_movie_id", "")),
            )
        )

    total = len(ordered)
    train_n = int(total * 0.8)
    val_n = int(total * 0.1)
    test_n = total - train_n - val_n
    return {
        "train.json": ordered[:train_n],
        "val.json": ordered[train_n:train_n + val_n],
        "test.json": ordered[train_n + val_n:train_n + val_n + test_n],
    }


def build_item_popularity(
    interactions: List[Tuple[str, str, int, int]],
    show_progress: bool = True,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for _, item_id, _, _ in tqdm(interactions, desc="Counting item popularity", disable=not show_progress):
        item_id = str(item_id)
        counts[item_id] = counts.get(item_id, 0) + 1
    return counts


def assign_item_buckets(
    item_popularity: Dict[str, int],
    hot_ratio: float = 0.2,
    medium_ratio: float = 0.3,
) -> Dict[str, str]:
    if hot_ratio < 0 or medium_ratio < 0 or hot_ratio + medium_ratio > 1:
        raise ValueError("hot_ratio and medium_ratio must satisfy: >=0 and hot+medium<=1")
    items = sorted(item_popularity.items(), key=lambda x: (-x[1], x[0]))
    n = len(items)
    hot_n = int(n * hot_ratio)
    medium_n = int(n * medium_ratio)
    bucket: Dict[str, str] = {}
    for i, (item_id, _) in enumerate(items):
        if i < hot_n:
            bucket[item_id] = "hot"
        elif i < hot_n + medium_n:
            bucket[item_id] = "medium"
        else:
            bucket[item_id] = "cold"
    return bucket


def _balanced_pick(
    indices: List[int],
    samples: List[Dict],
    n: int,
    rng: random.Random,
) -> List[int]:
    if n <= 0 or not indices:
        return []
    yes = [i for i in indices if samples[i].get("output") == "Yes."]
    no = [i for i in indices if samples[i].get("output") == "No."]
    rng.shuffle(yes)
    rng.shuffle(no)

    half = n // 2
    take_yes = min(len(yes), half)
    take_no = min(len(no), half)
    selected = yes[:take_yes] + no[:take_no]

    remaining = n - len(selected)
    if remaining > 0:
        leftovers = yes[take_yes:] + no[take_no:]
        rng.shuffle(leftovers)
        selected += leftovers[:remaining]
    rng.shuffle(selected)
    return selected


def balanced_sample(samples: List[Dict], max_samples: int, seed: int) -> List[Dict]:
    if max_samples <= 0:
        return samples
    if len(samples) <= max_samples:
        return samples
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    selected_indices = _balanced_pick(indices, samples, max_samples, rng)
    return [samples[i] for i in selected_indices]


def estimate_sample_count(
    interactions: List[Tuple[str, str, int, int]],
    seq_len: int,
) -> int:
    by_user_count: Dict[str, int] = {}
    for user_id, _, _, _ in interactions:
        by_user_count[user_id] = by_user_count.get(user_id, 0) + 1
    return sum(max(0, c - seq_len) for c in by_user_count.values())


def split_item_sets_random_with_min_test_samples(
    interactions: List[Tuple[str, str, int, int]],
    test_item_ratio: float = 0.2,
    val_item_ratio: float = 0.1,
    min_test_samples: int = 1000,
    min_val_samples: int = 800,
    seq_len: int = 10,
    seed: int = 42,
    max_split_attempts: int = 50,
) -> Tuple[Dict[str, Set[str]], int, int, int]:
    if test_item_ratio <= 0 or val_item_ratio < 0 or test_item_ratio + val_item_ratio >= 1:
        raise ValueError("invalid ratios: require test>0, val>=0, test+val<1")

    item_ids = sorted({str(x[1]) for x in interactions})
    n = len(item_ids)
    test_n = max(1, int(n * test_item_ratio))
    val_n = max(1, int(n * val_item_ratio))
    train_n = n - test_n - val_n
    if train_n <= 0:
        raise ValueError("invalid split sizes, train item count <= 0")

    best_sets = None
    best_test_est = -1
    best_val_est = -1
    best_attempt = -1

    for attempt in range(max_split_attempts):
        rng = random.Random(seed + attempt)
        shuffled = item_ids[:]
        rng.shuffle(shuffled)

        train_items = set(shuffled[:train_n])
        val_items = set(shuffled[train_n:train_n + val_n])
        test_items = set(shuffled[train_n + val_n:])

        test_interactions = [x for x in interactions if str(x[1]) in test_items]
        val_interactions = [x for x in interactions if str(x[1]) in val_items]
        est_test_samples = estimate_sample_count(test_interactions, seq_len=seq_len)
        est_val_samples = estimate_sample_count(val_interactions, seq_len=seq_len)

        if est_test_samples + est_val_samples > best_test_est + best_val_est:
            best_test_est = est_test_samples
            best_val_est = est_val_samples
            best_sets = {"train": train_items, "val": val_items, "test": test_items}
            best_attempt = attempt + 1
        if est_test_samples >= min_test_samples and est_val_samples >= min_val_samples:
            return (
                {"train": train_items, "val": val_items, "test": test_items},
                est_test_samples,
                est_val_samples,
                attempt + 1,
            )

    return best_sets, best_test_est, best_val_est, best_attempt


def filter_interactions_by_item_set(
    interactions: List[Tuple[str, str, int, int]],
    item_set: Set[str],
) -> List[Tuple[str, str, int, int]]:
    return [x for x in interactions if str(x[1]) in item_set]


def collect_item_ids(samples: List[Dict]) -> Set[str]:
    item_ids: Set[str] = set()
    for s in samples:
        item_ids.update(map(str, s.get("preference_ids", [])))
        item_ids.update(map(str, s.get("unpreference_ids", [])))
        item_ids.add(str(s.get("target_movie_id")))
    return item_ids


def popularity_aware_split(
    samples: List[Dict],
    item_bucket: Dict[str, str],
    max_samples: int,
    seed: int,
    train_val_cold_ratio: float = 0.0,
    show_progress: bool = True,
) -> Dict[str, List[Dict]]:
    rng = random.Random(seed)
    total = len(samples) if max_samples <= 0 else min(max_samples, len(samples))
    train_n = int(total * 0.8)
    val_n = int(total * 0.1)
    test_n = total - train_n - val_n

    hot_idx, medium_idx, cold_idx = [], [], []
    for i, s in tqdm(enumerate(samples), total=len(samples), desc="Assigning sample buckets", disable=not show_progress):
        item_id = str(s.get("target_movie_id", ""))
        b = item_bucket.get(item_id, "cold")
        if b == "hot":
            hot_idx.append(i)
        elif b == "medium":
            medium_idx.append(i)
        else:
            cold_idx.append(i)

    rng.shuffle(hot_idx)
    rng.shuffle(medium_idx)
    rng.shuffle(cold_idx)

    hot_set = set(hot_idx)
    medium_set = set(medium_idx)
    cold_set = set(cold_idx)
    hm_set = hot_set | medium_set

    selected_test = _balanced_pick(cold_idx, samples, test_n, rng)
    remaining: Set[int] = set(range(len(samples))) - set(selected_test)

    def pick_train_or_val(need: int) -> List[int]:
        if need <= 0:
            return []
        hm_pool = list(remaining & hm_set)
        cold_pool = list(remaining & cold_set)
        take_cold = int(need * train_val_cold_ratio)
        take_hm = need - take_cold
        picked = _balanced_pick(hm_pool, samples, take_hm, rng)
        remain_need = need - len(picked)
        if remain_need > 0:
            picked += _balanced_pick(cold_pool, samples, min(remain_need, take_cold + remain_need), rng)
        if len(picked) < need:
            fallback_pool = list(remaining - set(picked))
            picked += _balanced_pick(fallback_pool, samples, need - len(picked), rng)
        return picked[:need]

    selected_val = pick_train_or_val(val_n)
    remaining -= set(selected_val)
    selected_train = pick_train_or_val(train_n)

    splits = {
        "train.json": [samples[i] for i in selected_train],
        "val.json": [samples[i] for i in selected_val],
        "test.json": [samples[i] for i in selected_test],
    }
    for k in splits:
        rng.shuffle(splits[k])
    return splits


def print_label_stats(name: str, samples: List[Dict]):
    total = len(samples)
    yes = sum(1 for s in samples if s.get("output") == "Yes.")
    no = sum(1 for s in samples if s.get("output") == "No.")
    print(f"{name}: total={total}, yes={yes}, no={no}, yes_ratio={(yes / total if total else 0):.4f}")


def print_bucket_stats(name: str, samples: List[Dict], item_bucket: Dict[str, str]):
    hot = 0
    medium = 0
    cold = 0
    for s in samples:
        b = item_bucket.get(str(s.get("target_movie_id", "")), "cold")
        if b == "hot":
            hot += 1
        elif b == "medium":
            medium += 1
        else:
            cold += 1
    total = len(samples)
    print(
        f"{name}_bucket: total={total}, hot={hot}, medium={medium}, cold={cold}, "
        f"cold_ratio={(cold / total if total else 0):.4f}"
    )


def main(
    dataset: str = "ml-1m",
    data_root: str = "../../../data",
    output_root: str = "./data",
    seq_len: int = 10,
    max_samples: int = 10000,
    seed: int = 42,
    test_item_ratio: float = 0.2,
    val_item_ratio: float = 0.1,
    min_test_samples: int = 1000,
    min_val_samples: int = 800,
    max_split_attempts: int = 50,
    split_mode: str = "random",
    show_progress: bool = True,
):
    dataset = dataset.strip()
    raw_dir = os.path.join(data_root, dataset, "raw-0")
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"raw dir not found: {raw_dir}")

    if dataset == "ml-1m":
        positive_threshold = 3.0
        interactions, raw_title_map = load_ml1m(
            raw_dir,
            positive_threshold=positive_threshold,
            show_progress=show_progress,
        )
        title_map = load_id_item_map(data_root, dataset)
        if not title_map:
            title_map = raw_title_map
    elif dataset == "amazon-book":
        positive_threshold = 4.0
        interactions, title_map = load_amazon_book(
            raw_dir,
            data_root,
            dataset,
            positive_threshold=positive_threshold,
            show_progress=show_progress,
        )
    else:
        raise ValueError(f"unsupported dataset: {dataset}")

    print(
        f"dataset={dataset}, interactions={len(interactions)}, "
        f"positive_threshold={positive_threshold}"
    )
    print_interaction_stats("raw_interactions", interactions)
    all_samples = build_samples(
        dataset=dataset,
        interactions=interactions,
        title_map=title_map,
        seq_len=seq_len,
        show_progress=show_progress,
        progress_desc="Building rec-tuning samples",
    )
    print_label_stats("all_samples", all_samples)

    if max_samples > 0:
        all_samples = balanced_sample(all_samples, max_samples, seed=seed)
        print_label_stats("all_samples_after_max_samples", all_samples)

    splits = split_samples_811(all_samples, split_mode=split_mode, seed=seed)
    print(f"split_mode={split_mode}, split_ratio=8:1:1")

    output_dir = os.path.join(output_root, dataset)
    os.makedirs(output_dir, exist_ok=True)
    for filename, split_data in splits.items():
        out_path = os.path.join(output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print_label_stats(filename, split_data)
        print(f"saved {len(split_data)} samples -> {out_path}")

    train_items = collect_item_ids(splits["train.json"])
    val_items = collect_item_ids(splits["val.json"])
    test_items = collect_item_ids(splits["test.json"])
    print(
        "item overlap (expected in non-cold-start split): "
        f"train&val={len(train_items & val_items)}, "
        f"train&test={len(train_items & test_items)}, "
        f"val&test={len(val_items & test_items)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="amazon-book")
    parser.add_argument("--data_root", type=str, default="../../data/")
    parser.add_argument("--output_root", type=str, default="./data")
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_item_ratio", type=float, default=0.2)
    parser.add_argument("--val_item_ratio", type=float, default=0.1)
    parser.add_argument("--min_test_samples", type=int, default=1000)
    parser.add_argument("--min_val_samples", type=int, default=800)
    parser.add_argument("--max_split_attempts", type=int, default=50)
    parser.add_argument("--split_mode", type=str, default="random", choices=["random", "temporal"])
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()
    main(
        dataset=args.dataset,
        data_root=args.data_root,
        output_root=args.output_root,
        seq_len=args.seq_len,
        max_samples=args.max_samples,
        seed=args.seed,
        test_item_ratio=args.test_item_ratio,
        val_item_ratio=args.val_item_ratio,
        min_test_samples=args.min_test_samples,
        min_val_samples=args.min_val_samples,
        max_split_attempts=args.max_split_attempts,
        split_mode=args.split_mode,
        show_progress=not args.no_progress,
    )
