import argparse
import ast
import csv
import gzip
import json
import random
from pathlib import Path
from typing import Any

from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MOVIE_DIR = DATA_DIR / "movie"
BOOK_DIR = DATA_DIR / "book"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified dataset builder for movie and book in InvInst_dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["movie", "book", "all"],
        help="Choose which dataset pipeline to run.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Max number of train samples to keep (randomly sampled). Mainly used for book train.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by train sampling.",
    )
    return parser.parse_args()


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_edges(edge_payload: dict[str, Any]) -> str:
    segments: list[str] = []
    for key, value in edge_payload.items():
        if key == "item_name":
            continue
        if isinstance(value, list):
            value_text = ",".join(str(item) for item in value)
        else:
            value_text = str(value)
        segments.append(f"{key}:{value_text}")
    return " | ".join(segments)


def parse_movie_txt(file_path: Path) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("::", 2)
            if len(parts) != 3:
                continue
            movie_id, title, genres = parts
            result[movie_id] = (title, genres)
    return result


def build_movie_catalog() -> tuple[dict[str, tuple[str, str]], set[str]]:
    ml1m_path = MOVIE_DIR / "ml-1m" / "movies.dat"
    ml25m_path = MOVIE_DIR / "ml-25m" / "movies.csv"
    ml32m_path = MOVIE_DIR / "ml-32m" / "movies.csv"

    movie_dict: dict[str, tuple[str, str]] = {}
    ml1m_titles: set[str] = set()

    print("Processing movie: ml-1m ...")
    with ml1m_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc="movie ml-1m", leave=False):
            line = line.strip()
            if not line:
                continue
            parts = line.split("::")
            if len(parts) < 3:
                continue
            _, title, genres = parts[0], parts[1].strip(), parts[2]
            if title not in movie_dict:
                movie_dict[title] = (parts[0], genres)
            ml1m_titles.add(title)

    def merge_csv_movies(path: Path) -> None:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc=f"movie {path.parent.name}", leave=False):
                title = row["title"].strip()
                movie_id = row["movieId"].strip()
                genres = row["genres"].strip()
                if title not in movie_dict:
                    movie_dict[title] = (movie_id, genres)

    print("Processing movie: ml-25m ...")
    merge_csv_movies(ml25m_path)
    print("Processing movie: ml-32m ...")
    merge_csv_movies(ml32m_path)
    return movie_dict, ml1m_titles


def build_movie_datasets() -> None:
    movie_output = MOVIE_DIR / "movie.json"
    movie_train_txt = MOVIE_DIR / "movie_train.txt"
    movie_test_txt = MOVIE_DIR / "movie_test.txt"
    movie_edges_json = MOVIE_DIR / "movie_edges.json"
    train_json = MOVIE_DIR / "train.json"
    test_json = MOVIE_DIR / "test.json"

    movie_dict, ml1m_titles = build_movie_catalog()
    print(f"Movie unique titles: {len(movie_dict)}")

    movie_data: dict[str, str] = {}
    train_items: dict[str, tuple[str, str]] = {}
    test_items: dict[str, tuple[str, str]] = {}

    with movie_train_txt.open("w", encoding="utf-8") as train_f, movie_test_txt.open(
        "w", encoding="utf-8"
    ) as test_f:
        unique_id = 1
        for title, (_, genres) in tqdm(
            sorted(movie_dict.items()), desc="movie split train/test", leave=False
        ):
            item_id = str(unique_id)
            movie_data[item_id] = title
            line = f"{item_id}::{title}::{genres}\n"
            if title in ml1m_titles:
                test_f.write(line)
                test_items[item_id] = (title, genres)
            else:
                train_f.write(line)
                train_items[item_id] = (title, genres)
            unique_id += 1

    save_json(movie_data, movie_output)
    print(f"Saved: {movie_output} ({len(movie_data)} items)")
    print(f"Saved: {movie_train_txt} ({len(train_items)} items)")
    print(f"Saved: {movie_test_txt} ({len(test_items)} items)")

    if not movie_edges_json.exists():
        print(f"Skip train/test json build because missing: {movie_edges_json}")
        return

    edge_items = load_json(movie_edges_json)
    train_dataset: dict[str, list[str]] = {}
    for movie_id, (title, genres) in tqdm(
        train_items.items(), desc="movie build train.json", leave=False
    ):
        edge_payload = edge_items.get(movie_id)
        if not isinstance(edge_payload, dict):
            continue
        edge_text = format_edges(edge_payload)
        if not edge_text:
            continue
        train_dataset[movie_id] = [title, edge_text, genres]

    test_dataset: dict[str, list[str]] = {
        movie_id: [title, genres] for movie_id, (title, genres) in test_items.items()
    }

    save_json(train_dataset, train_json)
    save_json(test_dataset, test_json)
    print(f"Saved: {train_json} ({len(train_dataset)} items)")
    print(f"Saved: {test_json} ({len(test_dataset)} items)")


def parse_meta_line(line: str) -> dict[str, Any] | None:
    payload = line.strip()
    if not payload:
        return None
    try:
        obj = json.loads(payload)
    except Exception:
        try:
            obj = ast.literal_eval(payload)
        except Exception:
            return None
    if isinstance(obj, dict):
        return obj
    return None


def normalize_category(raw_category: Any, mode: str) -> str:
    if isinstance(raw_category, list):
        categories = [str(x).strip() for x in raw_category if str(x).strip()]
    elif isinstance(raw_category, str):
        text = raw_category.strip()
        categories = [text] if text else []
    else:
        categories = []

    # Drop the root node "Books", keep the meaningful sub-path.
    if categories and categories[0].strip().lower() == "books":
        categories = categories[1:]

    if not categories:
        return ""
    if mode == "leaf":
        return categories[-1]
    return "|".join(categories)


def extract_book_record(meta_obj: dict[str, Any], category_mode: str) -> tuple[str, str, str] | None:
    item_id = str(meta_obj.get("asin", "")).strip()
    title = str(meta_obj.get("title", "")).strip()
    category = normalize_category(meta_obj.get("category", []), mode=category_mode)
    if not item_id or not title or not category:
        return None
    return item_id, title, category


def load_book_split(
    gz_path: Path, category_mode: str
) -> tuple[dict[str, list[str]], int, int, int]:
    dataset: dict[str, list[str]] = {}
    parse_fail = 0
    invalid_record = 0
    duplicate_id = 0

    with gzip.open(gz_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc=f"book parse {gz_path.name}", leave=False):
            meta_obj = parse_meta_line(line)
            if meta_obj is None:
                parse_fail += 1
                continue
            parsed = extract_book_record(meta_obj, category_mode=category_mode)
            if parsed is None:
                invalid_record += 1
                continue
            item_id, title, category = parsed
            if item_id in dataset:
                duplicate_id += 1
                continue
            dataset[item_id] = [title, category]
    return dataset, parse_fail, invalid_record, duplicate_id


def build_book_datasets(max_train_samples: int | None, seed: int) -> None:
    test_meta_path = BOOK_DIR / "meta_Books_0.json.gz"
    all_meta_path = BOOK_DIR / "meta_Books.json.gz"
    train_output = BOOK_DIR / "train.json"
    test_output = BOOK_DIR / "test.json"
    metadata_output = BOOK_DIR / "split_metadata.json"
    category_mode = "full"

    test_dataset, test_parse_fail, test_invalid, test_dup = load_book_split(
        test_meta_path, category_mode=category_mode
    )
    test_ids = set(test_dataset.keys())
    all_dataset, all_parse_fail, all_invalid, all_dup = load_book_split(
        all_meta_path, category_mode=category_mode
    )

    train_dataset: dict[str, list[str]] = {}
    overlap_removed = 0
    for item_id, payload in tqdm(all_dataset.items(), desc="book remove test overlap", leave=False):
        if item_id in test_ids:
            overlap_removed += 1
            continue
        train_dataset[item_id] = payload

    train_samples_before_sampling = len(train_dataset)
    sampling_applied = False
    if max_train_samples is not None and max_train_samples > 0 and len(train_dataset) > max_train_samples:
        sampling_applied = True
        rng = random.Random(seed)
        sampled_items = rng.sample(list(train_dataset.items()), k=max_train_samples)
        train_dataset = dict(sampled_items)

    save_json(train_dataset, train_output)
    save_json(test_dataset, test_output)
    metadata = {
        "test_meta_path": str(test_meta_path),
        "all_meta_path": str(all_meta_path),
        "category_mode": category_mode,
        "test_samples": len(test_dataset),
        "train_samples": len(train_dataset),
        "sampling_applied": sampling_applied,
        "max_train_samples": max_train_samples,
        "seed": seed,
        "train_samples_before_sampling": train_samples_before_sampling,
        "overlap_removed_from_train": overlap_removed,
        "test_parse_fail_lines": test_parse_fail,
        "test_invalid_records": test_invalid,
        "test_duplicate_ids": test_dup,
        "all_parse_fail_lines": all_parse_fail,
        "all_invalid_records": all_invalid,
        "all_duplicate_ids": all_dup,
        "train_output": str(train_output),
        "test_output": str(test_output),
    }
    save_json(metadata, metadata_output)
    print(f"Saved: {test_output} ({len(test_dataset)} items)")
    print(f"Saved: {train_output} ({len(train_dataset)} items)")
    print(f"Saved: {metadata_output}")


def main() -> None:
    args = parse_args()
    if args.dataset in {"movie", "all"}:
        build_movie_datasets()
    if args.dataset in {"book", "all"}:
        build_book_datasets(max_train_samples=args.max_train_samples, seed=args.seed)


if __name__ == "__main__":
    main()
