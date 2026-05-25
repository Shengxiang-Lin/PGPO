import argparse
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build [id, item_category] process datasets for movie/book."
    )
    parser.add_argument("--dataset", type=str, default="movie", choices=["movie", "book"])
    parser.add_argument("--raw_dir", type=str, default="")
    parser.add_argument("--process_dir", type=str, default="")
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_train_dataset(raw_train_data: dict) -> dict:
    train = {}

    for item_id, item_payload in raw_train_data.items():
        if not isinstance(item_payload, list):
            continue

        # New format: [id, item_category]
        if len(item_payload) >= 3:
            item_category = item_payload[2]
        elif len(item_payload) >= 2:
            item_category = item_payload[1]
        else:
            continue
        train[item_id] = [str(item_id), item_category]

    return train


def format_edge_payload(edge_payload: dict) -> str:
    segments = []
    for key, value in edge_payload.items():
        if key == "item_name":
            continue
        if isinstance(value, list):
            value_text = ",".join(str(item) for item in value)
        else:
            value_text = str(value)
        segments.append(f"{key}:{value_text}")
    return " | ".join(segments)


def load_variants_map(variants_data: list[dict]) -> dict[str, str]:
    variants_map = {}
    for item in variants_data:
        if not isinstance(item, dict):
            continue
        original_word = item.get("original_word")
        variant_word = item.get("variant")
        if isinstance(original_word, str) and isinstance(variant_word, str):
            variants_map[original_word] = variant_word
    return variants_map


def apply_variants_to_edges(edge_payload: dict, variants_map: dict[str, str]) -> dict:
    replaced_payload = {}
    for key, value in edge_payload.items():
        if key == "item_name":
            replaced_payload[key] = value
            continue
        if isinstance(value, list):
            replaced_values = []
            for item in value:
                item_text = str(item)
                replaced_values.append(
                    variants_map.get(item_text, variants_map.get(item_text.strip(), item_text))
                )
            replaced_payload[key] = replaced_values
        else:
            value_text = str(value)
            replaced_payload[key] = variants_map.get(
                value_text, variants_map.get(value_text.strip(), value_text)
            )
    return replaced_payload


def create_train_datasets(raw_train_path: Path, train_path: Path) -> None:
    raw_train_data = load_json(raw_train_path)
    train = build_train_dataset(raw_train_data)
    save_json(train, train_path)
    print(f"train.json saved to {train_path} ({len(train)} items)")


def create_test_datasets(raw_test_path: Path, test_path: Path) -> None:
    raw_test_data = load_json(raw_test_path)
    test = {}

    for item_id, item_payload in raw_test_data.items():
        if not isinstance(item_payload, list) or len(item_payload) < 2:
            continue
        item_category = item_payload[1]
        pair = [str(item_id), item_category]
        test[item_id] = pair

    save_json(test, test_path)
    print(f"test.json saved to {test_path} ({len(test)} items)")


def main() -> None:
    args = parse_args()
    default_raw_dir = BASE_DIR / "data" / args.dataset / "raw"
    default_process_dir = BASE_DIR / "data" / args.dataset / "process"
    raw_dir = Path(args.raw_dir) if args.raw_dir else default_raw_dir
    process_dir = Path(args.process_dir) if args.process_dir else default_process_dir
    train_dir = process_dir / "train"
    test_dir = process_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    raw_train_path = raw_dir / "train.json"
    raw_test_path = raw_dir / "test.json"
    train_path = train_dir / "train.json"
    test_path = test_dir / "test.json"
    create_train_datasets(raw_train_path=raw_train_path, train_path=train_path)
    create_test_datasets(raw_test_path=raw_test_path, test_path=test_path)


if __name__ == "__main__":
    main()
