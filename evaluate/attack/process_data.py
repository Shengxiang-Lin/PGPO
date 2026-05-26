import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_EMBEDDING_ROOTS = {
    "movie": Path("../embedding/data/ml-1m"),
    "book": Path("../embedding/data/amazon-book"),
}
DEFAULT_TRAIN_EMBEDDING_MODEL = "../../base_models/bert-base-uncased"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def normalize_text(value: Any) -> str:
    return str(value).strip()


def format_edge_payload(edge_payload: dict[str, Any]) -> str:
    segments: list[str] = []
    for key, value in edge_payload.items():
        if key == "item_name":
            continue
        if isinstance(value, list):
            values = [normalize_text(v) for v in value if normalize_text(v)]
            if not values:
                continue
            value_text = ", ".join(values)
        else:
            value_text = normalize_text(value)
            if not value_text:
                continue
        segments.append(f"{key}: {value_text}")
    return " | ".join(segments)


class HFTextEncoder:
    def __init__(self, model_name_or_path: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).to(self.device)
        self.model.eval()

    def encode(
        self,
        texts: list[str],
        batch_size: int,
        max_length: int,
        normalize_embeddings: bool = True,
    ) -> torch.Tensor:
        if not texts:
            return torch.empty((0, 0), dtype=torch.float32)
        vectors: list[torch.Tensor] = []
        with torch.no_grad():
            for start in tqdm(range(0, len(texts), batch_size), desc="Encoding train texts", leave=False):
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
                vectors.append(pooled.detach().cpu().to(torch.float32))
        return torch.cat(vectors, dim=0)


def load_dataset_id_categories(
    dataset_path: Path, max_samples: int | None = None
) -> list[dict[str, str]]:
    raw_data = load_json(dataset_path)
    # Keep ordering deterministic while avoiding mixed int/str compare errors.
    # Numeric IDs are sorted numerically first, then non-numeric IDs lexicographically.
    ordered_ids = sorted(
        raw_data.keys(),
        key=lambda x: (0, int(str(x))) if str(x).isdigit() else (1, str(x)),
    )
    records: list[dict[str, str]] = []

    for item_id in ordered_ids:
        payload = raw_data[item_id]
        if not isinstance(payload, list) or len(payload) < 2:
            continue

        payload_id = str(payload[0]).strip() if payload[0] is not None else str(item_id)
        category = str(payload[1]).strip()
        if not payload_id:
            payload_id = str(item_id)
        if not category:
            continue
        if category.strip().lower() == "(no genres listed)":
            continue

        records.append(
            {
                "id": payload_id,
                "item_category": category,
            }
        )
        if max_samples is not None and len(records) >= max_samples:
            break

    return records


def load_train_text_map(raw_dir: Path, dataset: str, embedding_type: str) -> tuple[dict[str, str], Path]:
    if embedding_type == "v1":
        title_json_candidates = [raw_dir / f"{dataset}.json", raw_dir / "movie.json", raw_dir / "book.json"]
        for path in title_json_candidates:
            if not path.exists():
                continue
            title_data = load_json(path)
            if isinstance(title_data, dict):
                text_map = {str(k): normalize_text(v) for k, v in title_data.items() if normalize_text(v)}
                if text_map:
                    return text_map, path
        # Fallback: extract title from raw train.json payload [title, ...]
        train_path = raw_dir / "train.json"
        train_data = load_json(train_path)
        text_map = {}
        for item_id, payload in train_data.items():
            if isinstance(payload, list) and payload:
                text = normalize_text(payload[0])
                if text:
                    text_map[str(item_id)] = text
        return text_map, train_path

    if embedding_type == "v2":
        edges_candidates = [raw_dir / f"{dataset}_edges.json", raw_dir / "movie_edges.json", raw_dir / "book_edges.json"]
        for edges_path in edges_candidates:
            if not edges_path.exists():
                continue
            edges_data = load_json(edges_path)
            text_map: dict[str, str] = {}
            for item_id, payload in edges_data.items():
                if not isinstance(payload, dict):
                    continue
                edge_text = format_edge_payload(payload)
                if edge_text:
                    text_map[str(item_id)] = edge_text
            if text_map:
                return text_map, edges_path
        raise FileNotFoundError(
            f"Cannot find usable edge source for dataset={dataset}, embedding_type=v2 under {raw_dir}"
        )

    raise ValueError("Train split only supports embedding_type in {'v1', 'v2'}.")


def load_external_embeddings(
    embedding_root: Path, embedding_type: str
) -> tuple[torch.Tensor, dict[str, int], Path, Path]:
    type_dir = embedding_root / embedding_type
    embeddings_path = type_dir / "item_embeddings.npy"
    id_map_path = type_dir / "item_id_map.json"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {embeddings_path}")
    if not id_map_path.exists():
        raise FileNotFoundError(f"ID map file not found: {id_map_path}")

    embeddings_np = np.load(embeddings_path)
    if embeddings_np.ndim != 2:
        raise ValueError(f"item_embeddings.npy must be 2D, got shape={embeddings_np.shape}")
    embeddings = torch.from_numpy(embeddings_np).to(torch.float32)
    id_map_raw = load_json(id_map_path)
    id_map = {str(k): int(v) for k, v in id_map_raw.items()}
    return embeddings, id_map, embeddings_path, id_map_path


def process_train_dataset(
    dataset_path: Path,
    output_root: Path,
    raw_dir: Path,
    dataset: str,
    embedding_type: str,
    encoder: HFTextEncoder,
    batch_size: int,
    max_length: int,
    max_samples: int | None = None,
) -> dict[str, Any]:
    split = dataset_path.parent.name
    version = embedding_type
    output_dir = output_root / split / version

    rows = load_dataset_id_categories(dataset_path, max_samples=max_samples)
    if not rows:
        raise ValueError(f"No valid samples found: {dataset_path}")

    train_text_map, text_source_path = load_train_text_map(
        raw_dir=raw_dir, dataset=dataset, embedding_type=embedding_type
    )
    content_out_path = output_dir / f"{split}_{embedding_type}_item_content.json"

    valid_rows: list[dict[str, str]] = []
    train_texts: list[str] = []
    missing_ids: list[str] = []
    for row in tqdm(rows, desc=f"Matching train ids ({embedding_type})", leave=False):
        item_id = row["id"]
        text = train_text_map.get(item_id)
        if text is None:
            missing_ids.append(item_id)
            continue
        valid_rows.append(row)
        train_texts.append(text)

    if not valid_rows:
        raise ValueError(f"No sample IDs from {dataset_path} can be matched in {text_source_path}.")

    input_embeddings = encoder.encode(
        texts=train_texts,
        batch_size=batch_size,
        max_length=max_length,
        normalize_embeddings=True,
    )
    target_texts = [row["item_category"] for row in valid_rows]

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(input_embeddings, output_dir / "input_embeddings.pt")
    save_json({row["id"]: train_texts[i] for i, row in enumerate(valid_rows)}, content_out_path)

    records = []
    id_category_pairs = []
    for i, row in enumerate(valid_rows):
        records.append(
            {
                "index": i,
                "id": row["id"],
                "item_category": row["item_category"],
                "target_text": row["item_category"],
                "source_embedding_index": i,
            }
        )
        id_category_pairs.append([row["id"], row["item_category"]])

    save_jsonl(records, output_dir / "records.jsonl")
    save_json({"id_list": [row["id"] for row in valid_rows]}, output_dir / "id_list.json")
    save_json({"target_texts": target_texts}, output_dir / "target_texts.json")
    save_json({"pairs": id_category_pairs}, output_dir / "id_item_category.json")
    save_json({"pairs": [[row["id"], i] for i, row in enumerate(valid_rows)]}, output_dir / "id_embedding_index.json")
    torch.save(
        {"ids": [row["id"] for row in valid_rows], "embeddings": input_embeddings},
        output_dir / "id_embedding.pt",
    )

    # Save split-level version file expected by user:
    # train_v1[item_id] = [id, embedding] / train_v2[item_id] = [id, embedding]
    split_embed_pairs: dict[str, list[Any]] = {}
    for i, row in enumerate(valid_rows):
        split_embed_pairs[row["id"]] = [row["id"], input_embeddings[i].tolist()]
    split_version_path = dataset_path.parent / f"{dataset_path.stem}_{embedding_type}.json"
    save_json(split_embed_pairs, split_version_path)

    embedding_dim = int(input_embeddings.shape[1]) if input_embeddings.ndim == 2 else 0
    metadata = {
        "source_file": str(dataset_path),
        "split": split,
        "version": version,
        "embedding_type": embedding_type,
        "sample_count": len(valid_rows),
        "missing_id_count": len(missing_ids),
        "missing_ids_preview": missing_ids[:20],
        "embedding_dim": embedding_dim,
        "embedding_source_mode": "local_train_generation",
        "embedding_source_file": str(text_source_path),
        "train_item_content_path": str(content_out_path),
        "input_embeddings_path": str(output_dir / "input_embeddings.pt"),
        "id_embedding_path": str(output_dir / "id_embedding.pt"),
        "id_embedding_index_path": str(output_dir / "id_embedding_index.json"),
        "id_item_category_path": str(output_dir / "id_item_category.json"),
        "split_embedding_pairs_path": str(split_version_path),
        "records_path": str(output_dir / "records.jsonl"),
    }
    save_json(metadata, output_dir / "metadata.json")
    return metadata


def process_test_dataset(
    dataset_path: Path,
    output_root: Path,
    source_embeddings: torch.Tensor,
    source_id_map: dict[str, int],
    source_embeddings_path: Path,
    source_id_map_path: Path,
    embedding_type: str,
    max_samples: int | None = None,
) -> dict[str, Any]:
    split = dataset_path.parent.name
    version = embedding_type
    output_dir = output_root / split / version

    rows = load_dataset_id_categories(dataset_path, max_samples=max_samples)
    if not rows:
        raise ValueError(f"No valid samples found: {dataset_path}")

    valid_rows: list[dict[str, str]] = []
    selected_indices: list[int] = []
    missing_ids: list[str] = []
    for row in tqdm(rows, desc=f"Matching test ids ({embedding_type})", leave=False):
        item_id = row["id"]
        emb_idx = source_id_map.get(item_id)
        if emb_idx is None:
            missing_ids.append(item_id)
            continue
        if emb_idx < 0 or emb_idx >= len(source_embeddings):
            missing_ids.append(item_id)
            continue
        valid_rows.append(row)
        selected_indices.append(emb_idx)

    if not valid_rows:
        raise ValueError(
            f"No sample IDs from {dataset_path} can be matched in embedding map ({source_id_map_path})."
        )

    index_tensor = torch.tensor(selected_indices, dtype=torch.long)
    input_embeddings = source_embeddings[index_tensor]
    target_texts = [row["item_category"] for row in valid_rows]

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(input_embeddings, output_dir / "input_embeddings.pt")

    records = []
    id_category_pairs = []
    id_embedding_index_pairs = []
    for i, row in enumerate(valid_rows):
        records.append(
            {
                "index": i,
                "id": row["id"],
                "item_category": row["item_category"],
                "target_text": row["item_category"],
                "source_embedding_index": selected_indices[i],
            }
        )
        id_category_pairs.append([row["id"], row["item_category"]])
        id_embedding_index_pairs.append([row["id"], selected_indices[i]])

    save_jsonl(records, output_dir / "records.jsonl")
    save_json({"id_list": [row["id"] for row in valid_rows]}, output_dir / "id_list.json")
    save_json({"target_texts": target_texts}, output_dir / "target_texts.json")
    save_json({"pairs": id_category_pairs}, output_dir / "id_item_category.json")
    save_json({"pairs": id_embedding_index_pairs}, output_dir / "id_embedding_index.json")
    torch.save(
        {"ids": [row["id"] for row in valid_rows], "embeddings": input_embeddings},
        output_dir / "id_embedding.pt",
    )

    split_embed_pairs: dict[str, list[Any]] = {}
    for i, row in enumerate(valid_rows):
        split_embed_pairs[row["id"]] = [row["id"], input_embeddings[i].tolist()]
    split_version_path = dataset_path.parent / f"{dataset_path.stem}_{embedding_type}.json"
    save_json(split_embed_pairs, split_version_path)

    embedding_dim = int(input_embeddings.shape[1]) if input_embeddings.ndim == 2 else 0
    metadata = {
        "source_file": str(dataset_path),
        "split": split,
        "version": version,
        "embedding_type": embedding_type,
        "sample_count": len(valid_rows),
        "missing_id_count": len(missing_ids),
        "missing_ids_preview": missing_ids[:20],
        "embedding_dim": embedding_dim,
        "embedding_source_mode": "external_test_loading",
        "embedding_source_root": str(source_embeddings_path.parent),
        "embedding_source_file": str(source_embeddings_path),
        "embedding_id_map_file": str(source_id_map_path),
        "input_embeddings_path": str(output_dir / "input_embeddings.pt"),
        "id_embedding_path": str(output_dir / "id_embedding.pt"),
        "id_embedding_index_path": str(output_dir / "id_embedding_index.json"),
        "id_item_category_path": str(output_dir / "id_item_category.json"),
        "split_embedding_pairs_path": str(split_version_path),
        "records_path": str(output_dir / "records.jsonl"),
    }
    save_json(metadata, output_dir / "metadata.json")
    return metadata


def discover_datasets(process_dir: Path) -> list[Path]:
    candidates = []
    train_file = process_dir / "train" / "train.json"
    test_file = process_dir / "test" / "test.json"
    if train_file.exists():
        candidates.append(train_file)
    if test_file.exists():
        candidates.append(test_file)
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process [id, item_category] datasets and inject external embeddings."
    )
    parser.add_argument("--dataset", type=str, default="movie", choices=["movie", "book"])
    parser.add_argument("--process_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--raw_dir", type=str, default="")
    parser.add_argument("--embedding_root", type=str, default="")
    parser.add_argument("--embedding_type", type=str, default="v1")
    parser.add_argument(
        "--target_split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Choose which split to process. train uses local embedding generation; test loads external embeddings.",
    )
    parser.add_argument("--train_model_name_or_path", type=str, default=DEFAULT_TRAIN_EMBEDDING_MODEL)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Relative dataset paths under process_dir, e.g. train/train.json test/test.json. "
        "target_split always filters the selected paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_dir = (
        Path(args.process_dir)
        if args.process_dir
        else (BASE_DIR / "data" / args.dataset / "process")
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (BASE_DIR / "data" / args.dataset / "process" / "vector2text")
    )
    raw_dir = Path(args.raw_dir) if args.raw_dir else (BASE_DIR / "data" / args.dataset / "raw")
    embedding_root = (
        Path(args.embedding_root)
        if args.embedding_root
        else DEFAULT_EMBEDDING_ROOTS.get(args.dataset, DEFAULT_EMBEDDING_ROOTS["movie"])
    )

    if args.datasets:
        dataset_paths = [process_dir / rel_path for rel_path in args.datasets]
    else:
        dataset_paths = discover_datasets(process_dir)

    dataset_paths = [path for path in dataset_paths if path.exists()]
    dataset_paths = [path for path in dataset_paths if path.parent.name == args.target_split]
    if not dataset_paths:
        raise FileNotFoundError(
            f"No dataset json found in {process_dir} for target_split={args.target_split}"
        )

    train_encoder = None
    source_embeddings = None
    source_id_map = None
    source_embeddings_path = None
    source_id_map_path = None
    if any(path.parent.name == "train" for path in dataset_paths):
        train_encoder = HFTextEncoder(args.train_model_name_or_path)
    if any(path.parent.name == "test" for path in dataset_paths):
        source_embeddings, source_id_map, source_embeddings_path, source_id_map_path = load_external_embeddings(
            embedding_root=embedding_root,
            embedding_type=args.embedding_type,
        )

    summaries = []
    for dataset_path in tqdm(dataset_paths, desc="Datasets", leave=False):
        print(f"Processing {dataset_path} ...")
        if dataset_path.parent.name == "train":
            if train_encoder is None:
                raise RuntimeError("Train encoder is not initialized.")
            metadata = process_train_dataset(
                dataset_path=dataset_path,
                output_root=output_dir,
                raw_dir=raw_dir,
                dataset=args.dataset,
                embedding_type=args.embedding_type,
                encoder=train_encoder,
                batch_size=args.batch_size,
                max_length=args.max_length,
                max_samples=args.max_samples,
            )
        else:
            if (
                source_embeddings is None
                or source_id_map is None
                or source_embeddings_path is None
                or source_id_map_path is None
            ):
                raise RuntimeError("External test embedding source is not initialized.")
            metadata = process_test_dataset(
                dataset_path=dataset_path,
                output_root=output_dir,
                source_embeddings=source_embeddings,
                source_id_map=source_id_map,
                source_embeddings_path=source_embeddings_path,
                source_id_map_path=source_id_map_path,
                embedding_type=args.embedding_type,
                max_samples=args.max_samples,
            )
        summaries.append(metadata)
        print(
            f"Done: {metadata['split']}/{metadata['version']} -> "
            f"{metadata['sample_count']} samples, dim={metadata['embedding_dim']}"
        )

    summary_path = output_dir / "summary.json"
    save_json({"datasets": summaries}, summary_path)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
