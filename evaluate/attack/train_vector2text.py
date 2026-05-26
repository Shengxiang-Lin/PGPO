import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Adafactor, T5ForConditionalGeneration, T5Tokenizer


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare data and train vec2text-style inversion model.")
    parser.add_argument("--dataset", type=str, default="movie", choices=["movie", "book"])
    parser.add_argument("--stage", type=str, default="all", choices=["prepare", "train", "all"])
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_embeddings_in_jsonl", action="store_true")

    parser.add_argument("--model_name_or_path", type=str, default="../../base_models/t5-base")
    parser.add_argument("--num_repeat_tokens", type=int, default=8)
    parser.add_argument("--max_target_length", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adafactor", choices=["adamw", "adafactor"])
    parser.add_argument("--train_dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--train_projector_only", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--tokenizer_legacy", action="store_true")
    parser.add_argument("--model_output_dir", type=str, default="")
    return parser.parse_args()


def load_source_records(input_root: Path, split: str, version: str) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    source_dir = input_root / split / version
    embeddings = torch.load(source_dir / "input_embeddings.pt", map_location="cpu")
    records = []
    with (source_dir / "records.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "target_text" not in row and "item_category" in row:
                row["target_text"] = row["item_category"]
            records.append(row)
    if len(embeddings) != len(records):
        raise ValueError(f"Embeddings/records mismatch: {len(embeddings)} vs {len(records)}")
    return embeddings, records


def split_indices(total: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = int(total * val_ratio)
    if val_size <= 0:
        return indices, []
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def write_split_jsonl(
    path: Path,
    split_name: str,
    records: list[dict[str, Any]],
    indices: list[int],
    embeddings: torch.Tensor,
    include_embeddings_in_jsonl: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx in tqdm(indices, desc=f"Writing {split_name}.jsonl"):
            row = records[idx]
            output_row = {
                "id": row["id"],
                "text": row["target_text"],
                "embedding_index": idx,
            }
            if include_embeddings_in_jsonl:
                output_row["embedding"] = embeddings[idx].tolist()
            f.write(json.dumps(output_row, ensure_ascii=False))
            f.write("\n")


def prepare_dataset(args: argparse.Namespace) -> Path:
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("val_ratio must satisfy 0.0 <= val_ratio < 1.0")

    print(f"Loading source data: {input_root / args.split / args.version}")
    embeddings, records = load_source_records(input_root, args.split, args.version)
    print(f"Loaded samples: {len(records)}, embedding dim: {embeddings.shape[1]}")

    print("Splitting train/val...")
    train_indices, val_indices = split_indices(len(records), args.val_ratio, args.seed)
    save_dir = output_root / args.split / args.version

    write_split_jsonl(
        path=save_dir / "train.jsonl",
        split_name="train",
        records=records,
        indices=train_indices,
        embeddings=embeddings,
        include_embeddings_in_jsonl=args.include_embeddings_in_jsonl,
    )
    write_split_jsonl(
        path=save_dir / "val.jsonl",
        split_name="val",
        records=records,
        indices=val_indices,
        embeddings=embeddings,
        include_embeddings_in_jsonl=args.include_embeddings_in_jsonl,
    )

    print("Saving split embeddings...")
    train_embeddings = embeddings[torch.tensor(train_indices, dtype=torch.long)] if train_indices else torch.empty((0, embeddings.shape[1]))
    val_embeddings = embeddings[torch.tensor(val_indices, dtype=torch.long)] if val_indices else torch.empty((0, embeddings.shape[1]))
    torch.save(train_embeddings, save_dir / "train_embeddings.pt")
    torch.save(val_embeddings, save_dir / "val_embeddings.pt")

    train_ids = [records[i]["id"] for i in train_indices]
    val_ids = [records[i]["id"] for i in val_indices]
    train_categories = [records[i]["target_text"] for i in train_indices]
    val_categories = [records[i]["target_text"] for i in val_indices]

    save_json({"target_texts": train_categories}, save_dir / "train_texts.json")
    save_json({"target_texts": val_categories}, save_dir / "val_texts.json")
    save_json({"id_list": train_ids}, save_dir / "train_ids.json")
    save_json({"id_list": val_ids}, save_dir / "val_ids.json")
    save_json({"pairs": [[train_ids[i], train_categories[i]] for i in range(len(train_ids))]}, save_dir / "train_id_item_category.json")
    save_json({"pairs": [[val_ids[i], val_categories[i]] for i in range(len(val_ids))]}, save_dir / "val_id_item_category.json")
    torch.save({"ids": train_ids, "embeddings": train_embeddings}, save_dir / "train_id_embedding.pt")
    torch.save({"ids": val_ids, "embeddings": val_embeddings}, save_dir / "val_id_embedding.pt")

    metadata = {
        "source_input_dir": str(input_root / args.split / args.version),
        "output_dir": str(save_dir),
        "split": args.split,
        "version": args.version,
        "total_samples": len(records),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "official_vec2text_reference": "https://github.com/vec2text/vec2text",
    }
    save_json(metadata, save_dir / "dataset_metadata.json")
    print(f"Prepared vec2text dataset at: {save_dir}")
    print(f"Train/Val: {len(train_indices)}/{len(val_indices)}")
    return save_dir


class VectorTextDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, texts: list[str]) -> None:
        if len(embeddings) != len(texts):
            raise ValueError(f"Embeddings/texts mismatch: {len(embeddings)} vs {len(texts)}")
        self.embeddings = embeddings
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        return self.embeddings[idx], self.texts[idx]


class T5InversionModel(nn.Module):
    def __init__(self, model_name_or_path: str, input_dim: int, num_repeat_tokens: int) -> None:
        super().__init__()
        self.base_model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.num_repeat_tokens = num_repeat_tokens
        hidden_size = self.base_model.config.d_model
        self.projector = nn.Linear(input_dim, hidden_size * num_repeat_tokens)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        projected = self.projector(embeddings)
        projected = projected.view(embeddings.size(0), self.num_repeat_tokens, self.base_model.config.d_model)
        attention_mask = torch.ones(projected.size(0), projected.size(1), device=projected.device, dtype=torch.long)
        outputs = self.base_model(
            inputs_embeds=projected,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss

    def save(self, output_dir: Path, tokenizer: T5Tokenizer) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.base_model.save_pretrained(output_dir / "base_model")
        tokenizer.save_pretrained(output_dir / "base_model")
        torch.save(self.projector.state_dict(), output_dir / "projector.pt")
        save_json({"num_repeat_tokens": self.num_repeat_tokens}, output_dir / "inversion_config.json")


def collate_fn(tokenizer: T5Tokenizer, max_target_length: int):
    def _inner(batch: list[tuple[torch.Tensor, str]]) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = torch.stack([item[0] for item in batch])
        texts = [item[1] for item in batch]
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_target_length,
            return_tensors="pt",
        )
        labels = tokenized["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        return embeddings, labels

    return _inner


def evaluate_epoch(model: T5InversionModel, dataloader: DataLoader, device: str) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for embeddings, labels in tqdm(dataloader, desc="Validating", leave=False):
            embeddings = embeddings.to(device).float()
            labels = labels.to(device)
            loss = model(embeddings, labels)
            total_loss += float(loss.item())
    return total_loss / max(len(dataloader), 1)


def get_autocast_dtype(train_dtype: str) -> torch.dtype | None:
    if train_dtype == "fp16":
        return torch.float16
    if train_dtype == "bf16":
        return torch.bfloat16
    return None


def train_model(prepared_dir: Path, args: argparse.Namespace) -> Path:
    train_embeddings = torch.load(prepared_dir / "train_embeddings.pt", map_location="cpu")
    val_embeddings = torch.load(prepared_dir / "val_embeddings.pt", map_location="cpu")
    train_texts = load_json(prepared_dir / "train_texts.json")["target_texts"]
    val_texts = load_json(prepared_dir / "val_texts.json")["target_texts"]
    if len(train_embeddings) == 0:
        raise ValueError("No training samples found in prepared dataset.")

    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path, legacy=args.tokenizer_legacy)
    model = T5InversionModel(
        model_name_or_path=args.model_name_or_path,
        input_dim=int(train_embeddings.shape[1]),
        num_repeat_tokens=args.num_repeat_tokens,
    )
    if args.gradient_checkpointing:
        model.base_model.gradient_checkpointing_enable()
    if args.train_projector_only:
        for p in model.base_model.parameters():
            p.requires_grad = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    device_type = "cuda" if device == "cuda" else "cpu"
    autocast_dtype = get_autocast_dtype(args.train_dtype)
    use_amp = device == "cuda" and autocast_dtype is not None
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and args.train_dtype == "fp16"))

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "ALL")
    print(f"CUDA_VISIBLE_DEVICES={visible_devices}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Visible CUDA device {i}: {torch.cuda.get_device_name(i)}")
        print(f"Current CUDA device index: {torch.cuda.current_device()}")

    train_dataset = VectorTextDataset(train_embeddings, train_texts)
    val_dataset = VectorTextDataset(val_embeddings, val_texts)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn(tokenizer, args.max_target_length),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn(tokenizer, args.max_target_length),
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adafactor":
        optimizer = Adafactor(
            trainable_params,
            lr=args.learning_rate,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    model_dir = Path(args.model_output_dir) / "vec2text" / args.split / args.version
    best_dir = model_dir / "best"
    last_dir = model_dir / "last"
    model_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    history = []
    print(
        f"Start training on {device}: epochs={args.epochs}, batch_size={args.batch_size}, "
        f"optimizer={args.optimizer}, dtype={args.train_dtype}, grad_accum={args.gradient_accumulation_steps}, "
        f"projector_only={args.train_projector_only}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)
        for step_idx, (embeddings, labels) in enumerate(progress, start=1):
            embeddings = embeddings.to(device).float()
            labels = labels.to(device)
            with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=use_amp):
                raw_loss = model(embeddings, labels)
                loss = raw_loss / max(args.gradient_accumulation_steps, 1)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if step_idx % max(args.gradient_accumulation_steps, 1) == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            train_loss_sum += float(raw_loss.item())
            progress.set_postfix({"loss": f"{raw_loss.item():.4f}"})

        if len(train_loader) % max(args.gradient_accumulation_steps, 1) != 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_loss = train_loss_sum / max(len(train_loader), 1)
        val_loss = evaluate_epoch(model, val_loader, device) if len(val_dataset) > 0 else train_loss
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(best_dir, tokenizer)
            save_json({"best_val_loss": best_val_loss, "epoch": epoch}, best_dir / "best_metric.json")

    model.save(last_dir, tokenizer)
    save_json({"history": history}, model_dir / "train_history.json")
    save_json(
        {
            "prepared_dir": str(prepared_dir),
            "model_name_or_path": args.model_name_or_path,
            "num_repeat_tokens": args.num_repeat_tokens,
            "max_target_length": args.max_target_length,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "best_val_loss": best_val_loss,
        },
        model_dir / "train_config.json",
    )
    print(f"Training finished. Best model: {best_dir}")
    return best_dir


def main() -> None:
    args = parse_args()
    input_dir = (
        Path(args.input_dir)
        if args.input_dir
        else (DEFAULT_DATA_DIR / args.dataset / "process" / "vector2text")
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (DEFAULT_DATA_DIR / args.dataset / "vec2text_ready")
    )
    model_output_dir = (
        Path(args.model_output_dir)
        if args.model_output_dir
        else (DEFAULT_DATA_DIR / args.dataset / "models")
    )
    args.input_dir = str(input_dir)
    args.output_dir = str(output_dir)
    args.model_output_dir = str(model_output_dir)
    prepared_dir = output_dir / args.split / args.version

    if args.stage in {"prepare", "all"}:
        prepared_dir = prepare_dataset(args)

    if args.stage in {"train", "all"}:
        if not prepared_dir.exists():
            raise FileNotFoundError(f"Prepared dataset directory not found: {prepared_dir}")
        train_model(prepared_dir, args)


if __name__ == "__main__":
    main()
