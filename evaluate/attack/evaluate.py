import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer


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
    parser = argparse.ArgumentParser(description="Evaluate inversion using official vec2text API.")
    parser.add_argument("--dataset", type=str, default="movie", choices=["movie", "book"])
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--eval_mode", type=str, default="local", choices=["local", "vec2text"])
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--corrector", type=str, default="text-embedding-ada-002")
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--sequence_beam_width", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--save_predictions", action="store_true")
    return parser.parse_args()


def compute_metrics(predictions: list[str], targets: list[str]) -> dict[str, Any]:
    if len(predictions) != len(targets):
        raise ValueError(f"Length mismatch: {len(predictions)} vs {len(targets)}")

    exact = 0
    similarities = []
    set_exact = 0
    overlap_precision_sum = 0.0
    overlap_recall_sum = 0.0
    overlap_f1_sum = 0.0
    jaccard_sum = 0.0

    def to_label_set(text: str) -> set[str]:
        labels = [part.strip().lower() for part in text.split("|")]
        return {label for label in labels if label}

    iterator = zip(predictions, targets)
    for pred, target in tqdm(iterator, total=len(predictions), desc="Computing metrics"):
        pred_norm = pred.strip()
        target_norm = target.strip()
        if pred_norm.lower() == target_norm.lower():
            exact += 1
        similarities.append(SequenceMatcher(a=pred_norm.lower(), b=target_norm.lower()).ratio())

        pred_set = to_label_set(pred_norm)
        target_set = to_label_set(target_norm)
        inter = len(pred_set & target_set)
        union = len(pred_set | target_set)
        if pred_set == target_set:
            set_exact += 1

        precision = inter / len(pred_set) if len(pred_set) > 0 else 0.0
        recall = inter / len(target_set) if len(target_set) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        jaccard = inter / union if union > 0 else 0.0

        overlap_precision_sum += precision
        overlap_recall_sum += recall
        overlap_f1_sum += f1
        jaccard_sum += jaccard

    n = len(predictions)
    sim_tensor = torch.tensor(similarities, dtype=torch.float32) if similarities else torch.tensor([], dtype=torch.float32)
    return {
        "num_samples": n,
        "exact_match_accuracy": float(exact / n) if n > 0 else 0.0,
        "edit_similarity_mean": float(sim_tensor.mean().item()) if n > 0 else 0.0,
        "edit_similarity_median": float(sim_tensor.median().item()) if n > 0 else 0.0,
        "label_set_exact_match_accuracy": float(set_exact / n) if n > 0 else 0.0,
        "label_overlap_precision_mean": float(overlap_precision_sum / n) if n > 0 else 0.0,
        "label_overlap_recall_mean": float(overlap_recall_sum / n) if n > 0 else 0.0,
        "label_overlap_f1_mean": float(overlap_f1_sum / n) if n > 0 else 0.0,
        "label_jaccard_mean": float(jaccard_sum / n) if n > 0 else 0.0,
    }


def invert_with_vec2text(
    embeddings: torch.Tensor,
    corrector_name: str,
    num_steps: int,
    sequence_beam_width: int,
    batch_size: int,
) -> list[str]:
    try:
        import vec2text
    except Exception as e:
        raise ImportError("vec2text is required. Please install with `pip install vec2text`.") from e

    corrector = vec2text.load_pretrained_corrector(corrector_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictions: list[str] = []
    embeddings = embeddings.to(device)

    for i in tqdm(range(0, len(embeddings), batch_size), desc="Inverting with vec2text"):
        batch = embeddings[i : i + batch_size]
        kwargs = {"embeddings": batch, "corrector": corrector, "num_steps": num_steps}
        if sequence_beam_width > 0:
            kwargs["sequence_beam_width"] = sequence_beam_width
        batch_predictions = vec2text.invert_embeddings(**kwargs)
        predictions.extend(batch_predictions)
    return predictions


def resolve_model_dir(model_dir: str, split: str, version: str, dataset: str) -> Path:
    if model_dir:
        path = Path(model_dir)
    else:
        path = BASE_DIR / "data" / dataset / "models" / "vec2text" / split / version / "best"
    if path.is_dir() and (path / "best").exists():
        path = path / "best"
    required_files = [path / "base_model", path / "projector.pt", path / "inversion_config.json"]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Local model files missing: {missing}")
    return path


def invert_with_local_model(
    embeddings: torch.Tensor,
    model_dir: Path,
    batch_size: int,
    max_new_tokens: int,
    num_beams: int,
) -> list[str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained(model_dir / "base_model")
    model = T5ForConditionalGeneration.from_pretrained(model_dir / "base_model").to(device)
    model.eval()

    inversion_config = load_json(model_dir / "inversion_config.json")
    num_repeat_tokens = int(inversion_config["num_repeat_tokens"])

    projector_state = torch.load(model_dir / "projector.pt", map_location="cpu")
    out_dim, in_dim = projector_state["weight"].shape
    projector = nn.Linear(in_dim, out_dim).to(device)
    projector.load_state_dict(projector_state)
    projector.eval()

    predictions: list[str] = []
    for i in tqdm(range(0, len(embeddings), batch_size), desc="Inverting with local model"):
        batch = embeddings[i : i + batch_size].to(device).float()
        with torch.no_grad():
            projected = projector(batch)
            projected = projected.view(batch.size(0), num_repeat_tokens, model.config.d_model)
            attention_mask = torch.ones(projected.size(0), projected.size(1), device=device, dtype=torch.long)
            generated = model.generate(
                inputs_embeds=projected,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return predictions


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
        else (DEFAULT_DATA_DIR / args.dataset / "models")
    )
    source_dir = input_dir / args.split / args.version
    embeddings = torch.load(source_dir / "input_embeddings.pt", map_location="cpu")
    targets_data = load_json(source_dir / "target_texts.json")
    targets = targets_data["target_texts"]

    if args.eval_mode == "vec2text":
        predictions = invert_with_vec2text(
            embeddings=embeddings,
            corrector_name=args.corrector,
            num_steps=args.num_steps,
            sequence_beam_width=args.sequence_beam_width,
            batch_size=args.batch_size,
        )
        eval_backend = args.corrector
    else:
        model_dir = resolve_model_dir(args.model_dir, args.split, args.version, args.dataset)
        predictions = invert_with_local_model(
            embeddings=embeddings,
            model_dir=model_dir,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
        eval_backend = str(model_dir)

    metrics = compute_metrics(predictions, targets)

    result = {
        "source_dir": str(source_dir),
        "eval_mode": args.eval_mode,
        "backend": eval_backend,
        "num_steps": args.num_steps,
        "sequence_beam_width": args.sequence_beam_width,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "metrics": metrics,
    }

    if args.save_predictions:
        records = load_json(source_dir / "id_list.json")["id_list"]
        result["predictions"] = [
            {"id": records[i], "prediction": predictions[i], "target": targets[i]}
            for i in range(len(predictions))
        ]

    output_path = output_dir / f"eval_{args.eval_mode}_{args.dataset}_{args.split}_{args.version}.json"
    save_json(result, output_path)
    print(f"Saved evaluation result: {output_path}")
    print(f"Exact Match: {metrics['exact_match_accuracy']:.4f}")
    print(f"Edit Similarity Mean: {metrics['edit_similarity_mean']:.4f}")


if __name__ == "__main__":
    main()
