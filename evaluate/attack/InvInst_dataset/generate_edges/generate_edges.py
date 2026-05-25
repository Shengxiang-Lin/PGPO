import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
PROMPT_DIR = BASE_DIR / "prompt"

DATASET_CONFIGS = {
    "movie": {
        "id_item_file": str(BASE_DIR.parent / "data" / "movie" / "movie.json"),
        "output_file": str(BASE_DIR.parent / "data" / "movie" / "movie_edges.json"),
        "prompt_name": "movielens",
    },
    "book": {
        "id_item_file": str(BASE_DIR.parent / "data" / "book" / "train.json"),
        "output_file": str(BASE_DIR.parent / "data" / "book" / "book_edges.json"),
        "prompt_name": "amazon-book",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate edge attributes from train item titles for movie/book."
    )
    parser.add_argument("id_item_file", nargs="?", default=None)
    parser.add_argument("output_file", nargs="?", default=None)
    parser.add_argument(
        "model_path",
        nargs="?",
        default="../../../../base_models/Qwen2.5-14B-Instruct",
    )
    parser.add_argument("prompt_file", nargs="?", default=None)
    parser.add_argument(
        "system_prompt",
        nargs="?",
        default="You are a professional knowledge graph and recommendation systems expert.",
    )
    parser.add_argument("--dataset", choices=["movie", "book"], default="movie")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--use_api_for_retry", action="store_true")
    parser.add_argument("--max_items", type=int, default=None)
    return parser.parse_args()


def load_prompt_template(prompt_file: str) -> str:
    path = Path(prompt_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    return path.read_text(encoding="utf-8")


def get_default_prompt_file(prompt_name: str) -> str:
    prompt_file = PROMPT_DIR / f"{prompt_name}.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Default prompt file not found for dataset prompt name '{prompt_name}': {prompt_file}"
        )
    return str(prompt_file)


def parse_response_to_json(raw_response: str) -> dict[str, Any]:
    json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON object found in response")
    return json.loads(json_match.group(0))


def load_id_name_map(id_item_file: str, max_items: int | None = None) -> list[tuple[str, str]]:
    with open(id_item_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"id_item file must be dict JSON, got {type(raw)}")

    records: list[tuple[str, str]] = []
    for item_id, payload in raw.items():
        item_name = None
        if isinstance(payload, str):
            item_name = payload.strip()
        elif isinstance(payload, list) and payload:
            # compatible with [title, category] format
            item_name = str(payload[0]).strip()
        elif isinstance(payload, dict) and "item_name" in payload:
            item_name = str(payload["item_name"]).strip()
        if not item_name:
            continue
        records.append((str(item_id), item_name))
        if max_items is not None and len(records) >= max_items:
            break
    return records


def generate_edges_with_api(
    client: Any, item_name: str, prompt_template: str, system_prompt: str
) -> dict[str, Any] | None:
    user_content = prompt_template.format(item_name=item_name)
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.7,
            max_tokens=512,
        )
        return parse_response_to_json(response.choices[0].message.content.strip())
    except Exception:
        return None


def decode_batch_responses(
    batch: list[tuple[str, str]],
    batch_responses: list[str],
    item_edges: dict[str, dict[str, Any]],
    failed_items: list[dict[str, Any]],
) -> None:
    for idx, response in enumerate(batch_responses):
        item_id, item_name = batch[idx]
        try:
            edge_info = parse_response_to_json(response)
            if edge_info.get("item_name") != item_name:
                edge_info["item_name"] = item_name
            item_edges[item_id] = edge_info
        except Exception as e:
            print(f"Failed to parse {item_name} (ID: {item_id}): {e}")
            failed_items.append({"id": item_id, "name": item_name, "response": response})


def run_local_generation_batch(
    batch: list[tuple[str, str]],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt_template: str,
    system_prompt: str,
    max_new_tokens: int,
) -> list[str]:
    prompts = []
    for _, item_name in batch:
        user_content = prompt_template.format(item_name=item_name)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    inputs = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated_tokens = outputs[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


def save_edges(item_edges: dict[str, dict[str, Any]], output_file: str) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(item_edges, f, ensure_ascii=False, indent=2)


def generate_edges(
    id_item_file: str,
    output_file: str,
    model_path: str,
    prompt_file: str,
    system_prompt: str,
    batch_size: int = 32,
    max_retries: int = 3,
    use_api_for_retry: bool = False,
    max_items: int | None = None,
) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Loading id-item file: {id_item_file}")
    items = load_id_name_map(id_item_file, max_items=max_items)
    print(f"Loaded valid items: {len(items)}")
    if not items:
        raise ValueError("No valid items found to process.")

    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
    print(f"Batch size: {batch_size}, total batches: {len(batches)}")

    print(f"Loading prompt template: {prompt_file}")
    prompt_template = load_prompt_template(prompt_file)

    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True,
    )
    print("Model loaded")

    item_edges: dict[str, dict[str, Any]] = {}
    failed_items: list[dict[str, Any]] = []
    save_edges(item_edges, output_file)

    print("Start batch edge generation...")
    for batch in tqdm(batches, desc="Generating"):
        responses = run_local_generation_batch(
            batch=batch,
            tokenizer=tokenizer,
            model=model,
            prompt_template=prompt_template,
            system_prompt=system_prompt,
            max_new_tokens=1024,
        )
        decode_batch_responses(
            batch=batch,
            batch_responses=responses,
            item_edges=item_edges,
            failed_items=failed_items,
        )
        save_edges(item_edges, output_file)

    print(f"Initial pass done: success={len(item_edges)}, failed={len(failed_items)}")

    api_client = None
    if use_api_for_retry and failed_items:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("DEEPSEEK_API_KEY not found, fallback to local retry.")
            use_api_for_retry = False
        else:
            api_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            print("API retry enabled.")

    retry_count = 0
    while failed_items and retry_count < max_retries:
        retry_count += 1
        print(f"\n--- Retry {retry_count}/{max_retries} ---")
        retry_items = [(x["id"], x["name"]) for x in failed_items]
        current_failed: list[dict[str, Any]] = []

        if use_api_for_retry and api_client:
            for item_id, item_name in tqdm(retry_items, desc="Retry API"):
                edge_info = generate_edges_with_api(api_client, item_name, prompt_template, system_prompt)
                if edge_info is None:
                    current_failed.append({"id": item_id, "name": item_name})
                    continue
                if edge_info.get("item_name") != item_name:
                    edge_info["item_name"] = item_name
                item_edges[item_id] = edge_info
        else:
            retry_batches = [retry_items[i : i + batch_size] for i in range(0, len(retry_items), batch_size)]
            for batch in tqdm(retry_batches, desc="Retry local"):
                responses = run_local_generation_batch(
                    batch=batch,
                    tokenizer=tokenizer,
                    model=model,
                    prompt_template=prompt_template,
                    system_prompt=system_prompt,
                    max_new_tokens=2048,
                )
                decode_batch_responses(
                    batch=batch,
                    batch_responses=responses,
                    item_edges=item_edges,
                    failed_items=current_failed,
                )

        failed_items = current_failed
        save_edges(item_edges, output_file)
        print(f"Retry done: success={len(item_edges)}, failed={len(failed_items)}")

    print("\n--- Final Result ---")
    print(f"Successfully generated: {len(item_edges)}")
    print(f"Remaining failed: {len(failed_items)}")
    if failed_items:
        print("Some items still failed. You can rerun with --max_items or --use_api_for_retry.")


if __name__ == "__main__":
    args = parse_args()
    cfg = DATASET_CONFIGS[args.dataset]
    id_item_file = args.id_item_file or cfg["id_item_file"]
    output_file = args.output_file or cfg["output_file"]
    prompt_file = args.prompt_file or get_default_prompt_file(cfg["prompt_name"])

    print(f"dataset={args.dataset}, id_item_file={id_item_file}")
    print(f"output_file={output_file}")

    generate_edges(
        id_item_file=id_item_file,
        output_file=output_file,
        model_path=args.model_path,
        prompt_file=prompt_file,
        system_prompt=args.system_prompt,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        use_api_for_retry=args.use_api_for_retry,
        max_items=args.max_items,
    )
