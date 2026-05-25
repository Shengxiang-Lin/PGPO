import re
import random


PROMPT_PREFIX = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
RESPONSE_PREFIX = "### Response:"


def clean_plain_id_input(input_text: str) -> str:
    input_text = re.sub(r"\s*\[MOVIE\]", "", input_text)
    input_text = re.sub(r"\s*\[\]", "", input_text)
    return input_text


def clean_embedded_id_input(input_text: str) -> str:
    # Keep the [MOVIE] placeholder but remove explicit item ids from the prompt text.
    input_text = re.sub(r'"[^"\n]*\s*\[MOVIE\]"', '"[MOVIE]"', input_text)
    input_text = re.sub(r"\s{2,}", " ", input_text)
    return input_text


NO_EMBEDDING_TOKEN = "__NO_EMBEDDING__"


def build_prompt(instruction: str, input_text: str, output_text=None) -> str:
    if output_text is None:
        return f"""{PROMPT_PREFIX}

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    return f"""{PROMPT_PREFIX}

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output_text}"""


def apply_response_only_labels(tokenizer, tokenized, texts):
    labels = tokenized["input_ids"].clone()
    labels = labels.masked_fill(tokenized["input_ids"] == tokenizer.pad_token_id, -100)

    response_prefix_tokens = tokenizer.encode(RESPONSE_PREFIX, add_special_tokens=False)
    response_prefix_with_newline_tokens = tokenizer.encode(RESPONSE_PREFIX + "\n", add_special_tokens=False)
    for i, text in enumerate(texts):
        if RESPONSE_PREFIX not in text:
            continue
        prefix_end_idx = None
        for j in range(len(tokenized["input_ids"][i])):
            if j + len(response_prefix_tokens) <= len(tokenized["input_ids"][i]):
                if tokenized["input_ids"][i, j:j + len(response_prefix_tokens)].tolist() == response_prefix_tokens:
                    prefix_end_idx = j + len(response_prefix_tokens)
                    break
            if prefix_end_idx is None and j + len(response_prefix_with_newline_tokens) <= len(tokenized["input_ids"][i]):
                if tokenized["input_ids"][i, j:j + len(response_prefix_with_newline_tokens)].tolist() == response_prefix_with_newline_tokens:
                    prefix_end_idx = j + len(response_prefix_with_newline_tokens)
                    break

        if prefix_end_idx is not None:
            labels[i, :prefix_end_idx] = -100
    return labels


class EdgeEmbeddingCollator:
    def __init__(
        self,
        tokenizer,
        edge_embeddings_dir: str,
        projector,
        train: bool = True,
        max_steps: int = 1000,
        include_output: bool = True,
        hide_item_ids_in_input: bool = True,
        embedding_keep_schedule: str = "none",
        embedding_keep_prob_start: float = 1.0,
        embedding_keep_prob_end: float = 1.0,
    ):
        self.tokenizer = tokenizer
        self.edge_embeddings_dir = edge_embeddings_dir
        self.projector = projector
        self.train = train
        self.max_steps = max_steps
        self.include_output = include_output
        self.hide_item_ids_in_input = hide_item_ids_in_input
        self.embedding_keep_schedule = embedding_keep_schedule
        self.embedding_keep_prob_start = float(embedding_keep_prob_start)
        self.embedding_keep_prob_end = float(embedding_keep_prob_end)
        self._step_count = 0

        self.edge_embeddings = projector.load_edge_embeddings(embeddings_dir=edge_embeddings_dir)

        self.movie_token = "[MOVIE]"
        if self.movie_token not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [self.movie_token]})
        self.movie_token_id = tokenizer.convert_tokens_to_ids(self.movie_token)

    def _resolve_embedding_keep_prob(self) -> float:
        if self.train and self.embedding_keep_schedule == "linear":
            denom = max(1, self.max_steps - 1)
            progress = min(1.0, self._step_count / denom)
            keep_prob = self.embedding_keep_prob_start + (
                self.embedding_keep_prob_end - self.embedding_keep_prob_start
            ) * progress
            keep_prob = max(0.0, min(1.0, keep_prob))
            self._step_count += 1
            return keep_prob
        return max(0.0, min(1.0, self.embedding_keep_prob_end))

    def __call__(self, batch):
        texts = []
        history_movie_ids = []
        target_movie_ids = []
        edge_item_ids_list = []
        embedding_keep_prob = self._resolve_embedding_keep_prob()

        for item in batch:
            instruction = item["instruction"]
            input_text = item["input"]
            if self.hide_item_ids_in_input:
                input_text = clean_embedded_id_input(input_text)
            preference_ids = [str(x) for x in item.get("preference_ids", [])]
            unpreference_ids = [str(x) for x in item.get("unpreference_ids", [])]
            target_id = str(item.get("target_movie_id", ""))

            history_movie_ids.append(preference_ids + unpreference_ids)
            target_movie_ids.append(target_id)

            edge_item_ids = preference_ids + unpreference_ids + [target_id]
            if self.train and embedding_keep_prob < 1.0:
                edge_item_ids = [
                    edge_id if random.random() <= embedding_keep_prob else NO_EMBEDDING_TOKEN
                    for edge_id in edge_item_ids
                ]
            output_text = item.get("output", "") if self.include_output else None
            full_text = build_prompt(instruction=instruction, input_text=input_text, output_text=output_text)

            texts.append(full_text)
            edge_item_ids_list.append(edge_item_ids)

        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        if self.include_output:
            tokenized["labels"] = apply_response_only_labels(self.tokenizer, tokenized, texts)

        movie_positions = tokenized["input_ids"] == self.movie_token_id
        for i in range(len(edge_item_ids_list)):
            token_count = int(movie_positions[i].sum().item())
            edge_ids = edge_item_ids_list[i]
            if len(edge_ids) > token_count:
                edge_item_ids_list[i] = edge_ids[:token_count]
            elif len(edge_ids) < token_count:
                pad_id = edge_ids[-1] if edge_ids else "0"
                edge_item_ids_list[i] = edge_ids + [pad_id] * (token_count - len(edge_ids))

        return {
            **tokenized,
            "history_movie_ids": history_movie_ids,
            "target_movie_ids": target_movie_ids,
            "edge_item_ids": edge_item_ids_list,
        }


class SimpleCollator:
    def __init__(self, tokenizer, include_output: bool = True, pure_id_mode: bool = False):
        self.tokenizer = tokenizer
        self.include_output = include_output
        self.pure_id_mode = pure_id_mode

    def __call__(self, batch):
        texts = []
        for item in batch:
            input_text = item["input"]
            if self.pure_id_mode:
                input_text = clean_plain_id_input(input_text)
            output_text = item.get("output", "") if self.include_output else None
            full_text = build_prompt(
                instruction=item["instruction"],
                input_text=input_text,
                output_text=output_text,
            )
            texts.append(full_text)

        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        if self.include_output:
            tokenized["labels"] = apply_response_only_labels(self.tokenizer, tokenized, texts)

        return tokenized
