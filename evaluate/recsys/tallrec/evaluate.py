import sys
import os

import fire
import torch
torch.set_num_threads(1)
import transformers
import json
from typing import List
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np

from models import EdgeEmbeddingProjector, EdgeEmbeddingCollator, SimpleCollator, inject_edge_embeddings

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


def main(
    load_8bit: bool = True,
    base_model: str = "../../../base_models/Qwen2.5-7B-Instruct",
    dataset_name: str = "ml-1m",
    data_root: str = "./data",
    embedding_root: str = "../../../embedding/data",
    embedding_version: str = "v5",
    lora_weights: str = "",
    test_data_path: str = "",
    edge_embeddings_dir: str = "",
    result_json_data: str = "",
    batch_size: int = 32,           
    use_edge_embeddings: bool = True,
    use_compile: bool = False,
    hide_item_ids_in_input: bool = True
):
    if not lora_weights:
        lora_weights = os.path.join("./output", f"{dataset_name}_{embedding_version}")
    if not test_data_path:
        test_data_path = os.path.join(data_root, dataset_name, "test.json")
    if not result_json_data:
        result_json_data = os.path.join("./result", f"{dataset_name}_{embedding_version}.json")
    if use_edge_embeddings and not edge_embeddings_dir:
        edge_embeddings_dir = os.path.join(embedding_root, dataset_name, embedding_version)

    assert base_model, "Please specify a --base_model"
    print(
        f"dataset_name={dataset_name}, embedding_version={embedding_version}, "
        f"use_edge_embeddings={use_edge_embeddings}, "
        f"hide_item_ids_in_input={hide_item_ids_in_input}"
    )

    torch.set_grad_enabled(False)
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"Loading tokenizer from {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading model from {base_model}...")
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, 
            device_map={"": device}, 
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    if not load_8bit:
        model.half()

    model.eval()
    if use_compile and torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    projector = None
    collator = SimpleCollator(tokenizer=tokenizer, include_output=False, pure_id_mode=True)
    if use_edge_embeddings:
        print("Loading edge embedding projector...")
        item_embeddings_path = os.path.join(edge_embeddings_dir, "item_embeddings.npy")
        if not os.path.exists(item_embeddings_path):
            raise FileNotFoundError(f"item_embeddings not found: {item_embeddings_path}")
        sample_embedding = np.load(item_embeddings_path)
        if sample_embedding.ndim != 2 or sample_embedding.shape[0] == 0:
            raise ValueError(f"invalid embedding matrix shape: {sample_embedding.shape}")
        actual_dim = int(sample_embedding.shape[1])
        print(f"Detected edge embedding dimension: {actual_dim}")
        
        projector = EdgeEmbeddingProjector(
            edge_embedding_dim=actual_dim,
            llm_hidden_size=model.config.hidden_size,
        )
        
        projector_path = os.path.join(lora_weights, "projector.pt")
        if os.path.exists(projector_path):
            projector.load_state_dict(torch.load(projector_path, map_location=device))
            print(f"Loaded projector from {projector_path}")
        else:
            print(f"Warning: Projector not found at {projector_path}, using random initialization")
        
        projector.to(device)
        projector.eval()
        
        collator = EdgeEmbeddingCollator(
            tokenizer=tokenizer,
            edge_embeddings_dir=edge_embeddings_dir,
            projector=projector,
            include_output=False,
            hide_item_ids_in_input=hide_item_ids_in_input,
        )

    def evaluate(
        batch_data,
        temperature=0.7,
        top_p=1.0,
        top_k=40,
        num_beams=4,
        max_new_tokens=4,
        **kwargs,
    ):
        collated = collator(batch_data)
        input_ids = collated["input_ids"].to(device)
        attention_mask = collated.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        edge_item_ids = collated.get("edge_item_ids", None)
        
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        
        with torch.no_grad():
            inputs_embeds = None
            if use_edge_embeddings and projector is not None:
                inputs_embeds = inject_edge_embeddings(
                    model=model,
                    input_ids=input_ids,
                    edge_item_ids=edge_item_ids,
                    projector=projector,
                    movie_token_id=tokenizer.convert_tokens_to_ids("[MOVIE]"),
                    sanitize_projected=True,
                )

            if inputs_embeds is not None:
                model_outputs = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            else:
                model_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )

            next_token_logits = model_outputs.logits[:, -1, :]
            yes_token = tokenizer.encode("Yes.", add_special_tokens=False)[0]
            no_token = tokenizer.encode("No.", add_special_tokens=False)[0]
            logits = torch.softmax(next_token_logits[:, [no_token, yes_token]], dim=-1)
            logits = torch.nan_to_num(logits, nan=0.5)

            if inputs_embeds is not None:
                generation_output = model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=False,
                    max_new_tokens=max_new_tokens,
                )
            else:
                generation_output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=False,
                    max_new_tokens=max_new_tokens,
                )
        
        s = generation_output.sequences
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
        output = [_.split('Response:\n')[-1].strip() for _ in output]
        
        return output, logits.tolist()
    # =========================================================================================

    print(f"Loading test data from {test_data_path}...")
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    def batch(list_data, batch_size=32):
        for i in range(0, len(list_data), batch_size):
            yield list_data[i:i + batch_size]
    
    outputs = []
    logits = []
    
    print("Evaluating...")
    for batch_items in tqdm(batch(test_data, batch_size)):
        batch_outputs, batch_logits = evaluate(batch_items)
        outputs += batch_outputs
        logits += batch_logits
        if device == "cuda":
            torch.cuda.empty_cache()
    
    for i, test in enumerate(test_data):
        test_data[i]['predict'] = outputs[i]
        test_data[i]['logits'] = logits[i]
    
    pred = [logit[1] for logit in logits]
    pred_array = np.array(pred, dtype=np.float64)
    nan_count = np.sum(np.isnan(pred_array))
    if nan_count > 0:
        print(f"Warning: Found {nan_count} NaN predictions → set to 0.5")
    pred_array = np.nan_to_num(pred_array, nan=0.5)
    
    gold = [1 if _['output'] == "Yes." else 0 for _ in test_data]
    auc = roc_auc_score(gold, pred_array)
    print(f"\nAUC Score: {auc:.4f}")
    
    result_dir = os.path.dirname(result_json_data)
    if result_dir and not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    result = {'auc': auc, 'predictions': test_data}
    
    with open(result_json_data, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"Results saved to {result_json_data}")
    return auc


if __name__ == "__main__":
    fire.Fire(main)
