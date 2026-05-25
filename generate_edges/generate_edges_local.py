import argparse
import json
import os
import re 
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from openai import OpenAI

DATASET_CONFIGS = {
    'ml-1m': {
        'id_item_file': './data/ml-1m/handled/id_item.json',
        'output_file': './data/ml-1m/handled/item_edges.json',
        'prompt_dataset': 'movielens'
    },
    'amazon-book': {
        'id_item_file': './data/amazon-book/handled/id_item.json',
        'output_file': './data/amazon-book/handled/item_edges.json',
        'prompt_dataset': 'amazon-book'
    }
}

def load_prompt_template(prompt_file):
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

def get_default_prompt_file(dataset_name):
    prompt_dir = './generate_edges/prompt'
    prompt_file = os.path.join(prompt_dir, f'{dataset_name}.txt')
    if os.path.exists(prompt_file):
        return prompt_file
    else:
        raise FileNotFoundError(f"Default prompt file not found for dataset: {dataset_name}. Expected: {prompt_file}")

def detect_dataset_from_file(file_path):
    lower_path = file_path.lower()
    if 'amazon-book' in lower_path or 'books' in lower_path:
        return 'amazon-book'
    if 'ml-1m' in lower_path or 'ml-100k' in lower_path or 'movielens' in lower_path:
        return 'ml-1m'
    if 'movie' in lower_path:
        return 'ml-1m'
    return 'amazon-book'

def resolve_prompt_dataset(dataset_name):
    if dataset_name in DATASET_CONFIGS:
        return DATASET_CONFIGS[dataset_name]['prompt_dataset']
    if dataset_name == 'movielens':
        return 'movielens'
    else:
        return dataset_name

def generate_edges_with_api(client, item_name, prompt_template, system_prompt):
    """使用DeepSeek API生成边信息"""
    user_content = prompt_template.format(item_name=item_name)
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.7,
            max_tokens=512
        )
        
        content = response.choices[0].message.content.strip()
        
        # 尝试解析JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 尝试提取JSON部分
            if not content.startswith('{'):
                start_idx = content.find('{')
                if start_idx != -1:
                    content = content[start_idx:]
            if not content.endswith('}'):
                end_idx = content.rfind('}') + 1
                if end_idx != -1:
                    content = content[:end_idx]
            return json.loads(content)
    except Exception:
        return None

def generate_edges(id_item_file, output_file, model_path, prompt_file, system_prompt, batch_size=32, max_retries=3, use_api_for_retry=True):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Loading the id_item.json file: {id_item_file}")
    with open(id_item_file, 'r', encoding='utf-8') as f:
        id_name_map = json.load(f)
    print(f"A total of {len(id_name_map)} item names have been loaded.")
    
    test_items = list(id_name_map.items())
    print(f"Processing all {len(test_items)} item names")
    
    print(f"Batch size: {batch_size}")
    batches = [test_items[i:i + batch_size] for i in range(0, len(test_items), batch_size)]
    print(f"Total batches: {len(batches)}")
    
    print(f"Loading prompt template from: {prompt_file}")
    prompt_template = load_prompt_template(prompt_file)
    print("Prompt template loaded successfully")
    
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')  
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True
    )
    print("Model loading complete")
    
    item_edges = {}
    failed_items = []
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({}, f, ensure_ascii=False, indent=2)
    
    print("Begin true batch generation of edge information...")
    
    for batch in tqdm(batches, desc="Processing batches"):
        batch_ids = [item[0] for item in batch]
        batch_names = [item[1] for item in batch]
        batch_prompts = []
        for name in batch_names:
            user_content = prompt_template.format(item_name=name)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_prompts.append(formatted)
        inputs = tokenizer(
            batch_prompts,
            padding=True,
            return_tensors="pt",
            truncation=True,      
            max_length=2048   
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.3,
                do_sample=False,      
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated_tokens = outputs[:, inputs['input_ids'].shape[1]:]
        batch_responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        for idx, response in enumerate(batch_responses):
            item_id = batch_ids[idx]
            item_name = batch_names[idx]
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if not json_match:
                    raise ValueError("No JSON found")
                json_str = json_match.group(0)
                edge_info = json.loads(json_str)
                if edge_info.get("item_name") != item_name:
                    edge_info["item_name"] = item_name
                
                item_edges[item_id] = edge_info
                
            except Exception as e:
                print(f"Failed to parse {item_name} (ID: {item_id}): {e}")
                failed_items.append({"id": item_id, "name": item_name, "response": response})
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(item_edges, f, ensure_ascii=False, indent=2)
    
    print(f"Batch generation completed! Successfully generated {len(item_edges)} items, failed {len(failed_items)}")
    if failed_items:
        print(f"Number of failures: {len(failed_items)}")
    
    retry_count = 0
    
    # 初始化API客户端（如果需要）
    api_client = None
    if use_api_for_retry and failed_items:
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            print("DEEPSEEK_API_KEY is not set, skip API retry and keep local retry only.")
            use_api_for_retry = False
        else:
            api_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            print("API client initialized for retrying failed items")
    
    while failed_items and retry_count < max_retries:
        retry_count += 1
        print(f"\n--- Retry {retry_count}/{max_retries} ---\n")
        print(f"Retrying {len(failed_items)} failed items")
        
        retry_items = [(item["id"], item["name"]) for item in failed_items]
        
        current_failed = []
        
        # 使用API进行重试
        if use_api_for_retry and api_client:
            print("Using DeepSeek API for retry...")
            for item_id, item_name in tqdm(retry_items, desc="Retrying with API"):
                try:
                    edge_info = generate_edges_with_api(api_client, item_name, prompt_template, system_prompt)
                    
                    if edge_info:
                        item_edges[item_id] = edge_info
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(item_edges, f, ensure_ascii=False, indent=2)
                    else:
                        current_failed.append({"id": item_id, "name": item_name})
                except Exception as e:
                    print(f"API retry failed for {item_name} (ID: {item_id}): {e}")
                    current_failed.append({"id": item_id, "name": item_name})
        else:
            # 使用本地模型重试
            retry_batches = [retry_items[i:i + batch_size] for i in range(0, len(retry_items), batch_size)]
            print(f"Retry batches: {len(retry_batches)}")
            
            for retry_batch in tqdm(retry_batches, desc="Retrying batches"):
                retry_batch_ids = [item[0] for item in retry_batch]
                retry_batch_names = [item[1] for item in retry_batch]
                retry_batch_prompts = []
                for name in retry_batch_names:
                    user_content = prompt_template.format(item_name=name)
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ]
                    formatted = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    retry_batch_prompts.append(formatted)
                
                retry_inputs = tokenizer(
                    retry_batch_prompts,
                    padding=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(model.device)
                with torch.no_grad():
                    retry_outputs = model.generate(
                        **retry_inputs,
                        max_new_tokens=2048,
                        temperature=0.3,
                        do_sample=False,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id
                    )
                retry_generated_tokens = retry_outputs[:, retry_inputs['input_ids'].shape[1]:]
                retry_batch_responses = tokenizer.batch_decode(retry_generated_tokens, skip_special_tokens=True)
                
                for idx, response in enumerate(retry_batch_responses):
                    item_id = retry_batch_ids[idx]
                    item_name = retry_batch_names[idx]
                    try:
                        json_match = re.search(r'\{.*\}', response, re.DOTALL)
                        if not json_match:
                            raise ValueError("No JSON found")
                        json_str = json_match.group(0)
                        
                        edge_info = json.loads(json_str)
                        
                        if edge_info.get("item_name") != item_name:
                            edge_info["item_name"] = item_name
                        
                        item_edges[item_id] = edge_info
                        
                    except Exception as e:
                        print(f"Retry failed for {item_name} (ID: {item_id}): {e}")
                        current_failed.append({"id": item_id, "name": item_name, "response": response})
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(item_edges, f, ensure_ascii=False, indent=2)
        
        failed_items = current_failed
        print(f"Retry {retry_count} completed! Now {len(item_edges)} items succeeded, {len(failed_items)} items failed")
    
    print(f"\n--- Final Result ---\n")
    print(f"All processing completed! Total successfully generated: {len(item_edges)} items")
    if failed_items:
        print(f"Still failed items: {len(failed_items)}")
        print("Some items may require manual processing.")
    else:
        print("All items processed successfully!")

if __name__ == '__main__':
    default_model_path = './base_models/Qwen2.5-14B-Instruct'
    default_system_prompt = "You are a professional knowledge graph and recommendation systems expert."

    parser = argparse.ArgumentParser()
    parser.add_argument('id_item_file', nargs='?', default=None)
    parser.add_argument('output_file', nargs='?', default=None)
    parser.add_argument('model_path', nargs='?', default=default_model_path)
    parser.add_argument('prompt_file', nargs='?', default=None)
    parser.add_argument('system_prompt', nargs='?', default=default_system_prompt)
    parser.add_argument('--dataset', choices=list(DATASET_CONFIGS.keys()), default='ml-1m')
    args = parser.parse_args()

    dataset_config = DATASET_CONFIGS[args.dataset]
    id_item_file = args.id_item_file or dataset_config['id_item_file']
    output_file = args.output_file or dataset_config['output_file']
    model_path = args.model_path
    system_prompt = args.system_prompt

    if args.prompt_file:
        prompt_file = args.prompt_file
    else:
        dataset_name = args.dataset if args.dataset else detect_dataset_from_file(id_item_file)
        prompt_dataset = resolve_prompt_dataset(dataset_name)
        prompt_file = get_default_prompt_file(prompt_dataset)

    generate_edges(id_item_file, output_file, model_path, prompt_file, system_prompt)
