import unsloth
import json
import torch
import re
import random
import os
import time
import gc
import numpy as np
from unsloth import FastLanguageModel
from models.cache import CacheManager
from models.reward import RewardCalculator
from tqdm import tqdm
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

DATASET_CONFIGS = {
    'movielens': {
        'fine_details_file': './data/ml-1m/handled/extracted/fine_details.json',
        'similarities_file': './data/ml-1m/handled/extracted/fine_details_similarities.json',
        'embeddings_file': './data/ml-1m/handled/extracted/fine_details_embeddings.pkl',
        'category_statistics_file': './data/ml-1m/handled/category_statistics.pkl',
        'model_path': './output/grpo_model/movielens/checkpoint-3000',
        'output_file': './output/generated_variants_movielens.json',
        'prompt_file': './models/prompt/prompt.txt'
    },
    'amazon-book': {
        'fine_details_file': './data/amazon-book/handled/extracted/fine_details.json',
        'similarities_file': './data/amazon-book/handled/extracted/fine_details_similarities.json',
        'embeddings_file': './data/amazon-book/handled/extracted/fine_details_embeddings.pkl',
        'category_statistics_file': './data/amazon-book/handled/category_statistics.pkl',
        'model_path': './output/grpo_model/amazon-book/checkpoint-5000',
        'output_file': './output/generated_variants_amazon-book.json',
        'prompt_file': './models/prompt/prompt.txt'
    }
}


def load_reward_config_from_model_path(model_path):
    candidate_paths = [
        os.path.join(model_path, "grpo_reward_config.json"),
        os.path.join(os.path.dirname(model_path), "grpo_reward_config.json"),
    ]
    for cfg_path in candidate_paths:
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                print(f"Loaded reward config from {cfg_path}")
                return cfg
            except Exception as e:
                print(f"Failed to load reward config from {cfg_path}: {e}")
    print("Reward config not found, fallback to generate defaults")
    return None

def load_model_and_tokenizer(model_path, use_fast_inference=False):
    mode_name = "fast_inference(vLLM)" if use_fast_inference else "standard Transformers"
    print(f"Loading model and tokenizer with Unsloth ({mode_name})...")
    adapter_path = os.path.join(model_path, "adapter_model.safetensors")
    adapter_abs_path = os.path.abspath(adapter_path)
    if os.path.exists(adapter_path):
        print(f"Found adapter_model.safetensors at {adapter_abs_path}")
    else:
        print(f"Warning: adapter_model.safetensors not found at {adapter_abs_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        load_in_4bit=False,
        fast_inference=use_fast_inference,
        gpu_memory_utilization=0.8,
        device_map="cuda:0" if torch.cuda.is_available() else "auto",
        local_files_only=True,
    )
    if use_fast_inference:
        FastLanguageModel.for_inference(model)
    model.eval()
    if os.path.exists(adapter_path):
        print(f"Loaded adapter_model.safetensors from {adapter_abs_path}")
    return model, tokenizer

def load_prompt_template(prompt_file):
    with open(prompt_file, 'r', encoding='utf-8') as f:
        template = f.read()
    return template

def create_prompt(word, category, context_str, prompt_template=None):
    if context_str:
        similarity_note = """To ensure that term similarity is preserved before and after rewriting (originally similar terms remain similar in their rewritten forms, originally dissimilar terms remain dissimilar), examples of mappings for variants of related terms are provided below:

"""
    else:
        similarity_note = ""
    if not prompt_template:
        raise ValueError("prompt_template is required")
    prompt = prompt_template.format(
        word=word,
        category=category,
        similarity_note=similarity_note,
        context_str=context_str
    )
    return prompt


def get_generation_context_from_all_pairs(
    original_word_id,
    similarities_data,
    fine_details_data,
    all_generated_pairs,
    top_k=10,
):
    context = []

    try:
        word_id = int(original_word_id)
    except (TypeError, ValueError):
        return context

    original_word = fine_details_data.get(str(word_id), [None])[0]
    added = set()

    sim_entry = similarities_data.get(str(word_id), {})
    similar_ids = sim_entry.get("top_similar_ids", [])[:top_k]
    for sid in similar_ids:
        sid_str = str(sid)
        if sid_str not in fine_details_data:
            continue
        similar_word = fine_details_data[sid_str][0]
        if similar_word in all_generated_pairs and similar_word not in added:
            context.append({
                "original": similar_word,
                "variant": all_generated_pairs[similar_word],
            })
            added.add(similar_word)
        if len(context) >= top_k:
            break

    if len(context) < top_k and all_generated_pairs:
        candidates = [
            w for w in all_generated_pairs.keys()
            if w != original_word and w not in added
        ]
        random.seed(hash(str(original_word_id)) % 10000)
        random.shuffle(candidates)
        for w in candidates:
            context.append({"original": w, "variant": all_generated_pairs[w]})
            added.add(w)
            if len(context) >= top_k:
                break

    return context


def generate_variants_batch(
    batch_samples,
    model,
    tokenizer,
    num_generations=8,
    prompt_template=None,
    max_prompt_length=1024,
    temperature=1.2,
    min_p=0.05,
):
    prompts = []
    for sample in batch_samples:
        word = sample['word']
        category = sample['category']
        context_str = sample.get('context_str', '')
        prompt = create_prompt(word, category, context_str, prompt_template)
        prompts.append(prompt)

    inference_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length
    ).to(inference_device)

    input_length = inputs['input_ids'].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            temperature=temperature,
            top_p=0.95,
            min_p=min_p,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=num_generations,
            repetition_penalty=1.0,
            length_penalty=1.0,
            no_repeat_ngram_size=0
        )
    batch_variants = [[] for _ in range(len(batch_samples))]

    for i in range(outputs.shape[0]):
        sample_idx = i // num_generations
        generated_ids = outputs[i][input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        variant = ""
        if "</answer>" in generated_text:
            answer_end = generated_text.find("</answer>")
            if answer_end > 0:
                answer_content = generated_text[:answer_end].strip()
                if answer_content:
                    variant = answer_content
        if variant:
            batch_variants[sample_idx].append(variant)
    return batch_variants


def init_generate_debug_log(enable_debug_log: bool):
    if not enable_debug_log:
        return None
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_time = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"generate_log_{log_time}.log")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("GENERATION DEBUG LOG\n")
        f.write("=" * 80 + "\n")
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
    print(f"Generation debug log enabled: {log_file}")
    return log_file


def append_generate_debug_log(log_file, source, original_word, category, prompt, variants_with_rewards, best_variant, best_reward):
    if not log_file:
        return
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{source}] Original word: {original_word}\n")
        f.write(f"Category: {category}\n")
        f.write(f"Prompt:\n{prompt}\n\n")
        f.write("Variants and rewards:\n")
        for idx, info in enumerate(variants_with_rewards):
            f.write(f"  Variant {idx + 1}: {info['variant']} (reward: {info['reward']:.4f})\n")
        f.write(f"Selected: {best_variant} (reward: {best_reward:.4f})\n")
        f.write("-" * 80 + "\n\n")


def append_reward_step_metrics(metrics_file, phase, step_idx, sample_records, tensorboard_writer=None):
    if not metrics_file:
        return
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    if not os.path.exists(metrics_file):
        with open(metrics_file, "w", encoding="utf-8") as f:
            f.write("# reward step metrics jsonl\n")

    step_rewards = []
    per_sample = []
    for sample in sample_records:
        best_reward = float(sample.get("best_reward", 0.0))
        step_rewards.append(best_reward)
        variants_with_rewards = sample.get("variants_with_rewards", [])
        total_rewards = [
            float(v.get("reward", 0.0))
            for v in variants_with_rewards
            if isinstance(v.get("reward", None), (int, float))
        ]
        details = [
            v.get("reward_details")
            for v in variants_with_rewards
            if isinstance(v.get("reward_details", None), dict)
        ]
        per_sample.append({
            "original_word": sample.get("word", ""),
            "category": sample.get("category", ""),
            "num_variants": len(variants_with_rewards),
            "avg_total_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
            "max_total_reward": float(np.max(total_rewards)) if total_rewards else best_reward,
            "avg_r_dissimilar": float(np.mean([d.get("r_dissimilar", 0.0) for d in details])) if details else 0.0,
            "avg_r_structure": float(np.mean([d.get("r_structure", 0.0) for d in details])) if details else 0.0,
            "avg_r_category_valid": float(np.mean([d.get("r_category_valid", 0.0) for d in details])) if details else 0.0,
            "avg_weighted_dissimilar": float(np.mean([d.get("weighted_dissimilar", 0.0) for d in details])) if details else 0.0,
            "avg_weighted_structure": float(np.mean([d.get("weighted_structure", 0.0) for d in details])) if details else 0.0,
            "avg_weighted_category_valid": float(np.mean([d.get("weighted_category_valid", 0.0) for d in details])) if details else 0.0,
        })

    record = {
        "phase": phase,
        "step": int(step_idx),
        "reward_mean": float(np.mean(step_rewards)) if step_rewards else 0.0,
        "reward_max": float(np.max(step_rewards)) if step_rewards else 0.0,
        "num_samples": len(sample_records),
        "samples": per_sample,
    }
    with open(metrics_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    all_details = []
    for sample in sample_records:
        for info in sample.get("variants_with_rewards", []):
            details = info.get("reward_details")
            if isinstance(details, dict):
                all_details.append(details)
    r_dissimilar_vals = [float(d.get("r_dissimilar", 0.0)) for d in all_details]
    r_structure_vals = [float(d.get("r_structure", 0.0)) for d in all_details]
    r_category_valid_vals = [float(d.get("r_category_valid", 0.0)) for d in all_details]
    r_dissimilar_mean = float(np.mean(r_dissimilar_vals)) if r_dissimilar_vals else 0.0
    r_dissimilar_max = float(np.max(r_dissimilar_vals)) if r_dissimilar_vals else 0.0
    r_structure_mean = float(np.mean(r_structure_vals)) if r_structure_vals else 0.0
    r_structure_max = float(np.max(r_structure_vals)) if r_structure_vals else 0.0
    r_category_valid_mean = float(np.mean(r_category_valid_vals)) if r_category_valid_vals else 0.0
    r_category_valid_max = float(np.max(r_category_valid_vals)) if r_category_valid_vals else 0.0

    if tensorboard_writer is not None:
        tensorboard_writer.add_scalar(f"reward/{phase}_step_mean", record["reward_mean"], step_idx)
        tensorboard_writer.add_scalar(f"reward/{phase}_step_max", record["reward_max"], step_idx)
        tensorboard_writer.add_scalar(f"reward/{phase}_r_dissimilar_mean", r_dissimilar_mean, step_idx)
        tensorboard_writer.add_scalar(f"reward/{phase}_r_dissimilar_max", r_dissimilar_max, step_idx)
        tensorboard_writer.add_scalar(f"reward/{phase}_r_structure_mean", r_structure_mean, step_idx)
        tensorboard_writer.add_scalar(f"reward/{phase}_r_structure_max", r_structure_max, step_idx)
        tensorboard_writer.add_scalar(f"reward/{phase}_r_category_valid_mean", r_category_valid_mean, step_idx)
        tensorboard_writer.add_scalar(f"reward/{phase}_r_category_valid_max", r_category_valid_max, step_idx)


def is_cuda_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg


def cleanup_after_oom():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def detect_dataset_from_path(path):
    if 'amazon-book' in path.lower() or 'books' in path.lower():
        return 'amazon-book'
    elif 'ml-1m' in path.lower():
        return 'movielens'
    elif 'ml-100k' in path.lower() or 'movielens' in path.lower():
        return 'movielens'
    else:
        return 'movielens'  # 默认返回movielens

def batch_inference(
    fine_details_file,
    similarities_file,
    model_path,
    output_file,
    batch_size=4,
    num_generations=8,
    prompt_file=None,
    dataset_name=None,
    cache_manager=None,
    model=None,
    tokenizer=None,
    max_prompt_length=1024,
    temperature=1.2,
    min_p=0.05,
    warmup_fine_details_file=None,
    warmup_similarities_file=None,
    enable_debug_log=False,
    use_fast_inference=False,
    reward_metrics_file=None,
    tensorboard_log_dir=None,
    use_cache=True,
):
    use_in_memory_model = (
        model is not None and tokenizer is not None and use_fast_inference
    )
    if use_in_memory_model:
        print("Using in-memory model/tokenizer from training phase")
    else:
        if model is not None or tokenizer is not None:
            print("Reloading model for generation to honor inference mode setting")
        model, tokenizer = load_model_and_tokenizer(
            model_path,
            use_fast_inference=use_fast_inference,
        )
        print(f"Model loaded successfully from {model_path}")
    
    with open(fine_details_file, 'r', encoding='utf-8') as f:
        fine_details_data = json.load(f)

    with open(similarities_file, 'r', encoding='utf-8') as f:
        similarities_data = json.load(f)

    warmup_fine_details_file = warmup_fine_details_file or fine_details_file
    warmup_similarities_file = warmup_similarities_file or similarities_file

    with open(warmup_fine_details_file, 'r', encoding='utf-8') as f:
        warmup_fine_details_data = json.load(f)
    with open(warmup_similarities_file, 'r', encoding='utf-8') as f:
        warmup_similarities_data = json.load(f)

    print(f"Loaded main generation words: {len(fine_details_data)} from {fine_details_file}")
    print(f"Loaded warmup words: {len(warmup_fine_details_data)} from {warmup_fine_details_file}")
    debug_log_file = init_generate_debug_log(enable_debug_log)
    tb_writer = None
    if tensorboard_log_dir and SummaryWriter is not None:
        tb_writer = SummaryWriter(log_dir=tensorboard_log_dir)
        print(f"Generation TensorBoard writer enabled at {tensorboard_log_dir}")
    dataset_name = dataset_name or detect_dataset_from_path(fine_details_file)
    print(f"Detected dataset: {dataset_name}")
    prompt_template = None
    if prompt_file:
        print(f"Loading prompt template from {prompt_file}")
        prompt_template = load_prompt_template(prompt_file)

    if not use_cache:
        cache_manager = None
        print("Cache disabled in generation: no context injection and no cache updates.")
    elif cache_manager is None:
        cache_manager = CacheManager(per_word_max_variants=8, global_recent_max=1000)
        print("Created local cache manager for generation")
    else:
        print("Using shared cache manager from training phase")
    print("Loading reward calculator...")

    reward_cfg = load_reward_config_from_model_path(model_path)
    lambda1 = reward_cfg.get("lambda1", 0.5) if reward_cfg else 0.5
    lambda2 = reward_cfg.get("lambda2", 3.0) if reward_cfg else 3.0
    lambda3 = reward_cfg.get("lambda3", 1.0) if reward_cfg else 1.0
    use_r_dissimilar = reward_cfg.get("use_r_dissimilar", True) if reward_cfg else True
    use_r_structure = reward_cfg.get("use_r_structure", True) if reward_cfg else True
    if not use_cache and use_r_structure:
        print("Cache disabled: forcing use_r_structure=False in generation.")
    use_r_structure = use_r_structure and use_cache
    use_r_category_valid = reward_cfg.get("use_r_category_valid", True) if reward_cfg else True
    dissimilar_target = reward_cfg.get("dissimilar_target", -0.3) if reward_cfg else -0.3
    structure_target = reward_cfg.get("structure_target", -0.002) if reward_cfg else -0.002
    category_valid_target = reward_cfg.get("category_valid_target", 0.9) if reward_cfg else 0.9
    history_window = reward_cfg.get("history_window", 100) if reward_cfg else 100
    print(
        "Generation reward switches: "
        f"dissimilar={use_r_dissimilar}, "
        f"structure={use_r_structure}, "
        f"category_valid={use_r_category_valid}"
    )

    if dataset_name in DATASET_CONFIGS:
        embeddings_file = DATASET_CONFIGS[dataset_name]['embeddings_file']
        category_statistics_file = DATASET_CONFIGS[dataset_name]['category_statistics_file']
    else:
        embeddings_file = DATASET_CONFIGS['movielens']['embeddings_file']
        category_statistics_file = DATASET_CONFIGS['movielens']['category_statistics_file']
    
    embedding_model_path = "./base_models/bert-base-uncased"
    
    reward_calculator = RewardCalculator(
        embeddings_file=embeddings_file,
        embedding_model_path=embedding_model_path,
        category_statistics_file=category_statistics_file,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
        use_r_dissimilar=use_r_dissimilar,
        use_r_structure=use_r_structure,
        use_r_category_valid=use_r_category_valid,
        dissimilar_target=dissimilar_target,
        structure_target=structure_target,
        category_valid_target=category_valid_target,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        cache_manager=cache_manager,
        similarities_data=similarities_data,
        fine_details_data=fine_details_data,
        history_window=history_window
    )
    print("Reward calculator loaded successfully")
    
    warmup_all_samples = []
    for word_id_str, word_data in warmup_fine_details_data.items():
        word = word_data[0]
        categories = word_data[1]

        if isinstance(categories, str):
            category_list = [c.strip() for c in categories.split(',')]
        else:
            category_list = [categories]

        category = category_list[0]

        warmup_all_samples.append({
            'word_id': int(word_id_str),
            'word': word,
            'category': category
        })

    warmup_words_set = {s['word'] for s in warmup_all_samples}
    samples = []
    for word_id_str, word_data in fine_details_data.items():
        word = word_data[0]
        categories = word_data[1]
        
        if isinstance(categories, str):
            category_list = [c.strip() for c in categories.split(',')]
        else:
            category_list = [categories]
        
        category = category_list[0]
        if word in warmup_words_set:
            continue

        samples.append({
            'word_id': int(word_id_str),
            'word': word,
            'category': category
        })
    
    random.shuffle(warmup_all_samples)
    random.shuffle(samples)
    print(f"Shuffled warmup samples: {len(warmup_all_samples)}")
    print(f"Shuffled main generation samples (excluding warmup words): {len(samples)}")
    
    warmup_samples_list = warmup_all_samples

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([], f)

    all_generated_pairs = {}

    if warmup_samples_list:
        print(f"\n{'='*50}")
        print(f"核心词生成阶段：处理 {len(warmup_samples_list)} 个样本...")
        print(f"{'='*50}")
        core_total_samples = len(warmup_samples_list)
        core_cache_hit_total = 0
        core_to_gen_total = 0
        
        warmup_batches = (len(warmup_samples_list) + batch_size - 1) // batch_size
        warmup_iterator = tqdm(range(warmup_batches), desc="Core batches")

        for batch_idx in warmup_iterator:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(warmup_samples_list))
            full_batch_samples = warmup_samples_list[start_idx:end_idx]
            cur_bs = len(full_batch_samples)
            processed = False

            while not processed and cur_bs >= 1:
                batch_samples = full_batch_samples[:cur_bs]
            
                try:
                    for sample in batch_samples:
                        context_str = ""
                        if use_cache and cache_manager:
                            word_id = sample['word_id']
                            similar_context = cache_manager.get_similar_context(
                                word_id,
                                warmup_similarities_data,
                                warmup_fine_details_data,
                                top_k=10
                            )
                            if similar_context:
                                context_lines = []
                                for item in similar_context:
                                    context_lines.append(f"original: {item['original']}, output: {item['variant']}")
                                context_str = "\n".join(context_lines) + "\n"
                        sample['context_str'] = context_str

                    uncached_samples = []
                    cached_sample_to_pair = {}
                    for sample in batch_samples:
                        word = sample['word']
                        cached_pair = cache_manager.per_word_cache.get(word) if (use_cache and cache_manager) else None
                        if cached_pair is not None:
                            cached_sample_to_pair[id(sample)] = cached_pair
                        else:
                            uncached_samples.append(sample)

                    all_variants = []
                    if uncached_samples:
                        all_variants = generate_variants_batch(
                            uncached_samples,
                            model,
                            tokenizer,
                            num_generations,
                            prompt_template,
                            max_prompt_length,
                        temperature,
                        min_p,
                        )
                except Exception as e:
                    if is_cuda_oom_error(e):
                        new_bs = max(1, cur_bs // 2)
                        if new_bs == cur_bs:
                            raise
                        print(f"[OOM][core] batch_idx={batch_idx}, reducing sub-batch {cur_bs} -> {new_bs} and retrying")
                        cleanup_after_oom()
                        cur_bs = new_bs
                        continue
                    raise
                
                warmup_batch_results = []
                warmup_step_samples = []
                uncached_idx = 0
                cached_hits = 0
                for sample in batch_samples:
                    word_id = sample['word_id']
                    word = sample['word']
                    category = sample['category']
                    context_str = sample.get('context_str', '')
                    
                    prompt = create_prompt(word, category, context_str, prompt_template)

                    cached_pair = cached_sample_to_pair.get(id(sample))
                    if cached_pair is not None:
                        best_variant, best_reward = cached_pair
                        cached_hits += 1
                        variants_with_rewards = [{"variant": best_variant, "reward": float(best_reward), "reward_details": None}]
                        if use_cache and cache_manager:
                            cache_manager.update_cache(word, best_variant, best_reward)
                            all_generated_pairs[word] = best_variant
                        append_generate_debug_log(
                            log_file=debug_log_file,
                            source="core_cache",
                            original_word=word,
                            category=category,
                            prompt=prompt,
                            variants_with_rewards=variants_with_rewards,
                            best_variant=best_variant,
                            best_reward=best_reward,
                        )
                        warmup_batch_results.append({
                            'word_id': word_id,
                            'original_word': word,
                            'category': category,
                            'variant': best_variant,
                            'reward': best_reward,
                            'source': 'core'
                        })
                        warmup_step_samples.append({
                            "word": word,
                            "category": category,
                            "best_reward": float(best_reward),
                            "variants_with_rewards": variants_with_rewards,
                        })
                        continue
                    
                    variants = all_variants[uncached_idx] if uncached_idx < len(all_variants) else []
                    uncached_idx += 1
                    best_variant = None
                    best_reward = -float('inf')
                    variants_with_rewards = []
                    
                    for variant in variants:
                        try:
                            reward_dict = reward_calculator.compute_reward(
                                original_ids=[word_id],
                                original_words=[word],
                                generated_words=[variant],
                                target_category=category,
                                original_similarities_data={},
                                prompt=prompt
                            )
                            reward = reward_dict['total']
                            variants_with_rewards.append(
                                {"variant": variant, "reward": float(reward), "reward_details": reward_dict}
                            )
                            if reward > best_reward:
                                best_reward = reward
                                best_variant = variant
                        except Exception:
                            continue
                    
                    if best_variant is None and variants:
                        best_variant = variants[0]
                        best_reward = 0.0
                        variants_with_rewards.append({"variant": best_variant, "reward": float(best_reward)})

                    # 生成失败兜底：回退到原词，避免输出 null / -Infinity
                    if best_variant is None:
                        best_variant = word
                        best_reward = 0.0
                        variants_with_rewards.append({"variant": best_variant, "reward": float(best_reward)})
                    
                    if use_cache and cache_manager:
                        cache_manager.update_cache(word, best_variant, best_reward)
                        all_generated_pairs[word] = best_variant
                    append_generate_debug_log(
                        log_file=debug_log_file,
                        source="core",
                        original_word=word,
                        category=category,
                        prompt=prompt,
                        variants_with_rewards=variants_with_rewards,
                        best_variant=best_variant,
                        best_reward=best_reward,
                    )
                    warmup_batch_results.append({
                        'word_id': word_id,
                        'original_word': word,
                        'category': category,
                        'variant': best_variant,
                        'reward': best_reward,
                        'source': 'core'
                    })
                    warmup_step_samples.append({
                        "word": word,
                        "category": category,
                        "best_reward": float(best_reward),
                        "variants_with_rewards": variants_with_rewards,
                    })

                if warmup_batch_results:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        existing_results = json.load(f)
                    existing_results.extend(warmup_batch_results)
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(existing_results, f, ensure_ascii=False, indent=2)
                core_cache_hit_total += cached_hits
                core_to_gen_total += (len(batch_samples) - cached_hits)
                append_reward_step_metrics(
                    reward_metrics_file,
                    phase="generate_core",
                    step_idx=batch_idx,
                    sample_records=warmup_step_samples,
                    tensorboard_writer=tb_writer,
                )
                warmup_iterator.set_postfix({"batch": f"{batch_idx + 1}/{warmup_batches}", "cache_hit": cached_hits, "to_gen": len(batch_samples) - cached_hits})
                processed = True
        
        print(f"核心词生成完成，已写入 {len(warmup_samples_list)} 个样本（如有有效生成）")
        core_hit_ratio = (core_cache_hit_total / core_total_samples * 100.0) if core_total_samples > 0 else 0.0
        print(
            f"核心词缓存复用统计: total={core_total_samples}, "
            f"cache_hit={core_cache_hit_total}, generated={core_to_gen_total}, "
            f"hit_ratio={core_hit_ratio:.2f}%"
        )
        print(f"{'='*50}\n")
    
    # 阶段2：非核心词生成（生成文件减去核心词）
    total_batches = (len(samples) + batch_size - 1) // batch_size

    results = []
    
    batch_iterator = tqdm(range(total_batches), desc="Processing batches")
    
    for batch_idx in batch_iterator:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(samples))
        full_batch_samples = samples[start_idx:end_idx]
        cur_bs = len(full_batch_samples)
        processed = False

        while not processed and cur_bs >= 1:
            batch_samples = full_batch_samples[:cur_bs]
            
            try:
                for sample in batch_samples:
                    context_str = ""
                    if use_cache:
                        word_id = sample['word_id']
                        similar_context = get_generation_context_from_all_pairs(
                            original_word_id=word_id,
                            similarities_data=similarities_data,
                            fine_details_data=fine_details_data,
                            all_generated_pairs=all_generated_pairs,
                            top_k=10,
                        )
                        if similar_context:
                            context_lines = []
                            for item in similar_context:
                                context_lines.append(f"original: {item['original']}, output: {item['variant']}")
                            context_str = "\n".join(context_lines) + "\n"
                    sample['context_str'] = context_str
                
                all_variants = generate_variants_batch(
                    batch_samples,
                    model,
                    tokenizer,
                    num_generations,
                    prompt_template,
                    max_prompt_length,
                    temperature,
                    min_p,
                )
            except Exception as e:
                if is_cuda_oom_error(e):
                    new_bs = max(1, cur_bs // 2)
                    if new_bs == cur_bs:
                        raise
                    print(f"[OOM][non_core] batch_idx={batch_idx}, reducing sub-batch {cur_bs} -> {new_bs} and retrying")
                    cleanup_after_oom()
                    cur_bs = new_bs
                    continue
                raise
        
            batch_results = []
            step_samples = []
            for sample, variants in zip(batch_samples, all_variants):
                word_id = sample['word_id']
                word = sample['word']
                category = sample['category']
                context_str = sample.get('context_str', '')
                
                prompt = create_prompt(word, category, context_str, prompt_template)
                
                best_variant = None
                best_reward = -float('inf')
                variants_with_rewards = []
                
                for variant in variants:
                    try:
                        reward_dict = reward_calculator.compute_reward(
                            original_ids=[word_id],
                            original_words=[word],
                            generated_words=[variant],
                            target_category=category,
                            original_similarities_data={},
                            prompt=prompt
                        )
                        reward = reward_dict['total']
                        variants_with_rewards.append(
                            {"variant": variant, "reward": float(reward), "reward_details": reward_dict}
                        )
                        if reward > best_reward:
                            best_reward = reward
                            best_variant = variant
                    except Exception as e:
                        print(f"Error computing reward for {variant}: {e}")
                        continue
                
                if best_variant is None and variants:
                    best_variant = variants[0]
                    best_reward = 0.0
                    variants_with_rewards.append({"variant": best_variant, "reward": float(best_reward)})

                if best_variant is None:
                    best_variant = word
                    best_reward = 0.0
                    variants_with_rewards.append({"variant": best_variant, "reward": float(best_reward)})

                if use_cache and cache_manager:
                    cache_manager.update_cache(word, best_variant, best_reward)
                    all_generated_pairs[word] = best_variant
                append_generate_debug_log(
                    log_file=debug_log_file,
                    source="non_core",
                    original_word=word,
                    category=category,
                    prompt=prompt,
                    variants_with_rewards=variants_with_rewards,
                    best_variant=best_variant,
                    best_reward=best_reward,
                )
                
                batch_results.append({
                    'word_id': word_id,
                    'original_word': word,
                    'category': category,
                    'variant': best_variant,
                    'reward': best_reward,
                    'source': 'non_core'
                })
                step_samples.append({
                    "word": word,
                    "category": category,
                    "best_reward": float(best_reward),
                    "variants_with_rewards": variants_with_rewards,
                })
            
            results.extend(batch_results)
            
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            
            existing_results.extend(batch_results)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(existing_results, f, ensure_ascii=False, indent=2)
            
            append_reward_step_metrics(
                reward_metrics_file,
                phase="generate_non_core",
                step_idx=batch_idx,
                sample_records=step_samples,
                tensorboard_writer=tb_writer,
            )
            batch_iterator.set_postfix({"batch": f"{batch_idx + 1}/{total_batches}", "sub_bs": cur_bs})
            processed = True
    
    if tb_writer is not None:
        tb_writer.close()
    print(f"Generated variants for {len(results)} words")
    print(f"All results saved to {output_file}")
