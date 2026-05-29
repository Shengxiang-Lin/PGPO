import unsloth
import os
import torch
import gc
import json
from unsloth import FastLanguageModel, PatchFastRL
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

from models.dataset import FineDetailsDataset
from models.reward import create_reward_function
from models.cache import CacheManager

PatchFastRL("GRPO", FastLanguageModel)

def load_model_and_tokenizer(
    model_name_or_path: str,
    max_seq_length: int,
    lora_r: int,
    seed: int,
):
    print("Loading model and tokenizer with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        gpu_memory_utilization=0.8, 
        local_files_only=True,
    )

    if hasattr(model, 'generation_config'):
        model.generation_config.max_length = max_seq_length
        print(f"Updated generation config max_length to {max_seq_length}")

    print("Creating LoRA adapter with Unsloth...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_r * 2,
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    return model, tokenizer

class CustomGRPOTrainer(GRPOTrainer):
    def __init__(self, dataset_instance=None, **kwargs):
        super().__init__(**kwargs)
        self.dataset_instance = dataset_instance
        
    def training_step(self, model, inputs, optimizer=None):
        if self.dataset_instance is not None:
            current_step = self.state.global_step + 1
            self.dataset_instance.set_current_step(current_step)
            if current_step % 10 == 0:
                print(f"DEBUG: Global Step: {current_step}")
        
        if optimizer is not None:
            return super().training_step(model, inputs, optimizer)
        else:
            return super().training_step(model, inputs)

def create_grpo_trainer(
    model,
    tokenizer,
    fine_details_file: str,
    similarities_file: str,
    embeddings_file: str,
    embedding_model_path: str,
    category_statistics_file: str,
    output_dir: str,
    learning_rate: float,
    num_generations: int,
    max_seq_length: int,
    max_prompt_length: int,
    max_completion_length: int,
    num_samples: int,
    seed: int,
    lambda1: float,
    lambda2: float,
    lambda3: float,
    enable_dynamic_reward_weights: bool = False,
    use_r_dissimilar: bool = True,
    use_r_structure: bool = True,
    use_r_category_valid: bool = True,
    dissimilar_target: float = 0.7,
    structure_target: float = 0.9,
    category_valid_target: float = 0.9,
    batch_size: int = 8,
    context_injection_step: int = 1000,
    gradient_accumulation_steps: int = 1,
    beta: float = 0.1,
    temperature: float = 1.2,
    min_p: float = 0.05,
    per_word_max_variants: int = 8,
    global_recent_max: int = 500,
    history_window: int = 100,
    context_top_k: int = 10,
    prompt_file: str = None,
    cache_manager: CacheManager = None,
    use_cache: bool = True,
    max_steps: int = 10000,
    enable_tensorboard: bool = True,
):
    print("Loading fine details and similarities data...")
    with open(fine_details_file, 'r', encoding='utf-8') as f:
        fine_details_data = json.load(f)
    
    with open(similarities_file, 'r', encoding='utf-8') as f:
        similarities_data = json.load(f)

    if not use_cache:
        cache_manager = None
        print("Cache disabled: no context injection and no cache updates in training.")
    elif cache_manager is None:
        print("Creating cache manager...")
        cache_manager = CacheManager(
            per_word_max_variants=per_word_max_variants,
            global_recent_max=global_recent_max,
        )
    else:
        print("Using external shared cache manager...")
    
    print("Loading prompt template...")
    if prompt_file:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        print(f"Prompt template loaded from: {prompt_file}")
    else:
        prompt_template = None
        print("Using default prompt template")
    
    print("Creating dataset...")
    dataset_instance = FineDetailsDataset(
        fine_details_file=fine_details_file,
        similarities_file=similarities_file,
        cache_manager=cache_manager,
        num_samples=num_samples,
        seed=seed,
        context_injection_step=context_injection_step,
        context_top_k=context_top_k,
        prompt_template=prompt_template,
    )
    train_dataset, val_dataset = dataset_instance.split_train_val(val_ratio=0.1)
    dataset_instance = train_dataset
    print("Creating GRPO config...")
    trl_batch_size = batch_size * num_generations
    
    print(f"DEBUG: Desired Prompts per Batch: {batch_size}")
    print(f"DEBUG: Num Generations: {num_generations}")
    print(f"DEBUG: Setting per_device_train_batch_size to {trl_batch_size} (to compensate for internal division)")

    if hasattr(model, 'generation_config'):
        model.generation_config.max_length = max_prompt_length + max_completion_length

    vllm_sampling_params = SamplingParams(
        temperature=temperature,
        min_p=min_p,
        top_p=0.95,
        top_k=-1,
        presence_penalty=0.1,     
        frequency_penalty=0.2, 
        seed=seed,
        max_tokens=max_completion_length,
    )

    tb_log_dir = os.path.join(output_dir, "tensorboard")
    metrics_file_path = os.path.join(output_dir, "reward_step_metrics.txt")

    training_args = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        temperature=temperature,  
        learning_rate=learning_rate,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=trl_batch_size, 
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        save_steps=1000,
        report_to="none",
        logging_dir=tb_log_dir,
        output_dir=output_dir,
        beta=beta,
    )

    print("Creating reward function...")
    reward_func = create_reward_function(
        embeddings_file=embeddings_file,
        embedding_model_path=embedding_model_path,
        category_statistics_file=category_statistics_file,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
        enable_dynamic_reward_weights=enable_dynamic_reward_weights,
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
        num_generations=num_generations,
        history_window=history_window,
        metrics_file_path=metrics_file_path,
        tensorboard_log_dir=tb_log_dir,
        enable_tensorboard=enable_tensorboard,
    )

    print("Initializing GRPO trainer...")
    trainer = CustomGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_instance=dataset_instance,
    )

    return trainer

def train_grpo(
    model_name_or_path: str = "./base_models/Qwen3-4B-Base",
    sft_lora_path: str = None,
    fine_details_file: str = "./data/yelp/handled/extracted/grpo_dataset/fine_details.json",
    similarities_file: str = "./data/yelp/handled/extracted/grpo_dataset/fine_details_similarities.json",
    embeddings_file: str = "./data/yelp/handled/extracted/fine_details_embeddings.pkl",
    embedding_model_path: str = "./base_models/bert-base-uncased",
    category_statistics_file: str = "./data/ml-1m/handled/category_statistics.pkl",
    prompt_file: str = None,
    output_dir: str = "./output/grpo_model",
    lora_r: int = 16,
    lora_alpha: int = 32,
    learning_rate: float = 5e-6,
    num_generations: int = 8,
    beta: float = 0.1,
    temperature: float = 1.2,
    min_p: float = 0.05,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    lambda3: float = 1.0,
    enable_dynamic_reward_weights: bool = False,
    use_r_dissimilar: bool = True,
    use_r_structure: bool = True,
    use_r_category_valid: bool = True,
    dissimilar_target: float = 0.7,
    structure_target: float = 0.9,
    category_valid_target: float = 0.9,
    max_seq_length: int = 2048,
    max_prompt_length: int = 1024,
    max_completion_length: int = 128,
    num_samples: int = None,
    seed: int = 42,
    batch_size: int = 8,
    context_injection_step: int = 1000,
    gradient_accumulation_steps: int = 1,
    per_word_max_variants: int = 8,
    global_recent_max: int = 500,
    history_window: int = 100,
    context_top_k: int = 10,
    cache_manager: CacheManager = None,
    use_cache: bool = True,
    max_steps: int = 10000,
    return_model_for_inference: bool = False,
    enable_tensorboard: bool = True,
):
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("GRPO Training Configuration (Fixed for Unsloth/TRL Compatibility)")
    print("=" * 80)
    print(f"Model: {model_name_or_path}")
    print(f"Batch size (Prompts per step): {batch_size}")
    print(f"Num generations: {num_generations}")
    print(f"KL beta: {beta}")
    print(f"Sampling temperature: {temperature}")
    print(f"Min_p: {min_p}")
    print(f"Use rewards: dissimilar={use_r_dissimilar}, structure={use_r_structure}, category_valid={use_r_category_valid}")
    print(f"Dynamic reward weights: {enable_dynamic_reward_weights}")
    print(f"TensorBoard: {enable_tensorboard}")
    print(f"Reward step metrics file: {os.path.join(output_dir, 'reward_step_metrics.txt')}")
    print(f"Max steps: {max_steps}")
    print(f"Expected per_device_train_batch_size in config: {batch_size * num_generations}")
    print("=" * 80)

    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        max_seq_length=max_seq_length,
        lora_r=lora_r,
        seed=seed,
    )

    trainer = create_grpo_trainer(
        model=model,
        tokenizer=tokenizer,
        fine_details_file=fine_details_file,
        similarities_file=similarities_file,
        embeddings_file=embeddings_file,
        embedding_model_path=embedding_model_path,
        category_statistics_file=category_statistics_file,
        prompt_file=prompt_file,
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_generations=num_generations,
        max_seq_length=max_seq_length,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        num_samples=num_samples,
        seed=seed,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
        enable_dynamic_reward_weights=enable_dynamic_reward_weights,
        use_r_dissimilar=use_r_dissimilar,
        use_r_structure=use_r_structure,
        use_r_category_valid=use_r_category_valid,
        dissimilar_target=dissimilar_target,
        structure_target=structure_target,
        category_valid_target=category_valid_target,
        batch_size=batch_size,
        context_injection_step=context_injection_step,
        gradient_accumulation_steps=gradient_accumulation_steps,
        beta=beta,
        temperature=temperature,
        min_p=min_p,
        per_word_max_variants=per_word_max_variants,
        global_recent_max=global_recent_max,
        history_window=history_window,
        context_top_k=context_top_k,
        cache_manager=cache_manager,
        use_cache=use_cache,
        max_steps=max_steps,
        enable_tensorboard=enable_tensorboard,
    )

    print("\nStarting training...")
    print("=" * 80)
    trainer.train()

    print("\nSaving LoRA adapter...")
    model.save_lora(output_dir)
    tokenizer.save_pretrained(output_dir)

    reward_config_path = os.path.join(output_dir, "grpo_reward_config.json")
    reward_config = {
        "lambda1": lambda1,
        "lambda2": lambda2,
        "lambda3": lambda3,
        "enable_dynamic_reward_weights": enable_dynamic_reward_weights,
        "use_r_dissimilar": use_r_dissimilar,
        "use_r_structure": use_r_structure,
        "use_r_category_valid": use_r_category_valid,
        "dissimilar_target": dissimilar_target,
        "structure_target": structure_target,
        "category_valid_target": category_valid_target,
        "history_window": history_window,
    }
    with open(reward_config_path, "w", encoding="utf-8") as f:
        json.dump(reward_config, f, ensure_ascii=False, indent=2)
    print(f"Saved reward config to {reward_config_path}")

    print(f"\nTraining completed! LoRA adapter saved to {output_dir}")
    print("=" * 80)

    del trainer
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()
    gc.collect()

    if return_model_for_inference:
        model.zero_grad(set_to_none=True)
        for param in model.parameters():
            param.grad = None
        FastLanguageModel.for_inference(model)
        model.eval()
        return model, tokenizer

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return None, None
