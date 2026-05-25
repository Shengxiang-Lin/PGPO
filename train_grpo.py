import unsloth
import os
import argparse
import gc
import torch
import sys
from models.trainer import train_grpo as run_train_grpo
from models.cache import CacheManager
from models.generator import batch_inference

DATASET_CONFIGS = {
    'movielens': {
        'fine_details_file': './data/ml-1m/handled/extracted/grpo_dataset/fine_details.json',
        'similarities_file': './data/ml-1m/handled/extracted/grpo_dataset/fine_details_similarities.json',
        'embeddings_file': './data/ml-1m/handled/extracted/fine_details_embeddings.pkl',
        'embedding_model_path': './base_models/bert-base-uncased',
        'category_statistics_file': './data/ml-1m/handled/category_statistics.pkl',
        'output_dir': './output/grpo_model/movielens',
        'prompt_file': './models/prompt/prompt.txt',
        'generated_output_file': './output/generated_variants_movielens.json',
    },
    'amazon-book': {
        'fine_details_file': './data/amazon-book/handled/extracted/grpo_dataset/fine_details.json',
        'similarities_file': './data/amazon-book/handled/extracted/grpo_dataset/fine_details_similarities.json',
        'embeddings_file': './data/amazon-book/handled/extracted/fine_details_embeddings.pkl',
        'embedding_model_path': './base_models/bert-base-uncased',
        'category_statistics_file': './data/amazon-book/handled/category_statistics.pkl',
        'output_dir': './output/grpo_model/amazon-book',
        'prompt_file': './models/prompt/prompt.txt',
        'generated_output_file': './output/generated_variants_amazon-book.json',
    }
}

def detect_dataset_from_path(path):
    """根据文件路径自动检测数据集类型"""
    if 'amazon-book' in path.lower() or 'books' in path.lower():
        return 'amazon-book'
    elif 'ml-1m' in path.lower():
        return 'movielens'
    elif 'ml-100k' in path or 'movielens' in path.lower() or 'movie' in path.lower():
        return 'movielens'
    else:
        return 'movielens'

def get_default_paths(dataset_name):
    """根据数据集类型获取默认路径"""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_CONFIGS[dataset_name]

def load_prompt_template(prompt_file):
    """加载prompt模板文件"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GRPO model")
    
    parser.add_argument("--model_name_or_path", type=str, default="./base_models/Qwen3-4B-Base", help="Model path")
    parser.add_argument("--sft_lora_path", type=str, default=None, help="SFT LoRA path (optional)")
    
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (movielens or amazon-book)")
    parser.add_argument("--fine_details_file", type=str, default=None, help="Fine details file")
    parser.add_argument("--similarities_file", type=str, default=None, help="Similarities file")
    parser.add_argument("--embeddings_file", type=str, default=None, help="Embeddings file")
    parser.add_argument("--embedding_model_path", type=str, default=None, help="Embedding model path")
    parser.add_argument("--category_statistics_file", type=str, default=None, help="Category statistics file")
    parser.add_argument("--prompt_file", type=str, default=None, help="Prompt template file")
    
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    
    parser.add_argument("--learning_rate", type=float, default=5e-7, help="Learning rate")
    parser.add_argument("--num_generations", type=int, default=8, help="Number of generations per prompt")
    parser.add_argument("--beta", type=float, default=0.1, help="KL coefficient")
    
    parser.add_argument("--lambda1", type=float, default=1.0, help="R_dissimilar weight")
    parser.add_argument("--lambda2", type=float, default=1.0, help="R_structure weight")
    parser.add_argument("--lambda3", type=float, default=1.0, help="R_category_valid weight")
    parser.add_argument("--use_r_dissimilar", type=int, choices=[0, 1], default=1, help="Whether to use R_dissimilar (1=yes, 0=no)")
    parser.add_argument("--use_r_structure", type=int, choices=[0, 1], default=1, help="Whether to use R_structure (1=yes, 0=no)")
    parser.add_argument("--use_r_category_valid", type=int, choices=[0, 1], default=1, help="Whether to use R_category_valid (1=yes, 0=no)")
    parser.add_argument("--use_cache", type=int, choices=[0, 1], default=1, help="Whether to use cache/context injection in both training and generation (1=yes, 0=no)")
    
    parser.add_argument("--dissimilar_target", type=float, default=0.9, help="Target threshold for dissimilar reward")
    parser.add_argument("--structure_target", type=float, default=0.9, help="Target threshold for structure reward")
    parser.add_argument("--category_valid_target", type=float, default=0.9, help="Target threshold for category valid reward")
    
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Max prompt length")
    parser.add_argument("--max_completion_length", type=int, default=16, help="Max completion length")
    
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to use (None for all)")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--context_injection_step", type=int, default=2500, help="Step to start context injection")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--min_p", type=float, default=0.05, help="min_p sampling parameter")
    
    parser.add_argument("--per_word_max_variants", type=int, default=8, help="Max variants per word in cache")
    parser.add_argument("--global_recent_max", type=int, default=None, help="Max recent words in global cache (None for no limit)")
    parser.add_argument(
        "--core_train_cache_max",
        type=int,
        default=None,
        help="Cache cap used only in GRPO training phase (keep latest N words for context injection; None for no limit)",
    )
    parser.add_argument("--history_window", type=int, default=500, help="History window for reward calculation")
    parser.add_argument("--context_top_k", type=int, default=10, help="Top k similar words for context injection")
    parser.add_argument("--max_steps", type=int, default=10000, help="Max training steps")
    parser.add_argument("--enable_tensorboard", type=int, choices=[0, 1], default=1, help="Enable TensorBoard logging during training (1=yes, 0=no)")

    parser.add_argument("--generate_output_file", type=str, default=None, help="Output file for generated variants")
    parser.add_argument("--generate_fine_details_file", type=str, default=None, help="Fine details file for generation phase")
    parser.add_argument("--generate_similarities_file", type=str, default=None, help="Similarities file for generation phase")
    parser.add_argument("--generate_batch_size", type=int, default=8, help="Batch size for generation phase")
    parser.add_argument("--generate_num_generations", type=int, default=8, help="Number of generations per sample in generation phase")
    parser.add_argument("--generate_debug_log", action="store_true", help="Write detailed prompt/variant/reward logs during generation")
    parser.add_argument("--load_cache_file", type=str, default=None, help="Load cache state file before generation")
    parser.add_argument("--generation_model_path", type=str, default=None, help="Model path used in generation-only mode")
    
    args = parser.parse_args()
    
    # 检测数据集类型并设置默认路径
    if args.dataset:
        dataset_name = args.dataset
    elif args.fine_details_file:
        dataset_name = detect_dataset_from_path(args.fine_details_file)
    else:
        dataset_name = 'movielens'
    
    paths = get_default_paths(dataset_name)
    
    # 使用命令行参数或默认路径
    fine_details_file = args.fine_details_file if args.fine_details_file else paths['fine_details_file']
    similarities_file = args.similarities_file if args.similarities_file else paths['similarities_file']
    embeddings_file = args.embeddings_file if args.embeddings_file else paths['embeddings_file']
    embedding_model_path = args.embedding_model_path if args.embedding_model_path else paths['embedding_model_path']
    category_statistics_file = args.category_statistics_file if args.category_statistics_file else paths['category_statistics_file']
    output_dir = args.output_dir if args.output_dir else paths['output_dir']
    prompt_file = args.prompt_file if args.prompt_file else paths['prompt_file']
    
    # 检查prompt文件是否存在
    if prompt_file and not os.path.exists(prompt_file):
        print(f"Error: Prompt file not found: {prompt_file}")
        print("Prompt template is required")
        exit(1)
    
    print(f"Processing dataset: {dataset_name}")
    print(f"Prompt template: {prompt_file}")
    use_cache = bool(args.use_cache)
    effective_use_r_structure = bool(args.use_r_structure) and use_cache
    if bool(args.use_r_structure) and not use_cache:
        print("use_cache=0 -> force disabling use_r_structure in both training and generation.")
    print(
        "Reward switches: "
        f"dissimilar={bool(args.use_r_dissimilar)}, "
        f"structure={effective_use_r_structure}, "
        f"category_valid={bool(args.use_r_category_valid)}"
    )

    shared_cache_manager = None
    if use_cache:
        # 默认缓存无上限；仅在训练阶段可通过 core_train_cache_max 开启“最近N词”窗口。
        training_phase_cache_max = args.core_train_cache_max if args.max_steps > 0 else None
        shared_cache_manager = CacheManager(
            per_word_max_variants=args.per_word_max_variants,
            global_recent_max=training_phase_cache_max,
        )
        print(
            "Cache mode: "
            f"training_phase_cache_max={training_phase_cache_max}, "
            "generation_phase_cache_max=None (unbounded)"
        )
        if args.load_cache_file:
            if os.path.exists(args.load_cache_file):
                shared_cache_manager.load_from_file(args.load_cache_file)
                print(f"Loaded cache state from {args.load_cache_file}")
            else:
                print(f"Warning: cache file not found, skip loading: {args.load_cache_file}")
        # 仅生成阶段恢复为无上限，避免继续按训练窗口淘汰。
        if args.max_steps <= 0:
            shared_cache_manager.global_recent_max = None
            print("Generation-only mode: cache cap disabled (unbounded).")
    else:
        print("Cache mode: disabled (no context injection in training and generation).")
        if args.load_cache_file:
            print("Warning: --load_cache_file is ignored because --use_cache=0.")

    generation_model_path = output_dir
    if args.max_steps <= 0:
        print("max_steps <= 0, skipping GRPO training and LoRA creation.")
        generation_model_path = args.generation_model_path if args.generation_model_path else args.model_name_or_path
        print(f"Generation model path: {generation_model_path}")
    else:
        _, _ = run_train_grpo(
            model_name_or_path=args.model_name_or_path,
            sft_lora_path=args.sft_lora_path,
            fine_details_file=fine_details_file,
            similarities_file=similarities_file,
            embeddings_file=embeddings_file,
            embedding_model_path=embedding_model_path,
            category_statistics_file=category_statistics_file,
            prompt_file=prompt_file,
            output_dir=output_dir,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            learning_rate=args.learning_rate,
            num_generations=args.num_generations,
            beta=args.beta,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            lambda3=args.lambda3,
            use_r_dissimilar=bool(args.use_r_dissimilar),
            use_r_structure=effective_use_r_structure,
            use_r_category_valid=bool(args.use_r_category_valid),
            dissimilar_target=args.dissimilar_target,
            structure_target=args.structure_target,
            category_valid_target=args.category_valid_target,
            max_seq_length=args.max_seq_length,
            max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length,
            num_samples=args.num_samples,
            seed=args.seed,
            batch_size=args.batch_size,
            context_injection_step=args.context_injection_step,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            temperature=args.temperature,
            min_p=args.min_p,
            per_word_max_variants=args.per_word_max_variants,
            global_recent_max=args.global_recent_max,
            history_window=args.history_window,
            context_top_k=args.context_top_k,
            cache_manager=shared_cache_manager,
            use_cache=use_cache,
            max_steps=args.max_steps,
            enable_tensorboard=bool(args.enable_tensorboard),
            return_model_for_inference=False,
        )
        # 训练结束后，在主流程再做一次显式显存清理，避免残留占用影响后续生成加载。
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        # 将训练阶段累积的上下文缓存落盘，并重启为“仅生成”进程，彻底释放显存。
        cache_state_file = None
        if use_cache and shared_cache_manager is not None:
            cache_state_file = os.path.join(output_dir, "cache_state.json")
            shared_cache_manager.save_to_file(cache_state_file)
            print(f"Saved cache state to {cache_state_file}")
        print("Restarting process for generation-only phase to ensure clean GPU memory...")
        raw_args = sys.argv[1:]
        restart_args = []
        skip_next = False
        skip_keys = {"--max_steps", "--load_cache_file", "--generation_model_path"}
        for i, arg in enumerate(raw_args):
            if skip_next:
                skip_next = False
                continue
            if arg in skip_keys:
                skip_next = True
                continue
            if any(arg.startswith(f"{k}=") for k in skip_keys):
                continue
            restart_args.append(arg)
        exec_argv = [
            sys.executable,
            __file__,
            *restart_args,
            "--max_steps", "0",
            "--generation_model_path", output_dir,
        ]
        if cache_state_file:
            exec_argv.extend(["--load_cache_file", cache_state_file])
        os.execv(sys.executable, exec_argv)

    generate_output_file = args.generate_output_file if args.generate_output_file else paths['generated_output_file']
    generate_fine_details_file = args.generate_fine_details_file if args.generate_fine_details_file else fine_details_file
    generate_similarities_file = args.generate_similarities_file if args.generate_similarities_file else similarities_file

    print("\n" + "=" * 80)
    print("Entering eval-phase generation...")
    print(f"Generate output file: {generate_output_file}")
    print(f"Generate fine details file: {generate_fine_details_file}")
    print(f"Generate similarities file: {generate_similarities_file}")
    print(f"Generate debug log: {args.generate_debug_log}")
    print("Generate inference mode: standard Transformers (fast_inference=False)")
    print("=" * 80)

    batch_inference(
        fine_details_file=generate_fine_details_file,
        similarities_file=generate_similarities_file,
        model_path=generation_model_path,
        output_file=generate_output_file,
        batch_size=args.generate_batch_size,
        num_generations=args.generate_num_generations,
        prompt_file=prompt_file,
        dataset_name=dataset_name,
        cache_manager=shared_cache_manager,
        model=None,
        tokenizer=None,
        max_prompt_length=args.max_prompt_length,
        temperature=args.temperature,
        min_p=args.min_p,
        warmup_fine_details_file=fine_details_file,
        warmup_similarities_file=similarities_file,
        enable_debug_log=args.generate_debug_log,
        use_fast_inference=False,
        reward_metrics_file=os.path.join(output_dir, "reward_step_metrics.txt"),
        tensorboard_log_dir=os.path.join(output_dir, "tensorboard"),
        use_cache=use_cache,
    )
