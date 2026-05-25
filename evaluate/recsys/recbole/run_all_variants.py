import argparse
import json
import os
import subprocess

MAX_EPOCH_CAP = 50


def parse_str_list(value):
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_int_list(value):
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_float_list(value):
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_bin", type=str, default="python")
    parser.add_argument("--script_path", type=str, default="./train_eval_recbole.py")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--dataset", type=str, default="amazon_book_recbole") #amazon_book_recbole, ml1m_recbole
    parser.add_argument("--models", type=str, default="sasrec") #default="lightgcn,sasrec"
    parser.add_argument("--embedding_variants", type=str, default="v3")
    parser.add_argument("--batch_sizes", type=str, default="1024")
    parser.add_argument("--learning_rates", type=str, default="1e-4,5e-4,1e-3,5e-3")
    parser.add_argument("--embedding_sizes", type=str, default="128")
    parser.add_argument("--hidden_sizes", type=str, default="64")
    parser.add_argument("--max_seq_lens", type=str, default="20")
    parser.add_argument("--epochs", type=str, default="50")
    parser.add_argument("--seeds", type=str, default="2026")
    parser.add_argument("--gpu_id", type=str, default="1")
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--stop_on_error", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(dataset_output_dir, exist_ok=True)
    models = parse_str_list(args.models)
    embedding_variants = parse_str_list(args.embedding_variants)
    batch_sizes = parse_int_list(args.batch_sizes)
    learning_rates = parse_float_list(args.learning_rates)
    embedding_sizes = parse_int_list(args.embedding_sizes)
    hidden_sizes = parse_int_list(args.hidden_sizes)
    max_seq_lens = parse_int_list(args.max_seq_lens)
    epochs_list = parse_int_list(args.epochs)
    seeds = parse_int_list(args.seeds)

    print("=" * 100)
    print("RecBole Grid Search Config")
    print("models", models)
    print("variants", embedding_variants)
    print("batch_sizes", batch_sizes)
    print("learning_rates", learning_rates)
    print("embedding_sizes", embedding_sizes)
    print("hidden_sizes", hidden_sizes)
    print("max_seq_lens", max_seq_lens)
    print("epochs", epochs_list)
    print("seeds", seeds)
    print("=" * 100)

    runs = []
    total = 0
    for model in models:
        for variant in embedding_variants:
            variant_output_dir = os.path.join(dataset_output_dir, model, variant)
            os.makedirs(variant_output_dir, exist_ok=True)
            log_dir = os.path.join(variant_output_dir, "group_logs")
            os.makedirs(log_dir, exist_ok=True)
            group_log_path = os.path.join(log_dir, f"{model}_{variant}.log")
            for batch_size in batch_sizes:
                for lr in learning_rates:
                    for embedding_size in embedding_sizes:
                        for hidden_size in hidden_sizes:
                            for max_seq_len in max_seq_lens:
                                for epochs in epochs_list:
                                    effective_epochs = min(epochs, MAX_EPOCH_CAP)
                                    for seed in seeds:
                                        total += 1
                                        run_name = (
                                            f"{model}_{variant}_bs{batch_size}_lr{lr}"
                                            f"_emb{embedding_size}_hid{hidden_size}"
                                            f"_seq{max_seq_len}_ep{effective_epochs}_seed{seed}"
                                        )
                                        output_json = os.path.join(variant_output_dir, f"{run_name}.json")
                                        cmd = [
                                            args.python_bin,
                                            args.script_path,
                                            "--model", model,
                                            "--embedding_variant", variant,
                                            "--dataset", args.dataset,
                                            "--train_batch_size", str(batch_size),
                                            "--learning_rate", str(lr),
                                            "--embedding_size", str(embedding_size),
                                            "--hidden_size", str(hidden_size),
                                            "--max_seq_len", str(max_seq_len),
                                            "--epochs", str(effective_epochs),
                                            "--seed", str(seed),
                                            "--gpu_id", args.gpu_id,
                                            "--output_json", output_json,
                                        ]
                                        if args.use_gpu:
                                            cmd.append("--use_gpu")
                                        print("RUN", total, run_name)
                                        print(" ".join(cmd))
                                        try:
                                            with open(group_log_path, "a", encoding="utf-8") as log_f:
                                                log_f.write("=" * 120 + "\n")
                                                log_f.write(f"RUN {total}: {run_name}\n")
                                                log_f.write("CMD: " + " ".join(cmd) + "\n")
                                                log_f.flush()
                                                subprocess.run(cmd, check=True, stdout=log_f, stderr=subprocess.STDOUT)
                                            with open(output_json, "r", encoding="utf-8") as f:
                                                result = json.load(f)
                                                runs.append(result)
                                            with open(group_log_path, "a", encoding="utf-8") as log_f:
                                                log_f.write("RESULT_JSON: " + output_json + "\n")
                                                if "test_result" in result:
                                                    log_f.write("TEST_RESULT: " + json.dumps(result["test_result"], ensure_ascii=False) + "\n")
                                                if "test_result_best_valid" in result:
                                                    log_f.write("TEST_RESULT_BEST_VALID: " + json.dumps(result["test_result_best_valid"], ensure_ascii=False) + "\n")
                                                if "test_result_best_test" in result:
                                                    log_f.write("TEST_RESULT_BEST_TEST: " + json.dumps(result["test_result_best_test"], ensure_ascii=False) + "\n")
                                                log_f.write("\n")
                                        except Exception as e:
                                            failed = {
                                                "run_name": run_name,
                                                "error": str(e),
                                                "model": model,
                                                "embedding_variant": variant,
                                                "train_batch_size": batch_size,
                                                "learning_rate": lr,
                                                "embedding_size": embedding_size,
                                                "hidden_size": hidden_size,
                                                "max_seq_len": max_seq_len,
                                                "epochs": effective_epochs,
                                                "seed": seed,
                                            }
                                            runs.append(failed)
                                            print("FAILED", run_name, str(e))
                                            with open(group_log_path, "a", encoding="utf-8") as log_f:
                                                log_f.write("FAILED: " + str(e) + "\n\n")
                                            if args.stop_on_error:
                                                raise
    summary_path = os.path.join(dataset_output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(runs, f, ensure_ascii=False, indent=2)
    print(f"saved summary to {summary_path}")


if __name__ == "__main__":
    main()
