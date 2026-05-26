import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime


def parse_list(raw, cast):
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def ensure_paths(project_root, dataset):
    data_dir = os.path.join(project_root, "data", dataset)
    if not os.path.exists(data_dir):
        data_dir = os.path.join(project_root, "Data", dataset)
    required = [
        os.path.join(data_dir, "train.npy"),
        os.path.join(data_dir, "val.npy"),
        os.path.join(data_dir, "test.npy"),
        os.path.join(data_dir, "user_item_dict.npy"),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing dataset files: {missing}")


def parse_args():
    parser = argparse.ArgumentParser(description="Grid-search runner for LightGT.")
    parser.add_argument("--dataset", type=str, default="ml-1m", choices=["ml-1m", "amazon-book"])
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lrs", type=str, default="1e-2,5e-3,1e-3,5e-4")
    parser.add_argument("--weight_decays", type=str, default="0,1e-3,5e-4")
    parser.add_argument("--batch_sizes", type=str, default="2048")
    parser.add_argument("--dims", type=str, default="64")
    parser.add_argument("--src_lens", type=str, default="50")
    parser.add_argument("--transformer_layers", type=str, default="4")
    parser.add_argument("--lightgcn_layers", type=str, default="4")
    parser.add_argument("--score_weights", type=str, default="0.5")
    parser.add_argument("--nheads", type=str, default="1")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_trials", type=int, default=0, help="0 means run all.")
    parser.add_argument("--log_root", type=str, default="./logs/lightgt")
    parser.add_argument("--has_v", type=str, default="False")
    parser.add_argument("--has_a", type=str, default="False")
    parser.add_argument("--has_t", type=str, default="True")
    parser.add_argument("--has_e", type=str, default="False")
    parser.add_argument("--edge_variant", type=str, default="v3")
    parser.add_argument("--embedding_root", type=str, default="../../../embedding/data")
    return parser.parse_args()


def run_trial(model_dir, args, trial_id, hp, run_dir):
    command = [
        args.python_bin,
        "main.py",
        "--dataset", args.dataset,
        "--seed", str(args.seed),
        "--l_r", str(hp["lr"]),
        "--weight_decay", str(hp["wd"]),
        "--batch_size", str(hp["bs"]),
        "--dim_E", str(hp["dim"]),
        "--src_len", str(hp["src_len"]),
        "--transformer_layers", str(hp["tf_layers"]),
        "--lightgcn_layers", str(hp["gcn_layers"]),
        "--score_weight", str(hp["score_weight"]),
        "--nhead", str(hp["nhead"]),
        "--num_epoch", str(args.num_epoch),
        "--num_workers", str(args.num_workers),
        "--has_v", args.has_v,
        "--has_a", args.has_a,
        "--has_t", args.has_t,
        "--has_e", args.has_e,
        "--edge_variant", args.edge_variant,
        "--embedding_root", args.embedding_root,
    ]

    env = os.environ.copy()
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTHONUNBUFFERED", "1")

    metric_re = re.compile(
        r"Precition:(?P<p>\d+\.\d+)\s+Recall:(?P<r>\d+\.\d+)\s+NDCG:(?P<n>\d+\.\d+)\s+HR:(?P<h>\d+\.\d+)"
    )
    phase = None
    current_val = None
    best = {
        "val_recall": -1.0,
        "test_precision": 0.0,
        "test_recall": 0.0,
        "test_ndcg": 0.0,
        "test_hr": 0.0,
    }

    log_path = os.path.join(run_dir, f"trial_{trial_id}.log")
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        process = subprocess.Popen(
            command,
            cwd=model_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            logf.write(line)
            print(f"[trial {trial_id}] {line}", end="")
            if "Val start" in line:
                phase = "val"
                continue
            if "Test start" in line:
                phase = "test"
                continue
            m = metric_re.search(line)
            if not m:
                continue
            precision = float(m.group("p"))
            recall = float(m.group("r"))
            ndcg = float(m.group("n"))
            hit = float(m.group("h"))
            if phase == "val":
                current_val = {"precision": precision, "recall": recall, "ndcg": ndcg, "hr": hit}
            elif phase == "test" and current_val is not None:
                if current_val["recall"] > best["val_recall"]:
                    best["val_recall"] = current_val["recall"]
                    best["test_precision"] = precision
                    best["test_recall"] = recall
                    best["test_ndcg"] = ndcg
                    best["test_hr"] = hit
                current_val = None
        ret = process.wait()

    elapsed = time.time() - start
    result = {
        "trial_id": trial_id,
        "hp": hp,
        "return_code": ret,
        "elapsed_sec": elapsed,
        **best,
        "log_path": log_path,
    }
    return result


def main():
    args = parse_args()
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(project_root, "LightGT")
    ensure_paths(project_root, args.dataset)

    lrs = parse_list(args.lrs, float)
    wds = parse_list(args.weight_decays, float)
    bss = parse_list(args.batch_sizes, int)
    dims = parse_list(args.dims, int)
    src_lens = parse_list(args.src_lens, int)
    tf_layers = parse_list(args.transformer_layers, int)
    gcn_layers = parse_list(args.lightgcn_layers, int)
    score_weights = parse_list(args.score_weights, float)
    nheads = parse_list(args.nheads, int)
    grid = [
        {
            "lr": lr,
            "wd": wd,
            "bs": bs,
            "dim": dim,
            "src_len": src_len,
            "tf_layers": tf_layer,
            "gcn_layers": gcn_layer,
            "score_weight": sw,
            "nhead": nhead,
        }
        for lr, wd, bs, dim, src_len, tf_layer, gcn_layer, sw, nhead in itertools.product(
            lrs, wds, bss, dims, src_lens, tf_layers, gcn_layers, score_weights, nheads
        )
    ]
    if args.max_trials > 0:
        grid = grid[: args.max_trials]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.abspath(os.path.join(args.log_root, f"{args.dataset}_{timestamp}"))
    os.makedirs(run_dir, exist_ok=True)

    all_results = []
    print(f"Running {len(grid)} LightGT trials on dataset={args.dataset}")
    for i, hp in enumerate(grid, start=1):
        print(f"\n=== Trial {i}/{len(grid)} | hp={hp} ===")
        result = run_trial(model_dir, args, i, hp, run_dir)
        all_results.append(result)

    valid = [x for x in all_results if x["return_code"] == 0 and x["val_recall"] >= 0]
    if not valid:
        raise RuntimeError("No successful trial with parsed metrics. Check logs.")

    best = max(valid, key=lambda x: x["val_recall"])
    summary = {
        "dataset": args.dataset,
        "num_trials": len(all_results),
        "best_trial": best,
        "all_results": all_results,
    }
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== LightGT Best Trial =====")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
