import argparse
import itertools
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_list(raw, cast):
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Grid-search runner for MMRec FREEDOM/BPR.")
    parser.add_argument("--dataset", type=str, default="ml-1m", choices=["ml-1m", "amazon-book"])
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lrs", type=str, default="5e-3,1e-3")
    parser.add_argument("--weight_decays", type=str, default="5e-2,1e-2,5e-3")
    parser.add_argument("--batch_sizes", type=str, default="2048")
    parser.add_argument("--dims", type=str, default="64")
    parser.add_argument("--num_layer", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_trials", type=int, default=0, help="0 means run all.")
    parser.add_argument("--log_root", type=str, default="./logs/mmrec_freedom")
    parser.add_argument("--edge_variant", type=str, default="v2")
    parser.add_argument("--embedding_root", type=str, default="../../embedding/data")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=4096)
    parser.add_argument("--stopping_step", type=int, default=20)
    parser.add_argument("--prepare_only", action="store_true", help="Only prepare MMRec-format data and exit.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate prepared files even if they already exist.")
    parser.add_argument("--save_model", action="store_true", help="Keep MMRec checkpoints.")
    return parser.parse_args()


def expand_holdout_rows(rows, split_label):
    interactions = []
    for row in rows:
        user = int(row[0])
        for item in row[1:]:
            interactions.append((user, int(item), split_label))
    return interactions


def load_item2id(path):
    with open(path, "r", encoding="utf-8") as f:
        item2id = json.load(f)

    local_ids = sorted(int(v) for v in item2id.values())
    expected = list(range(len(local_ids)))
    if local_ids != expected:
        raise ValueError(f"Item ids in {path} are not contiguous from 0 to n-1.")
    return item2id


def build_text_features(item2id, embed_map, embeddings):
    num_items = len(item2id)
    features = np.empty((num_items, embeddings.shape[1]), dtype=embeddings.dtype)

    for raw_item_id, local_item_id in item2id.items():
        if raw_item_id not in embed_map:
            raise KeyError(f"Missing raw item id {raw_item_id} in embedding map.")
        features[int(local_item_id)] = embeddings[int(embed_map[raw_item_id])]

    return features


def prepare_mmrec_dataset(project_root, dataset, edge_variant, embedding_root, overwrite=False):
    source_data_dir = project_root / "data" / dataset
    embed_root_abs = (project_root / embedding_root).resolve()
    embed_dir = embed_root_abs / dataset / edge_variant
    mmrec_root = project_root / "MMRec"
    prepared_root = mmrec_root / "data"
    prepared_name = f"{dataset}_freedom_{edge_variant}"
    prepared_dir = prepared_root / prepared_name
    prepared_dir.mkdir(parents=True, exist_ok=True)

    inter_path = prepared_dir / "inter.tsv"
    text_feat_path = prepared_dir / "text_feat.npy"
    manifest_path = prepared_dir / "manifest.json"
    inject_text = edge_variant != "v0"

    if inter_path.exists() and manifest_path.exists() and not overwrite:
        if (not inject_text) or text_feat_path.exists():
            return prepared_root, prepared_name, prepared_dir, inject_text

    if text_feat_path.exists() and (not inject_text):
        text_feat_path.unlink()

    required_source_files = [
        source_data_dir / "train.npy",
        source_data_dir / "val_full.npy",
        source_data_dir / "test_full.npy",
        source_data_dir / "item2id.json",
    ]
    if inject_text:
        required_source_files.extend(
            [
                embed_dir / "item_embeddings.npy",
                embed_dir / "item_id_map.json",
            ]
        )

    missing_files = [str(path) for path in required_source_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {missing_files}")

    train = np.load(source_data_dir / "train.npy", allow_pickle=True)
    val = np.load(source_data_dir / "val_full.npy", allow_pickle=True)
    test = np.load(source_data_dir / "test_full.npy", allow_pickle=True)
    item2id = load_item2id(source_data_dir / "item2id.json")

    train_rows = [(int(user), int(item), 0) for user, item in train.tolist()]
    val_rows = expand_holdout_rows(val, 1)
    test_rows = expand_holdout_rows(test, 2)
    all_rows = train_rows + val_rows + test_rows

    with open(inter_path, "w", encoding="utf-8") as f:
        f.write("user_id:token\titem_id:token\tx_label\n")
        for user, item, split in all_rows:
            f.write(f"{user}\t{item}\t{split}\n")

    text_feat_shape = None
    if inject_text:
        item_embeddings = np.load(embed_dir / "item_embeddings.npy")
        with open(embed_dir / "item_id_map.json", "r", encoding="utf-8") as f:
            item_id_map = json.load(f)
        text_features = build_text_features(item2id, item_id_map, item_embeddings)
        np.save(text_feat_path, text_features)
        text_feat_shape = list(text_features.shape)

    manifest = {
        "source_dataset": dataset,
        "edge_variant": edge_variant,
        "inject_text": inject_text,
        "prepared_dataset": prepared_name,
        "num_users": int(max(max(int(train[:, 0].max()) + 1, len(val)), len(test))),
        "num_items": len(item2id),
        "num_train_interactions": len(train_rows),
        "num_val_interactions": len(val_rows),
        "num_test_interactions": len(test_rows),
        "text_feat_shape": text_feat_shape,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return prepared_root, prepared_name, prepared_dir, inject_text


def run_single_trial(project_root, prepared_root, prepared_name, inject_text, args, trial_id, hp, run_dir):
    mmrec_src = project_root / "MMRec" / "src"
    model_name = "FREEDOM" if inject_text else "BPR"
    checkpoint_dir = Path(run_dir) / "saved"
    topk_dir = Path(run_dir) / "recommend_topk"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    topk_dir.mkdir(parents=True, exist_ok=True)

    lr, reg_weight, train_bs, dim = hp["lr"], hp["wd"], hp["bs"], hp["dim"]
    print(f"\n=== Trial {trial_id} | model={model_name} | hp={hp} ===")

    sys.path.insert(0, str(mmrec_src))
    old_cwd = os.getcwd()
    start = time.time()
    status = "ok"
    error = ""
    mmrec_summary = None
    try:
        os.chdir(mmrec_src)
        from utils.quick_start import quick_start

        config_dict = {
            "gpu_id": args.gpu_id,
            "use_gpu": not args.cpu,
            "seed": [args.seed],
            "hyper_parameters": [],
            "metrics": ["Recall", "NDCG", "Precision", "MAP", "HR"],
            "data_path": str(prepared_root) + os.sep,
            "inter_file_name": "inter.tsv",
            "is_multimodal_model": bool(inject_text),
            "end2end": False,
            "filter_out_cod_start_users": True,
            "use_neighborhood_loss": False,
            "embedding_size": dim,
            "epochs": args.num_epoch,
            "stopping_step": args.stopping_step,
            "train_batch_size": train_bs,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": lr,
            # 复用 run_mmgcn 的 weight_decays 命名，对应 MMRec 的 reg_weight
            "reg_weight": reg_weight,
            "checkpoint_dir": str(checkpoint_dir),
            "save_recommended_topk": False,
            "recommend_topk": str(topk_dir),
            "vision_feature_file": "image_feat.npy",
            "text_feature_file": "text_feat.npy",
        }
        if model_name == "FREEDOM":
            # Align defaults with MMRec FREEDOM config while keeping runner CLI unchanged.
            config_dict.update(
                {
                    "feat_embed_dim": dim,
                    "knn_k": 10,
                    "lambda_coeff": 0.9,
                    "cf_model": "lightgcn",
                    "n_mm_layers": 1,
                    "n_ui_layers": 2,
                    "mm_image_weight": 0.1,
                    "dropout": 0.8,
                    "degree_ratio": 0.0,
                }
            )
        mmrec_summary = quick_start(model=model_name, dataset=prepared_name, config_dict=config_dict, save_model=args.save_model)
        if isinstance(mmrec_summary, dict):
            bv = mmrec_summary.get("best_valid_then_test", {})
            bt = mmrec_summary.get("best_test_any", {})
            print(f"[trial {trial_id}] best_valid_then_test={bv}")
            print(f"[trial {trial_id}] best_test_any={bt}")
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error = repr(exc)
        print(f"[trial {trial_id}] failed: {error}")
    finally:
        os.chdir(old_cwd)

    return {
        "trial_id": trial_id,
        "status": status,
        "error": error,
        "elapsed_sec": round(time.time() - start, 3),
        "model": model_name,
        "hp": hp,
        "mmrec_summary": mmrec_summary,
    }


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    prepared_root, prepared_name, prepared_dir, inject_text = prepare_mmrec_dataset(
        project_root=project_root,
        dataset=args.dataset,
        edge_variant=args.edge_variant,
        embedding_root=args.embedding_root,
        overwrite=args.overwrite,
    )

    print(f"Prepared MMRec dataset: {prepared_dir}")
    print(f"Edge variant: {args.edge_variant}, inject_text={inject_text}")
    if args.prepare_only:
        return prepared_root, prepared_name, prepared_dir

    lrs = parse_list(args.lrs, float)
    wds = parse_list(args.weight_decays, float)
    bss = parse_list(args.batch_sizes, int)
    dims = parse_list(args.dims, int)
    grid = [{"lr": lr, "wd": wd, "bs": bs, "dim": dim} for lr, wd, bs, dim in itertools.product(lrs, wds, bss, dims)]
    if args.max_trials > 0:
        grid = grid[: args.max_trials]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.abspath(os.path.join(args.log_root, f"{args.dataset}_{args.edge_variant}_{timestamp}"))
    os.makedirs(run_dir, exist_ok=True)

    print(f"Running {len(grid)} MMRec trials on dataset={args.dataset}, edge_variant={args.edge_variant}")
    all_results = []
    for idx, hp in enumerate(grid, start=1):
        result = run_single_trial(
            project_root=project_root,
            prepared_root=prepared_root,
            prepared_name=prepared_name,
            inject_text=inject_text,
            args=args,
            trial_id=idx,
            hp=hp,
            run_dir=run_dir,
        )
        all_results.append(result)

    metric_key = "recall@20"
    valid_then_test_rank = []
    test_any_rank = []
    for r in all_results:
        if r.get("status") != "ok" or not isinstance(r.get("mmrec_summary"), dict):
            continue
        mmrec_summary = r["mmrec_summary"]
        best_valid_then_test = mmrec_summary.get("best_valid_then_test", {})
        best_test_any = mmrec_summary.get("best_test_any", {})
        valid_then_test_score = float(best_valid_then_test.get("test", {}).get(metric_key, -1.0))
        test_any_score = float(best_test_any.get("test", {}).get(metric_key, -1.0))
        valid_then_test_rank.append((valid_then_test_score, r))
        test_any_rank.append((test_any_score, r))

    best_valid_then_test_trial = max(valid_then_test_rank, key=lambda x: x[0])[1] if valid_then_test_rank else None
    best_test_any_trial = max(test_any_rank, key=lambda x: x[0])[1] if test_any_rank else None

    summary = {
        "dataset": args.dataset,
        "edge_variant": args.edge_variant,
        "inject_text": inject_text,
        "ranking_metric": metric_key,
        "num_trials": len(all_results),
        "best_valid_then_test_trial": best_valid_then_test_trial,
        "best_test_any_trial": best_test_any_trial,
        "all_results": all_results,
    }
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    ok_count = sum(1 for x in all_results if x["status"] == "ok")
    print(f"\nTrials done: {ok_count}/{len(all_results)} success")
    if best_valid_then_test_trial is not None:
        print(f"Best valid->test trial ({metric_key}): {best_valid_then_test_trial['trial_id']}")
    if best_test_any_trial is not None:
        print(f"Best test-any trial ({metric_key}): {best_test_any_trial['trial_id']}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
