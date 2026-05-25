import argparse
import copy
import json
import os
import sys
import logging
import shutil
import numpy as np
import torch
from scipy.sparse import dok_matrix

# Fix for PyTorch 2.6+ compatibility with RecBole model loading
def patch_torch_load():
    original_load = torch.load
    
    def patched_load(f, map_location=None, **kwargs):
        # For RecBole model files, use weights_only=False for compatibility
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_load(f, map_location=map_location, **kwargs)
    
    torch.load = patched_load

patch_torch_load()

if not hasattr(dok_matrix, "_update"):
    def _compat_dok_update(self, data_dict):
        for key, value in data_dict.items():
            self[key] = value
    dok_matrix._update = _compat_dok_update

if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_
if not hasattr(np, "int_"):
    np.int_ = np.int64
if not hasattr(np, "bool_"):
    np.bool_ = np.bool

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, get_trainer, set_color
from recbole.model.sequential_recommender.sasrec import SASRec
from recbole.model.context_aware_recommender.deepfm import DeepFM
from recbole.model.general_recommender.neumf import NeuMF
from recbole.model.general_recommender.lightgcn import LightGCN
import recbole.trainer.trainer as recbole_trainer_module
from models import (
    TwoTower,
    PretrainedSASRec,
    PretrainedDeepFM,
    PretrainedNeuMF,
    PretrainedLightGCN,
)

MAX_EPOCH_CAP = 50


class _NoOpSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def add_hparams(self, *args, **kwargs):
        return None

    def close(self):
        return None


def _disable_recbole_tensorboard():
    recbole_trainer_module.get_tensorboard = lambda logger: _NoOpSummaryWriter()


def _init_console_logger(config_state):
    logger = logging.getLogger()
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)
    if config_state is None or str(config_state).lower() == "info":
        level = logging.INFO
    elif str(config_state).lower() == "debug":
        level = logging.DEBUG
    elif str(config_state).lower() == "error":
        level = logging.ERROR
    elif str(config_state).lower() == "warning":
        level = logging.WARNING
    elif str(config_state).lower() == "critical":
        level = logging.CRITICAL
    else:
        level = logging.INFO
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)-15s %(levelname)s  %(message)s", "%d %b %H:%M")
    handler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["sasrec", "two_tower", "deepfm", "neumf", "lightgcn"], default="sasrec")
    parser.add_argument("--embedding_variant", type=str, default="v3")
    parser.add_argument("--dataset", type=str, default="ml1m_recbole")
    parser.add_argument("--data_path", type=str, default="./data/recbole")
    parser.add_argument("--embedding_root", type=str, default="../embedding")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train_batch_size", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--gpu_id", type=str, default="1")
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output_json", type=str, default="./data/results.json")
    return parser.parse_args()


def infer_raw_dataset_name(recbole_dataset_name):
    name = str(recbole_dataset_name).lower()
    if "amazon" in name:
        return "amazon-book"
    return "ml-1m"


def build_embedding_paths(args):
    if args.embedding_variant == "none":
        return "", ""
    raw_dataset_name = infer_raw_dataset_name(args.dataset)
    variant = args.embedding_variant
    embedding_candidates = [
        os.path.join(args.embedding_root, "data", raw_dataset_name, variant, "item_embeddings.npy"),
        os.path.join(args.embedding_root, variant, "item_embeddings.npy"),
    ]
    map_candidates = [
        os.path.join(args.embedding_root, "data", raw_dataset_name, variant, "item_id_map.json"),
        os.path.join(args.embedding_root, variant, "item_id_map.json"),
    ]
    emb_path = next((p for p in embedding_candidates if os.path.exists(p)), embedding_candidates[0])
    map_path = next((p for p in map_candidates if os.path.exists(p)), map_candidates[0])
    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"pretrained embedding file not found for dataset={raw_dataset_name}, variant={variant}. "
            f"Tried: {embedding_candidates}"
        )
    if not os.path.exists(map_path):
        raise FileNotFoundError(
            f"pretrained id map file not found for dataset={raw_dataset_name}, variant={variant}. "
            f"Tried: {map_candidates}"
        )
    return emb_path, map_path


def has_presplit_dataset(data_path, dataset):
    dataset_dir = os.path.join(data_path, dataset)
    required_files = [
        os.path.join(dataset_dir, f"{dataset}.train.inter"),
        os.path.join(dataset_dir, f"{dataset}.valid.inter"),
        os.path.join(dataset_dir, f"{dataset}.test.inter"),
    ]
    return all(os.path.exists(path) for path in required_files)


def has_presplit_seq_dataset(data_path, dataset):
    dataset_dir = os.path.join(data_path, dataset)
    required_files = [
        os.path.join(dataset_dir, f"{dataset}.train_seq.inter"),
        os.path.join(dataset_dir, f"{dataset}.valid_seq.inter"),
        os.path.join(dataset_dir, f"{dataset}.test_seq.inter"),
    ]
    return all(os.path.exists(path) for path in required_files)


def extend_token_vocab(dataset, field, extra_tokens):
    token2id = dataset.field2token_id[field]
    id2token = dataset.field2id_token[field]
    new_tokens = [token for token in extra_tokens if token and token not in token2id]
    if not new_tokens:
        return 0
    start_idx = len(id2token)
    for offset, token in enumerate(new_tokens):
        token2id[token] = start_idx + offset
    dataset.field2id_token[field] = np.concatenate(
        [id2token, np.asarray(new_tokens, dtype=id2token.dtype)]
    )
    return len(new_tokens)


def load_all_item_tokens(data_path, dataset_name):
    inter_path = os.path.join(data_path, dataset_name, f"{dataset_name}.inter")
    if not os.path.exists(inter_path):
        return []
    tokens = []
    with open(inter_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        field_names = [column.split(":")[0] for column in header]
        if "item_id" not in field_names:
            return []
        item_col = field_names.index("item_id")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if item_col < len(parts) and parts[item_col]:
                tokens.append(parts[item_col])
    return tokens


def sync_sasrec_item_vocab(dataset, data_path, dataset_name):
    item_field = getattr(dataset, "iid_field", None)
    item_seq_field = getattr(dataset, "item_id_list_field", None)
    if not item_field or not item_seq_field:
        return 0
    alias_fields = [item_field, item_seq_field]
    token2id = dataset.field2token_id[item_field]
    id2token = dataset.field2id_token[item_field]
    for field in alias_fields:
        dataset.field2token_id[field] = token2id
        dataset.field2id_token[field] = id2token
    added = extend_token_vocab(
        dataset, item_field, load_all_item_tokens(data_path, dataset_name)
    )
    shared_token2id = dataset.field2token_id[item_field]
    shared_id2token = dataset.field2id_token[item_field]
    for field in alias_fields:
        dataset.field2token_id[field] = shared_token2id
        dataset.field2id_token[field] = shared_id2token
    return added


def build_config_dict(args):
    emb_path, map_path = build_embedding_paths(args)
    effective_epochs = min(args.epochs, MAX_EPOCH_CAP)
    eval_args = {
        "split": {"TS": [0.8, 0.1, 0.1]},
        "group_by": "none",
        "order": "TO",
        "mode": "full",
    }
    config_dict = {
        "data_path": args.data_path,
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        "RATING_FIELD": "rating",
        "load_col": {"inter": ["user_id", "item_id", "rating", "timestamp"]},
        "eval_args": eval_args,
        "metrics": ["Recall", "NDCG", "MRR", "Hit"],
        "topk": [10, 20],
        "valid_metric": "NDCG@10",
        "eval_step": 1,
        "epochs": effective_epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.train_batch_size,
        "learning_rate": args.learning_rate,
        "embedding_size": args.embedding_size,
        "hidden_size": args.hidden_size,
        "MAX_ITEM_LIST_LENGTH": args.max_seq_len,
        "stopping_step": 10,
        "use_gpu": args.use_gpu,
        "gpu_id": args.gpu_id,
        "seed": args.seed,
        "reproducibility": True,
        "pretrained_item_embedding_path": emb_path,
        "pretrained_item_id_map_path": map_path,
        "freeze_pretrained_item_embedding": False,
        "dropout_prob": 0.1,
    }
    if args.model == "sasrec" and has_presplit_seq_dataset(args.data_path, args.dataset):
        config_dict["load_col"] = {"inter": ["user_id", "item_id", "item_id_list", "rating", "timestamp"]}
        config_dict["benchmark_filename"] = ["train_seq", "valid_seq", "test_seq"]
        config_dict["alias_of_item_id"] = ["item_id", "item_id_list"]
    elif has_presplit_dataset(args.data_path, args.dataset):
        config_dict["benchmark_filename"] = ["train", "valid", "test"]
    if args.model == "two_tower":
        config_dict["neg_sampling"] = {"uniform": 1}
        config_dict["train_neg_sample_args"] = {
            "distribution": "uniform",
            "sample_num": 1,
            "alpha": 1.0,
            "dynamic": False,
            "candidate_num": 0,
        }
    elif args.model == "deepfm":
        config_dict["neg_sampling"] = {"uniform": 1}
        config_dict["train_neg_sample_args"] = {
            "distribution": "uniform",
            "sample_num": 1,
            "alpha": 1.0,
            "dynamic": False,
            "candidate_num": 0,
        }
        config_dict["LABEL_FIELD"] = "label"
    elif args.model == "sasrec":
        config_dict["neg_sampling"] = None
        config_dict["train_neg_sample_args"] = None
    return config_dict


def select_backbone(args):
    if args.model == "sasrec":
        return "SASRec", PretrainedSASRec if args.embedding_variant != "none" else SASRec
    if args.model == "two_tower":
        return "BPR", TwoTower
    if args.model == "deepfm":
        return "DeepFM", PretrainedDeepFM if args.embedding_variant != "none" else DeepFM
    if args.model == "neumf":
        return "NeuMF", PretrainedNeuMF if args.embedding_variant != "none" else NeuMF
    return "LightGCN", PretrainedLightGCN if args.embedding_variant != "none" else LightGCN


def ensure_dataset_layout(args):
    dataset_dir = os.path.join(args.data_path, args.dataset)
    target_inter = os.path.join(dataset_dir, f"{args.dataset}.inter")
    target_train = os.path.join(dataset_dir, f"{args.dataset}.train.inter")
    target_valid = os.path.join(dataset_dir, f"{args.dataset}.valid.inter")
    target_test = os.path.join(dataset_dir, f"{args.dataset}.test.inter")
    target_train_seq = os.path.join(dataset_dir, f"{args.dataset}.train_seq.inter")
    target_valid_seq = os.path.join(dataset_dir, f"{args.dataset}.valid_seq.inter")
    target_test_seq = os.path.join(dataset_dir, f"{args.dataset}.test_seq.inter")
    legacy_inter = os.path.join(args.data_path, f"{args.dataset}.inter")
    if args.model == "sasrec" and all(
        os.path.exists(path) for path in [target_train_seq, target_valid_seq, target_test_seq]
    ):
        return
    if os.path.exists(target_inter):
        return
    if os.path.exists(legacy_inter):
        os.makedirs(dataset_dir, exist_ok=True)
        shutil.copy2(legacy_inter, target_inter)
        return
    if all(os.path.exists(path) for path in [target_train, target_valid, target_test]):
        return
    raise FileNotFoundError(
        f"RecBole dataset file not found. Expected {target_inter}, the pre-split files "
        f"{target_train}/{target_valid}/{target_test}, or legacy {legacy_inter}. "
        f"Please run preprocess_recbole_data.py first."
    )


def train_and_eval(args):
    if args.use_gpu and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU is requested but CUDA is unavailable in current environment. "
            "Please align NVIDIA driver with PyTorch CUDA build or install a compatible PyTorch build."
        )
    ensure_dataset_layout(args)
    config_model_name, model_class = select_backbone(args)
    config_dict = build_config_dict(args)
    argv_backup = list(sys.argv)
    try:
        sys.argv = [sys.argv[0]]
        config = Config(model=config_model_name, dataset=args.dataset, config_dict=config_dict)
    finally:
        sys.argv = argv_backup
    init_seed(config["seed"], config["reproducibility"])
    logger = _init_console_logger(config["state"])
    logger.info(config)
    _disable_recbole_tensorboard()
    dataset = create_dataset(config)
    if args.model == "sasrec" and has_presplit_seq_dataset(args.data_path, args.dataset):
        added_item_count = sync_sasrec_item_vocab(dataset, args.data_path, args.dataset)
        if added_item_count:
            logger.info(
                "Expanded SASRec item vocabulary with %d items from full .inter file.",
                added_item_count,
            )
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = model_class(config, train_data.dataset).to(config["device"])
    trainer_cls = get_trainer(config["MODEL_TYPE"], config_model_name)
    trainer = trainer_cls(config, model)
    valid_metric_name = str(config["valid_metric"])
    valid_metric_bigger = bool(config["valid_metric_bigger"])
    epoch_test_records = []
    best_test_result = None
    best_test_score = None
    best_test_epoch = None

    def _extract_metric(result_dict, metric_name):
        target = str(metric_name).lower()
        for key, value in result_dict.items():
            if str(key).lower() == target:
                return float(value)
        return None

    def _is_better(new_score, old_score, bigger):
        if old_score is None:
            return True
        return new_score > old_score if bigger else new_score < old_score

    def on_epoch_end(epoch_idx, valid_score):
        nonlocal best_test_result, best_test_score, best_test_epoch
        test_result_epoch = trainer.evaluate(
            test_data, load_best_model=False, show_progress=config["show_progress"]
        )
        test_metric_score = _extract_metric(test_result_epoch, valid_metric_name)
        epoch_record = {
            "epoch": int(epoch_idx) + 1,
            "valid_score": float(valid_score),
            "test_result": test_result_epoch,
        }
        if test_metric_score is not None:
            epoch_record["test_metric"] = valid_metric_name
            epoch_record["test_metric_score"] = float(test_metric_score)
            if _is_better(test_metric_score, best_test_score, valid_metric_bigger):
                best_test_score = float(test_metric_score)
                best_test_result = copy.deepcopy(test_result_epoch)
                best_test_epoch = int(epoch_idx) + 1
        epoch_test_records.append(epoch_record)
        logger.info(
            "epoch %d test result: %s",
            int(epoch_idx) + 1,
            test_result_epoch,
        )

    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=True,
        show_progress=config["show_progress"],
        callback_fn=on_epoch_end,
    )
    test_result_best_valid = trainer.evaluate(
        test_data, load_best_model=True, show_progress=config["show_progress"]
    )
    if best_test_result is None:
        best_test_result = copy.deepcopy(test_result_best_valid)
        best_test_score = _extract_metric(best_test_result, valid_metric_name)
        best_test_epoch = None
    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result (best valid)", "yellow") + f": {test_result_best_valid}")
    logger.info(set_color("test result (best test)", "yellow") + f": {best_test_result}")
    result = {
        "model": args.model,
        "embedding_variant": args.embedding_variant,
        "best_valid_score": float(best_valid_score),
        "best_valid_result": best_valid_result,
        "test_result": test_result_best_valid,
        "test_result_best_valid": test_result_best_valid,
        "test_result_best_test": best_test_result,
        "best_test_epoch": best_test_epoch,
        "best_test_metric": valid_metric_name,
        "best_test_score": best_test_score,
        "epoch_test_records": epoch_test_records,
        "dataset": args.dataset,
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    train_and_eval(parse_args())
