import json
import os
import warnings
from argparse import ArgumentParser

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)

import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from dataset.data_interface import DInterface
from model.model_interface import MInterface


def load_callbacks(args):
    callbacks = []
    callbacks.append(
        plc.EarlyStopping(
            monitor="metric",
            mode="max",
            patience=10,
            min_delta=0.001,
        )
    )
    callbacks.append(
        plc.ModelCheckpoint(
            monitor="metric",
            dirpath=args.ckpt_dir,
            filename="{epoch:02d}-{metric:.3f}",
            save_top_k=-1,
            mode="max",
            save_last=True,
            every_n_epochs=1,
        )
    )
    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(logging_interval="step"))
    return callbacks


def infer_semantic_paths(args):
    if args.rec_embed != "semantic":
        return None, None
    if args.semantic_emb_path and args.semantic_id_map_path:
        return args.semantic_emb_path, args.semantic_id_map_path
    emb_base = os.path.join(args.embedding_root, args.dataset_name, args.embedding_variant)
    semantic_emb_path = os.path.join(emb_base, "item_embeddings.npy")
    semantic_id_map_path = os.path.join(emb_base, "item_id_map.json")
    if not os.path.exists(semantic_emb_path):
        raise FileNotFoundError(f"semantic embedding not found: {semantic_emb_path}")
    if not os.path.exists(semantic_id_map_path):
        raise FileNotFoundError(f"semantic id map not found: {semantic_id_map_path}")
    return semantic_emb_path, semantic_id_map_path


def infer_data_dir(args):
    if args.data_dir:
        return args.data_dir
    return os.path.join("./data", args.dataset_name, args.embedding_variant)


def load_padding_item_id(args):
    meta_path = os.path.join(args.data_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta file not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return int(meta["padding_item_id"])


def main(args):
    pl.seed_everything(args.seed)
    model = MInterface(**vars(args))
    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print("load checkpoints from {}".format(args.ckpt_path))
    data_module = DInterface(llm_tokenizer=model.llama_tokenizer, **vars(args))
    logger = TensorBoardLogger(save_dir="./log/", name=args.log_dir)
    args.callbacks = load_callbacks(args)
    args.logger = logger
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    trainer = Trainer.from_argparse_args(args)
    if args.auto_lr_find:
        lr_finder = trainer.tuner.lr_find(
            model=model,
            datamodule=data_module,
            min_lr=1e-10,
            max_lr=1e-3,
            num_training=100,
        )
        fig = lr_finder.plot(suggest=True)
        fig_path = "lr_finder.png"
        fig.savefig(fig_path)
        model.hparams.lr = lr_finder.suggestion()
    if args.mode == "train":
        trainer.fit(model=model, datamodule=data_module)
    else:
        trainer.test(model=model, datamodule=data_module)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default="gpu", type=str)
    parser.add_argument("--devices", default=-1, type=int)
    parser.add_argument("--precision", default="bf16", type=str)
    parser.add_argument("--amp_backend", default="native", type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--accumulate_grad_batches", default=8, type=int)
    parser.add_argument("--check_val_every_n_epoch", default=1, type=int)
    parser.add_argument("--lr_scheduler", default="cosine", choices=["cosine"], type=str)
    parser.add_argument("--lr_decay_min_lr", default=1e-9, type=float)
    parser.add_argument("--lr_warmup_start_lr", default=1e-7, type=float)
    parser.add_argument("--dataset", default="sequence_data", type=str)
    parser.add_argument("--dataset_name", default="ml-1m", choices=["ml-1m", "amazon-book"], type=str)
    parser.add_argument("--embedding_root", default="../embedding/data", type=str)
    parser.add_argument("--embedding_variant", default="v1", choices=["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"], type=str)
    parser.add_argument("--data_dir", default="", type=str)
    parser.add_argument("--model_name", default="mlp_projector", type=str)
    parser.add_argument("--task_type", default="generate", choices=["generate", "candidate_cls"], type=str)
    parser.add_argument("--loss", default="lm", choices=["lm", "candidate_ce"], type=str)
    parser.add_argument("--align_weight", default=0.1, type=float)
    parser.add_argument("--align_temperature", default=0.07, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--ckpt_dir", default="./checkpoints/", type=str)
    parser.add_argument("--log_dir", default="llara_logs", type=str)
    parser.add_argument("--padding_item_id", default=-1, type=int)
    parser.add_argument("--llm_path", type=str, required=True)
    parser.add_argument("--semantic_emb_path", default=None, type=str)
    parser.add_argument("--semantic_id_map_path", default=None, type=str)
    parser.add_argument("--prompt_path", default="./prompt/item_id.txt", type=str)
    parser.add_argument("--output_dir", default="./output/", type=str)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--rec_embed", default="semantic", choices=["semantic", "none"], type=str)
    parser.add_argument("--aug_prob", default=0.5, type=float)
    parser.add_argument("--mode", default="train", choices=["train", "test"], type=str)
    parser.add_argument("--auto_lr_find", default=False, action="store_true")
    parser.add_argument("--metric", default="hr", choices=["hr"], type=str)
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--save", default="part", choices=["part", "all"], type=str)
    parser.add_argument("--cans_num", default=10, type=int)
    parser.add_argument("--llm_tuning", default="lora", choices=["lora", "freeze", "freeze_lora"], type=str)
    parser.add_argument("--peft_dir", default=None, type=str)
    parser.add_argument("--peft_config", default=None, type=str)
    parser.add_argument("--lora_r", default=8, type=float)
    parser.add_argument("--lora_alpha", default=32, type=float)
    parser.add_argument("--lora_dropout", default=0.1, type=float)
    args = parser.parse_args()
    args.data_dir = infer_data_dir(args)
    args.semantic_emb_path, args.semantic_id_map_path = infer_semantic_paths(args)
    if args.padding_item_id < 0:
        args.padding_item_id = load_padding_item_id(args)
    if args.task_type == "candidate_cls" and args.loss != "candidate_ce":
        args.loss = "candidate_ce"
    if args.task_type == "generate" and args.loss != "lm":
        args.loss = "lm"
    main(args)
