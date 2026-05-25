import os
import sys
from typing import List, Dict, Any, Optional
import random
import json

import numpy as np
import fire
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from transformers import EarlyStoppingCallback, TrainerCallback, TrainerState, TrainerControl, Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from sklearn.metrics import roc_auc_score
from transformers import BitsAndBytesConfig

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from models import EdgeEmbeddingProjector, EdgeEmbeddingCollator, SimpleCollator, inject_edge_embeddings


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        
        projector = kwargs.get("projector", None)
        if projector is not None:
            projector_path = os.path.join(checkpoint_folder, "projector.pt")
            torch.save(projector.state_dict(), projector_path)
        
        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


class EdgeEmbeddingTrainer(Trainer):
    def __init__(self, *args, projector=None, projector_lr=1e-3, eval_data_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.projector = projector
        self.projector_lr = projector_lr
        self.eval_data_collator = eval_data_collator
    
    def _save_checkpoint(self, model, trial):
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        
        super()._save_checkpoint(model, trial)
        
        if self.projector is not None:
            projector_path = os.path.join(output_dir, "projector.pt")
            torch.save(self.projector.state_dict(), projector_path)
    
    def create_optimizer(self):
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(self.model)
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                    "weight_decay": 0.0,
                },
            ]
            if self.projector is not None:
                optimizer_grouped_parameters.append({
                    "params": self.projector.parameters(),
                    "lr": self.projector_lr,
                    "weight_decay": 0.0,
                })
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.projector is not None and "input_ids" in inputs:
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", None)
            labels = inputs.get("labels", None)
            edge_item_ids = inputs.get("edge_item_ids", None)

            inputs_embeds = inject_edge_embeddings(
                model=model,
                input_ids=input_ids,
                edge_item_ids=edge_item_ids,
                projector=self.projector,
                movie_token_id=self.processing_class.convert_tokens_to_ids("[MOVIE]"),
                sanitize_projected=False,
            )
            
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False
            )
            loss = outputs.loss
            
            if self.state.global_step % 10 == 0:
                non_ignored = (labels != -100).sum().item()
                total = labels.numel()
                #print(f"Step {self.state.global_step}: Raw loss={loss.item():.4f}, Non-ignored tokens={non_ignored}/{total}")
            
            return (loss, outputs) if return_outputs else loss
        else:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        return loss

    def get_eval_dataloader(self, eval_dataset=None):
        if self.eval_data_collator is None:
            return super().get_eval_dataloader(eval_dataset)
        original_collator = self.data_collator
        try:
            self.data_collator = self.eval_data_collator
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = original_collator


def train(
    base_model: str = "../../base_models/Qwen2.5-7B-Instruct",
    dataset_name: str = "ml-1m",
    data_root: str = "./data",
    embedding_root: str = "../embedding/data",
    embedding_version: str = "v4",
    train_data_path: str = "",
    val_data_path: str = "",
    edge_embeddings_dir: str = "",
    output_dir: str = "",
    sample: int = 8000,
    seed: int = 42,
    batch_size: int = 256,
    micro_batch_size: int = 4,
    num_epochs: int = 5,
    learning_rate: float = 1e-4,
    projector_lr: float = 1e-4,
    cutoff_len: int = 512,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "v_proj"],
    train_on_inputs: bool = True,
    group_by_length: bool = False,
    use_edge_embeddings: bool = True,
    hide_item_ids_in_input: bool = True,
    embedding_keep_schedule: str = "linear",
    embedding_keep_prob_start: float = 1.0,
    embedding_keep_prob_end: float = 1.0,
):
    if not train_data_path:
        train_data_path = os.path.join(data_root, dataset_name, "train.json")
    if not val_data_path:
        val_data_path = os.path.join(data_root, dataset_name, "val.json")
    if not output_dir:
        output_dir = os.path.join("./output", f"{dataset_name}_{embedding_version}")
    if use_edge_embeddings and not edge_embeddings_dir:
        edge_embeddings_dir = os.path.join(embedding_root, dataset_name, embedding_version)

    print(
        f"Training LLaRA-RecSys model with params:\n"
        f"dataset_name: {dataset_name}\n"
        f"base_model: {base_model}\n"
        f"train_data_path: {train_data_path}\n"
        f"val_data_path: {val_data_path}\n"
        f"embedding_version: {embedding_version}\n"
        f"edge_embeddings_dir: {edge_embeddings_dir}\n"
        f"output_dir: {output_dir}\n"
        f"use_edge_embeddings: {use_edge_embeddings}\n"
        f"hide_item_ids_in_input: {hide_item_ids_in_input}\n"
        f"embedding_keep_schedule: {embedding_keep_schedule}\n"
        f"embedding_keep_prob_start: {embedding_keep_prob_start}\n"
        f"embedding_keep_prob_end: {embedding_keep_prob_end}\n"
    )
    
    assert base_model, "Please specify a --base_model"
    
    gradient_accumulation_steps = batch_size // micro_batch_size
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        local_files_only=True,
        trust_remote_code=True
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True, trust_remote_code=True)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    print("Loading datasets...")
    if train_data_path.endswith(".json"):
        train_data = load_dataset("json", data_files=train_data_path)
    else:
        train_data = load_dataset(train_data_path)
    if val_data_path.endswith(".json"):
        val_data = load_dataset("json", data_files=val_data_path)
    else:
        val_data = load_dataset(val_data_path)

    train_data["train"] = train_data["train"].shuffle(seed=seed).select(range(sample)) if sample > -1 else train_data["train"].shuffle(seed=seed)
    train_data["train"] = train_data["train"].shuffle(seed=seed)
    
    val_data = val_data["train"]

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    projector = None
    eval_data_collator = None
    if use_edge_embeddings:
        print("Initializing edge embedding projector...")
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
        projector.to(model.device)
        
        max_steps = num_epochs * max(1, len(train_data["train"]) // batch_size)
        print(f"Max training steps: {max_steps}")
        
        data_collator = EdgeEmbeddingCollator(
            tokenizer=tokenizer,
            edge_embeddings_dir=edge_embeddings_dir,
            projector=projector,
            train=True,
            max_steps=max_steps,
            include_output=True,
            hide_item_ids_in_input=hide_item_ids_in_input,
            embedding_keep_schedule=embedding_keep_schedule,
            embedding_keep_prob_start=embedding_keep_prob_start,
            embedding_keep_prob_end=embedding_keep_prob_end,
        )
        eval_data_collator = EdgeEmbeddingCollator(
            tokenizer=tokenizer,
            edge_embeddings_dir=edge_embeddings_dir,
            projector=projector,
            train=False,
            max_steps=max_steps,
            include_output=True,
            hide_item_ids_in_input=False,
            embedding_keep_schedule="none",
            embedding_keep_prob_start=1.0,
            embedding_keep_prob_end=1.0,
        )
    else:
        data_collator = SimpleCollator(tokenizer=tokenizer, include_output=True, pure_id_mode=True)
        eval_data_collator = data_collator

    def compute_metrics(eval_preds):
        pre, labels = eval_preds
        auc = roc_auc_score(pre[1], pre[0])
        return {'auc': auc}
    
    def preprocess_logits_for_metrics(logits, labels):
        def filter_last_indices(labels_index):
            unique_values, indices = torch.unique(labels_index[:, 0], return_inverse=True)
            max_indices = torch.zeros(len(unique_values), dtype=torch.long)
            for i in range(len(unique_values)):
                group = torch.nonzero(indices == i, as_tuple=False)
                if group.dim() > 1:
                    group = group.squeeze(1)
                max_in_group = torch.argmax(labels_index[group, 1])
                max_indices[i] = group[max_in_group]
            return labels_index[max_indices]

        yes_token = tokenizer.encode("Yes.", add_special_tokens=False)[0]
        no_token = tokenizer.encode("No.", add_special_tokens=False)[0]
        
        labels_index = torch.argwhere(torch.bitwise_or(labels == yes_token, labels == no_token))
        labels_index = filter_last_indices(labels_index)
        gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == no_token, 0, 1)
        labels_index[:, 1] = labels_index[:, 1] - 1
        logits = logits.softmax(dim=-1)
        logits = torch.softmax(logits[labels_index[:, 0], labels_index[:, 1]][:, [no_token, yes_token]], dim=-1)
        return logits[:, 1], gold

    os.environ["WANDB_DISABLED"] = "true"
    
    trainer = EdgeEmbeddingTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_data["train"],
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=False,
            logging_steps=8,
            optim="adamw_torch",
            eval_strategy="epoch",
            save_strategy="epoch",
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_auc",
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=None,
            remove_unused_columns=False,
            max_grad_norm=1.0,
        ),
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5), SavePeftModelCallback()],
        projector=projector,
        projector_lr=projector_lr,
        eval_data_collator=eval_data_collator,
    )
    
    if projector is not None:
        trainer.projector = projector
    
    model.config.use_cache = False
    
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    trainer.train()
    model.save_pretrained(output_dir)
    
    if projector is not None:
        projector_path = os.path.join(output_dir, "projector.pt")
        torch.save(projector.state_dict(), projector_path)
        print(f"Saved projector to {projector_path}")
    
    print("\n Training complete!")


if __name__ == "__main__":
    fire.Fire(train)
