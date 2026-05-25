import importlib
import inspect
import json
import os
import os.path as op

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pandas.core.frame import DataFrame
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from transformers.utils import logging as hf_logging

from optims import LinearWarmupCosineLRScheduler


class MInterface(pl.LightningModule):
    def __init__(self, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_llm(self.hparams.llm_path)
        if self.hparams.rec_embed == "semantic":
            self.load_semantic_embeddings(self.hparams.semantic_emb_path, self.hparams.semantic_id_map_path)
            self.load_projector()
        else:
            self.num_items = int(self.hparams.padding_item_id) + 1
        self.load_output_projection()

    def forward(self, batch):
        if self.hparams.task_type == "candidate_cls":
            return self.forward_candidate_cls(batch)
        targets = batch["tokens"].input_ids.masked_fill(
            batch["tokens"].input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        targets = targets.masked_fill((batch["tokens"].token_type_ids == 0)[:, 1:], -100)
        input_embeds = self.wrap_emb(batch)
        outputs = self.llama_model(
            inputs_embeds=input_embeds,
            attention_mask=batch["tokens"].attention_mask,
            return_dict=True,
            labels=targets,
            use_cache=False,
        )
        return outputs

    def forward_candidate_cls(self, batch):
        input_embeds = self.wrap_emb(batch)
        outputs = self.llama_model(
            inputs_embeds=input_embeds,
            attention_mask=batch["tokens"].attention_mask,
            return_dict=True,
            use_cache=False,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]
        if hasattr(batch["tokens"], "token_type_ids") and batch["tokens"].token_type_ids is not None:
            prompt_mask = batch["tokens"].token_type_ids == 0
        else:
            prompt_mask = batch["tokens"].attention_mask == 1
        query_index = prompt_mask.long().sum(dim=1) - 1
        query_index = query_index.clamp(min=0, max=hidden.size(1) - 1).long()
        batch_index = torch.arange(hidden.size(0), device=hidden.device, dtype=torch.long)
        query = hidden[batch_index, query_index]
        full_logits = self.item_output(query)
        cans = batch["cans"].to(full_logits.device).long()
        candidate_logits = torch.gather(full_logits, 1, cans)
        rec_loss = F.cross_entropy(candidate_logits, batch["target_idx"].to(candidate_logits.device))
        pos_item_embeds = self.get_item_align_embeddings(batch["item_id"].to(query.device).long())
        query_norm = F.normalize(query, dim=-1)
        item_norm = F.normalize(pos_item_embeds, dim=-1)
        align_logits = torch.matmul(query_norm, item_norm.t()) / self.hparams.align_temperature
        align_targets = torch.arange(query.size(0), device=query.device)
        align_loss_q2i = F.cross_entropy(align_logits, align_targets)
        align_loss_i2q = F.cross_entropy(align_logits.t(), align_targets)
        align_loss = 0.5 * (align_loss_q2i + align_loss_i2q)
        loss = rec_loss + self.hparams.align_weight * align_loss
        return {"loss": loss, "logits": candidate_logits, "rec_loss": rec_loss, "align_loss": align_loss}

    def generate(
        self,
        batch,
        temperature=0.8,
        do_sample=False,
        num_beams=1,
        max_gen_length=64,
        min_gen_length=1,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_return_sequences=1,
    ):
        input_embeds = self.wrap_emb(batch)
        generate_ids = self.llama_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=batch["tokens"].attention_mask,
            temperature=temperature,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_gen_length,
            min_new_tokens=min_gen_length,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
        )
        output_text = self.llama_tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        outputs = [text.strip() for text in output_text]
        return outputs

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.global_step, self.current_epoch, self.trainer.max_steps)
        if hasattr(self, "projector"):
            if batch["flag"]:
                for _, param in self.projector.named_parameters():
                    param.requires_grad = False
            else:
                for _, param in self.projector.named_parameters():
                    param.requires_grad = True
        out = self(batch)
        loss = self.configure_loss(out)
        if self.hparams.task_type == "candidate_cls":
            pred_idx = torch.argmax(out["logits"], dim=-1)
            cls_acc = (pred_idx == batch["target_idx"].to(pred_idx.device)).float().mean()
            self.log("train_cls_acc", cls_acc, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train_rec_loss", out["rec_loss"], on_step=True, on_epoch=True, prog_bar=True)
            self.log("train_align_loss", out["align_loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("lr", self.scheduler.optimizer.param_groups[0]["lr"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("global_step_num", float(self.trainer.global_step), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self):
        self.val_content = {
            "generate": [],
            "real": [],
            "cans": [],
        }

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if self.hparams.task_type == "candidate_cls":
            out = self.forward_candidate_cls(batch)
            pred_idx = torch.argmax(out["logits"], dim=-1).detach().cpu().tolist()
            cls_acc = (torch.tensor(pred_idx) == batch["target_idx"].cpu()).float().mean()
            self.log("val_cls_acc", cls_acc, on_step=False, on_epoch=True, prog_bar=True)
            generate_output = [batch["cans_name"][i][pred_idx[i]] for i in range(len(pred_idx))]
        else:
            generate_output = self.generate(batch)
        output = []
        for i, generate in enumerate(generate_output):
            real = batch["correct_answer"][i]
            cans = batch["cans_name"][i]
            generate = generate.strip().split("\n")[0]
            output.append((generate, real, cans))
        return output

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        for generate, real, cans in outputs:
            self.val_content["generate"].append(generate)
            self.val_content["real"].append(real)
            self.val_content["cans"].append(cans)

    def on_validation_epoch_end(self):
        df = DataFrame(self.val_content)
        if not os.path.exists(self.hparams.output_dir):
            os.makedirs(self.hparams.output_dir)
        df.to_csv(op.join(self.hparams.output_dir, "valid.csv"))
        prediction_valid_ratio, hr = self.calculate_hr1(self.val_content)
        metric = hr * prediction_valid_ratio
        self.log("val_prediction_valid", prediction_valid_ratio, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_hr", hr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("metric", metric, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_start(self):
        self.test_content = {
            "generate": [],
            "real": [],
            "cans": [],
        }

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        if self.hparams.task_type == "candidate_cls":
            out = self.forward_candidate_cls(batch)
            pred_idx = torch.argmax(out["logits"], dim=-1).detach().cpu().tolist()
            cls_acc = (torch.tensor(pred_idx) == batch["target_idx"].cpu()).float().mean()
            self.log("test_cls_acc", cls_acc, on_step=False, on_epoch=True, prog_bar=True)
            generate_output = [batch["cans_name"][i][pred_idx[i]] for i in range(len(pred_idx))]
        else:
            generate_output = self.generate(batch)
        output = []
        for i, generate in enumerate(generate_output):
            real = batch["correct_answer"][i]
            cans = batch["cans_name"][i]
            generate = generate.strip().split("\n")[0]
            output.append((generate, real, cans))
        return output

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        for generate, real, cans in outputs:
            self.test_content["generate"].append(generate)
            self.test_content["real"].append(real)
            self.test_content["cans"].append(cans)

    def on_test_epoch_end(self):
        df = DataFrame(self.test_content)
        if not os.path.exists(self.hparams.output_dir):
            os.makedirs(self.hparams.output_dir)
        df.to_csv(op.join(self.hparams.output_dir, "test.csv"))
        prediction_valid_ratio, hr = self.calculate_hr1(self.test_content)
        metric = hr * prediction_valid_ratio
        self.log("test_prediction_valid", prediction_valid_ratio, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_hr", hr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("metric", metric, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        weight_decay = getattr(self.hparams, "weight_decay", 0.0)
        trainable_params = list(self.item_output.parameters())
        if hasattr(self, "projector"):
            trainable_params += list(self.projector.parameters())
        optimizer = torch.optim.Adam(
            [
                {
                    "params": trainable_params,
                    "lr": self.hparams.lr,
                    "weight_decay": weight_decay,
                },
                {"params": self.llama_model.parameters(), "lr": self.hparams.lr},
            ]
        )
        if self.hparams.lr_scheduler is None:
            return optimizer
        max_step = int(self.trainer.estimated_stepping_batches)
        warmup_steps = max_step // 20
        if self.hparams.lr_scheduler == "cosine":
            self.scheduler = LinearWarmupCosineLRScheduler(
                optimizer,
                max_step=max_step,
                min_lr=self.hparams.lr_decay_min_lr,
                init_lr=self.hparams.lr,
                warmup_steps=warmup_steps,
                warmup_start_lr=self.hparams.lr_warmup_start_lr,
            )
        else:
            self.scheduler = None
            raise ValueError("Invalid lr_scheduler type")
        return optimizer

    def configure_loss(self, out, labels=None):
        loss = self.hparams.loss.lower()
        if loss == "lm":
            return out.loss
        if loss == "candidate_ce":
            return out["loss"]
        raise ValueError("Invalid Loss Type")

    def on_save_checkpoint(self, checkpoint):
        if self.hparams.save == "part":
            checkpoint.pop("optimizer_states")
            to_be_removed = []
            for key, _ in checkpoint["state_dict"].items():
                try:
                    if not self.get_parameter(key).requires_grad:
                        to_be_removed.append(key)
                except AttributeError:
                    to_be_removed.append(key)
            for key in to_be_removed:
                checkpoint["state_dict"].pop(key)

    def load_llm(self, llm_path):
        hf_prev_verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            try:
                self.llama_tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False)
            except ValueError:
                self.llama_tokenizer = LlamaTokenizer.from_pretrained(llm_path, use_fast=False)
        finally:
            hf_logging.set_verbosity(hf_prev_verbosity)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.llama_tokenizer.padding_side = "right"
        self.llama_tokenizer.add_special_tokens({"additional_special_tokens": ["[PH]", "[HistoryEmb]", "[CansEmb]"]})
        history_ids = self.llama_tokenizer("[HistoryEmb]", return_tensors="pt", add_special_tokens=False).input_ids[0]
        cans_ids = self.llama_tokenizer("[CansEmb]", return_tensors="pt", add_special_tokens=False).input_ids[0]
        if history_ids.numel() != 1 or cans_ids.numel() != 1:
            raise ValueError("Special tokens are not single tokens")
        self.history_token_id = history_ids.item()
        self.cans_token_id = cans_ids.item()
        self.llama_model = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.bfloat16)
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        if self.hparams.llm_tuning == "lora":
            if self.hparams.peft_dir:
                self.llama_model = PeftModel.from_pretrained(self.llama_model, self.hparams.peft_dir, is_trainable=True)
            else:
                if self.hparams.peft_config:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(self.hparams.peft_config))
                else:
                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        r=self.hparams.lora_r,
                        lora_alpha=self.hparams.lora_alpha,
                        lora_dropout=self.hparams.lora_dropout,
                        target_modules=[
                            "k_proj",
                            "v_proj",
                            "q_proj",
                            "o_proj",
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                        ],
                    )
                self.peft_config = peft_config
                self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
        elif self.hparams.llm_tuning == "freeze":
            for _, param in self.llama_model.named_parameters():
                param.requires_grad = False
        elif self.hparams.llm_tuning == "freeze_lora":
            if self.hparams.peft_dir:
                self.llama_model = PeftModel.from_pretrained(self.llama_model, self.hparams.peft_dir, is_trainable=True)
            else:
                if self.hparams.peft_config:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(self.hparams.peft_config))
                else:
                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        r=self.hparams.lora_r,
                        lora_alpha=self.hparams.lora_alpha,
                        lora_dropout=self.hparams.lora_dropout,
                        target_modules=[
                            "k_proj",
                            "v_proj",
                            "q_proj",
                            "o_proj",
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                        ],
                    )
                self.peft_config = peft_config
                self.llama_model = get_peft_model(self.llama_model, peft_config)
            for _, param in self.llama_model.named_parameters():
                param.requires_grad = False
            self.llama_model.print_trainable_parameters()
        else:
            raise NotImplementedError()

    def load_projector(self):
        name = self.hparams.model_name
        camel_name = "".join([i.capitalize() for i in name.split("_")])
        try:
            Model = getattr(importlib.import_module("." + name, package=__package__), camel_name)
        except Exception as e:
            raise ValueError(f"Invalid Module File Name or Invalid Class Name {name}.{camel_name}") from e
        self.projector = self.instancialize(
            Model, rec_size=self.rec_input_size, llm_size=self.llama_model.config.hidden_size
        )

    def load_output_projection(self):
        self.item_output = nn.Linear(self.llama_model.config.hidden_size, self.num_items, bias=False)
        if self.hparams.rec_embed == "semantic":
            with torch.no_grad():
                all_ids = torch.arange(self.num_items, dtype=torch.long)
                init_item_embeds = self.encode_items(all_ids)
                self.item_output.weight.copy_(init_item_embeds)
        else:
            nn.init.xavier_uniform_(self.item_output.weight)

    def instancialize(self, Model, **other_args):
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)

    def load_semantic_embeddings(self, semantic_emb_path, semantic_id_map_path):
        if semantic_emb_path is None or semantic_id_map_path is None:
            raise ValueError("semantic_emb_path and semantic_id_map_path are required")
        embeddings = np.load(semantic_emb_path)
        with open(semantic_id_map_path, "r", encoding="utf-8") as f:
            item_id_map = json.load(f)
        row_indices = [int(v) for v in item_id_map.values()]
        if not row_indices:
            raise ValueError("semantic id map is empty")
        num_items = max(row_indices) + 1
        emb_dim = embeddings.shape[1]
        if embeddings.shape[0] < num_items:
            raise ValueError("embedding rows are fewer than required item indices")
        padding_item_id = int(self.hparams.padding_item_id)
        expected_padding = num_items
        if padding_item_id != expected_padding:
            raise ValueError(
                f"padding_item_id mismatch: got {padding_item_id}, expected {expected_padding}. "
                "Use preprocess_llara_data.py generated meta.json value."
            )
        full_weight = np.zeros((num_items + 1, emb_dim), dtype=np.float32)
        full_weight[:num_items] = embeddings[:num_items]
        self.semantic_embeddings = nn.Embedding.from_pretrained(
            torch.tensor(full_weight, dtype=torch.float32),
            freeze=True,
            padding_idx=padding_item_id,
        )
        self.rec_input_size = emb_dim
        self.num_items = num_items

    def encode_items(self, seq):
        if self.hparams.rec_embed != "semantic":
            raise ValueError("encode_items is only available when rec_embed=semantic")
        item_rec_embs = self.semantic_embeddings(seq)
        item_txt_embs = self.projector(item_rec_embs)
        return item_txt_embs

    def get_item_align_embeddings(self, item_ids):
        if self.hparams.rec_embed == "semantic":
            return self.encode_items(item_ids)
        return self.item_output.weight[item_ids]

    def wrap_emb(self, batch):
        input_embeds = self.llama_model.get_input_embeddings()(batch["tokens"].input_ids)
        if self.hparams.rec_embed != "semantic":
            return input_embeds
        his_item_embeds = self.encode_items(batch["seq"])
        cans_item_embeds = self.encode_items(batch["cans"])
        for i in range(len(batch["len_seq"])):
            if (batch["tokens"].input_ids[i] == self.history_token_id).nonzero().shape[0] > 0:
                idx_tensor = (batch["tokens"].input_ids[i] == self.history_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor, his_item_embeds[i, : batch["len_seq"][i].item()]):
                    input_embeds[i, idx] = item_emb
            if (batch["tokens"].input_ids[i] == self.cans_token_id).nonzero().shape[0] > 0:
                idx_tensor = (batch["tokens"].input_ids[i] == self.cans_token_id).nonzero().view(-1)
                for idx, item_emb in zip(idx_tensor, cans_item_embeds[i, : batch["len_cans"][i].item()]):
                    input_embeds[i, idx] = item_emb
        return input_embeds

    def calculate_hr1(self, eval_content):
        correct_num = 0
        valid_num = 0
        total_num = 0
        for i, generate in enumerate(eval_content["generate"]):
            real = eval_content["real"][i]
            cans = eval_content["cans"][i]
            total_num += 1
            generate = generate.strip().lower().strip()
            real = real.strip().lower().strip()
            cans = [item.strip().lower().strip() for item in cans]
            gen_cans_list = []
            for cans_item in cans:
                if cans_item in generate:
                    gen_cans_list.append(cans_item)
            if len(gen_cans_list) == 1:
                valid_num += 1
                if real == gen_cans_list[0]:
                    correct_num += 1
        valid_ratio = valid_num / total_num
        if valid_num > 0:
            hr1 = correct_num / valid_num
        else:
            hr1 = 0
        return valid_ratio, hr1
