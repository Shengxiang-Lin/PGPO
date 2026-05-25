import json
import os
import numpy as np
import torch
from sklearn.decomposition import PCA
from recbole.model.general_recommender.lightgcn import LightGCN


class PretrainedLightGCN(LightGCN):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self._load_pretrained_item_embeddings(config, dataset)

    def _align_dim(self, embeddings):
        dim = embeddings.shape[1]
        target_dim = self.latent_dim
        
        if dim == target_dim:
            return embeddings
        if dim > target_dim:
            pca = PCA(n_components=target_dim, random_state=42)
            return pca.fit_transform(embeddings)
        
        if target_dim > dim:
            pca = PCA(n_components=min(dim, target_dim), random_state=42)
            reduced = pca.fit_transform(embeddings)
            
            if target_dim > reduced.shape[1]:
                pad_width = target_dim - reduced.shape[1]
                pad = np.zeros((reduced.shape[0], pad_width), dtype=reduced.dtype)
                return np.concatenate([reduced, pad], axis=1)
            return reduced
        
        return embeddings

    def _load_pretrained_item_embeddings(self, config, dataset):
        emb_path = config["pretrained_item_embedding_path"]
        map_path = config["pretrained_item_id_map_path"]
        if not emb_path or not map_path:
            return
        if not os.path.exists(emb_path) or not os.path.exists(map_path):
            return
        embeddings = np.load(emb_path)
        embeddings = self._align_dim(embeddings)
        with open(map_path, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
        token2id = dataset.field2token_id[self.ITEM_ID]
        weight = self.item_embedding.weight.data
        for raw_item_id, row_idx in raw_map.items():
            token = str(raw_item_id)
            if token not in token2id:
                continue
            inner_id = token2id[token]
            if row_idx >= embeddings.shape[0]:
                continue
            weight[inner_id] = torch.from_numpy(embeddings[row_idx]).to(weight.device)
