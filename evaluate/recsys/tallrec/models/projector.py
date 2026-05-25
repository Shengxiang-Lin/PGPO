import json
import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


class EdgeEmbeddingProjector(nn.Module):
    def __init__(
        self,
        edge_embedding_dim: int = 4096,
        llm_hidden_size: int = 4096,
    ):
        super().__init__()

        self.edge_embedding_dim = edge_embedding_dim
        self.llm_hidden_size = llm_hidden_size
        self.edge_embeddings = {}

        self.mlp_proj = nn.Sequential(
            nn.Linear(edge_embedding_dim, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        return self.mlp_proj(x)

    def load_edge_embeddings(
        self,
        embeddings_path: str = "",
        embeddings_dir: str = "",
        item_id_map_path: str = "",
        item_embeddings_path: str = "",
    ) -> Dict[str, torch.Tensor]:
        embeddings = {}

        use_json = bool(embeddings_path) and embeddings_path.endswith(".json")
        if use_json:
            with open(embeddings_path, "r", encoding="utf-8") as f:
                embeddings_dict = json.load(f)
            for item_id, embedding_list in embeddings_dict.items():
                emb = torch.tensor(embedding_list, dtype=torch.float16)
                if torch.isnan(emb).any() or torch.isinf(emb).any():
                    emb = torch.zeros_like(emb)
                embeddings[str(item_id)] = emb
            self.edge_embeddings = embeddings
            return embeddings

        resolved_dir = embeddings_dir or embeddings_path
        if resolved_dir:
            item_id_map_path = item_id_map_path or os.path.join(resolved_dir, "item_id_map.json")
            item_embeddings_path = item_embeddings_path or os.path.join(resolved_dir, "item_embeddings.npy")

        if not item_id_map_path or not item_embeddings_path:
            raise ValueError("item_id_map_path and item_embeddings_path must be provided")
        if not os.path.exists(item_id_map_path):
            raise FileNotFoundError(f"item_id_map not found: {item_id_map_path}")
        if not os.path.exists(item_embeddings_path):
            raise FileNotFoundError(f"item_embeddings not found: {item_embeddings_path}")

        with open(item_id_map_path, "r", encoding="utf-8") as f:
            id_map = json.load(f)
        matrix = np.load(item_embeddings_path)

        for item_id, row_idx in id_map.items():
            idx = int(row_idx)
            if idx < 0 or idx >= len(matrix):
                continue
            emb = torch.tensor(matrix[idx], dtype=torch.float16)
            if torch.isnan(emb).any() or torch.isinf(emb).any():
                emb = torch.zeros_like(emb)
            embeddings[str(item_id)] = emb

        self.edge_embeddings = embeddings
        return embeddings
