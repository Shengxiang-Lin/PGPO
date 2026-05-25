import random
import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)

def _infer_user_count(user_item_dict):
    if not user_item_dict:
        return 0
    return int(max(user_item_dict.keys())) + 1


def _infer_item_count(data_dir, train_edge):
    item2id_path = os.path.join(data_dir, 'item2id.json')
    if os.path.exists(item2id_path):
        with open(item2id_path, 'r', encoding='utf-8') as f:
            item2id = json.load(f)
        if item2id:
            return int(max(int(v) for v in item2id.values())) + 1
    if len(train_edge) == 0:
        return 0
    return int(np.max(train_edge[:, 1])) + 1


def _resolve_data_dir(dataset):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    candidates = [
        os.path.join(project_dir, "data", dataset),
        os.path.join(project_dir, "Data", dataset),
        os.path.join(base_dir, "data", dataset),
        os.path.join(base_dir, "Data", dataset),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return os.path.join(project_dir, "data", dataset)


def _resolve_variant_dir(dataset, edge_variant, embedding_root):
    candidates = [
        os.path.join(embedding_root, dataset, edge_variant),
        os.path.join(embedding_root, "data", dataset, edge_variant),
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, 'item_embeddings.npy')):
            return path
    raise FileNotFoundError(f'Cannot find item embeddings for {dataset}. Tried: {candidates}')


def _load_feature(feature_dir, file_name):
    return torch.tensor(np.load(os.path.join(feature_dir, file_name)), dtype=torch.float)


def _random_feature_like(feature_dir, file_name, rng):
    reference = np.load(os.path.join(feature_dir, file_name), mmap_mode="r")
    random_feat = rng.standard_normal(reference.shape, dtype=np.float32)
    return torch.from_numpy(random_feat)


def _build_local_to_embed_map(data_dir, feature_dir):
    item2id_path = os.path.join(data_dir, 'item2id.json')
    embed_map_candidates = [
        os.path.join(feature_dir, 'item_id_map.json'),
        os.path.join(os.path.dirname(feature_dir), 'item_id_map.json'),
    ]
    embed_map_path = None
    for candidate in embed_map_candidates:
        if os.path.exists(candidate):
            embed_map_path = candidate
            break
    missing = []
    if not os.path.exists(item2id_path):
        missing.append(item2id_path)
    if embed_map_path is None:
        missing.extend(embed_map_candidates)
    if missing:
        raise FileNotFoundError(
            f"Missing required id mapping files for remap: {missing}. "
            "For ml-1m/amazon-book, mapping files are mandatory."
        )
    with open(item2id_path, 'r', encoding='utf-8') as f:
        item2id = json.load(f)
    with open(embed_map_path, 'r', encoding='utf-8') as f:
        embed_map = json.load(f)
    print(f"[MMGCN][remap] using embed map: {embed_map_path}")
    if not item2id:
        raise ValueError(f"item2id is empty: {item2id_path}")
    max_local = max(int(v) for v in item2id.values())
    local_to_embed = np.full(max_local + 1, -1, dtype=np.int64)
    for raw_item_id, local_idx in item2id.items():
        local_idx = int(local_idx)
        if str(raw_item_id) in embed_map:
            local_to_embed[local_idx] = int(embed_map[str(raw_item_id)])
    return local_to_embed


def _remap_interactions(train_edge, user_item_dict, local_to_embed, min_keep_ratio=0.95):
    if local_to_embed is None:
        return train_edge, user_item_dict
    raw_edge_count = int(len(train_edge))
    raw_item_ids = set(int(i) for i in train_edge[:, 1].tolist()) if raw_edge_count > 0 else set()
    raw_item_count = int(len(raw_item_ids))
    remapped_edges = []
    for u, i in train_edge:
        i = int(i)
        if 0 <= i < len(local_to_embed):
            ni = int(local_to_embed[i])
            if ni >= 0:
                remapped_edges.append([int(u), ni])
    if not remapped_edges:
        raise ValueError('No train edges left after remapping items to embedding index space.')
    train_edge = np.asarray(remapped_edges, dtype=np.int64)
    kept_edge_count = int(len(train_edge))
    kept_item_count = int(len(set(int(i) for i in train_edge[:, 1].tolist())))
    edge_keep_ratio = kept_edge_count / max(raw_edge_count, 1)
    item_keep_ratio = kept_item_count / max(raw_item_count, 1)
    print(
        "[MMGCN][remap] "
        f"kept_edges/raw_edges={kept_edge_count}/{raw_edge_count} ({edge_keep_ratio:.4f}), "
        f"kept_items/raw_items={kept_item_count}/{raw_item_count} ({item_keep_ratio:.4f})"
    )
    if edge_keep_ratio < min_keep_ratio or item_keep_ratio < min_keep_ratio:
        raise ValueError(
            "Remap keep ratio below threshold. "
            f"edge_keep_ratio={edge_keep_ratio:.4f}, item_keep_ratio={item_keep_ratio:.4f}, "
            f"min_keep_ratio={min_keep_ratio:.4f}. "
            "Please check item2id.json and item_id_map.json consistency."
        )
    remapped_dict = {}
    for u, items in user_item_dict.items():
        mapped = []
        for i in items:
            i = int(i)
            if 0 <= i < len(local_to_embed):
                ni = int(local_to_embed[i])
                if ni >= 0:
                    mapped.append(ni)
        remapped_dict[int(u)] = mapped
    return train_edge, remapped_dict


def _to_global_item_space(train_edge, user_item_dict, num_user):
    if len(train_edge) > 0 and int(np.min(train_edge[:, 1])) >= num_user:
        return train_edge, user_item_dict
    train_edge = train_edge.copy()
    train_edge[:, 1] += num_user
    shifted = {}
    for u, items in user_item_dict.items():
        shifted[int(u)] = [int(i) + num_user for i in items]
    return train_edge, shifted


def data_load(
    dataset,
    has_v=True,
    has_a=True,
    has_t=True,
    has_e=True,
    edge_variant='v3',
    embedding_root='../embedding/data',
    random_edge_features=False,
    random_seed=1,
):
    dir_str = _resolve_data_dir(dataset)
    train_edge = np.load(dir_str+'/train.npy', allow_pickle=True)
    user_item_dict = np.load(dir_str+'/user_item_dict.npy', allow_pickle=True).item()
    e_feat = None
    local_to_embed = None
    random_edge_features = _coerce_bool(random_edge_features)

    if dataset == 'movielens':
        num_user = 55485
        num_item = 5986
        v_feat = np.load(dir_str+'/FeatureVideo_normal.npy', allow_pickle=True) if has_v else None
        a_feat = np.load(dir_str+'/FeatureAudio_avg_normal.npy', allow_pickle=True) if has_a else None
        t_feat = np.load(dir_str+'/FeatureText_stl_normal.npy', allow_pickle=True) if has_t else None
        v_feat = torch.tensor(v_feat, dtype=torch.float).cuda() if has_v else None
        a_feat = torch.tensor(a_feat, dtype=torch.float).cuda() if has_a else None
        t_feat = torch.tensor(t_feat, dtype=torch.float).cuda() if has_t else None
    elif dataset == 'Tiktok':
        num_user = 36656
        num_item = 76085
        if has_v:
            v_feat = torch.load(dir_str+'/feat_v.pt')
            v_feat = torch.tensor(v_feat, dtype=torch.float).cuda()
        else:
            v_feat = None

        if has_a:
            a_feat = torch.load(dir_str+'/feat_a.pt')
            a_feat = torch.tensor(a_feat, dtype=torch.float).cuda() 
        else:
            a_feat = None
        
        t_feat = torch.load(dir_str+'/feat_t.pt') if has_t else None
    elif dataset == 'Kwai':
        num_user = 7010
        num_item = 86483
        v_feat = torch.load(dir_str+'/feat_v.pt')
        v_feat = torch.tensor(v_feat, dtype=torch.float).cuda()
        a_feat = t_feat = None
    elif dataset in ['ml-1m', 'amazon-book']:
        num_user = _infer_user_count(user_item_dict)
        try:
            variant_dir = _resolve_variant_dir(dataset, edge_variant, embedding_root)
        except FileNotFoundError:
            # v0 fallback: run MMGCN as pure ID collaborative model without modality GCN branches.
            variant_dir = None
            local_to_embed = None
            v_feat = a_feat = t_feat = e_feat = None
            num_item = _infer_item_count(dir_str, train_edge)
            train_edge, user_item_dict = _to_global_item_space(train_edge, user_item_dict, num_user)
            print(
                f"[MMGCN][fallback] Item embeddings not found for dataset={dataset}, "
                f"variant={edge_variant}. Running pure ID collaborative mode."
            )
        if variant_dir is not None:
            local_to_embed = _build_local_to_embed_map(dir_str, variant_dir)
            train_edge, user_item_dict = _remap_interactions(
                train_edge,
                user_item_dict,
                local_to_embed,
                min_keep_ratio=0.95,
            )
            train_edge, user_item_dict = _to_global_item_space(train_edge, user_item_dict, num_user)
            rng = np.random.default_rng(random_seed)
            v_feat = None
            a_feat = None
            if has_t:
                t_feat = _random_feature_like(variant_dir, 'item_embeddings.npy', rng) if random_edge_features else _load_feature(variant_dir, 'item_embeddings.npy')
            else:
                t_feat = None
            e_feat = None
            num_item = int(t_feat.size(0)) if t_feat is not None else _infer_item_count(dir_str, train_edge)
            print(
                f"[MMGCN][text-only] dataset={dataset}, variant={edge_variant}, "
                "using item_embeddings.npy as the only modality feature."
            )
    else:
        raise ValueError(f'Unsupported dataset: {dataset}')

    return num_user, num_item, train_edge, user_item_dict, v_feat, a_feat, t_feat, e_feat, local_to_embed

class TrainingDataset(Dataset):
    def __init__(self, num_user, num_item, user_item_dict, edge_index):
        self.edge_index = edge_index
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_dict = user_item_dict
        self.all_set = set(range(num_user, num_user+num_item))

    def __len__(self):
        return len(self.edge_index)

    def __getitem__(self, index):
        user, pos_item = self.edge_index[index]
        while True:
            neg_item = random.sample(self.all_set, 1)[0]
            if neg_item not in self.user_item_dict[user]:
                break
        return torch.LongTensor([user,user]), torch.LongTensor([pos_item, neg_item])
