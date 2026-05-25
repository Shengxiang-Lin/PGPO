import numpy as np
import random
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp

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
    # Default to lowercase path for new preprocessing.
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
    print(f"[LightGT][remap] using embed map: {embed_map_path}")

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
        "[LightGT][remap] "
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


def data_load(dataset, has_v=True, has_a=True, has_t=True, has_e=True, edge_variant='v3', embedding_root='../embedding/data'):
    dir_str = _resolve_data_dir(dataset)
    e_feat = None
    local_to_embed = None

    if dataset == 'movielens':
        user_num = 55485
        item_num = 5986
        train_edge = np.load(dir_str + '/train.npy', allow_pickle=True)
        user_item_dict = np.load(dir_str + '/user_item_dict.npy', allow_pickle=True).item()
        v_feat = torch.tensor(np.load(dir_str + '/FeatureVideo_normal.npy', allow_pickle=True), dtype=torch.float).cuda() if has_v else None
        a_feat = torch.tensor(np.load(dir_str + '/FeatureAudio_avg_normal.npy', allow_pickle=True), dtype=torch.float).cuda() if has_a else None
        t_feat = torch.tensor(np.load(dir_str + '/FeatureText_stl_normal.npy', allow_pickle=True), dtype=torch.float).cuda() if has_t else None
    elif dataset == 'tiktok':
        user_num = 36656
        item_num = 76085
        train_edge = np.load(dir_str + '/train.npy', allow_pickle=True)
        user_item_dict = np.load(dir_str + '/user_item_dict.npy', allow_pickle=True).item()
        v_feat = torch.load(dir_str + '/visual_feat_new.pt').to(dtype=torch.float).cuda() if has_v else None
        a_feat = torch.load(dir_str + '/audio_feat_new.pt').to(dtype=torch.float).cuda() if has_a else None
        t_feat = torch.tensor(np.load(dir_str + '/tiktok_t_64.npy')).to(dtype=torch.float).cuda() if has_t else None
    elif dataset == 'kwai':
        user_num = 7010
        item_num = 86483
        train_edge = np.load(dir_str + '/train.npy', allow_pickle=True)
        user_item_dict = np.load(dir_str + '/user_item_dict.npy', allow_pickle=True).item()
        v_feat = torch.load(dir_str + '/v_feat.pt').to(dtype=torch.float).cuda() if has_v else None
        a_feat = None
        t_feat = torch.tensor(np.load(dir_str + '/kwai_t_64.npy')).to(dtype=torch.float).cuda() if has_t else None
    elif dataset in ['ml-1m', 'amazon-book']:
        train_edge = np.load(dir_str + '/train.npy', allow_pickle=True)
        user_item_dict = np.load(dir_str + '/user_item_dict.npy', allow_pickle=True).item()
        user_num = _infer_user_count(user_item_dict)
        try:
            variant_dir = _resolve_variant_dir(dataset, edge_variant, embedding_root)
        except FileNotFoundError:
            # Fallback mode for variants like v0: run pure LightGCN without edge modalities.
            variant_dir = None
            local_to_embed = None
            v_feat = a_feat = t_feat = e_feat = None
            item_num = _infer_item_count(dir_str, train_edge)
            print(
                f"[LightGT][fallback] Item embeddings not found for dataset={dataset}, "
                f"variant={edge_variant}. Running pure LightGCN without v/a/t/e features."
            )
        if variant_dir is not None:
            local_to_embed = _build_local_to_embed_map(dir_str, variant_dir)
            train_edge, user_item_dict = _remap_interactions(
                train_edge,
                user_item_dict,
                local_to_embed,
                min_keep_ratio=0.95,
            )
            v_feat = None
            a_feat = None
            t_feat = torch.tensor(np.load(os.path.join(variant_dir, 'item_embeddings.npy')), dtype=torch.float).cuda() if has_t else None
            e_feat = None
            item_num = int(t_feat.size(0)) if t_feat is not None else _infer_item_count(dir_str, train_edge)
            print(
                f"[LightGT][text-only] dataset={dataset}, variant={edge_variant}, "
                "using item_embeddings.npy as the only modality feature."
            )
    else:
        raise ValueError(f'Unsupported dataset: {dataset}')

    train_edge[:, 1] += user_num
    user_item_dict = {i:[j+user_num for j in user_item_dict[i]] for i in user_item_dict.keys()}

    local_to_embed = local_to_embed if dataset in ['ml-1m', 'amazon-book'] else None
    return user_num, item_num, train_edge, user_item_dict, v_feat, a_feat, t_feat, e_feat, local_to_embed

class TrainingDataset(Dataset):
    def __init__(self, dataset, user_num, item_num, user_item_dict, edge_index, src_len):
        self.dir_str = _resolve_data_dir(dataset)
        self.user_num = user_num
        self.item_num = item_num
        self.user_item_dict = user_item_dict
        self.edge_index = edge_index
        self.src_len = src_len
        self.all_set = set(range(user_num, user_num + item_num))
        self.graph = None

    def __len__(self):
        return len(self.edge_index)

    def __getitem__(self, index):
        user, pos_item = self.edge_index[index]
        while True:
            neg_item = random.sample(self.all_set, 1)[0]
            if neg_item not in self.user_item_dict[user]:
                break

        temp = list(self.user_item_dict[user])
        if len(temp) > self.src_len:
            mask = torch.ones(self.src_len + 1) == 0
            temp = temp[-self.src_len:]
        else:
            mask = torch.cat((torch.ones(len(temp) + 1), torch.zeros(self.src_len - len(temp)))) == 0
            temp.extend([self.user_num for i in range(self.src_len - len(temp))])

        user_item = torch.tensor(temp) - self.user_num
        user_item = torch.cat((torch.tensor([-1]), user_item))

        return torch.LongTensor([user,user]), torch.LongTensor([pos_item, neg_item]), user_item, mask

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def get_sparse_graph(self):
        # print('loading adjacency matrix')
        if self.graph is None:
            expected_size = self.user_num + self.item_num
            need_rebuild = False
            try:
                pre_adj_mat = sp.load_npz(self.dir_str + '/s_pre_adj_mat.npz')
                if pre_adj_mat.shape != (expected_size, expected_size):
                    need_rebuild = True
                else:
                    norm_adj = pre_adj_mat
            except Exception:
                need_rebuild = True

            if need_rebuild:
                # Cached adjacency may come from a different item indexing space
                # (e.g., remapped edge-embedding mode vs. v0 fallback mode).
                adj_mat = sp.dok_matrix((expected_size, expected_size), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                train_user = self.edge_index[:, 0]
                train_item = self.edge_index[:, 1] - self.user_num
                R = csr_matrix((np.ones(len(train_user)), (train_user, train_item)), shape=(self.user_num, self.item_num)).tolil()
                adj_mat[:self.user_num, self.user_num:] = R
                adj_mat[self.user_num:, :self.user_num] = R.T
                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum + 1e-5, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()

                sp.save_npz(self.dir_str + '/s_pre_adj_mat.npz', norm_adj)
                print(
                    f"[LightGT][graph] rebuilt cached adjacency with shape={norm_adj.shape} "
                    f"(expected={expected_size}x{expected_size})."
                )
            
            self.graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.graph = self.graph.coalesce().cuda()

        return self.graph

class EvalDataset(Dataset):
    def __init__(self, dataset, user_num, item_num, user_item_dict, src_len):
        self.dir_str = _resolve_data_dir(dataset)
        self.user_num = user_num
        self.item_num = item_num
        self.user_item_dict = user_item_dict
        self.src_len = src_len

    def __len__(self):
        return self.user_num

    def __getitem__(self, index):
        user = index

        temp = list(self.user_item_dict[user])
        if len(temp) > self.src_len:
            mask = torch.ones(self.src_len + 1) == 0
            temp = temp[-self.src_len:]
        else:
            mask = torch.cat((torch.ones(len(temp) + 1), torch.zeros(self.src_len - len(temp)))) == 0
            temp.extend([self.user_num for i in range(self.src_len - len(temp))])

        user_item = torch.tensor(temp) - self.user_num
        user_item = torch.cat((torch.tensor([-1]), user_item))
        
        return torch.LongTensor([user]), user_item, mask
