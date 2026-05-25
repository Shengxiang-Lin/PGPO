import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys
from time import time
from transformer import TransformerEncoder, TransformerEncoderLayer

class LightGCN(nn.Module):
    def __init__(self, user_num, item_num, graph, transformer_layers, latent_dim=64, n_layers=3):
        super(LightGCN, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.graph = graph
        self.transformer_layers = transformer_layers
        self.latent_dim = latent_dim
        self.n_layers = n_layers

        self.user_emb = nn.Embedding(user_num, latent_dim)
        nn.init.xavier_normal_(self.user_emb.weight)
        self.item_emb = nn.Embedding(item_num, latent_dim)
        nn.init.xavier_normal_(self.item_emb.weight)

    def cal_mean(self, embs):
        if (len(embs) > 1):
            embs = torch.stack(embs, dim=1)
            embs = torch.mean(embs, dim=1)
        else:
            embs = embs[0]
        users_emb, items_emb = torch.split(embs, [self.user_num, self.item_num])

        return users_emb, items_emb

    def forward(self):
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight])
        embs = [all_emb]

        embs_mean = []
        for i in range(self.n_layers):
            embs_mean.append([all_emb])

        for layer in range(self.transformer_layers):
            all_emb = torch.sparse.mm(self.graph, all_emb)
            if layer < self.n_layers:
                embs.append(all_emb)

            for i in range(layer, self.transformer_layers):
                embs_mean[i].append(all_emb)
            # embs_mean[layer].append(all_emb)

        users, items = self.cal_mean(embs)

        users_mean, items_mean = [], []
        for i in range(self.transformer_layers):
            a, b = self.cal_mean(embs_mean[i])
            users_mean.append(a)
            items_mean.append(b)

        return users, items, users_mean, items_mean


class Net(nn.Module):
    def __init__(self, user_num, item_num, graph, user_item_dict, v_feat, a_feat, t_feat, e_feat, eval_dataloader, reg_weight, src_len, batch_size=2048, latent_dim=64, transformer_layers=4, nhead=1, lightgcn_layers=3, score_weight=0.5):
        super(Net, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.graph = graph
        self.user_item_dict = user_item_dict
        self.v_feat = F.normalize(v_feat) if v_feat != None else None
        self.a_feat = F.normalize(a_feat) if a_feat != None else None
        self.t_feat = F.normalize(t_feat) if t_feat != None else None
        self.e_feat = F.normalize(e_feat) if e_feat != None else None
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.src_len = src_len
        self.weight = torch.tensor([[1.], [-1.]]).cuda()
        self.reg_weight = reg_weight
        self.score_weight1 = score_weight
        self.score_weight2 = 1-score_weight
        self.eval_dataloader = eval_dataloader
        self.has_modal_features = any(feat is not None for feat in (self.v_feat, self.a_feat, self.t_feat, self.e_feat))

        self.transformer_layers = transformer_layers
        self.nhead = nhead
        self.lightgcn_layers = lightgcn_layers

        self.lightgcn = LightGCN(user_num, item_num, graph, transformer_layers, latent_dim, lightgcn_layers)

        self.user_exp = nn.Parameter(torch.rand(user_num, latent_dim))
        nn.init.xavier_normal_(self.user_exp)

        if self.v_feat != None:
            self.v_mlp = nn.Linear(latent_dim, latent_dim)
            self.v_linear = nn.Linear(self.v_feat.size(1), latent_dim)
            self.v_encoder_layer = TransformerEncoderLayer(d_model=latent_dim, nhead=nhead)
            self.v_encoder = TransformerEncoder(self.v_encoder_layer, num_layers=transformer_layers)
            self.v_dense = nn.Linear(latent_dim, latent_dim)

        if self.a_feat != None:
            self.a_mlp = nn.Linear(latent_dim, latent_dim)
            self.a_linear = nn.Linear(self.a_feat.size(1), latent_dim)
            self.a_encoder_layer = TransformerEncoderLayer(d_model=latent_dim, nhead=nhead)
            self.a_encoder = TransformerEncoder(self.a_encoder_layer, num_layers=transformer_layers)
            self.a_dense = nn.Linear(latent_dim, latent_dim)

        if self.t_feat != None:
            self.t_mlp = nn.Linear(latent_dim, latent_dim)
            self.t_linear = nn.Linear(self.t_feat.size(1), latent_dim)
            self.t_encoder_layer = TransformerEncoderLayer(d_model=latent_dim, nhead=nhead)
            self.t_encoder = TransformerEncoder(self.t_encoder_layer, num_layers=transformer_layers)
            self.t_dense = nn.Linear(latent_dim, latent_dim)

        if self.e_feat != None:
            self.e_mlp = nn.Linear(latent_dim, latent_dim)
            self.e_linear = nn.Linear(self.e_feat.size(1), latent_dim)
            self.e_encoder_layer = TransformerEncoderLayer(d_model=latent_dim, nhead=nhead)
            self.e_encoder = TransformerEncoder(self.e_encoder_layer, num_layers=transformer_layers)
            self.e_dense = nn.Linear(latent_dim, latent_dim)

    def forward(self, users, user_item, mask):
        if torch.min(users).item() < 0 or torch.max(users).item() >= self.user_num:
            raise ValueError(
                f"user index out of range: min={int(torch.min(users).item())}, "
                f"max={int(torch.max(users).item())}, user_num={self.user_num}"
            )
        if torch.min(user_item).item() < -1 or torch.max(user_item).item() >= self.item_num:
            raise ValueError(
                f"item index out of range: min={int(torch.min(user_item).item())}, "
                f"max={int(torch.max(user_item).item())}, item_num={self.item_num}"
            )
        user_emb, item_emb, users_mean, items_mean = self.lightgcn()

        v_src, a_src, t_src, e_src = [], [], [], []
        for i in range(self.transformer_layers):
            temp = items_mean[i][user_item].detach()
            temp[:, 0] = users_mean[i][users].detach()
            if self.v_feat != None:
                v_src.append(torch.sigmoid(self.v_mlp(temp).transpose(0, 1)))
            if self.a_feat != None:
                a_src.append(torch.sigmoid(self.a_mlp(temp).transpose(0, 1)))
            if self.t_feat != None:
                t_src.append(torch.sigmoid(self.t_mlp(temp).transpose(0, 1)))
            if self.e_feat != None:
                e_src.append(torch.sigmoid(self.e_mlp(temp).transpose(0, 1)))

        v, a, t, e = None, None, None, None
        v_out, a_out, t_out, e_out = None, None, None, None

        if self.v_feat != None:
            v = self.v_linear(self.v_feat)
            v_in = v[user_item]
            v_in[:, 0] = self.user_exp[users]
            v_out = self.v_encoder(v_in.transpose(0, 1), v_src, src_key_padding_mask=mask).transpose(0, 1)[:, 0]
            v_out = F.leaky_relu(self.v_dense(v_out))

        if self.a_feat != None:
            a = self.a_linear(self.a_feat)
            a_in = a[user_item]
            a_in[:, 0] = self.user_exp[users]
            a_out = self.a_encoder(a_in.transpose(0, 1), a_src, src_key_padding_mask=mask).transpose(0, 1)[:, 0]
            a_out = F.leaky_relu(self.a_dense(a_out))

        if self.t_feat != None:
            t = self.t_linear(self.t_feat)
            t_in = t[user_item]
            t_in[:, 0] = self.user_exp[users]
            t_out = self.t_encoder(t_in.transpose(0, 1), t_src, src_key_padding_mask=mask).transpose(0, 1)[:, 0]
            t_out = F.leaky_relu(self.t_dense(t_out))

        if self.e_feat != None:
            e = self.e_linear(self.e_feat)
            e_in = e[user_item]
            e_in[:, 0] = self.user_exp[users]
            e_out = self.e_encoder(e_in.transpose(0, 1), e_src, src_key_padding_mask=mask).transpose(0, 1)[:, 0]
            e_out = F.leaky_relu(self.e_dense(e_out))

        return user_emb, item_emb, v, a, t, e, v_out, a_out, t_out, e_out

    def loss(self, users, items, user_item, mask):
        user_emb, item_emb, v, a, t, e, v_out, a_out, t_out, e_out = self.forward(users[:, 0], user_item, mask.cuda())

        users = users.view(-1)
        items = items - self.user_num

        pos_items = items[:, 0].view(-1)
        neg_items = items[:, 1].view(-1)
        items = items.view(-1)

        score1 = torch.sum(user_emb[users] * item_emb[items], dim=1).view(-1, 2)

        modal_pairs = [(v, v_out), (a, a_out), (t, t_out), (e, e_out)]
        score2_1 = None
        score2_2 = None
        for item_modal, user_modal in modal_pairs:
            if item_modal is None or user_modal is None:
                continue
            cur_pos = torch.sum(user_modal * item_modal[pos_items], dim=1).view(-1, 1)
            cur_neg = torch.sum(user_modal * item_modal[neg_items], dim=1).view(-1, 1)
            score2_1 = cur_pos if score2_1 is None else score2_1 + cur_pos
            score2_2 = cur_neg if score2_2 is None else score2_2 + cur_neg
        if score2_1 is None or score2_2 is None:
            score = score1
        else:
            score = self.score_weight1 * score1 + self.score_weight2 * torch.cat((score2_1, score2_2), dim=1)

        loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight)))).cuda()
        reg_embedding_loss = (user_emb**2).mean() + (item_emb**2).mean()
        reg_loss = self.reg_weight * reg_embedding_loss

        if torch.isnan(loss):
            print('Loss is Nan.')
            exit()

        return loss + reg_loss, reg_loss, loss, reg_embedding_loss, reg_embedding_loss

    def get_score_matrix(self, users, user_item, mask):
        user_emb, item_emb, v, a, t, e, v_out, a_out, t_out, e_out = self.forward(users, user_item, mask.cuda())

        score1 = torch.matmul(user_emb[users], item_emb.T)

        modal_pairs = [(v, v_out), (a, a_out), (t, t_out), (e, e_out)]
        score2 = None
        for item_modal, user_modal in modal_pairs:
            if item_modal is None or user_modal is None:
                continue
            cur = torch.matmul(user_modal, item_modal.T)
            score2 = cur if score2 is None else score2 + cur
        if score2 is None:
            score_matrix = score1
        else:
            score_matrix = self.score_weight1 * score1 + self.score_weight2 * score2

        return score_matrix

    def accuracy(self, step=2000, topk=10):
        start_index = 0
        end_index = self.user_num if step == None else step

        all_index_of_rank_list = torch.LongTensor([])
        for users, user_item, mask in self.eval_dataloader:
            score_matrix = self.get_score_matrix(users.view(-1), user_item, mask)
            _, index_of_rank_list = torch.topk(score_matrix, topk)
            all_index_of_rank_list = torch.cat((all_index_of_rank_list, index_of_rank_list.cpu()+self.user_num), dim=0)
            
            start_index = end_index
            if end_index + step < self.user_num:
                end_index += step
            else:
                end_index = self.user_num

        length = self.user_num
        precision = recall = ndcg = 0.0

        for row, col in self.user_item_dict.items():
            user = row
            pos_items = set(col)
            num_pos = len(pos_items)
            items_list = all_index_of_rank_list[user].tolist()
    
            items = set(items_list)

            num_hit = len(pos_items.intersection(items))

            precision += float(num_hit / topk)
            recall += float(num_hit / num_pos)

            ndcg_score = 0.0
            max_ndcg_score = 0.0

            for i in range(min(num_hit, topk)):
                max_ndcg_score += 1 / math.log2(i+2)
            
            if max_ndcg_score == 0:
                continue
                
            for i, temp_item in enumerate(items_list):
                if temp_item in pos_items:
                    ndcg_score += 1 / math.log2(i+2)
            
            ndcg += ndcg_score / max_ndcg_score

        return precision / length, recall / length, ndcg / length

    def full_accuracy(self, val_data, step=2000, topk=10):
        start_index = 0
        end_index = self.user_num if step == None else step


        all_index_of_rank_list = torch.LongTensor([])
        for users, user_item, mask in self.eval_dataloader:
            score_matrix = self.get_score_matrix(users.view(-1), user_item, mask)

            for row, col in self.user_item_dict.items():
                if row >= start_index and row < end_index:
                    row -= start_index
                    col = torch.LongTensor(list(col)) - self.user_num
                    score_matrix[row][col] = 1e-5
                
            _, index_of_rank_list = torch.topk(score_matrix, topk)
            all_index_of_rank_list = torch.cat((all_index_of_rank_list, index_of_rank_list.cpu()+self.user_num), dim=0)
            
            start_index = end_index
            if end_index + step < self.user_num:
                end_index += step
            else:
                end_index = self.user_num
        
        length = 0
        precision = recall = ndcg = 0.0
        hit_users = 0

        for data in val_data:
            user = data[0]
            pos_items = set(data[1:])
            num_pos = len(pos_items)
            if num_pos == 0:
                continue
            length += 1
            items_list = all_index_of_rank_list[user].tolist()

            items = set(items_list)

            num_hit = len(pos_items.intersection(items))
            if num_hit > 0:
                hit_users += 1

            precision += float(num_hit / topk)
            recall += float(num_hit / num_pos)
            
            ndcg_score = 0.0
            max_ndcg_score = 0.0

            for i in range(min(num_pos, topk)):
                max_ndcg_score += 1 / math.log2(i+2)
            if max_ndcg_score == 0:
                continue

            for i, temp_item in enumerate(items_list):
                if temp_item in pos_items:
                    ndcg_score += 1 / math.log2(i+2)

            ndcg += ndcg_score / max_ndcg_score

        return precision / length, recall / length, ndcg / length, hit_users / length
