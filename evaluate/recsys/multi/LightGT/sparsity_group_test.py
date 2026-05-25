import enum
import os
import json
import argparse
from tqdm import tqdm
import numpy as np
from datetime import datetime
import time
import torch
import torch.optim as optim
import math


def sparsity_group_test(model, configure, data_ori):
    conf = {
        "step": 5,
        "batch_size_test": 2048,
        "device": None,
        "src_len": None,
        "user_num": None,
        "item_num": None,
        "topk": None,
    }
    data = {
        "graph": None,
        "graph_train": None, # graph: sp_graph, [#users, #bundles]
    }
    
    for c in configure: conf[c] = configure[c]
    for d in data_ori: data[d] = data_ori[d]
    
    assert data['graph_train'] is not None and data['graph'] is not None 
    
    # 1. U-B interaction statistic
    # inter_count = data['graph_train'].sum(axis=1)
    # print(data['graph_train'])
    inter_count = []
    for user, item in data['graph_train'].items():
        inter_count.append(len(item))
    inter_count = np.array(inter_count)

    max_interaction = int(inter_count.max(axis=0)) + 1

    # 2. Define the groups of sparsity 
    num_group = math.ceil(max_interaction / conf['step'])
    groups = []
    min_ = max_ = 0
    for g in range(num_group):
        min_ = max_
        max_ = min_ + conf['step'] if min_ + conf['step'] <= max_interaction else max_interaction
        groups.append({'min': min_, 'max': max_})

    def group_idx(value):
        return int(value) // conf['step']

    user_groups = [[] for _ in groups]
    for u_idx,value in enumerate(inter_count):
        user_groups[group_idx(value)].append(u_idx)

    for g in range(num_group):
        groups[g]['count'] = len(user_groups[g])
        
    def split_by_batch_size(idxes):
        n_batch = math.ceil(len(idxes) / conf['batch_size_test'])
        bs = []
        start_idx = 0
        for b in range(n_batch):
            end_idx = start_idx + conf['batch_size_test']
            bs.append(torch.tensor(idxes[start_idx : end_idx if end_idx < len(idxes) else len(idxes)]))
            # end_idx = start_idx + conf['batch_size_test']
            start_idx = end_idx
        return bs
            
    user_groups = [split_by_batch_size(g) for g in user_groups]
    
    # 3. test each group
    
    device = conf["device"]
    model.eval()
    # rs = model.propagate(test=True)
    
    # config the test function >>>
    def test(user_group):
        tmp_metrics = {}
        for m in ["recall", "ndcg"]:
            tmp_metrics[m] = {}
            for topk in conf["topk"]:
                tmp_metrics[m][topk] = [0, 0]
        for batch_i, batch in enumerate(user_group):
            s_time = time.time()
            # >>>
            users = batch
            # ground_truth_u_b = torch.from_numpy(data['graph'][users].toarray()) #.squeeze()
            # train_mask_u_b = torch.from_numpy(data['graph_train'][users].toarray()) #.squeeze()
            ground_truth_u_b = get_ground_truth(users, data['graph'], conf['user_num'], conf['item_num'])
            train_mask_u_b = get_train_mask(users, data['graph_train'], conf['user_num'], conf['item_num'])

            user_item, mask = get_data(conf['user_num'], users, data['graph_train'], conf['src_len'])
            # <<<
            pred_b = model.get_score_matrix(users, user_item, mask)
            # pred_b = model.evaluate(rs, users.to(device))

            e_time = time.time() - s_time
            pred_b -= 1e8 * train_mask_u_b.to(device)
            tmp_metrics = get_metrics(tmp_metrics, ground_truth_u_b, pred_b, conf["topk"])
            # d_time = time.time()-  s_time
            # t_bar.set_description("e_time: %.5f d_time: %.5f" %(e_time, d_time))

        metrics = {}
        for m, topk_res in tmp_metrics.items():
            metrics[m] = {}
            for topk, res in topk_res.items():
                metrics[m][topk] = res[0] / (res[1] + 1e-12)

        return metrics
    # config the test function <<<
    
    # 4. store the results
    for g in range(num_group):
        groups[g]['metric'] = test(user_groups[g])

    return groups


def print_group_metrics(group_metrics):
    store = ""
    for gm in group_metrics:
        m = gm['metric']
        str_ = "[%d, %d) count: %d "%(gm['min'], gm['max'], gm['count'])
        str_ += ", ".join("R@%d: %.6f N@%d: %.6f"%(topk, m['recall'][topk], topk, m['ndcg'][topk]) for topk in m['recall'])
        print(str_)
        store += str_ + "\n"
    return store
        

def get_metrics(metrics, grd, pred, topks):
    tmp = {"recall": {}, "ndcg": {}}
    for topk in topks:
        _, col_indice = torch.topk(pred, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(pred.shape[0], device=pred.device, dtype=torch.long).view(-1, 1)
        is_hit = grd[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)
        # print(grd.sum(dim=1))
        # print(is_hit.sum(dim=1))

        tmp["recall"][topk] = get_recall(pred, grd, is_hit, topk)
        tmp["ndcg"][topk] = get_ndcg(pred, grd, is_hit, topk)

    for m, topk_res in tmp.items():
        for topk, res in topk_res.items():
            for i, x in enumerate(res):
                metrics[m][topk][i] += x

    return metrics


def get_recall(pred, grd, is_hit, topk):
    epsilon = 1e-8
    hit_cnt = is_hit.sum(dim=1)
    num_pos = grd.sum(dim=1)

    # remove those test cases who don't have any positive items
    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = (hit_cnt/(num_pos+epsilon)).sum().item()

    return [nomina, denorm]

def get_ndcg(pred, grd, is_hit, topk):
    def DCG(hit, topk, device):
        hit = hit/torch.log2(torch.arange(2, topk+2, device=device, dtype=torch.float))
        return hit.sum(-1)

    def IDCG(num_pos, topk, device):
        hit = torch.zeros(topk, dtype=torch.float)
        hit[:num_pos] = 1
        return DCG(hit, topk, device)

    device = grd.device
    IDCGs = torch.empty(1+topk, dtype=torch.float)
    IDCGs[0] = 1  # avoid 0/0
    for i in range(1, topk+1):
        IDCGs[i] = IDCG(i, topk, device)

    num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
    dcg = DCG(is_hit, topk, device)

    idcg = IDCGs[num_pos]
    ndcg = dcg/idcg.to(device)

    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = ndcg.sum().item()

    return [nomina, denorm]

def get_data(user_num, user, user_item_dict, src_len):
    user_item, mask = torch.tensor([]), torch.tensor([])
    user = user.numpy()
    for i in range(len(user)):
        temp = list(user_item_dict[user[i]])
        if len(temp) > src_len:
            mask0 = torch.ones(src_len + 1) == 0
            temp = temp[-src_len:]
        else:
            mask0 = torch.cat((torch.ones(len(temp) + 1), torch.zeros(src_len - len(temp)))) == 0
            temp.extend([user_num for i in range(src_len - len(temp))])

        temp = torch.tensor(temp) - user_num
        temp = torch.cat((torch.tensor([-1]), temp)) # 添加cls

        user_item = torch.cat((user_item, temp), dim=0)
        mask = torch.cat((mask, mask0), dim=0)

    user_item = user_item.view(len(user), -1).numpy()
    mask = mask.view(len(user), -1).numpy()

    return torch.LongTensor(user_item), torch.BoolTensor(mask)


def get_ground_truth(users, test_graph, user_num, item_num):
    cnt = 0
    graph = []
    for i in range(user_num):
        if (test_graph[cnt][0] == i):
            graph.append(test_graph[cnt])
            cnt += 1
        else:
            graph.append([])

    score_matrix = torch.zeros((len(users), item_num))

    for i in range(len(users)):
        for j in range(1, len(graph[users[i]])):
            score_matrix[i][graph[users[i]][j]] = 1

    return score_matrix

def get_train_mask(users, graph, user_num, item_num):
    users = users.numpy()
    score_matrix = torch.zeros((len(users), item_num))
    for i in range(len(users)):
        for j in graph[users[i]]:
            score_matrix[i][j-user_num] = 1
    return score_matrix
