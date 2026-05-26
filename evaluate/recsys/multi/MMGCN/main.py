import argparse
import os
import random
import time
import numpy as np
import torch
from Dataset import TrainingDataset, data_load
from Model_MMGCN import Net
from torch.utils.data import DataLoader
from Train import train
from Full_vt import full_vt
# from torch.utils.tensorboard import SummaryWriter
###############################248###########################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Seed init.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--data_path', default='movielens', help='Dataset path')
    parser.add_argument('--save_file', default='', help='Filename')

    parser.add_argument('--PATH_weight_load', default=None, help='Loading weight filename.')
    parser.add_argument('--PATH_weight_save', default=None, help='Writing weight filename.')

    parser.add_argument('--l_r', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay.')

    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--num_epoch', type=int, default=100, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=1, help='Workers number.')

    parser.add_argument('--dim_E', type=int, default=128, help='Embedding dimension.')
    parser.add_argument('--prefix', default='', help='Prefix of save_file.')
    parser.add_argument('--aggr_mode', default='mean', help='Aggregation Mode.')
    parser.add_argument('--concate', default='False', help='Whether to use concatenation in the combination layer.')
    parser.add_argument('--num_layer', type=int, default=1, help='GCN layer count.')
    parser.add_argument('--topK', type=int, default=10, help='Workers number.')

    parser.add_argument('--has_entropy_loss', default='False', help='Has Cross Entropy loss.')
    parser.add_argument('--has_weight_loss', default='False', help='Has Weight Loss.')
    parser.add_argument('--has_v', default='False', help='Has Visual Features.')
    parser.add_argument('--has_a', default='False', help='Has Acoustic Features.')
    parser.add_argument('--has_t', default='True', help='Has Textual Features.')
    parser.add_argument('--has_e', default='False', help='Has 4th Edge Features.')
    parser.add_argument('--edge_variant', default='v3', help='Embedding variant name.')
    parser.add_argument('--embedding_root', default='../../../embedding/data', help='Root of precomputed embeddings.')
    parser.add_argument('--random_edge_features', default='False', help='Use random edge features as a lower-bound baseline.')

    args = parser.parse_args()
    
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    ##########################################################################################################################################
    data_path = args.data_path
    save_file = args.save_file

    learning_rate = args.l_r
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_epoch = args.num_epoch
    topK = args.topK
    prefix = args.prefix
    aggr_mode = args.aggr_mode
    concate = True if args.concate == 'True' else False
    num_layer = args.num_layer

    has_v = True if args.has_v == 'True' else False
    has_a = True if args.has_a == 'True' else False
    has_t = True if args.has_t == 'True' else False
    has_e = True if args.has_e == 'True' else False
    edge_variant = args.edge_variant
    embedding_root = args.embedding_root
    random_edge_features = True if args.random_edge_features == 'True' else False
    has_entropy_loss = True if args.has_entropy_loss == 'True' else False
    has_weight_loss = True if args.has_weight_loss == 'True' else False
    dim_E = args.dim_E
    writer = None#SummaryWriter()
    # with open(data_path+'/result/result{0}_{1}.txt'.format(l_r, weight_decay), 'w') as save_file:
    #     save_file.write('---------------------------------lr: {0} \t Weight_decay:{1} ---------------------------------\r\n'.format(l_r, weight_decay))
    ##########################################################################################################################################
    print('Data loading ...')

    num_user, num_item, train_edge, user_item_dict, v_feat, a_feat, t_feat, e_feat, local_to_embed = data_load(
        data_path,
        has_v=has_v,
        has_a=has_a,
        has_t=has_t,
        has_e=has_e,
        edge_variant=edge_variant,
        embedding_root=embedding_root,
        random_edge_features=random_edge_features,
        random_seed=seed,
    )

    v_feat = v_feat.to(dtype=torch.float, device=device) if has_v and v_feat is not None else None
    a_feat = a_feat.to(dtype=torch.float, device=device) if has_a and a_feat is not None else None
    t_feat = t_feat.to(dtype=torch.float, device=device) if has_t and t_feat is not None else None
    e_feat = e_feat.to(dtype=torch.float, device=device) if has_e and e_feat is not None else None

    train_dataset = TrainingDataset(num_user, num_item, user_item_dict, train_edge)
    def _seed_worker(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        generator=generator,
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir = os.path.join(project_dir, 'data', data_path)
    if not os.path.exists(data_dir):
        data_dir = os.path.join(project_dir, 'Data', data_path)
    val_data = np.load(os.path.join(data_dir, 'val_full.npy'), allow_pickle=True)
    test_data = np.load(os.path.join(data_dir, 'test_full.npy'), allow_pickle=True)

    def _maybe_offset_eval_items(eval_data):
        fixed = []
        for row in eval_data:
            row_list = list(row)
            if len(row_list) <= 1:
                fixed.append(row_list)
                continue
            items = [int(x) for x in row_list[1:]]
            if local_to_embed is not None:
                remapped = []
                for it in items:
                    if 0 <= it < len(local_to_embed):
                        new_it = int(local_to_embed[it])
                        if new_it >= 0:
                            remapped.append(new_it)
                items = remapped
            if not items:
                row_list = [int(row_list[0])]
                fixed.append(row_list)
                continue
            # If ids are not already in global node space, shift to global item ids.
            if min(items) < num_user:
                row_list = [int(row_list[0])] + [it + num_user for it in items]
            fixed.append(row_list)
        return np.array(fixed, dtype=object)

    val_data = _maybe_offset_eval_items(val_data)
    test_data = _maybe_offset_eval_items(test_data)
    print('Data has been loaded.')
    ##########################################################################################################################################
    model = Net(
        v_feat,
        a_feat,
        t_feat,
        e_feat,
        None,
        train_edge,
        batch_size,
        num_user,
        num_item,
        aggr_mode,
        concate,
        num_layer,
        True,
        user_item_dict,
        weight_decay,
        dim_E,
    ).cuda()
    ##########################################################################################################################################
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rate}])
    ##########################################################################################################################################
    max_precision = 0.0
    max_recall = 0.0
    max_NDCG = 0.0
    val_max_recall = 0.0
    num_decreases = 0 
    for epoch in range(num_epoch):
        loss = train(
            epoch,
            len(train_dataset),
            train_dataloader,
            model,
            optimizer,
            batch_size,
            writer=writer,
        )
        if torch.isnan(loss):
            with open(os.path.join(data_dir, 'result_{0}.txt'.format(save_file)), 'a') as save_file:
                    save_file.write('lr: {0} \t Weight_decay:{1} is Nan\r\n'.format(learning_rate, weight_decay))
            break
        torch.cuda.empty_cache()

        val_precision, val_recall, val_ndcg, val_hr = full_vt(epoch, model, val_data, 'Val', writer)
        test_precision, test_recall, test_ndcg, test_hr = full_vt(epoch, model, test_data, 'Test', writer)

        if val_recall > val_max_recall:
            val_max_recall = val_recall
            max_precision = test_precision
            max_recall = test_recall
            max_NDCG = test_ndcg
            num_decreases = 0
        else:
            if num_decreases > 20:
                with open(os.path.join(data_dir, 'result_{0}.txt'.format(save_file)), 'a') as save_file:
                    save_file.write('lr: {0} \t Weight_decay:{1} =====> Precision:{2} \t Recall:{3} \t NDCG:{4}\r\n'.
                                    format(learning_rate, weight_decay, max_precision, max_recall, max_NDCG))
                break
            else:
                num_decreases += 1
