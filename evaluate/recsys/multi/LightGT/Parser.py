import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2000, help='Seed init.')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--dataset', type=str, default='movielens', help='Dataset.')
    
    parser.add_argument('--PATH_weight_load', type=str, default=None, help='Loading weight filename.')
    parser.add_argument('--PATH_weight_save', type=str, default=None, help='Writing weight filename.')
    
    parser.add_argument('--l_r', type=float, default=1e-2, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay.')
    
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size.')
    parser.add_argument('--num_epoch', type=int, default=100, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=8, help='Workers number.')

    parser.add_argument('--dim_E', type=int, default=64, help='Embedding dimension.')
    parser.add_argument('--src_len', type=int, default=50, help='The length of transformer src.')
    parser.add_argument('--transformer_layers', type=int, default=3, help='Number of layers in transformer.')
    parser.add_argument('--nhead', type=int, default=1, help='Number of heads in transformer.')
    parser.add_argument('--lightgcn_layers', type=int, default=3, help='Layer number of lightgcn.')
    parser.add_argument('--score_weight', type=float, default=0.5, help='The weight of score1 and score2.')
    
    parser.add_argument('--prefix', type=str, default='', help='Prefix of save_file.')
    parser.add_argument('--aggr_mode', type=str, default='add', help='Aggregation mode.')
    parser.add_argument('--topK', type=int, default=10, help='@k test list.')

    parser.add_argument('--has_entropy_loss', default='False', help='Has Cross Entropy loss.')
    parser.add_argument('--has_weight_loss', default='False', help='Has Weight Loss.')
    parser.add_argument('--has_v', default='False', help='Has Visual Features.')
    parser.add_argument('--has_a', default='False', help='Has Acoustic Features.')
    parser.add_argument('--has_t', default='True', help='Has Textual Features.')
    parser.add_argument('--has_e', default='False', help='Has 4th Edge Features.')
    parser.add_argument('--edge_variant', type=str, default='v3', help='Embedding variant name (e.g., v2/v3).')
    parser.add_argument('--embedding_root', type=str, default='../../../embedding/data', help='Root of precomputed embeddings.')

    return parser.parse_args()
