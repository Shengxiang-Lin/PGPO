import argparse
import ast
import csv
from datetime import datetime
import gzip
import json
import os
import random
from collections import Counter
from tqdm import tqdm

DATASET_CONFIGS = {
    'amazon-book': {
        'input_csv': './data/amazon-book/raw/Books.csv',
        'input_meta': './data/amazon-book/raw/meta_Books.json.gz',
        'output_dir': './data/amazon-book/raw-0'
    }
}


def read_interactions(csv_path):
    interactions = []
    with open(csv_path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
        reader = csv.reader(f)
        for row in tqdm(reader, desc='Reading interactions', unit='rows'):
            if len(row) < 2:
                continue
            item_id = row[0].strip()
            user_id = row[1].strip()
            if not item_id or not user_id:
                continue
            interactions.append((user_id, item_id, row))
    return interactions


def k_core_filter(interactions, core_k, max_rounds=100, min_drop=0):
    current = interactions
    round_idx = 0
    while round_idx < max_rounds:
        round_idx += 1
        user_counts = Counter(u for u, _, _ in current)
        item_counts = Counter(i for _, i, _ in current)
        filtered = []
        for u, i, row in tqdm(current, desc=f'Core filtering round {round_idx}', unit='rows'):
            if user_counts[u] >= core_k and item_counts[i] >= core_k:
                filtered.append((u, i, row))
        removed = len(current) - len(filtered)
        users_after = len(set(u for u, _, _ in filtered))
        items_after = len(set(i for _, i, _ in filtered))
        print(f'Core round {round_idx}: {len(current)} -> {len(filtered)} (removed={removed}, users={users_after}, items={items_after})')
        if removed == 0:
            return filtered
        if removed <= min_drop:
            print(f'Stop core filtering due to min_drop={min_drop}, current removed={removed}')
            return filtered
        current = filtered
    print(f'Stop core filtering due to max_rounds={max_rounds}')
    return current


def sample_items(interactions, target_items, seed):
    item_ids = sorted(set(i for _, i, _ in interactions))
    if len(item_ids) <= target_items:
        return set(item_ids)
    rng = random.Random(seed)
    sampled = rng.sample(item_ids, target_items)
    return set(sampled)


def filter_interactions_by_items(interactions, sampled_items):
    filtered = []
    for u, i, row in tqdm(interactions, desc='Filtering sampled items', unit='rows'):
        if i in sampled_items:
            filtered.append((u, i, row))
    return filtered


def write_interactions(output_csv, interactions):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for _, _, row in tqdm(interactions, desc='Writing interactions', unit='rows'):
            writer.writerow(row)


def filter_and_write_meta(input_meta_gz, output_meta_gz, keep_items):
    os.makedirs(os.path.dirname(output_meta_gz), exist_ok=True)
    kept = 0
    with gzip.open(input_meta_gz, 'rt', encoding='utf-8', errors='ignore') as fin, gzip.open(output_meta_gz, 'wt', encoding='utf-8') as fout:
        for line in tqdm(fin, desc='Filtering meta', unit='lines'):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                try:
                    obj = ast.literal_eval(line)
                except (ValueError, SyntaxError):
                    continue
            asin = obj.get('asin')
            if asin in keep_items:
                fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
                kept += 1
    return kept


def calc_stats(interactions):
    users = len(set(u for u, _, _ in interactions))
    items = len(set(i for _, i, _ in interactions))
    num_interactions = len(interactions)
    denom = users * items
    sparsity = 1.0 - (num_interactions / denom) if denom > 0 else 1.0
    return {
        'users': users,
        'items': items,
        'interactions': num_interactions,
        'sparsity': sparsity
    }


def get_timestamp_range(interactions):
    min_ts = None
    max_ts = None
    for _, _, row in interactions:
        if len(row) < 4:
            continue
        try:
            ts = int(row[3])
        except (ValueError, TypeError):
            continue
        if min_ts is None or ts < min_ts:
            min_ts = ts
        if max_ts is None or ts > max_ts:
            max_ts = ts
    return min_ts, max_ts


def write_stats(stats_path, stats):
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def resolve_min_timestamp(min_timestamp, min_date):
    if min_timestamp is not None:
        return int(min_timestamp)
    if min_date:
        dt = datetime.strptime(min_date, '%Y-%m-%d')
        return int(dt.timestamp())
    raise ValueError('time_after mode requires --min_timestamp or --min_date')


def filter_interactions_by_min_timestamp(interactions, min_timestamp):
    filtered = []
    for u, i, row in tqdm(interactions, desc='Filtering by time', unit='rows'):
        if len(row) < 4:
            continue
        try:
            ts = int(row[3])
        except (ValueError, TypeError):
            continue
        if ts >= min_timestamp:
            filtered.append((u, i, row))
    return filtered


def resolve_time_window(start_date, end_date):
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp()) + 24 * 60 * 60 - 1
    if end_ts < start_ts:
        raise ValueError('end_date must be greater than or equal to start_date')
    return start_ts, end_ts


def filter_interactions_by_time_window(interactions, start_ts, end_ts):
    filtered = []
    for u, i, row in tqdm(interactions, desc='Filtering by time window', unit='rows'):
        if len(row) < 4:
            continue
        try:
            ts = int(row[3])
        except (ValueError, TypeError):
            continue
        if start_ts <= ts <= end_ts:
            filtered.append((u, i, row))
    return filtered


def apply_min_frequency_filter(interactions, min_item_freq, min_user_freq, max_rounds=50):
    current = interactions
    round_idx = 0
    while round_idx < max_rounds:
        round_idx += 1
        item_counts = Counter(i for _, i, _ in current)
        user_counts = Counter(u for u, _, _ in current)
        filtered = []
        for u, i, row in tqdm(current, desc=f'Frequency filtering round {round_idx}', unit='rows'):
            if item_counts[i] >= min_item_freq and user_counts[u] >= min_user_freq:
                filtered.append((u, i, row))
        removed = len(current) - len(filtered)
        users_after = len(set(u for u, _, _ in filtered))
        items_after = len(set(i for _, i, _ in filtered))
        print(f'Frequency round {round_idx}: {len(current)} -> {len(filtered)} (removed={removed}, users={users_after}, items={items_after})')
        if removed == 0:
            return filtered
        current = filtered
    return current


def select_last_n_interactions(interactions, last_n):
    with_ts = []
    without_ts = []
    for idx, (u, i, row) in enumerate(tqdm(interactions, desc='Collecting timestamps', unit='rows')):
        if len(row) >= 4:
            try:
                ts = int(row[3])
                with_ts.append((ts, idx, u, i, row))
                continue
            except (ValueError, TypeError):
                pass
        without_ts.append((idx, u, i, row))

    if with_ts:
        with_ts.sort(key=lambda x: (x[0], x[1]))
        selected = with_ts[-last_n:] if len(with_ts) > last_n else with_ts
        selected.sort(key=lambda x: (x[0], x[1]))
        return [(u, i, row) for _, _, u, i, row in selected]

    return [(u, i, row) for _, u, i, row in without_ts[-last_n:]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=list(DATASET_CONFIGS.keys()), default='amazon-book')
    parser.add_argument('--input_csv', default=None)
    parser.add_argument('--input_meta', default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--mode', choices=['core_sample', 'time_after', 'last_n', 'window_freq_sample'], default='window_freq_sample')
    parser.add_argument('--core_k', type=int, default=5)
    parser.add_argument('--target_items', type=int, default=10000)
    parser.add_argument('--start_date', default='2018-01-01')
    parser.add_argument('--end_date', default='2018-03-31')
    parser.add_argument('--min_item_freq', type=int, default=3)
    parser.add_argument('--min_user_freq', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_core_rounds', type=int, default=100)
    parser.add_argument('--min_core_drop', type=int, default=0)
    parser.add_argument('--min_timestamp', type=int, default=None)
    parser.add_argument('--min_date', default=None)
    parser.add_argument('--last_n', type=int, default=100000)
    args = parser.parse_args()
    defaults = DATASET_CONFIGS[args.dataset]
    args.input_csv = args.input_csv or defaults['input_csv']
    args.input_meta = args.input_meta or defaults['input_meta']
    args.output_dir = args.output_dir or defaults['output_dir']

    if args.target_items <= 0:
        raise ValueError('target_items must be greater than 0')
    if args.core_k < 1:
        raise ValueError('core_k must be at least 1')
    if args.max_core_rounds < 1:
        raise ValueError('max_core_rounds must be at least 1')
    if args.min_core_drop < 0:
        raise ValueError('min_core_drop must be at least 0')
    if args.last_n <= 0:
        raise ValueError('last_n must be greater than 0')
    if args.min_item_freq < 1:
        raise ValueError('min_item_freq must be at least 1')
    if args.min_user_freq < 1:
        raise ValueError('min_user_freq must be at least 1')

    print(f'Loading interactions from: {args.input_csv}')
    interactions = read_interactions(args.input_csv)
    raw_stats = calc_stats(interactions)
    print(f'Raw stats: users={raw_stats["users"]}, items={raw_stats["items"]}, interactions={raw_stats["interactions"]}, sparsity={raw_stats["sparsity"]:.6f}')
    raw_min_ts, raw_max_ts = get_timestamp_range(interactions)
    if raw_min_ts is not None and raw_max_ts is not None:
        min_date_str = datetime.utcfromtimestamp(raw_min_ts).strftime('%Y-%m-%d')
        max_date_str = datetime.utcfromtimestamp(raw_max_ts).strftime('%Y-%m-%d')
        print(f'Timestamp range in data: [{raw_min_ts}, {raw_max_ts}] ({min_date_str} ~ {max_date_str}, UTC)')

    core_stats = None
    if args.mode == 'time_after':
        threshold_ts = resolve_min_timestamp(args.min_timestamp, args.min_date)
        print(f'Applying time filter: timestamp >= {threshold_ts}')
        final_interactions = filter_interactions_by_min_timestamp(interactions, threshold_ts)
        final_items = set(i for _, i, _ in final_interactions)
        final_stats = calc_stats(final_interactions)
        print(f'Final stats: users={final_stats["users"]}, items={final_stats["items"]}, interactions={final_stats["interactions"]}, sparsity={final_stats["sparsity"]:.6f}')
    elif args.mode == 'last_n':
        print(f'Selecting last {args.last_n} interactions by timestamp...')
        final_interactions = select_last_n_interactions(interactions, args.last_n)
        final_items = set(i for _, i, _ in final_interactions)
        final_stats = calc_stats(final_interactions)
        print(f'Final stats: users={final_stats["users"]}, items={final_stats["items"]}, interactions={final_stats["interactions"]}, sparsity={final_stats["sparsity"]:.6f}')
    elif args.mode == 'window_freq_sample':
        start_ts, end_ts = resolve_time_window(args.start_date, args.end_date)
        print(f'Applying time window filter: {args.start_date} ~ {args.end_date} (timestamp {start_ts} ~ {end_ts})')
        window_interactions = filter_interactions_by_time_window(interactions, start_ts, end_ts)
        window_stats = calc_stats(window_interactions)
        print(f'After window filter: users={window_stats["users"]}, items={window_stats["items"]}, interactions={window_stats["interactions"]}, sparsity={window_stats["sparsity"]:.6f}')

        print(f'Applying frequency filter before sampling: min_item_freq={args.min_item_freq}, min_user_freq={args.min_user_freq}')
        pre_sample_interactions = apply_min_frequency_filter(window_interactions, args.min_item_freq, args.min_user_freq)
        pre_sample_stats = calc_stats(pre_sample_interactions)
        print(f'After pre-sample frequency filter: users={pre_sample_stats["users"]}, items={pre_sample_stats["items"]}, interactions={pre_sample_stats["interactions"]}, sparsity={pre_sample_stats["sparsity"]:.6f}')

        sampled_items = sample_items(pre_sample_interactions, args.target_items, args.seed)
        print(f'Sampled item count: {len(sampled_items)}')
        sampled_interactions = filter_interactions_by_items(pre_sample_interactions, sampled_items)
        sampled_stats = calc_stats(sampled_interactions)
        print(f'After item sampling: users={sampled_stats["users"]}, items={sampled_stats["items"]}, interactions={sampled_stats["interactions"]}, sparsity={sampled_stats["sparsity"]:.6f}')

        print(f'Applying frequency filter after sampling: min_item_freq={args.min_item_freq}, min_user_freq={args.min_user_freq}')
        final_interactions = apply_min_frequency_filter(sampled_interactions, args.min_item_freq, args.min_user_freq)
        final_items = set(i for _, i, _ in final_interactions)
        final_stats = calc_stats(final_interactions)
        print(f'Final stats: users={final_stats["users"]}, items={final_stats["items"]}, interactions={final_stats["interactions"]}, sparsity={final_stats["sparsity"]:.6f}')
    else:
        print(f'Applying core-{args.core_k} filtering...')
        core_interactions = k_core_filter(
            interactions,
            args.core_k,
            max_rounds=args.max_core_rounds,
            min_drop=args.min_core_drop
        )
        core_stats = calc_stats(core_interactions)
        print(f'Core stats: users={core_stats["users"]}, items={core_stats["items"]}, interactions={core_stats["interactions"]}, sparsity={core_stats["sparsity"]:.6f}')

        print(f'Sampling items to target count: {args.target_items}')
        sampled_items = sample_items(core_interactions, args.target_items, args.seed)
        final_interactions = filter_interactions_by_items(core_interactions, sampled_items)
        final_items = sampled_items
        final_stats = calc_stats(final_interactions)
        print(f'Final stats: users={final_stats["users"]}, items={final_stats["items"]}, interactions={final_stats["interactions"]}, sparsity={final_stats["sparsity"]:.6f}')

    output_csv = os.path.join(args.output_dir, os.path.basename(args.input_csv))
    output_meta = os.path.join(args.output_dir, os.path.basename(args.input_meta))
    output_stats = os.path.join(args.output_dir, 'stats.json')

    print(f'Writing filtered interactions to: {output_csv}')
    write_interactions(output_csv, final_interactions)
    print(f'Filtering metadata to: {output_meta}')
    kept_meta = filter_and_write_meta(args.input_meta, output_meta, final_items)
    print(f'Metadata kept: {kept_meta}')

    stats = {
        'config': {
            'core_k': args.core_k,
            'target_items': args.target_items,
            'start_date': args.start_date,
            'end_date': args.end_date,
            'min_item_freq': args.min_item_freq,
            'min_user_freq': args.min_user_freq,
            'seed': args.seed,
            'max_core_rounds': args.max_core_rounds,
            'min_core_drop': args.min_core_drop,
            'mode': args.mode,
            'min_timestamp': args.min_timestamp,
            'min_date': args.min_date,
            'last_n': args.last_n
        },
        'raw': raw_stats,
        'raw_timestamp_range': {
            'min_timestamp': raw_min_ts,
            'max_timestamp': raw_max_ts
        },
        'after_core': core_stats,
        'final': final_stats,
        'meta_items_kept': kept_meta
    }
    write_stats(output_stats, stats)
    print(f'Stats written to: {output_stats}')


if __name__ == '__main__':
    main()
