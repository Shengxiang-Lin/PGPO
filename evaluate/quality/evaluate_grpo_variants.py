import argparse
import json
import os
import numpy as np
from scipy.stats import wasserstein_distance, spearmanr
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def cosine_sim(vec1, vec2):
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)

def normalize_distribution(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    min_v = np.min(arr)
    if min_v < 0:
        arr = arr - min_v
    arr = arr + 1e-10
    s = np.sum(arr)
    if s <= 0:
        return np.ones_like(arr) / len(arr)
    return arr / s

def l2_distance(vec1, vec2):
    return float(np.linalg.norm(vec1 - vec2))

def split_base_categories(category):
    return [c.strip() for c in str(category).split(',') if c and c.strip()]

def evaluate(args):
    print("Loading inputs...")
    generated_data = load_json(args.generated_file)
    print(f"Loading embedding model from: {args.embedding_model_path}")
    model = SentenceTransformer(args.embedding_model_path)

    valid_samples = []
    for item in tqdm(generated_data, desc='Collecting samples', unit='items'):
        original_word = item.get('original_word')
        variant = item.get('variant')
        category = item.get('category')
        if not original_word or not variant or not category:
            continue
        base_categories = split_base_categories(category)
        if not base_categories:
            continue
        valid_samples.append({
            'word_id': item.get('word_id'),
            'categories': base_categories,
            'original_word': original_word,
            'variant': variant
        })

    embedding_cache = {}

    def get_emb(text):
        if text not in embedding_cache:
            embedding_cache[text] = model.encode(text, convert_to_numpy=True)
        return embedding_cache[text]

    category_to_samples = {}
    for sample in valid_samples:
        for category in sample['categories']:
            category_to_samples.setdefault(category, []).append(sample)

    category_centers = {}
    for category, samples in tqdm(category_to_samples.items(), desc='Computing category centers', unit='cat'):
        original_embeddings = [get_emb(s['original_word']) for s in samples]
        if len(original_embeddings) == 0:
            continue
        category_centers[category] = np.mean(np.stack(original_embeddings, axis=0), axis=0)

    similarity_scores = []
    structure_scores = []
    structure_wasserstein = []
    structure_spearman = []
    category_scores = []
    category_pass_rate = []

    processed = 0
    for sample in tqdm(valid_samples, desc='Evaluating variants', unit='items'):
        original_word = sample['original_word']
        variant = sample['variant']
        categories = sample['categories']
        processed += 1
        orig_emb = get_emb(original_word)
        gen_emb = get_emb(variant)

        sim = cosine_sim(orig_emb, gen_emb)
        similarity_scores.append(sim)

        for category in categories:
            if category in category_centers:
                center = category_centers[category]
                orig_dist = l2_distance(orig_emb, center)
                gen_dist = l2_distance(gen_emb, center)
                base = max(orig_dist, 1e-8)
                relative_diff = abs(gen_dist - orig_dist) / base
                category_score = float(np.exp(-relative_diff))
                category_scores.append(category_score)
                category_pass_rate.append(1.0 if category_score >= args.category_pass_threshold else 0.0)

            category_pool = category_to_samples.get(category, [])
            neighbor_candidates = [s for s in category_pool if s is not sample]
            if len(neighbor_candidates) < 2:
                continue
            neighbor_count = min(args.top_k, len(neighbor_candidates))
            scored_neighbors = []
            for neighbor in neighbor_candidates:
                neighbor_orig_emb = get_emb(neighbor['original_word'])
                score = cosine_sim(orig_emb, neighbor_orig_emb)
                scored_neighbors.append((score, neighbor))
            scored_neighbors.sort(key=lambda x: x[0], reverse=True)
            neighbors = [neighbor for _, neighbor in scored_neighbors[:neighbor_count]]

            orig_sims = []
            gen_sims = []
            for neighbor in neighbors:
                neighbor_orig = neighbor['original_word']
                neighbor_gen = neighbor['variant']
                neighbor_orig_emb = get_emb(neighbor_orig)
                neighbor_gen_emb = get_emb(neighbor_gen)
                orig_sims.append(cosine_sim(orig_emb, neighbor_orig_emb))
                gen_sims.append(cosine_sim(gen_emb, neighbor_gen_emb))

            if len(orig_sims) < 2 or len(gen_sims) < 2:
                continue

            orig_dist = normalize_distribution(orig_sims)
            gen_dist = normalize_distribution(gen_sims)
            w_dist = float(wasserstein_distance(orig_dist, gen_dist))
            w_score = max(0.0, 1.0 - w_dist / 0.5)

            corr, _ = spearmanr(orig_sims, gen_sims)
            if np.isnan(corr):
                corr = 0.0
            s_score = (float(corr) + 1.0) / 2.0
            structure_score = w_score * s_score

            structure_wasserstein.append(w_dist)
            structure_spearman.append(float(corr))
            structure_scores.append(structure_score)

    report = {
        'processed_samples': processed,
        'valid_similarity_samples': len(similarity_scores),
        'valid_structure_samples': len(structure_scores),
        'valid_category_samples': len(category_scores),
        'metrics': {
            'cosine_similarity_mean': float(np.mean(similarity_scores)) if similarity_scores else None,
            'structure_score_mean': float(np.mean(structure_scores)) if structure_scores else None,
            'wasserstein_distance_mean': float(np.mean(structure_wasserstein)) if structure_wasserstein else None,
            'spearman_correlation_mean': float(np.mean(structure_spearman)) if structure_spearman else None,
            'category_valid_score_mean': float(np.mean(category_scores)) if category_scores else None,
            'category_pass_rate': float(np.mean(category_pass_rate)) if category_pass_rate else None
        },
        'config': {
            'top_k': args.top_k,
            'category_pass_threshold': args.category_pass_threshold,
            'embedding_model_path': args.embedding_model_path
        }
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated_file', required=True)
    parser.add_argument('--embedding_model_path', default=os.path.join(PROJECT_ROOT, 'base_models/bert-base-uncased'))
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--category_pass_threshold', type=float, default=0.8)
    parser.add_argument('--output_file', default=None)
    args = parser.parse_args()

    evaluate(args)

if __name__ == '__main__':
    main()
