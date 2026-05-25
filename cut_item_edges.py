import argparse
import json
import math
import os
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

DATASET_CONFIGS = {
    'ml-1m': {
        'input_file': './data/ml-1m/handled/item_edges.json',
        'output_file': './data/ml-1m/handled/cleaned_item_edges.json'
    },
    'amazon-book': {
        'input_file': './data/amazon-book/handled/item_edges.json',
        'output_file': './data/amazon-book/handled/cleaned_item_edges.json'
    }
}

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_global_word_freq(data):
    word_df = defaultdict(set)
    category_words = defaultdict(set)

    for movie_id, movie_data in data.items():
        for category, words in movie_data.items():
            if category == 'item_name' or not isinstance(words, list):
                continue
            for word in words:
                if isinstance(word, str) and len(word.strip()) >= 2:
                    w = word.strip()
                    word_df[w].add(movie_id)
                    category_words[category].add(w)

    word_df = {w: len(items) for w, items in word_df.items()}
    return word_df, category_words

def calculate_idf(word_df, total_movies):
    return {w: math.log(total_movies / (1 + df)) for w, df in word_df.items()}

def calculate_category_scores_and_variance(category_words, idf_scores):
    category_scores = {}
    category_variances = {}
    for cat, words in category_words.items():
        if not words:
            category_scores[cat] = 0
            category_variances[cat] = 0
            continue
        scores = [idf_scores.get(w, 0) for w in words]
        category_scores[cat] = sum(scores) / len(scores) if scores else 0
        category_variances[cat] = np.var(scores) if len(scores) > 1 else 0
    return category_scores, category_variances

def calculate_semantic_similarity(model, category_words_map):
    category_embeddings = {}
    total_categories = len(category_words_map)

    with tqdm(total=total_categories, desc="Computing category semantic embeddings") as pbar:
        for cat, words in category_words_map.items():
            if not words:
                category_embeddings[cat] = np.zeros(model.get_sentence_embedding_dimension())
                pbar.update(1)
                continue
            
            embeddings = []
            for word in words:
                embedding = model.encode(word, convert_to_numpy=True)
                embeddings.append(embedding)
            
            avg_embedding = np.mean(embeddings, axis=0)
            category_embeddings[cat] = avg_embedding
            pbar.update(1)

    return category_embeddings

def get_semantic_similarity(category_embeddings, cat1, cat2):
    emb1 = category_embeddings.get(cat1)
    emb2 = category_embeddings.get(cat2)
    if emb1 is None or emb2 is None:
        return 0.0 
    return cosine_similarity([emb1], [emb2])[0][0]

def iterative_pruning(data, total_movies, model, 
                      target_min_categories=4, 
                      target_max_categories=4,
                      max_words_per_category=2,
                      initial_redundancy_threshold=0.60,
                      variance_pruning_factor=0.5):
    print("\n" + "="*60)
    print("Phase 1: Initialization and Precomputation")
    print("="*60)
    word_df, category_words_initial = calculate_global_word_freq(data)
    idf_scores = calculate_idf(word_df, total_movies)
    category_scores, category_variances = calculate_category_scores_and_variance(category_words_initial, idf_scores)
    current_categories = list(category_scores.keys())
    removed_categories_info = [] 
    all_category_embeddings = calculate_semantic_similarity(model, category_words_initial)
    print("\n" + "="*60)
    print(f"Phase 2: Iterative Category Pruning (Target: {target_min_categories}-{target_max_categories} categories)")
    print("="*60)
    iteration_count = 0
    while len(current_categories) > target_max_categories or \
          (len(current_categories) < target_min_categories and iteration_count < 100):
        iteration_count += 1
        print(f"\n--- Iteration {iteration_count} ---")
        print(f"  Current number of categories: {len(current_categories)}. Target: {target_min_categories}-{target_max_categories}")
        
        if len(current_categories) <= target_max_categories and len(current_categories) >= target_min_categories:
            print("  Target category range reached. Stopping iterative pruning.")
            break 
        removal_candidates = [] 
        if len(current_categories) > target_max_categories and len(current_categories) > 1:
            for i in range(len(current_categories)):
                cat1 = current_categories[i]
                redundancy_score = 0
                most_similar_cat = None
                max_similarity = 0

                for j in range(len(current_categories)):
                    if i == j: continue
                    cat2 = current_categories[j]
                    similarity = get_semantic_similarity(all_category_embeddings, cat1, cat2)
                    if similarity > initial_redundancy_threshold:
                        redundancy_score += (similarity - initial_redundancy_threshold)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_cat = cat2
                
                priority = redundancy_score * 10 - category_scores.get(cat1, 0) * 1 - category_variances.get(cat1, 0) * 0.5 
                
                if redundancy_score > 0: 
                    removal_candidates.append({
                        'priority': priority,
                        'category': cat1,
                        'method': 'semantic',
                        'reason_base': f"Semantically redundant with {most_similar_cat} (cosine={max_similarity:.3f})"
                    })

        if len(current_categories) > target_max_categories:
            current_variances_values = [category_variances.get(cat, 0) for cat in current_categories if cat in category_variances]
            avg_current_variance = np.mean(current_variances_values) if current_variances_values else 0
            variance_threshold = avg_current_variance * variance_pruning_factor

            for cat in current_categories:
                var = category_variances.get(cat, 0)
                if var < variance_threshold and cat not in [c['category'] for c in removal_candidates]:
                    priority = -category_scores.get(cat, 0) - var * 0.5
                    removal_candidates.append({
                        'priority': priority,
                        'category': cat,
                        'method': 'variance',
                        'reason_base': f'Low distinctiveness (variance={var:.3f} < threshold={variance_threshold:.3f})'
                    })

        if len(current_categories) > target_max_categories:
             for cat in current_categories:
                if cat not in [c['category'] for c in removal_candidates]: 
                    score = category_scores.get(cat, 0)
                    priority = -score * 100 
                    removal_candidates.append({
                        'priority': priority,
                        'category': cat,
                        'method': 'catscore',
                        'reason_base': f'Lowest CatScore (score={score:.2f})'
                    })
        
        if removal_candidates and len(current_categories) > target_max_categories:
            removal_candidates.sort(key=lambda x: x['priority'], reverse=True)
            chosen_to_remove = removal_candidates[0]
            
            cat_to_remove = chosen_to_remove['category']
            current_categories.remove(cat_to_remove)
            removed_categories_info.append({
                'category': cat_to_remove,
                'score': category_scores.get(cat_to_remove, 0),
                'method': chosen_to_remove['method'],
                'reason': chosen_to_remove['reason_base']
            })
            print(f"    - Removed category '{cat_to_remove}': {chosen_to_remove['reason_base']}")
        elif len(current_categories) > target_max_categories:
            print("  Warning: No suitable removal candidates found, but category count is still too high. Breaking to avoid deadlock.")
            break
        elif len(current_categories) < target_min_categories:
            if removed_categories_info:
                candidates_to_add = sorted(
                    [item for item in removed_categories_info if item['reason'] != 'Exceeded maximum keep limit'], 
                    key=lambda x: (category_scores.get(x['category'], 0), category_variances.get(x['category'], 0)), 
                    reverse=True
                )
                if candidates_to_add:
                    cat_to_add_info = candidates_to_add.pop(0)
                    current_categories.append(cat_to_add_info['category'])
                    removed_categories_info = [item for item in removed_categories_info if item['category'] != cat_to_add_info['category']]
                    print(f"    - Restored category: '{cat_to_add_info['category']}' (count below target minimum)")
                else:
                    print("  Note: Category count too low, but no suitable categories to restore.")
                    break
            else:
                print("  Note: Category count too low and no removed categories available to restore.")
                break
        else:
            print("  No removal or restoration performed.")
            break 
    while len(current_categories) > target_max_categories:
        current_categories.sort(key=lambda cat: (category_scores.get(cat, 0), category_variances.get(cat, 0)))
        cat_to_remove = current_categories.pop(0)
        removed_categories_info.append({
            'category': cat_to_remove,
            'score': category_scores.get(cat_to_remove, 0),
            'method': 'limit',
            'reason': 'Exceeded maximum keep limit'
        })
        print(f"  Forced removal of extra category: '{cat_to_remove}' (reached max {target_max_categories})")

    while len(current_categories) < target_min_categories:
        print(f"\n  Final restoration: category count below minimum ({len(current_categories)} < {target_min_categories})...")
        candidates_to_add = sorted(
            [item for item in removed_categories_info if item['reason'] != 'Exceeded maximum keep limit'], 
            key=lambda x: (category_scores.get(x['category'], 0), category_variances.get(x['category'], 0)), 
            reverse=True
        )
        if not candidates_to_add:
            print("  No suitable categories to restore.")
            break
        
        cat_to_add_info = candidates_to_add.pop(0)
        current_categories.append(cat_to_add_info['category'])
        removed_categories_info = [item for item in removed_categories_info if item['category'] != cat_to_add_info['category']]
        print(f"    - Finally restored category: '{cat_to_add_info['category']}'")

    selected_categories = current_categories
    print(f"\nCategory pruning completed. Final retained {len(selected_categories)} categories: {selected_categories}")

    print("\n" + "="*60)
    print(f"Phase 3: Item-level Word Pruning (keep up to {max_words_per_category} words per movie per category)")
    print("="*60)

    print("\n" + "="*60)
    print("Phase 4: Generating Final Pruned Data")
    print("="*60)
    pruned_data = {}
    after_total_edges = 0
    
    for movie_id, movie_data in tqdm(data.items(), desc="Generating pruned movie data"):
        pruned_movie = {'item_name': movie_data.get('item_name', '')}
        for cat in selected_categories:
            if cat in movie_data:
                raw_words_for_movie_cat = [w.strip() for w in movie_data[cat] if isinstance(w, str) and len(w.strip()) >= 2]
                
                if raw_words_for_movie_cat:
                    valid_words = [(w, idf_scores.get(w, 0)) for w in set(raw_words_for_movie_cat) if w in idf_scores]
                    
                    if valid_words:
                        sorted_words_by_idf = sorted(valid_words, key=lambda x: x[1], reverse=True)
                        words_to_keep = [word for word, _ in sorted_words_by_idf[:max_words_per_category]]
                        
                        if words_to_keep:
                            pruned_movie[cat] = words_to_keep
                            after_total_edges += len(pruned_movie[cat])
        
        pruned_data[movie_id] = pruned_movie

    after_stats = {
        'total_edges': after_total_edges,
        'avg_edges': after_total_edges / total_movies
    }

    return pruned_data, selected_categories, after_stats, removed_categories_info

def print_comparison(before_stats, after_stats, selected_categories, category_scores_all, category_variances_all, removed_info=None):
    print("\n" + "="*60)
    print("Before/After Pruning Comparison Report")
    print("="*60)
    print(f"Original total categories          : {len(category_scores_all)}")
    print(f"Original total edges (keywords)    : {before_stats['total_edges']:,}")
    print(f"Original avg edges per movie       : {before_stats['avg_edges']:.2f}")
    print(f"Original unique keywords count     : {before_stats['unique_words']:,}")
    print("-" * 40)
    print(f"Retained categories after pruning  : {len(selected_categories)}")
    print(f"Total edges after pruning          : {after_stats['total_edges']:,}")
    print(f"Avg edges per movie after pruning  : {after_stats['avg_edges']:.2f}")
    print(f"Edge reduction ratio               : {(1 - after_stats['total_edges']/before_stats['total_edges'])*100 :.1f}%")
    print("-" * 40)
    print("Retained categories and their CatScore and variance:")
    for cat in selected_categories:
        score = category_scores_all.get(cat, 0)
        variance = category_variances_all.get(cat, 0)
        print(f"  {cat:18} : score={score:.2f}, variance={variance:.3f}")

    if removed_info:
        print("\nRemoved categories and reasons:")
        removed_info_sorted = sorted(removed_info, key=lambda x: x['score'], reverse=True)
        for item in removed_info_sorted:
            reason = item['reason']
            print(f"  {item['category']:18} : {item['score']:.2f} - {reason}")
    else:
        print("\nNo categories were removed or removal info not recorded.")
    print("="*60)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', nargs='?', default=None)
    parser.add_argument('output_file', nargs='?', default=None)
    parser.add_argument('--dataset', choices=list(DATASET_CONFIGS.keys()), default='ml-1m')
    args = parser.parse_args()

    defaults = DATASET_CONFIGS[args.dataset]
    input_file = args.input_file or defaults['input_file']
    output_file = args.output_file or defaults['output_file']

    print("Starting movie knowledge graph pruning pipeline...")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    print("Loading data...")
    data = load_data(input_file)
    total_movies = len(data)

    print("\nComputing original data statistics...")
    total_edges = 0
    unique_words = set()
    word_df_global, _ = calculate_global_word_freq(data) 

    for movie in data.values():
        for cat, words in movie.items():
            if cat == 'item_name' or not isinstance(words, list):
                continue
            total_edges += len(words)
            for w in words:
                if isinstance(w, str) and len(w.strip()) >= 2:
                    unique_words.add(w.strip())

    before_stats = {
        'total_edges': total_edges,
        'avg_edges': total_edges / total_movies,
        'unique_words': len(unique_words)
    }
    print(f"Original total edges: {before_stats['total_edges']:,}")
    print(f"Original unique keywords: {before_stats['unique_words']:,}")

    model_path = './base_models/all-MiniLM-L6-v2'
    print(f"\nLoading semantic model: {model_path}...")
    try:
        model = SentenceTransformer(model_path)
        print("Semantic model loaded successfully.")
    except Exception as e:
        print(f"Error: Failed to load model: {e}")
        print("Please ensure the model path is correct and sentence-transformers is installed.")
        return

    pruned_data, selected_categories, after_stats, removed_info = iterative_pruning(
        data=data,
        total_movies=total_movies,
        model=model,
        target_min_categories=4,
        target_max_categories=4,
        max_words_per_category=2,
        initial_redundancy_threshold=0.60,
        variance_pruning_factor=0.5
    )

    _, category_words_all = calculate_global_word_freq(data) 
    idf_scores_all = calculate_idf(word_df_global, total_movies) 
    category_scores_all, category_variances_all = calculate_category_scores_and_variance(category_words_all, idf_scores_all)

    print_comparison(before_stats, after_stats, selected_categories, category_scores_all, category_variances_all, removed_info)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pruned_data, f, ensure_ascii=False, indent=2)
    print(f"\nPruning completed! Final result saved to: {output_file}")
    print("Pipeline finished.")

if __name__ == "__main__":
    main()