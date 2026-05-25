import argparse
import json
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

DATASET_CONFIGS = {
    'ml-1m': {
        'edges_file': './data/ml-1m/handled/cleaned_item_edges.json',
        'extracted_dir': './data/ml-1m/handled/extracted',
        'grpo_dataset_dir': './data/ml-1m/handled/extracted/grpo_dataset'
    },
    'amazon-book': {
        'edges_file': './data/amazon-book/handled/cleaned_item_edges.json',
        'extracted_dir': './data/amazon-book/handled/extracted',
        'grpo_dataset_dir': './data/amazon-book/handled/extracted/grpo_dataset'
    }
}

def get_default_paths(dataset='ml-1m'):
    return DATASET_CONFIGS[dataset]


def extract_fine_details(edges_file):
    print(f"Loading edges file: {edges_file}")
    with open(edges_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} items")
    
    fine_details_with_categories = {}
    
    for item_id, item_info in data.items():
        for category, details in item_info.items():
            if category == 'item_name':
                continue
            if isinstance(details, list):
                for detail in details:
                    if detail not in fine_details_with_categories:
                        fine_details_with_categories[detail] = set()
                    fine_details_with_categories[detail].add(category)
    
    fine_details_list = sorted(fine_details_with_categories.keys())
    
    fine_details_map = {}
    for i, detail in enumerate(fine_details_list):
        categories = sorted(list(fine_details_with_categories[detail]))
        category_str = ",".join(categories) if len(categories) > 1 else categories[0]
        fine_details_map[str(i+1)] = [detail, category_str]
    
    print(f"Extracted {len(fine_details_list)} unique fine details")
    return fine_details_map


def save_fine_details(fine_details_map, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fine_details_map, f, ensure_ascii=False, indent=2)
    
    print(f"fine_details.json saved to: {output_file}")
    print(f"Total entries: {len(fine_details_map)}")
    
    print("\nSample entries:")
    for i, (key, value) in enumerate(list(fine_details_map.items())[:5]):
        categories = value[1] if isinstance(value[1], str) else ",".join(value[1])
        print(f'  {key}: {value[0]} [{categories}]')


def ensure_model_downloaded(model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model_short_name = model_name.split('/')[-1]
    local_model_path = f'./base_models/{model_short_name}'
    
    if not os.path.exists(local_model_path):
        print(f"Downloading {model_name} to {local_model_path}...")
        model = SentenceTransformer(model_name)
        os.makedirs(local_model_path, exist_ok=True)
        model.save(local_model_path)
        print(f"Model saved to {local_model_path}")
    else:
        print(f"Using local model: {local_model_path}")
    
    return local_model_path


def load_fine_details(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    id_to_word = {}
    id_to_attribute = {}
    for key, value in data.items():
        word_id = int(key)
        word = value[0]
        attribute = value[1]
        id_to_word[word_id] = word
        id_to_attribute[word_id] = attribute
    
    return id_to_word, id_to_attribute


def compute_embeddings(id_to_word, model_path, batch_size=32):
    print(f"\nLoading model from: {model_path}")
    model = SentenceTransformer(model_path)
    
    sorted_ids = sorted(id_to_word.keys())
    words = [id_to_word[word_id] for word_id in sorted_ids]
    
    print(f"Computing embeddings for {len(words)} words...")
    embeddings = model.encode(
        words,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    id_to_embedding = {}
    for i, word_id in enumerate(sorted_ids):
        id_to_embedding[word_id] = embeddings[i]
    
    return id_to_embedding, embeddings


def save_embeddings(id_to_embedding, id_to_word, output_path):
    print(f"\nSaving embeddings to {output_path}")
    
    with open(output_path, 'wb') as f:
        pickle.dump({
            'id_to_word': id_to_word,
            'id_to_embedding': id_to_embedding
        }, f)
    
    print(f"Embeddings saved successfully! Total: {len(id_to_embedding)}")


def find_top_similarities(id_to_embedding, id_to_word, top_k=5, batch_size=1000):
    print("\nComputing similarities...")
    
    sorted_ids = sorted(id_to_embedding.keys())
    embeddings = np.array([id_to_embedding[word_id] for word_id in sorted_ids])
    
    results = {}
    n = len(embeddings)
    
    for i in tqdm(range(0, n, batch_size), desc="Finding similarities"):
        batch_end = min(i + batch_size, n)
        batch_embeddings = embeddings[i:batch_end]
        
        batch_similarity = cosine_similarity(batch_embeddings, embeddings)
        
        for j in range(batch_end - i):
            idx = i + j
            word_id = sorted_ids[idx]
            similarities = batch_similarity[j]
            
            top_similar_indices = np.argsort(similarities)[-top_k-1:-1][::-1]
            top_similar_ids = [sorted_ids[k] for k in top_similar_indices]
            
            top_dissimilar_indices = np.argsort(similarities)[:top_k]
            top_dissimilar_ids = [sorted_ids[k] for k in top_dissimilar_indices]
            
            results[word_id] = {
                'top_similar_ids': top_similar_ids,
                'top_similar_scores': [float(similarities[k]) for k in top_similar_indices],
                'top_dissimilar_ids': top_dissimilar_ids,
                'top_dissimilar_scores': [float(similarities[k]) for k in top_dissimilar_indices]
            }
    
    return results


def save_similarity_results(results, output_path):
    print(f"\nSaving similarity results to {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Similarity results saved successfully!")


def compute_influence_scores(similarity_results, top_k=10):
    influence_scores = {}
    for _, result in similarity_results.items():
        similar_ids = result.get('top_similar_ids', [])[:top_k]
        similar_scores = result.get('top_similar_scores', [])[:top_k]
        for sid, score in zip(similar_ids, similar_scores):
            try:
                sid_int = int(sid)
                score_float = float(score)
            except (TypeError, ValueError):
                continue
            influence_scores[sid_int] = influence_scores.get(sid_int, 0.0) + score_float
    return influence_scores


def select_top_influential_samples(
    influence_scores,
    id_to_word,
    id_to_attribute,
    id_to_embedding,
    top_n=5000,
):
    print(f"\nSelecting top {top_n} influential samples...")
    all_word_ids = sorted(id_to_word.keys())
    ranked_ids = sorted(
        all_word_ids,
        key=lambda wid: (-influence_scores.get(wid, 0.0), wid)
    )
    selected_ids = ranked_ids[:min(top_n, len(ranked_ids))]

    selected_samples = {wid: id_to_word[wid] for wid in selected_ids}
    selected_attributes = {wid: id_to_attribute[wid] for wid in selected_ids}
    selected_embeddings = {wid: id_to_embedding[wid] for wid in selected_ids}

    print(f"Selected {len(selected_samples)} samples by influence")
    return selected_samples, selected_attributes, selected_embeddings, selected_ids


def compute_selected_similarities(selected_embeddings, top_k=10):
    print(f"\nComputing similarities between selected samples...")
    
    sorted_ids = sorted(selected_embeddings.keys())
    embeddings = np.array([selected_embeddings[word_id] for word_id in sorted_ids])
    
    results = {}
    n = len(embeddings)
    
    similarities = cosine_similarity(embeddings, embeddings)
    
    for i in range(n):
        word_id = sorted_ids[i]
        sim_scores = similarities[i]
        
        top_similar_indices = np.argsort(sim_scores)[-top_k-1:-1][::-1]
        top_similar_ids = [sorted_ids[k] for k in top_similar_indices]
        top_similar_scores = [float(sim_scores[k]) for k in top_similar_indices]
        
        top_dissimilar_indices = np.argsort(sim_scores)[:top_k]
        top_dissimilar_ids = [sorted_ids[k] for k in top_dissimilar_indices]
        top_dissimilar_scores = [float(sim_scores[k]) for k in top_dissimilar_indices]
        
        results[word_id] = {
            'top_similar_ids': top_similar_ids,
            'top_similar_scores': top_similar_scores,
            'top_dissimilar_ids': top_dissimilar_ids,
            'top_dissimilar_scores': top_dissimilar_scores
        }
    
    print(f"Similarity computation completed for {len(results)} samples")
    return results


def visualize_influence_distribution(influence_scores, all_word_ids, output_path):
    scores = np.array([float(influence_scores.get(wid, 0.0)) for wid in all_word_ids], dtype=np.float32)
    sorted_scores = np.sort(scores)[::-1]
    ranks = np.arange(1, len(sorted_scores) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.hist(scores, bins=60, color="#4C72B0", alpha=0.85, edgecolor="black")
    plt.title("Influence Score Distribution")
    plt.xlabel("Influence Score")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.2)

    plt.subplot(1, 2, 2)
    plt.plot(ranks, sorted_scores, color="#DD8452", linewidth=1.5)
    plt.title("Influence Rank Curve (Descending)")
    plt.xlabel("Rank")
    plt.ylabel("Influence Score")
    plt.grid(alpha=0.2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()
    print(f"Saved influence visualization to: {output_path}")


def visualize_tsne_selected(id_to_embedding, selected_ids, output_path, random_state=42):
    sorted_ids = sorted(id_to_embedding.keys())
    embeddings = np.array([id_to_embedding[word_id] for word_id in sorted_ids], dtype=np.float32)
    n = embeddings.shape[0]
    selected_set = set(selected_ids)
    labels = np.array([1 if wid in selected_set else 0 for wid in sorted_ids], dtype=np.int32)

    if n < 5:
        print("Skip t-SNE visualization: too few samples.")
        return

    perplexity = min(30, max(5, (n - 1) // 3))
    print(f"Running t-SNE for {n} words (perplexity={perplexity})...")
    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    )
    coords = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    mask_other = labels == 0
    mask_selected = labels == 1

    plt.scatter(
        coords[mask_other, 0],
        coords[mask_other, 1],
        s=8,
        c="#4C78A8",
        alpha=0.25,
        label="Not selected (outside top-N)",
    )
    plt.scatter(
        coords[mask_selected, 0],
        coords[mask_selected, 1],
        s=10,
        c="#E45756",
        alpha=0.7,
        label="Selected (top-N influence)",
    )

    plt.title("t-SNE of All Words (Selected vs Not Selected)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(loc="best")
    plt.grid(alpha=0.15)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()
    print(f"Saved t-SNE visualization to: {output_path}")


def save_grpo_dataset(selected_samples, selected_attributes, similarity_results, output_dir):
    print(f"\nSaving GRPO dataset to {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fine_details_data = {}
    for i, (word_id, word) in enumerate(selected_samples.items(), 1):
        fine_details_data[str(i)] = [word, selected_attributes[word_id]]
    
    fine_details_path = os.path.join(output_dir, 'fine_details.json')
    with open(fine_details_path, 'w', encoding='utf-8') as f:
        json.dump(fine_details_data, f, ensure_ascii=False, indent=2)
    print(f"Saved fine_details.json to: {fine_details_path}")
    
    similarities_data = {}
    for i, (word_id, result) in enumerate(similarity_results.items(), 1):
        new_similar_ids = []
        for sim_id in result['top_similar_ids']:
            for j, (s_id, _) in enumerate(selected_samples.items(), 1):
                if s_id == sim_id:
                    new_similar_ids.append(j)
                    break
        
        new_dissimilar_ids = []
        for dissim_id in result['top_dissimilar_ids']:
            for j, (s_id, _) in enumerate(selected_samples.items(), 1):
                if s_id == dissim_id:
                    new_dissimilar_ids.append(j)
                    break
        
        similarities_data[str(i)] = {
            'top_similar_ids': new_similar_ids,
            'top_similar_scores': result['top_similar_scores'],
            'top_dissimilar_ids': new_dissimilar_ids,
            'top_dissimilar_scores': result['top_dissimilar_scores']
        }
    
    similarities_path = os.path.join(output_dir, 'fine_details_similarities.json')
    with open(similarities_path, 'w', encoding='utf-8') as f:
        json.dump(similarities_data, f, ensure_ascii=False, indent=2)
    print(f"Saved fine_details_similarities.json to: {similarities_path}")
    
    print("GRPO dataset saved successfully!")


def process_dataset(edges_file, extracted_dir, grpo_dataset_dir, model_name, batch_size=512, top_k=10, top_n=5000):
    print("="*60)
    print("Starting dataset processing pipeline")
    print("="*60)
    
    os.makedirs(extracted_dir, exist_ok=True)
    
    fine_details_file = os.path.join(extracted_dir, 'fine_details.json')
    output_embeddings = os.path.join(extracted_dir, 'fine_details_embeddings.pkl')
    output_similarities = os.path.join(extracted_dir, 'fine_details_similarities.json')
    influence_output_file = os.path.join(grpo_dataset_dir, 'influence_scores.json')
    vis_dir = os.path.join(grpo_dataset_dir, 'visualizations')
    influence_vis_file = os.path.join(vis_dir, 'influence_distribution.png')
    tsne_vis_file = os.path.join(vis_dir, 'tsne_selected_vs_other.png')
    
    print(f"\n[Step 1/6] Extracting fine details from edges file...")
    fine_details_map = extract_fine_details(edges_file)
    save_fine_details(fine_details_map, fine_details_file)
    
    print(f"\n[Step 2/6] Ensuring model is available...")
    model_path = ensure_model_downloaded(model_name)
    
    print(f"\n[Step 3/6] Loading fine details and computing embeddings...")
    id_to_word, id_to_attribute = load_fine_details(fine_details_file)
    print(f"Loaded {len(id_to_word)} fine detail words")
    
    id_to_embedding, embeddings = compute_embeddings(
        id_to_word,
        model_path=model_path,
        batch_size=batch_size
    )
    save_embeddings(id_to_embedding, id_to_word, output_embeddings)
    
    print(f"\n[Step 4/6] Computing similarities...")
    similarity_results = find_top_similarities(id_to_embedding, id_to_word, top_k=top_k)
    save_similarity_results(similarity_results, output_similarities)
    
    print(f"\n[Step 5/6] Computing influence scores and selecting top samples...")
    influence_scores = compute_influence_scores(similarity_results, top_k=top_k)
    selected_samples, selected_attributes, selected_embeddings, selected_ids = select_top_influential_samples(
        influence_scores,
        id_to_word,
        id_to_attribute,
        id_to_embedding,
        top_n=top_n,
    )
    influence_payload = [
        {
            "old_id": int(wid),
            "influence": float(influence_scores.get(wid, 0.0)),
        }
        for wid in selected_ids
    ]
    with open(influence_output_file, 'w', encoding='utf-8') as f:
        json.dump(influence_payload, f, ensure_ascii=False, indent=2)
    print(f"Saved influence scores to: {influence_output_file}")

    visualize_influence_distribution(
        influence_scores=influence_scores,
        all_word_ids=sorted(id_to_word.keys()),
        output_path=influence_vis_file,
    )
    visualize_tsne_selected(
        id_to_embedding=id_to_embedding,
        selected_ids=selected_ids,
        output_path=tsne_vis_file,
    )

    print(f"\n[Step 6/6] Recomputing selected similarities and generating GRPO dataset...")
    
    selected_similarity_results = compute_selected_similarities(
        selected_embeddings,
        top_k=top_k
    )
    
    save_grpo_dataset(
        selected_samples,
        selected_attributes,
        selected_similarity_results,
        grpo_dataset_dir
    )
    
    print("\n" + "="*60)
    print("Processing completed successfully!")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  - Fine details: {fine_details_file}")
    print(f"  - Embeddings: {output_embeddings}")
    print(f"  - Similarities: {output_similarities}")
    print(f"  - Influence scores: {influence_output_file}")
    print(f"  - Influence visualization: {influence_vis_file}")
    print(f"  - t-SNE visualization: {tsne_vis_file}")
    print(f"  - GRPO dataset: {grpo_dataset_dir}")
    
    print("\nExample results:")
    sorted_ids = sorted(id_to_word.keys())[:3]
    for word_id in sorted_ids:
        word = id_to_word[word_id]
        result = similarity_results[word_id]
        print(f"\nID {word_id}: '{word}'")
        print(f"  Top {top_k} similar IDs: {result['top_similar_ids']}")
        print(f"  Top {top_k} dissimilar IDs: {result['top_dissimilar_ids']}")


if __name__ == "__main__":
    default_model_name = './base_models/bert-base-uncased'
    default_batch_size = 512
    default_top_k = 10
    default_top_n = 10000

    parser = argparse.ArgumentParser()
    parser.add_argument('edges_file', nargs='?', default=None)
    parser.add_argument('extracted_dir', nargs='?', default=None)
    parser.add_argument('grpo_dataset_dir', nargs='?', default=None)
    parser.add_argument('model_name', nargs='?', default=default_model_name)
    parser.add_argument('batch_size', nargs='?', type=int, default=default_batch_size)
    parser.add_argument('top_k', nargs='?', type=int, default=default_top_k)
    parser.add_argument('top_n', nargs='?', type=int, default=default_top_n)
    parser.add_argument('--dataset', choices=list(DATASET_CONFIGS.keys()), default='ml-1m')
    args = parser.parse_args()

    paths = get_default_paths(args.dataset)
    edges_file = args.edges_file or paths['edges_file']
    extracted_dir = args.extracted_dir or paths['extracted_dir']
    grpo_dataset_dir = args.grpo_dataset_dir or paths['grpo_dataset_dir']
    model_name = args.model_name
    batch_size = args.batch_size
    top_k = args.top_k
    top_n = args.top_n

    process_dataset(
        edges_file=edges_file,
        extracted_dir=extracted_dir,
        grpo_dataset_dir=grpo_dataset_dir,
        model_name=model_name,
        batch_size=batch_size,
        top_k=top_k,
        top_n=top_n
    )