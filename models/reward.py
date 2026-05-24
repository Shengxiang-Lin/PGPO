import json
import pickle
import os
import re
import time
import atexit
import math
import torch
import numpy as np
from scipy.stats import wasserstein_distance, spearmanr
from collections import deque
from typing import Callable, List, Dict, Any
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import cosine_similarity
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


class RewardCalculator:
    def __init__(
        self,
        embeddings_file: str,
        embedding_model_path: str,
        category_statistics_file: str,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        lambda3: float = 1.0,
        use_r_dissimilar: bool = True,
        use_r_structure: bool = True,
        use_r_category_valid: bool = True,
        dissimilar_target: float = 0.7,
        structure_target: float = 0.9,
        category_valid_target: float = 0.9,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        cache_manager=None,
        similarities_data=None,
        fine_details_data=None,
        history_window: int = 100,
    ):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.use_r_dissimilar = use_r_dissimilar
        self.use_r_structure = use_r_structure
        self.use_r_category_valid = use_r_category_valid
        self.dissimilar_target = dissimilar_target
        self.structure_target = structure_target
        self.category_valid_target = category_valid_target
        self.device = device
        self.cache_manager = cache_manager
        self.similarities_data = similarities_data
        self.fine_details_data = fine_details_data
        
        self.base_lambda1 = lambda1 if use_r_dissimilar else 0.0
        self.base_lambda2 = lambda2 if use_r_structure else 0.0
        self.base_lambda3 = lambda3 if use_r_category_valid else 0.0
        
        self.dynamic_lambda1 = self.base_lambda1
        self.dynamic_lambda2 = self.base_lambda2
        self.dynamic_lambda3 = self.base_lambda3
        
        self.history_window = history_window
        self.dissimilar_history = deque(maxlen=history_window)
        self.structure_history = deque(maxlen=history_window)
        self.category_valid_history = deque(maxlen=history_window)
        
        self.current_step = 0
        
        self.reward_ranges = {
            'dissimilar': {'min': 0.0, 'max': 1.0},
            'structure': {'min': 0.0, 'max': 1.0},
            'category_valid': {'min': 0.0, 'max': 1.0},
        }
        
        print(f"Loading pre-computed embeddings from {embeddings_file}...")
        with open(embeddings_file, 'rb') as f:
            data = pickle.load(f)
            self.id_to_word = data['id_to_word']
            self.id_to_embedding = data['id_to_embedding']
        
        print(f"Loading embedding model from {embedding_model_path}...")
        self.embedding_model = SentenceTransformer(embedding_model_path)
        self.embedding_model.to(device)
        
        print(f"Loading category statistics from {category_statistics_file}...")
        with open(category_statistics_file, 'rb') as f:
            self.category_statistics = pickle.load(f)
        print(f"Loaded statistics for {len(self.category_statistics)} categories")
        
        self.category_reference_distances = {}
        for category, stats in self.category_statistics.items():
            if 'reference_distance' in stats:
                self.category_reference_distances[category] = stats['reference_distance']
            else:
                self.category_reference_distances[category] = 384.0
        
        print(
            "Reward switches: "
            f"dissimilar={self.use_r_dissimilar}, "
            f"structure={self.use_r_structure}, "
            f"category_valid={self.use_r_category_valid}"
        )
        print("Reward calculator initialized successfully!")
    
    def get_embedding(self, text: str) -> torch.Tensor:
        embedding = self.embedding_model.encode([text], convert_to_tensor=True, show_progress_bar=False)
        return embedding[0].to(self.device)
    
    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        return cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    
    def normalize_reward(self, reward: float, reward_type: str) -> float:
        if reward_type not in self.reward_ranges:
            return reward
        
        r_min = self.reward_ranges[reward_type]['min']
        r_max = self.reward_ranges[reward_type]['max']
        epsilon = 1e-10
        
        if r_max - r_min < epsilon:
            return 0.0
        normalized = (reward - r_min) / (r_max - r_min)
        
        return max(0.0, min(1.0, normalized))
    
    def update_history(self, r_dissimilar: float, r_structure: float, r_category_valid: float):
        self.dissimilar_history.append(r_dissimilar)
        self.structure_history.append(r_structure)
        self.category_valid_history.append(r_category_valid)
    
    def compute_statistics(self, history: deque) -> Dict[str, float]:
        if len(history) == 0:
            return {'avg': 0.0, 'std': 0.0, 'var': 0.0}
        
        history_list = list(history)
        avg = np.mean(history_list)
        var = np.var(history_list) if len(history_list) > 1 else 0.0
        std = np.sqrt(var)
        
        return {'avg': avg, 'std': std, 'var': var}
    
    def update_dynamic_lambda(self):
        dissimilar_stats = self.compute_statistics(self.dissimilar_history)
        structure_stats = self.compute_statistics(self.structure_history)
        category_valid_stats = self.compute_statistics(self.category_valid_history)
        
        dissimilar_target = self.dissimilar_target
        structure_target = self.structure_target
        category_valid_target = self.category_valid_target
        
        epsilon = 1e-10
        
        if self.use_r_dissimilar and len(self.dissimilar_history) > 0:
            gap_dissimilar = dissimilar_stats['avg'] - dissimilar_target
            sigmoid_gap = 1.0 / (1.0 + math.exp(-gap_dissimilar * 10))
            adjustment_factor1 = 2.0 - sigmoid_gap 
            if dissimilar_stats['var'] > 0:
                norm_var_dissimilar = min(dissimilar_stats['var'], 0.2) / 0.2
                adjustment_factor1 += 0.3 * norm_var_dissimilar 
        
        adjustment_factor2 = 1.0  
        if len(self.structure_history) > 0:
            gap_structure = structure_stats['avg'] - structure_target
            
            sigmoid_gap = 1.0 / (1.0 + math.exp(-gap_structure * 100))
            adjustment_factor2 = 2.0 - sigmoid_gap 
            if structure_stats['var'] > 0:
                norm_var_structure = min(structure_stats['var'], 0.2) / 0.2
                adjustment_factor2 += 0.3 * norm_var_structure 
        if self.use_r_category_valid and len(self.category_valid_history) > 0:
            gap_category = category_valid_stats['avg'] - category_valid_target
            
            sigmoid_gap = 1.0 / (1.0 + math.exp(-gap_category * 10))
            adjustment_factor3 = 2.0 - sigmoid_gap  
            if category_valid_stats['var'] > 0:
                norm_var_category = min(category_valid_stats['var'], 0.2) / 0.2
                adjustment_factor3 += 0.3 * norm_var_category
        if self.use_r_dissimilar and len(self.dissimilar_history) > 0:
            self.dynamic_lambda1 = self.base_lambda1 * adjustment_factor1
        if self.use_r_structure and len(self.structure_history) > 0:
            self.dynamic_lambda2 = self.base_lambda2 * adjustment_factor2
        if self.use_r_category_valid and len(self.category_valid_history) > 0:
            self.dynamic_lambda3 = self.base_lambda3 * adjustment_factor3
        
        total_base = 0.0
        if self.use_r_dissimilar and len(self.dissimilar_history) > 0:
            total_base += self.base_lambda1
        if self.use_r_structure and len(self.structure_history) > 0:
            total_base += self.base_lambda2
        if self.use_r_category_valid and len(self.category_valid_history) > 0:
            total_base += self.base_lambda3
        active_weights = []
        if self.use_r_dissimilar and len(self.dissimilar_history) > 0:
            active_weights.append(self.dynamic_lambda1)
        if self.use_r_structure and len(self.structure_history) > 0:
            active_weights.append(self.dynamic_lambda2)
        if self.use_r_category_valid and len(self.category_valid_history) > 0:
            active_weights.append(self.dynamic_lambda3)
        
        total_dynamic = sum(active_weights)
        if total_dynamic > 0:
            scale_factor = total_base / total_dynamic
            if self.use_r_dissimilar and len(self.dissimilar_history) > 0:
                self.dynamic_lambda1 *= scale_factor
            if self.use_r_structure and len(self.structure_history) > 0:
                self.dynamic_lambda2 *= scale_factor
            if self.use_r_category_valid and len(self.category_valid_history) > 0:
                self.dynamic_lambda3 *= scale_factor
        
        if self.use_r_dissimilar and self.base_lambda1 > 0:
            self.dynamic_lambda1 = max(self.dynamic_lambda1, 0.1)
        else:
            self.dynamic_lambda1 = 0.0
        if self.use_r_structure and self.base_lambda2 > 0:
            self.dynamic_lambda2 = max(self.dynamic_lambda2, 0.1)
        else:
            self.dynamic_lambda2 = 0.0
        if self.use_r_category_valid and self.base_lambda3 > 0:
            self.dynamic_lambda3 = max(self.dynamic_lambda3, 0.1)
        else:
            self.dynamic_lambda3 = 0.0
        
        if self.current_step % 100 == 0:
            print(f"[Step {self.current_step}] Dynamic Lambdas:")
            print(f"  Lambda1 (Dissimilar): {self.dynamic_lambda1:.4f} (base: {self.base_lambda1:.4f})")
            print(f"  Lambda2 (Structure): {self.dynamic_lambda2:.4f} (base: {self.base_lambda2:.4f})")
            print(f"  Lambda3 (Category Valid): {self.dynamic_lambda3:.4f} (base: {self.base_lambda3:.4f})")
            print(f"  Stats - Dissimilar: avg={dissimilar_stats['avg']:.4f}, std={dissimilar_stats['std']:.4f}, target={self.dissimilar_target:.4f}")
            print(f"  Stats - Structure: avg={structure_stats['avg']:.4f}, std={structure_stats['std']:.4f}, target={self.structure_target:.4f}")
            print(f"  Stats - Category Valid: avg={category_valid_stats['avg']:.4f}, std={category_valid_stats['std']:.4f}, target={self.category_valid_target:.4f}")
            print(f"  History lengths - Dissimilar: {len(self.dissimilar_history)}, Structure: {len(self.structure_history)}, Category Valid: {len(self.category_valid_history)}")
    
    def compute_r_dissimilar(
        self,
        original_words: List[str],
        generated_words: List[str],
    ) -> float:
        if len(original_words) != len(generated_words):
            return 0.0
        
        similarities = []
        for orig_word, gen_word in zip(original_words, generated_words):
            try:
                orig_emb = self.get_embedding(orig_word)
                gen_emb = self.get_embedding(gen_word)
                sim = self.compute_similarity(orig_emb, gen_emb)
                similarities.append(sim)
            except:
                similarities.append(1.0)
        
        if not similarities:
            return 0.0
        
        avg_similarity = sum(similarities) / len(similarities)
        reward = 1.0 - avg_similarity
        
        return reward
    
    def compute_r_structure(
        self,
        original_ids: List[int],
        generated_words: List[str],
        original_similarities_data: Dict,
        prompt: str = None,
    ) -> float:
        if len(generated_words) < 1:
            return 0.0
        
        try:
            context_originals = []
            context_generateds = []
            
            if prompt:
                lines = prompt.split('\n')
                for line in lines:
                    if 'original:' in line and 'output:' in line:
                        try:
                            parts = line.split(',')
                            original_part = parts[0].strip()
                            output_part = parts[1].strip()
                            
                            original_word = original_part.replace('original:', '').strip()
                            variant_word = output_part.replace('output:', '').strip()
                            
                            if original_word and variant_word:
                                context_originals.append(original_word)
                                context_generateds.append(variant_word)
                        except:
                            continue
            
            if len(context_originals) == 0:
                return 0.0
            
            current_word_id = original_ids[0] if original_ids else 0
            current_word = self.id_to_word.get(current_word_id, "")
            current_generated = generated_words[0] if generated_words else ""
            
            if len(context_originals) < 1 or len(context_generateds) < 1:
                return 0.0
            
            current_orig_emb = self.get_embedding(current_word)
            orig_similarities = []
            for word in context_originals:
                emb = self.get_embedding(word)
                sim = self.compute_similarity(current_orig_emb, emb)
                orig_similarities.append(sim)
            
            current_gen_emb = self.get_embedding(current_generated)
            gen_similarities = []
            for word in context_generateds:
                emb = self.get_embedding(word)
                sim = self.compute_similarity(current_gen_emb, emb)
                gen_similarities.append(sim)
            
            epsilon = 1e-10
            orig_sum = sum(orig_similarities) + epsilon
            gen_sum = sum(gen_similarities) + epsilon
            
            orig_norm = [x / orig_sum for x in orig_similarities]
            gen_norm = [x / gen_sum for x in gen_similarities]
            
            w_dist = wasserstein_distance(orig_norm, gen_norm)
            
            w_norm = max(0.0, 1.0 - w_dist / 0.5) 
            corr, _ = spearmanr(orig_similarities, gen_similarities)
            if np.isnan(corr):
                corr = 0.0
            s_norm = (corr + 1.0) / 2.0 
            reward = w_norm * s_norm
            
            return reward
        
        except Exception as e:
            print(f"Error computing R_structure: {e}")
            return 0.0
    
    def compute_r_category_valid(
        self,
        generated_words: List[str],
        target_category: str,
        original_words: List[str] = None,
    ) -> float:
        if not generated_words:
            return 0.0
        
        try:
            if target_category not in self.category_statistics:
                return 0.0
            
            category_stats = self.category_statistics[target_category]
            mean = torch.tensor(category_stats['center'], dtype=torch.float32, device=self.device)
            variance = torch.tensor(category_stats['variance'], dtype=torch.float32, device=self.device)
            
            epsilon = 1e-6
            variance = torch.clamp(variance, min=epsilon)
            
            orig_distance = None
            if original_words and len(original_words) > 0:
                total_orig_distance = 0.0
                orig_count = 0
                for word in original_words:
                    try:
                        orig_emb = self.get_embedding(word)
                        diff = orig_emb - mean
                        mahalanobis_sq = torch.sum((diff ** 2) / variance).item()
                        total_orig_distance += mahalanobis_sq
                        orig_count += 1
                    except:
                        continue
                
                if orig_count > 0:
                    orig_distance = total_orig_distance / orig_count
            
            if orig_distance is None:
                orig_distance = self.category_reference_distances.get(target_category, 384.0)
            
            orig_distance = max(orig_distance, 10.0)
            
            total_score = 0.0
            count = 0
            
            for word in generated_words:
                gen_emb = self.get_embedding(word)
                
                diff = gen_emb - mean
                gen_distance = torch.sum((diff ** 2) / variance).item()
                
                distance_diff = abs(gen_distance - orig_distance)
                relative_diff = distance_diff / orig_distance
                
                score = np.exp(-relative_diff)
                
                total_score += score
                count += 1
            
            if count == 0:
                return 0.0
            
            avg_score = total_score / count
            
            return avg_score
        
        except Exception as e:
            print(f"Error computing R_category_valid: {e}")
            return 0.0
    
    def compute_reward(
        self,
        original_ids: List[int],
        original_words: List[str],
        generated_words: List[str],
        target_category: str,
        original_similarities_data: Dict,
        prompt: str = None,
    ) -> Dict[str, float]:
        r_dissimilar = (
            self.compute_r_dissimilar(original_words, generated_words)
            if self.use_r_dissimilar else 0.0
        )
        r_structure = (
            self.compute_r_structure(original_ids, generated_words, original_similarities_data, prompt)
            if self.use_r_structure else 0.0
        )
        r_category_valid = (
            self.compute_r_category_valid(generated_words, target_category, original_words)
            if self.use_r_category_valid else 0.0
        )
        
        norm_r_dissimilar = self.normalize_reward(r_dissimilar, 'dissimilar') if self.use_r_dissimilar else 0.0
        norm_r_structure = self.normalize_reward(r_structure, 'structure') if self.use_r_structure else 0.0
        norm_r_category_valid = self.normalize_reward(r_category_valid, 'category_valid') if self.use_r_category_valid else 0.0
        
        has_context = self.use_r_structure and (r_structure > 0.0)
        
        effective_lambda1 = self.dynamic_lambda1 if self.use_r_dissimilar else 0.0
        effective_lambda2 = self.dynamic_lambda2 if has_context else 0.0
        effective_lambda3 = self.dynamic_lambda3 if self.use_r_category_valid else 0.0
        
        total_reward = (
            effective_lambda1 * norm_r_dissimilar +
            effective_lambda2 * norm_r_structure +
            effective_lambda3 * norm_r_category_valid
        )
        
        if self.use_r_dissimilar:
            self.dissimilar_history.append(r_dissimilar)
        if self.use_r_structure and has_context:
            self.structure_history.append(r_structure)
        if self.use_r_category_valid:
            self.category_valid_history.append(r_category_valid)
        
        self.update_dynamic_lambda()
        
        self.current_step += 1
        
        return {
            'total': total_reward,
            'r_dissimilar': r_dissimilar,
            'r_structure': r_structure,
            'r_category_valid': r_category_valid,
            'norm_r_dissimilar': norm_r_dissimilar,
            'norm_r_structure': norm_r_structure,
            'norm_r_category_valid': norm_r_category_valid,
            'weighted_dissimilar': effective_lambda1 * norm_r_dissimilar,
            'weighted_structure': effective_lambda2 * norm_r_structure,
            'weighted_category_valid': effective_lambda3 * norm_r_category_valid,
            'lambda1': effective_lambda1,
            'lambda2': effective_lambda2,
            'lambda3': effective_lambda3,
            'has_context': has_context,
        }
    
    def extract_generated_words(self, completion: str) -> List[str]:
        try:
            if "</answer>" not in completion:
                return []
            
            answer_end = completion.find("</answer>")
            if answer_end > 0:
                answer_content = completion[:answer_end].strip()
                if answer_content:
                    return [answer_content]
            
            return []
        
        except Exception as e:
            print(f"Error extracting generated words: {e}")
            return []


def create_reward_function(
    embeddings_file: str,
    embedding_model_path: str,
    category_statistics_file: str,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    lambda3: float = 1.0,
    use_r_dissimilar: bool = True,
    use_r_structure: bool = True,
    use_r_category_valid: bool = True,
    dissimilar_target: float = 0.7,
    structure_target: float = 0.9,
    category_valid_target: float = 0.9,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    cache_manager=None,
    similarities_data=None,
    fine_details_data=None,
    num_generations: int = 8,
    history_window: int = 100,
    metrics_file_path: str = None,
    tensorboard_log_dir: str = None,
    enable_tensorboard: bool = False,
) -> Callable[[List[str], List[str], Dict[str, Any]], List[float]]:
    calculator = RewardCalculator(
        embeddings_file=embeddings_file,
        embedding_model_path=embedding_model_path,
        category_statistics_file=category_statistics_file,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
        use_r_dissimilar=use_r_dissimilar,
        use_r_structure=use_r_structure,
        use_r_category_valid=use_r_category_valid,
        dissimilar_target=dissimilar_target,
        structure_target=structure_target,
        category_valid_target=category_valid_target,
        device=device,
        cache_manager=cache_manager,
        similarities_data=similarities_data,
        fine_details_data=fine_details_data,
        history_window=history_window,
    )
    
    log_file_initialized = False
    log_start_time = time.strftime("%Y%m%d_%H%M%S")
    step_idx = 0
    step_summaries = []

    writer = None
    if enable_tensorboard and SummaryWriter is not None and tensorboard_log_dir:
        writer = SummaryWriter(log_dir=tensorboard_log_dir)
        print(f"TensorBoard writer enabled at {tensorboard_log_dir}")
    elif enable_tensorboard and SummaryWriter is None:
        print("Warning: TensorBoard SummaryWriter is unavailable, skip custom scalar logging.")

    if metrics_file_path:
        os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)
        if not os.path.exists(metrics_file_path):
            with open(metrics_file_path, "w", encoding="utf-8") as f:
                f.write("# reward step metrics jsonl\n")

    def _close_writer():
        if metrics_file_path and step_summaries:
            summary = {
                "phase": "train_summary",
                "num_steps": len(step_summaries),
                "mean_of_step_means": float(np.mean([x["reward_mean"] for x in step_summaries])),
                "max_of_step_max": float(np.max([x["reward_max"] for x in step_summaries])),
            }
            with open(metrics_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        if writer is not None:
            writer.close()

    atexit.register(_close_writer)
    
    def reward_function(
            prompts: List[str],
            completions: List[str],
            **kwargs
        ) -> List[float]:
            nonlocal step_idx
            batch_size = len(prompts)
            rewards = [-10.0] * batch_size
            
            anchor_ids_list = kwargs.get('anchor_ids', [])
            anchor_words_list = kwargs.get('anchor_words', [])
            category_list = kwargs.get('category', [])
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            log_dir = os.path.join(project_root, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f'log_{log_start_time}.log')
            
            nonlocal log_file_initialized
            if not log_file_initialized:
                try:
                    with open(log_file, 'w', encoding='utf-8') as f:
                        f.write('=' * 80 + '\n')
                        f.write('DEBUG LOG INITIALIZED (Prompts & Generated Variants)\n')
                        f.write('=' * 80 + '\n')
                        f.write(f'Time: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
                        f.write(f'Num generations: {num_generations}\n')
                        f.write('=' * 80 + '\n\n')
                    log_file_initialized = True
                except Exception as e:
                    pass
            
            try:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[BATCH START] Batch size: {batch_size} (prompts={len(prompts)}, completions={len(completions)})\n")
                    f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write('-' * 80 + '\n')
            except Exception as e:
                pass
            
            prompt_groups = {}
            
            for i in range(batch_size):
                try:
                    prompt = prompts[i] if i < len(prompts) else ""
                    completion = completions[i] if i < len(completions) else ""
                    
                    anchor_ids = anchor_ids_list[i] if i < len(anchor_ids_list) else None
                    anchor_words = anchor_words_list[i] if i < len(anchor_words_list) else None
                    category = category_list[i] if i < len(category_list) else None
                    
                    if not anchor_ids or not anchor_words or not category:
                        continue
                    
                    generated_words = calculator.extract_generated_words(completion)
                    
                    if len(generated_words) == 0:
                        reward = -10.0
                        generated = "(Extraction failed)"
                        reward_details = None
                    else:
                        if len(generated_words) != len(anchor_words):
                            generated_words = generated_words[:len(anchor_words)]
                        
                        reward_details = calculator.compute_reward(
                            original_ids=anchor_ids,
                            original_words=anchor_words,
                            generated_words=generated_words,
                            target_category=category,
                            original_similarities_data={},
                            prompt=prompt,
                        )
                        
                        reward = reward_details['total']
                        
                        if cache_manager and similarities_data and fine_details_data:
                            current_word = anchor_words[0]
                            generated_word = generated_words[0]
                            cache_manager.update_cache(current_word, generated_word, reward)
                        
                        generated = generated_words[0]
                    
                    rewards[i] = reward
                    
                    original_word = anchor_words[0]
                    
                    if prompt not in prompt_groups:
                        prompt_groups[prompt] = {
                            'original_word': original_word,
                            'category': category,
                            'variants': []
                        }
                    
                    variant_info = {
                        'variant': generated,
                        'reward': reward,
                        'reward_details': reward_details,
                        'raw_completion': completion
                    }
                    
                    if generated == "(Extraction failed)":
                        variant_info['extraction_failed'] = True
                    
                    prompt_groups[prompt]['variants'].append(variant_info)
                    
                except Exception as e:
                    try:
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"Error processing sample {i+1}: {e}\n")
                            f.write("-" * 80 + '\n\n')
                    except Exception as log_error:
                        pass
            
            try:
                with open(log_file, 'a', encoding='utf-8') as f:
                    for prompt, data in prompt_groups.items():
                        f.write(f"--- Prompt Group ---\n")
                        f.write(f"Original word: {data['original_word']}\n")
                        f.write(f"Category: {data['category']}\n")
                        f.write(f"Full prompt:\n{prompt}\n\n")
                        f.write(f"Variants and rewards:\n")
                        for idx, variant_info in enumerate(data['variants']):
                            f.write(f"  Variant {idx+1}: {variant_info['variant']} (reward: {variant_info['reward']:.4f})\n")
                            
                            if variant_info.get('reward_details') and not variant_info.get('extraction_failed'):
                                details = variant_info['reward_details']
                                f.write(f"    - Dissimilar: {details.get('r_dissimilar', 0):.4f} (norm: {details.get('norm_r_dissimilar', 0):.4f}, weighted: {details.get('weighted_dissimilar', 0):.4f}, lambda: {details.get('lambda1', 0):.4f})\n")
                                f.write(f"    - Structure: {details.get('r_structure', 0):.4f} (norm: {details.get('norm_r_structure', 0):.4f}, weighted: {details.get('weighted_structure', 0):.4f}, lambda: {details.get('lambda2', 0):.4f})\n")
                                f.write(f"    - Category Valid: {details.get('r_category_valid', 0):.4f} (norm: {details.get('norm_r_category_valid', 0):.4f}, weighted: {details.get('weighted_category_valid', 0):.4f}, lambda: {details.get('lambda3', 0):.4f})\n")
                            
                            if variant_info.get('extraction_failed'):
                                f.write(f"    Raw completion: {variant_info['raw_completion'][:100]}...\n")
                        f.write(f"Cache size: {len(cache_manager.best_variants) if cache_manager else 0}\n")
                        if cache_manager:
                            f.write(f"Cache content: {list(cache_manager.best_variants.items())[:5]}...\n")
                        f.write("-" * 80 + "\n\n")
            except Exception as e:
                pass
            
            try:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[BATCH END]\n")
                    f.write('=' * 80 + '\n\n')
            except Exception as e:
                pass

            numeric_rewards = [float(x) for x in rewards if isinstance(x, (int, float))]
            step_reward_mean = float(np.mean(numeric_rewards)) if numeric_rewards else 0.0
            step_reward_max = float(np.max(numeric_rewards)) if numeric_rewards else 0.0
            all_details = []
            for _, data in prompt_groups.items():
                for variant_info in data.get("variants", []):
                    details = variant_info.get("reward_details")
                    if isinstance(details, dict):
                        all_details.append(details)

            r_dissimilar_vals = [float(d.get("r_dissimilar", 0.0)) for d in all_details]
            r_structure_vals = [float(d.get("r_structure", 0.0)) for d in all_details]
            r_category_valid_vals = [float(d.get("r_category_valid", 0.0)) for d in all_details]
            train_r_dissimilar_mean = float(np.mean(r_dissimilar_vals)) if r_dissimilar_vals else 0.0
            train_r_dissimilar_max = float(np.max(r_dissimilar_vals)) if r_dissimilar_vals else 0.0
            train_r_structure_mean = float(np.mean(r_structure_vals)) if r_structure_vals else 0.0
            train_r_structure_max = float(np.max(r_structure_vals)) if r_structure_vals else 0.0
            train_r_category_valid_mean = float(np.mean(r_category_valid_vals)) if r_category_valid_vals else 0.0
            train_r_category_valid_max = float(np.max(r_category_valid_vals)) if r_category_valid_vals else 0.0

            sample_metrics = []
            for _, data in prompt_groups.items():
                variants = data.get("variants", [])
                total_rewards = [
                    float(v.get("reward", 0.0))
                    for v in variants
                    if isinstance(v.get("reward", None), (int, float))
                ]
                details = [
                    v.get("reward_details")
                    for v in variants
                    if isinstance(v.get("reward_details", None), dict)
                ]
                sample_metrics.append({
                    "original_word": data.get("original_word", ""),
                    "category": data.get("category", ""),
                    "num_variants": len(variants),
                    "avg_total_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
                    "max_total_reward": float(np.max(total_rewards)) if total_rewards else 0.0,
                    "avg_r_dissimilar": float(np.mean([d.get("r_dissimilar", 0.0) for d in details])) if details else 0.0,
                    "avg_r_structure": float(np.mean([d.get("r_structure", 0.0) for d in details])) if details else 0.0,
                    "avg_r_category_valid": float(np.mean([d.get("r_category_valid", 0.0) for d in details])) if details else 0.0,
                    "avg_weighted_dissimilar": float(np.mean([d.get("weighted_dissimilar", 0.0) for d in details])) if details else 0.0,
                    "avg_weighted_structure": float(np.mean([d.get("weighted_structure", 0.0) for d in details])) if details else 0.0,
                    "avg_weighted_category_valid": float(np.mean([d.get("weighted_category_valid", 0.0) for d in details])) if details else 0.0,
                })

            step_record = {
                "phase": "train",
                "step": int(step_idx),
                "reward_mean": step_reward_mean,
                "reward_max": step_reward_max,
                "r_dissimilar_mean": train_r_dissimilar_mean,
                "r_dissimilar_max": train_r_dissimilar_max,
                "r_structure_mean": train_r_structure_mean,
                "r_structure_max": train_r_structure_max,
                "r_category_valid_mean": train_r_category_valid_mean,
                "r_category_valid_max": train_r_category_valid_max,
                "num_prompts": len(prompt_groups),
                "num_completions": batch_size,
                "samples": sample_metrics,
            }
            step_summaries.append({"reward_mean": step_reward_mean, "reward_max": step_reward_max})

            if writer is not None:
                writer.add_scalar("reward/train_step_mean", step_reward_mean, step_idx)
                writer.add_scalar("reward/train_step_max", step_reward_max, step_idx)
                writer.add_scalar("reward/train_r_dissimilar_mean", train_r_dissimilar_mean, step_idx)
                writer.add_scalar("reward/train_r_dissimilar_max", train_r_dissimilar_max, step_idx)
                writer.add_scalar("reward/train_r_structure_mean", train_r_structure_mean, step_idx)
                writer.add_scalar("reward/train_r_structure_max", train_r_structure_max, step_idx)
                writer.add_scalar("reward/train_r_category_valid_mean", train_r_category_valid_mean, step_idx)
                writer.add_scalar("reward/train_r_category_valid_max", train_r_category_valid_max, step_idx)

            if metrics_file_path:
                with open(metrics_file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(step_record, ensure_ascii=False) + "\n")
            
            step_idx += 1
            return rewards
    
    return reward_function
