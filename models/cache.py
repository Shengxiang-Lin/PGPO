import json

class CacheManager:
    def __init__(self, per_word_max_variants=8, global_recent_max=None):
        self.per_word_cache = {}
        self.global_recent_words = {}
        self.global_recent_max = global_recent_max
        self.best_variants = {}
        self.best_rewards = {}
    
    def update_cache(self, original_word, generated_word, reward):
        self.per_word_cache[original_word] = (generated_word, reward)
        self.best_variants[original_word] = generated_word
        self.best_rewards[original_word] = reward
        if original_word in self.global_recent_words:
            del self.global_recent_words[original_word]
        self.global_recent_words[original_word] = True
        if self.global_recent_max is not None and len(self.global_recent_words) > self.global_recent_max:
            oldest_word = next(iter(self.global_recent_words))
            del self.global_recent_words[oldest_word]
            if oldest_word in self.per_word_cache:
                del self.per_word_cache[oldest_word]
            if oldest_word in self.best_variants:
                del self.best_variants[oldest_word]
            if oldest_word in self.best_rewards:
                del self.best_rewards[oldest_word]
    
    def get_similar_context(self, original_word_or_id, similarities_data, fine_details_data, top_k=8):
        context = []
        try:
            original_word_id = int(original_word_or_id)
        except (ValueError, TypeError):
            original_word_id = None
            for word_id_str, word_data in fine_details_data.items():
                if word_data[0] == original_word_or_id:
                    original_word_id = int(word_id_str)
                    break
        if original_word_id is not None and str(original_word_id) in similarities_data:
            similar_ids = similarities_data[str(original_word_id)]['top_similar_ids']
            for similar_id in similar_ids:
                if str(similar_id) in fine_details_data:
                    similar_word = fine_details_data[str(similar_id)][0]
                    if similar_word in self.per_word_cache:
                        best_pair = self.per_word_cache[similar_word]
                        context.append({'original': similar_word, 'variant': best_pair[0]})
                    if len(context) >= top_k:
                        break
        return context

    def state_dict(self):
        global_recent_max = None if self.global_recent_max is None else int(self.global_recent_max)
        return {
            "per_word_cache": {
                k: [v[0], float(v[1])] for k, v in self.per_word_cache.items()
            },
            "global_recent_words": list(self.global_recent_words.keys()),
            "global_recent_max": global_recent_max,
            "best_variants": dict(self.best_variants),
            "best_rewards": {k: float(v) for k, v in self.best_rewards.items()},
        }

    def load_state_dict(self, state):
        self.per_word_cache = {}
        for k, v in state.get("per_word_cache", {}).items():
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                self.per_word_cache[k] = (v[0], float(v[1]))

        self.global_recent_words = {}
        for word in state.get("global_recent_words", []):
            self.global_recent_words[word] = True

        loaded_global_recent_max = state.get("global_recent_max", self.global_recent_max)
        self.global_recent_max = None if loaded_global_recent_max is None else int(loaded_global_recent_max)
        self.best_variants = dict(state.get("best_variants", {}))
        self.best_rewards = {k: float(v) for k, v in state.get("best_rewards", {}).items()}

    def save_to_file(self, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, ensure_ascii=False, indent=2)

    def load_from_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.load_state_dict(state)
