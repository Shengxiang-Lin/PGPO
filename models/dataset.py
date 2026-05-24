import json
import random
from typing import List, Dict, Optional, Tuple
import copy
from datasets import Dataset
from .cache import CacheManager


class FineDetailsDataset(Dataset):
    def __init__(
        self,
        fine_details_file: str,
        similarities_file: str,
        cache_manager: Optional[CacheManager] = None,
        num_samples: Optional[int] = None,
        seed: int = 42,
        context_injection_step: int = 1000,
        context_top_k: int = 10,
        prompt_template: Optional[str] = None,
    ):
        self.fine_details_file = fine_details_file
        self.similarities_file = similarities_file
        self.cache_manager = cache_manager
        self.seed = seed
        self.context_injection_step = context_injection_step
        self.context_top_k = context_top_k
        self.current_step = 0
        self.prompt_template = prompt_template
        random.seed(seed)
        self.context_cache = {}
        
        with open(fine_details_file, 'r', encoding='utf-8') as f:
            self.fine_details_data = json.load(f)
        
        with open(similarities_file, 'r', encoding='utf-8') as f:
            self.similarities_data = json.load(f)
        
        self.raw_samples = []
        for word_id_str, word_data in self.fine_details_data.items():
            word_id = int(word_id_str)
            word = word_data[0]
            categories = word_data[1]
            
            if isinstance(categories, str):
                categories_list = [c.strip() for c in categories.split(',')]
            else:
                categories_list = [categories]
            
            category = categories_list[0]
            
            if str(word_id) not in self.similarities_data:
                continue
            
            self.raw_samples.append({
                'word_id': word_id,
                'word': word,
                'category': category,
            })
        
        if num_samples is not None and num_samples < len(self.raw_samples):
            self.raw_samples = random.sample(self.raw_samples, num_samples)
        
        print(f"Total raw samples: {len(self.raw_samples)}")
    
    def set_current_step(self, step: int):
        self.current_step = step
    
    def __len__(self) -> int:
        return len(self.raw_samples)
    
    def __getitem__(self, idx):
        if isinstance(idx, list):
            samples = []
            for i in idx:
                samples.append(self._get_single_sample(i))
            result = {}
            if samples:
                for key in samples[0].keys():
                    result[key] = [sample[key] for sample in samples]
            return result
        return self._get_single_sample(idx)
    
    def _get_single_sample(self, idx: int) -> Dict:
        raw_sample = self.raw_samples[idx]
        word_id = raw_sample['word_id']
        word = raw_sample['word']
        category = raw_sample['category']
        
        cache_key = (idx, self.current_step)
        
        if cache_key not in self.context_cache:
            context_str = ""
            if self.current_step >= self.context_injection_step and self.cache_manager:
                import random
                random.seed(self.seed + idx + self.current_step)
                
                similar_context = self.cache_manager.get_similar_context(
                    word_id, 
                    self.similarities_data, 
                    self.fine_details_data, 
                    top_k=self.context_top_k
                )
                
                if similar_context:
                    context_lines = []
                    for item in similar_context:
                        context_lines.append(f"original: {item['original']}, output: {item['variant']}")
                    context_str = "\n".join(context_lines) + "\n"
            
            self.context_cache[cache_key] = context_str
        else:
            context_str = self.context_cache[cache_key]
        
        if context_str:
            similarity_note = """To ensure that term similarity is preserved before and after rewriting (originally similar terms remain similar in their rewritten forms, originally dissimilar terms remain dissimilar), examples of mappings for variants of related terms are provided below:

"""
        else:
            similarity_note = ""
        
        if not self.prompt_template:
            raise ValueError("prompt_template is required")
        
        prompt = self.prompt_template.format(
            word=word,
            category=category,
            similarity_note=similarity_note,
            context_str=context_str
        )
        
        sample = {
            "prompt": prompt,
            "anchor_ids": [word_id],
            "anchor_words": [word],
            "category": category,
        }
        return sample
    
    def split_train_val(self, val_ratio: float = 0.1) -> Tuple[Dataset, Dataset]:
        """
        分割数据集为训练集和验证集
        """
        random.seed(self.seed)
        shuffled_samples = copy.deepcopy(self.raw_samples)
        random.shuffle(shuffled_samples)
        val_size = 0
        train_size = len(shuffled_samples)
        
        train_samples = shuffled_samples[:train_size]
        val_samples = shuffled_samples[train_size:]
        
        train_dataset = FineDetailsDataset(
            fine_details_file=self.fine_details_file,
            similarities_file=self.similarities_file,
            cache_manager=self.cache_manager,
            num_samples=None,
            seed=self.seed,
            context_injection_step=self.context_injection_step,
            context_top_k=self.context_top_k,
            prompt_template=self.prompt_template,
        )
        train_dataset.raw_samples = train_samples
        train_dataset.current_step = self.current_step

        val_dataset = FineDetailsDataset(
            fine_details_file=self.fine_details_file,
            similarities_file=self.similarities_file,
            cache_manager=self.cache_manager,
            num_samples=None,
            seed=self.seed,
            context_injection_step=self.context_injection_step,
            context_top_k=self.context_top_k,
            prompt_template=self.prompt_template,
        )
        val_dataset.raw_samples = val_samples
        val_dataset.current_step = self.current_step
        
        print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
        
        return train_dataset, val_dataset
