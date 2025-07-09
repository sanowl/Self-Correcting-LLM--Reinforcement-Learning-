import logging
import json
import pickle
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from abc import ABC, abstractmethod
import secrets

logger = logging.getLogger(__name__)

@dataclass
class DatasetMetrics:
    """Dataset statistics and metrics."""
    total_samples: int = 0
    task_distribution: Dict[str, int] = None
    avg_sequence_length: float = 0.0
    max_sequence_length: int = 0
    min_sequence_length: int = 0
    vocabulary_size: int = 0
    class_balance: Dict[str, float] = None
    cache_hit_rate: float = 0.0
    processing_time_ms: float = 0.0
    
    def __post_init__(self):
        if self.task_distribution is None:
            self.task_distribution = {}
        if self.class_balance is None:
            self.class_balance = {}

class SmartCache:
    """Intelligent caching system for dataset items."""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.access_counts = defaultdict(int)
        self.lock = threading.RLock()
        
    def _generate_key(self, item: Any) -> str:
        """Generate cache key from item."""
        if isinstance(item, dict):
            return hashlib.md5(json.dumps(item, sort_keys=True).encode()).hexdigest()
        return hashlib.md5(str(item).encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                current_time = time.time()
                if current_time - self.access_times[key] < self.ttl_seconds:
                    self.access_counts[key] += 1
                    self.access_times[key] = current_time
                    return self.cache[key]
                else:
                    # Expired
                    del self.cache[key]
                    del self.access_times[key]
                    del self.access_counts[key]
            return None
    
    def put(self, key: str, value: Any):
        """Put item in cache."""
        with self.lock:
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.access_counts[key] = 1
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.cache:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]
        del self.access_counts[lru_key]
    
    def clear(self):
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': sum(self.access_counts.values()) / max(1, len(self.access_counts)),
                'avg_access_count': np.mean(list(self.access_counts.values())) if self.access_counts else 0
            }

class DataProcessor(ABC):
    """Abstract base class for data processors."""
    
    @abstractmethod
    def process(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single data item."""
        pass
    
    @abstractmethod
    def validate(self, item: Dict[str, Any]) -> bool:
        """Validate a data item."""
        pass

class MathDataProcessor(DataProcessor):
    """Processor for mathematical data."""
    
    def __init__(self, normalize_expressions: bool = True):
        self.normalize_expressions = normalize_expressions
    
    def process(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process math data item."""
        processed = item.copy()
        
        # Normalize mathematical expressions
        if self.normalize_expressions:
            if 'question' in processed:
                processed['question'] = self._normalize_math_expression(processed['question'])
            if 'answer' in processed:
                processed['answer'] = self._normalize_math_expression(processed['answer'])
        
        # Add difficulty level based on content
        processed['difficulty'] = self._estimate_difficulty(processed.get('question', ''))
        
        # Extract mathematical concepts
        processed['concepts'] = self._extract_math_concepts(processed.get('question', ''))
        
        return processed
    
    def validate(self, item: Dict[str, Any]) -> bool:
        """Validate math data item."""
        required_fields = ['question', 'answer']
        return all(field in item and item[field] for field in required_fields)
    
    def _normalize_math_expression(self, expression: str) -> str:
        """Normalize mathematical expressions."""
        # Basic normalization (can be enhanced)
        expression = expression.strip()
        expression = expression.replace('\\cdot', '*')
        expression = expression.replace('\\times', '*')
        expression = expression.replace('\\div', '/')
        return expression
    
    def _estimate_difficulty(self, question: str) -> str:
        """Estimate question difficulty."""
        if any(keyword in question.lower() for keyword in ['integral', 'derivative', 'limit']):
            return 'hard'
        elif any(keyword in question.lower() for keyword in ['equation', 'solve', 'graph']):
            return 'medium'
        else:
            return 'easy'
    
    def _extract_math_concepts(self, question: str) -> List[str]:
        """Extract mathematical concepts from question."""
        concepts = []
        concept_keywords = {
            'algebra': ['equation', 'variable', 'solve', 'linear'],
            'calculus': ['derivative', 'integral', 'limit', 'continuous'],
            'geometry': ['triangle', 'circle', 'angle', 'area', 'volume'],
            'statistics': ['probability', 'mean', 'variance', 'distribution']
        }
        
        question_lower = question.lower()
        for concept, keywords in concept_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                concepts.append(concept)
        
        return concepts

class CodeDataProcessor(DataProcessor):
    """Processor for code data."""
    
    def __init__(self, extract_ast: bool = True):
        self.extract_ast = extract_ast
    
    def process(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process code data item."""
        processed = item.copy()
        
        # Extract code metrics
        if 'code' in processed:
            processed['code_metrics'] = self._extract_code_metrics(processed['code'])
        
        # Detect programming language
        processed['language'] = self._detect_language(processed.get('code', ''))
        
        # Extract complexity
        processed['complexity'] = self._estimate_complexity(processed.get('code', ''))
        
        # Parse function signature
        if 'code' in processed:
            processed['function_signature'] = self._extract_function_signature(processed['code'])
        
        return processed
    
    def validate(self, item: Dict[str, Any]) -> bool:
        """Validate code data item."""
        required_fields = ['text', 'code']
        return all(field in item and item[field] for field in required_fields)
    
    def _extract_code_metrics(self, code: str) -> Dict[str, Any]:
        """Extract basic code metrics."""
        lines = code.split('\n')
        return {
            'lines_of_code': len([line for line in lines if line.strip()]),
            'total_lines': len(lines),
            'avg_line_length': np.mean([len(line) for line in lines]),
            'max_line_length': max([len(line) for line in lines]) if lines else 0,
            'indentation_levels': len(set(len(line) - len(line.lstrip()) for line in lines if line.strip()))
        }
    
    def _detect_language(self, code: str) -> str:
        """Detect programming language."""
        if 'def ' in code and ':' in code:
            return 'python'
        elif 'function' in code and '{' in code:
            return 'javascript'
        elif 'public static' in code:
            return 'java'
        else:
            return 'unknown'
    
    def _estimate_complexity(self, code: str) -> str:
        """Estimate code complexity."""
        complexity_indicators = ['for', 'while', 'if', 'else', 'elif', 'try', 'except']
        count = sum(code.lower().count(indicator) for indicator in complexity_indicators)
        
        if count > 10:
            return 'high'
        elif count > 5:
            return 'medium'
        else:
            return 'low'
    
    def _extract_function_signature(self, code: str) -> Optional[str]:
        """Extract function signature from code."""
        lines = code.split('\n')
        for line in lines:
            if 'def ' in line or 'function ' in line:
                return line.strip()
        return None

class AdvancedDataset(Dataset):
    """
    Advanced dataset with intelligent caching, preprocessing, and multi-modal support.
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        task_type: str = 'MATH',
        cache_size: int = 10000,
        preprocessing: bool = True,
        augmentation: bool = False,
        validation: bool = True,
        num_workers: int = 4,
        seed: int = 42
    ):
        self.raw_data = data
        self.task_type = task_type.upper()
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.validation = validation
        self.num_workers = num_workers
        self.seed = seed
        
        # Initialize components
        self.cache = SmartCache(max_size=cache_size)
        self.processor = self._get_processor()
        self.metrics = DatasetMetrics()
        
        # Set random seed
        secrets.SystemRandom().seed(seed)
        np.random.seed(seed)
        
        # Process data
        self.data = self._process_dataset()
        
        # Compute metrics
        self._compute_metrics()
        
        logger.info(f"Advanced dataset initialized with {len(self.data)} samples")
        logger.info(f"Task: {self.task_type}, Cache size: {cache_size}")
    
    def _get_processor(self) -> DataProcessor:
        """Get appropriate data processor."""
        if self.task_type == 'MATH':
            return MathDataProcessor()
        elif self.task_type == 'CODE':
            return CodeDataProcessor()
        else:
            # Default processor
            class DefaultProcessor(DataProcessor):
                def process(self, item):
                    return item
                def validate(self, item):
                    return True
            return DefaultProcessor()
    
    def _process_dataset(self) -> List[Dict[str, Any]]:
        """Process entire dataset with parallel processing."""
        start_time = time.time()
        
        if self.num_workers > 1:
            processed_data = self._parallel_process()
        else:
            processed_data = self._sequential_process()
        
        processing_time = (time.time() - start_time) * 1000
        self.metrics.processing_time_ms = processing_time
        
        logger.info(f"Dataset processing completed in {processing_time:.2f}ms")
        return processed_data
    
    def _parallel_process(self) -> List[Dict[str, Any]]:
        """Process dataset in parallel."""
        processed_data = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._process_single_item, item, idx) 
                for idx, item in enumerate(self.raw_data)
            ]
            
            for future in futures:
                try:
                    result = future.result(timeout=30)  # 30 second timeout
                    if result is not None:
                        processed_data.append(result)
                except Exception as e:
                    logger.error(f"Error processing item: {e}")
        
        return processed_data
    
    def _sequential_process(self) -> List[Dict[str, Any]]:
        """Process dataset sequentially."""
        processed_data = []
        
        for idx, item in enumerate(self.raw_data):
            try:
                result = self._process_single_item(item, idx)
                if result is not None:
                    processed_data.append(result)
            except Exception as e:
                logger.error(f"Error processing item {idx}: {e}")
        
        return processed_data
    
    def _process_single_item(self, item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        """Process a single data item."""
        try:
            # Validation
            if self.validation and not self.processor.validate(item):
                logger.warning(f"Invalid item at index {idx}, skipping")
                return None
            
            # Add metadata
            processed_item = item.copy()
            processed_item['_index'] = idx
            processed_item['_task_type'] = self.task_type
            
            # Preprocessing
            if self.preprocessing:
                processed_item = self.processor.process(processed_item)
            
            # Data augmentation
            if self.augmentation:
                processed_item = self._augment_item(processed_item)
            
            return processed_item
            
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            return None
    
    def _augment_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Apply data augmentation to item."""
        augmented = item.copy()
        
        if self.task_type == 'MATH':
            # Math-specific augmentation
            if secrets.SystemRandom().random() < 0.1:  # 10% chance
                augmented = self._augment_math_item(augmented)
        elif self.task_type == 'CODE':
            # Code-specific augmentation
            if secrets.SystemRandom().random() < 0.1:  # 10% chance
                augmented = self._augment_code_item(augmented)
        
        return augmented
    
    def _augment_math_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Augment mathematical data."""
        # Simple augmentation: rephrase question
        if 'question' in item:
            question = item['question']
            # Add variation in wording
            variations = [
                f"Solve: {question}",
                f"Find the answer to: {question}",
                f"Calculate: {question}",
                f"Determine: {question}"
            ]
            item['question'] = secrets.choice(variations)
        
        return item
    
    def _augment_code_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Augment code data."""
        # Simple augmentation: add comments
        if 'code' in item:
            code = item['code']
            if '# ' not in code:  # Add comment if none exists
                lines = code.split('\n')
                if lines:
                    lines[0] = f"# Solution\n{lines[0]}"
                    item['code'] = '\n'.join(lines)
        
        return item
    
    def _compute_metrics(self):
        """Compute dataset metrics."""
        if not self.data:
            return
        
        self.metrics.total_samples = len(self.data)
        
        # Sequence length statistics
        if self.task_type == 'MATH':
            lengths = [len(item.get('question', '').split()) for item in self.data]
        else:
            lengths = [len(item.get('text', '').split()) for item in self.data]
        
        if lengths:
            self.metrics.avg_sequence_length = np.mean(lengths)
            self.metrics.max_sequence_length = max(lengths)
            self.metrics.min_sequence_length = min(lengths)
        
        # Task distribution
        task_counts = Counter(item.get('_task_type', 'unknown') for item in self.data)
        self.metrics.task_distribution = dict(task_counts)
        
        # Difficulty distribution (if available)
        if 'difficulty' in self.data[0]:
            difficulty_counts = Counter(item.get('difficulty', 'unknown') for item in self.data)
            total = sum(difficulty_counts.values())
            self.metrics.class_balance = {
                k: v / total for k, v in difficulty_counts.items()
            }
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            # Check cache first
            cache_key = f"item_{idx}"
            cached_item = self.cache.get(cache_key)
            
            if cached_item is not None:
                return cached_item
            
            # Get item from data
            if idx >= len(self.data):
                raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
            
            item = self.data[idx].copy()
            
            # Cache the item
            self.cache.put(cache_key, item)
            
            return item
            
        except Exception as e:
            logger.error(f"Error retrieving item at index {idx}: {e}")
            raise
    
    def get_batch(self, indices: List[int]) -> List[Dict[str, Any]]:
        """Get multiple items efficiently."""
        return [self[idx] for idx in indices]
    
    def filter_by_difficulty(self, difficulty: str) -> 'AdvancedDataset':
        """Filter dataset by difficulty level."""
        filtered_data = [
            item for item in self.data 
            if item.get('difficulty', '').lower() == difficulty.lower()
        ]
        
        return AdvancedDataset(
            data=filtered_data,
            task_type=self.task_type,
            preprocessing=False,  # Already processed
            validation=False
        )
    
    def filter_by_concepts(self, concepts: List[str]) -> 'AdvancedDataset':
        """Filter dataset by mathematical concepts."""
        filtered_data = [
            item for item in self.data 
            if any(concept in item.get('concepts', []) for concept in concepts)
        ]
        
        return AdvancedDataset(
            data=filtered_data,
            task_type=self.task_type,
            preprocessing=False,
            validation=False
        )
    
    def sample_balanced(self, n_samples: int, balance_key: str = 'difficulty') -> 'AdvancedDataset':
        """Sample balanced subset of dataset."""
        if balance_key not in self.data[0]:
            logger.warning(f"Balance key '{balance_key}' not found, using random sampling")
            indices = secrets.SystemRandom().sample(range(len(self.data)), min(n_samples, len(self.data)))
            sampled_data = [self.data[i] for i in indices]
        else:
            # Group by balance key
            groups = defaultdict(list)
            for item in self.data:
                groups[item[balance_key]].append(item)
            
            # Sample equally from each group
            samples_per_group = n_samples // len(groups)
            sampled_data = []
            
            for group_items in groups.values():
                group_sample_size = min(samples_per_group, len(group_items))
                sampled_data.extend(secrets.SystemRandom().sample(group_items, group_sample_size))
        
        return AdvancedDataset(
            data=sampled_data,
            task_type=self.task_type,
            preprocessing=False,
            validation=False
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        stats = asdict(self.metrics)
        stats.update(self.cache.get_stats())
        return stats
    
    def save_cache(self, filepath: str):
        """Save cache to disk."""
        try:
            cache_data = {
                'cache': dict(self.cache.cache),
                'access_times': dict(self.cache.access_times),
                'access_counts': dict(self.cache.access_counts)
            }
            with open(filepath, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Cache saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def load_cache(self, filepath: str):
        """Load cache from disk."""
        try:
            with open(filepath, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.cache.cache = cache_data['cache']
            self.cache.access_times = cache_data['access_times']
            self.cache.access_counts = defaultdict(int, cache_data['access_counts'])
            logger.info(f"Cache loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
    
    def clear_cache(self):
        """Clear dataset cache."""
        self.cache.clear()
        logger.info("Dataset cache cleared")

def create_smart_dataloader(
    dataset: AdvancedDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2
) -> DataLoader:
    """Create optimized DataLoader with smart settings."""
    
    def smart_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Smart collation function."""
        collated = defaultdict(list)
        
        for item in batch:
            for key, value in item.items():
                collated[key].append(value)
        
        return dict(collated)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        collate_fn=smart_collate_fn,
        persistent_workers=num_workers > 0
    )

# Backward compatibility
BaseDataset = AdvancedDataset
