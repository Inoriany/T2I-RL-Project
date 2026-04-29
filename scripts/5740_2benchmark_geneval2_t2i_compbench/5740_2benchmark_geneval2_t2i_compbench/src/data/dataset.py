"""
Dataset Classes
===============

Dataset implementations for T2I-RL training.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import json

from torch.utils.data import Dataset


class T2IDataset(Dataset):
    """
    General dataset for T2I training.
    
    Supports:
    - JSON files with prompt lists
    - JSONL files with prompt-image pairs
    - CSV files
    """
    
    def __init__(
        self,
        data_path: str,
        max_samples: Optional[int] = None,
        prompt_key: str = "prompt",
        max_prompts_per_category: Optional[int] = None,
    ):
        self.data_path = Path(data_path)
        self.max_samples = max_samples
        self.prompt_key = prompt_key
        self.max_prompts_per_category = max_prompts_per_category
        
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from file."""
        suffix = self.data_path.suffix.lower()
        
        if suffix == ".json":
            with open(self.data_path) as f:
                data = json.load(f)
                data = self._apply_per_category_limit(data)
                data = self._normalize_prompt_data(data)
                    
        elif suffix == ".jsonl":
            data = []
            with open(self.data_path) as f:
                for line in f:
                    data.append(json.loads(line.strip()))
                    
        elif suffix == ".csv":
            import csv
            data = []
            with open(self.data_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(dict(row))
                    
        elif suffix == ".txt":
            data = []
            with open(self.data_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append({self.prompt_key: line})
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
            
        if self.max_samples:
            data = data[:self.max_samples]
            
        return data

    def _apply_per_category_limit(self, data: Any) -> Any:
        """Take the first k list entries under each category key in ``prompts`` (nested JSON)."""
        if self.max_prompts_per_category is None:
            return data
        if not isinstance(data, dict) or "prompts" not in data:
            return data
        prompts = data["prompts"]
        if not isinstance(prompts, dict):
            return data
        k = self.max_prompts_per_category
        new_prompts: Dict[str, Any] = {}
        for cat, value in prompts.items():
            if isinstance(value, list):
                new_prompts[cat] = value[:k]
            else:
                new_prompts[cat] = value
        out = dict(data)
        out["prompts"] = new_prompts
        return out

    def _normalize_prompt_data(self, raw: Any) -> List[Dict[str, Any]]:
        """Normalize multiple JSON prompt schemas into list-of-dicts.

        Supported inputs:
        - ["prompt a", "prompt b"]
        - [{"prompt": "..."}, ...]
        - {"prompts": ["...", ...]}
        - {"prompts": {"category_a": ["..."], "category_b": ["..."]}}
        """
        items = self._extract_prompt_items(raw)

        normalized: List[Dict[str, Any]] = []
        for item in items:
            if isinstance(item, str):
                normalized.append({self.prompt_key: item})
            elif isinstance(item, dict):
                if self.prompt_key in item and isinstance(item[self.prompt_key], str):
                    normalized.append(item)
                elif "text" in item and isinstance(item["text"], str):
                    converted = dict(item)
                    converted[self.prompt_key] = converted.pop("text")
                    normalized.append(converted)

        return normalized

    def _extract_prompt_items(self, raw: Any, category: Optional[str] = None) -> List[Any]:
        """Recursively extract prompt-like items from nested JSON structures."""
        if raw is None:
            return []

        if isinstance(raw, str):
            return [{self.prompt_key: raw, "category": category} if category else raw]

        if isinstance(raw, list):
            results: List[Any] = []
            for entry in raw:
                results.extend(self._extract_prompt_items(entry, category=category))
            return results

        if isinstance(raw, dict):
            if "prompts" in raw:
                return self._extract_prompt_items(raw["prompts"], category=category)

            if self.prompt_key in raw and isinstance(raw[self.prompt_key], str):
                item = dict(raw)
                if category and "category" not in item:
                    item["category"] = category
                return [item]

            results = []
            for key, value in raw.items():
                if key in {"metadata", "meta"}:
                    continue
                if isinstance(value, (list, dict)):
                    results.extend(self._extract_prompt_items(value, category=key))
            return results

        return []
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]
        

class PromptDataset(Dataset):
    """
    Simple dataset that holds a list of prompts.
    """
    
    def __init__(self, prompts: List[str]):
        self.prompts = prompts
        
    def __len__(self) -> int:
        return len(self.prompts)
        
    def __getitem__(self, idx: int) -> Dict[str, str]:
        return {"prompt": self.prompts[idx]}
        
    @classmethod
    def from_benchmark(cls, benchmark_name: str) -> "PromptDataset":
        """Create dataset from benchmark prompts."""
        from src.evaluation.benchmarks import (
            T2ICompBench, TIFABench, GenEvalBench
        )
        
        if benchmark_name == "t2i_compbench":
            benchmark = T2ICompBench()
        elif benchmark_name == "tifa":
            benchmark = TIFABench()
        elif benchmark_name == "geneval":
            benchmark = GenEvalBench()
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
            
        prompts = benchmark.get_prompts()
        if isinstance(prompts, dict):
            # Flatten if categorized
            all_prompts = []
            for p in prompts.values():
                all_prompts.extend(p)
            prompts = all_prompts
            
        return cls(prompts)
