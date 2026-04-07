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
    ):
        self.data_path = Path(data_path)
        self.max_samples = max_samples
        self.prompt_key = prompt_key
        
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from file."""
        suffix = self.data_path.suffix.lower()
        
        if suffix == ".json":
            with open(self.data_path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    if isinstance(data[0], str):
                        # List of prompts
                        data = [{self.prompt_key: p} for p in data]
                    # List of dicts
                    pass
                elif isinstance(data, dict):
                    # Dict with prompts key
                    data = [{self.prompt_key: p} for p in data.get("prompts", [])]
                    
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
