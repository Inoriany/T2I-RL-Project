"""
Unit Tests for Data Module
==========================

Tests for T2IDataset and PromptDataset classes.
"""

import os
import json
import tempfile
from pathlib import Path

import pytest

from src.data.dataset import T2IDataset, PromptDataset


class TestT2IDataset:
    """Tests for T2IDataset class."""
    
    def test_load_json_list_of_strings(self, temp_prompts_dir):
        """Test loading JSON file with list of strings."""
        json_path = Path(temp_prompts_dir) / "prompts.json"
        dataset = T2IDataset(str(json_path))
        
        assert len(dataset) == 3
        assert dataset[0]["prompt"] == "prompt 1"
        assert dataset[1]["prompt"] == "prompt 2"
        assert dataset[2]["prompt"] == "prompt 3"
        
    def test_load_json_dict_format(self):
        """Test loading JSON file with dict format (prompts key)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {"prompts": ["prompt A", "prompt B"]}
            json.dump(data, f)
            temp_path = f.name
            
        try:
            dataset = T2IDataset(temp_path)
            assert len(dataset) == 2
            assert dataset[0]["prompt"] == "prompt A"
        finally:
            os.unlink(temp_path)
            
    def test_load_json_list_of_dicts(self):
        """Test loading JSON file with list of dicts."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = [
                {"prompt": "p1", "id": 1},
                {"prompt": "p2", "id": 2},
            ]
            json.dump(data, f)
            temp_path = f.name
            
        try:
            dataset = T2IDataset(temp_path)
            assert len(dataset) == 2
            assert dataset[0]["prompt"] == "p1"
            assert dataset[0]["id"] == 1
        finally:
            os.unlink(temp_path)
            
    def test_load_jsonl(self, temp_jsonl_file):
        """Test loading JSONL file."""
        dataset = T2IDataset(temp_jsonl_file)
        
        assert len(dataset) == 3
        assert dataset[0]["prompt"] == "prompt 1"
        assert dataset[0]["id"] == 0
        
    def test_load_txt(self, temp_txt_file):
        """Test loading TXT file."""
        dataset = T2IDataset(temp_txt_file)
        
        assert len(dataset) == 3
        assert dataset[0]["prompt"] == "prompt 1"
        
    def test_load_csv(self, temp_csv_file):
        """Test loading CSV file."""
        dataset = T2IDataset(temp_csv_file)
        
        assert len(dataset) == 3
        assert dataset[0]["prompt"] == "prompt 1"
        assert dataset[0]["category"] == "color"
        
    def test_max_samples(self, temp_prompts_dir):
        """Test max_samples parameter."""
        json_path = Path(temp_prompts_dir) / "prompts.json"
        dataset = T2IDataset(str(json_path), max_samples=2)
        
        assert len(dataset) == 2
        
    def test_custom_prompt_key(self):
        """Test custom prompt key."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = [{"text": "custom prompt"}]
            json.dump(data, f)
            temp_path = f.name
            
        try:
            dataset = T2IDataset(temp_path, prompt_key="text")
            assert dataset[0]["text"] == "custom prompt"
        finally:
            os.unlink(temp_path)
            
    def test_unsupported_format(self):
        """Test that unsupported format raises error."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_path = f.name
            
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                T2IDataset(temp_path)
        finally:
            os.unlink(temp_path)
            
    def test_getitem(self, temp_jsonl_file):
        """Test __getitem__ method."""
        dataset = T2IDataset(temp_jsonl_file)
        
        item = dataset[1]
        assert isinstance(item, dict)
        assert "prompt" in item
        
    def test_len(self, temp_prompts_dir):
        """Test __len__ method."""
        json_path = Path(temp_prompts_dir) / "prompts.json"
        dataset = T2IDataset(str(json_path))
        
        assert len(dataset) == 3


class TestPromptDataset:
    """Tests for PromptDataset class."""
    
    def test_init(self, sample_prompts):
        """Test initialization with prompt list."""
        dataset = PromptDataset(sample_prompts)
        
        assert len(dataset) == len(sample_prompts)
        
    def test_getitem(self, sample_prompts):
        """Test __getitem__ returns dict with prompt key."""
        dataset = PromptDataset(sample_prompts)
        
        item = dataset[0]
        assert isinstance(item, dict)
        assert "prompt" in item
        assert item["prompt"] == sample_prompts[0]
        
    def test_len(self, sample_prompts):
        """Test __len__ method."""
        dataset = PromptDataset(sample_prompts)
        
        assert len(dataset) == 4
        
    def test_empty_list(self):
        """Test with empty list."""
        dataset = PromptDataset([])
        
        assert len(dataset) == 0
        
    def test_single_prompt(self):
        """Test with single prompt."""
        dataset = PromptDataset(["single prompt"])
        
        assert len(dataset) == 1
        assert dataset[0]["prompt"] == "single prompt"
        
    def test_iteration(self, sample_prompts):
        """Test iteration over dataset."""
        dataset = PromptDataset(sample_prompts)
        
        items = list(dataset)
        assert len(items) == len(sample_prompts)
        for i, item in enumerate(items):
            assert item["prompt"] == sample_prompts[i]


class TestDatasetIntegration:
    """Integration tests for dataset with DataLoader."""
    
    def test_with_dataloader(self, sample_prompts):
        """Test dataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader
        
        dataset = PromptDataset(sample_prompts)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        batches = list(dataloader)
        assert len(batches) == 2
        
        # First batch
        assert len(batches[0]["prompt"]) == 2
        assert batches[0]["prompt"][0] == sample_prompts[0]
        
    def test_shuffle(self, sample_prompts):
        """Test DataLoader shuffle."""
        from torch.utils.data import DataLoader
        import torch
        
        torch.manual_seed(42)
        dataset = PromptDataset(sample_prompts)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # Collect all prompts
        all_prompts = []
        for batch in dataloader:
            all_prompts.extend(batch["prompt"])
            
        # Should contain all prompts (possibly reordered)
        assert set(all_prompts) == set(sample_prompts)
