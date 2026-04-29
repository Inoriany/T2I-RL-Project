"""
Unit Tests for Evaluation Module
================================

Tests for metrics and benchmarks.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import numpy as np
from PIL import Image

from src.evaluation.metrics import (
    BaseMetric,
    CLIPScore,
    VLMScore,
    CompositionScore,
)
from src.evaluation.benchmarks import (
    BaseBenchmark,
    T2ICompBench,
    TIFABench,
    GenEvalBench,
    GenAIBench,
)


# =============================================================================
# Metrics Tests
# =============================================================================

class TestCLIPScore:
    """Tests for CLIPScore metric."""
    
    def test_init(self):
        """Test initialization."""
        metric = CLIPScore(
            model_name="ViT-B-32",
            pretrained="openai",
            device="cpu",
        )
        
        assert metric.model_name == "ViT-B-32"
        assert metric.pretrained == "openai"
        assert metric.device == "cpu"
        assert metric.model is None
        
    @patch('src.evaluation.metrics.open_clip')
    def test_load(self, mock_open_clip):
        """Test model loading."""
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_tokenizer = Mock()
        mock_open_clip.create_model_and_transforms.return_value = (
            mock_model, None, mock_preprocess
        )
        mock_open_clip.get_tokenizer.return_value = mock_tokenizer
        
        metric = CLIPScore(device="cpu")
        metric.load()
        
        assert metric.model is not None
        assert metric.preprocess is not None
        assert metric.tokenizer is not None
        
    @patch('src.evaluation.metrics.open_clip')
    def test_compute(self, mock_open_clip, sample_images, sample_prompts):
        """Test CLIP score computation."""
        # Setup mocks
        mock_model = Mock()
        mock_preprocess = Mock(side_effect=lambda x: torch.randn(3, 224, 224))
        mock_tokenizer = Mock(return_value=torch.zeros(len(sample_prompts), 77, dtype=torch.long))
        
        # Normalized features
        image_features = torch.randn(len(sample_images), 768)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = torch.randn(len(sample_prompts), 768)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        mock_model.encode_image.return_value = image_features
        mock_model.encode_text.return_value = text_features
        
        mock_open_clip.create_model_and_transforms.return_value = (
            mock_model, None, mock_preprocess
        )
        mock_open_clip.get_tokenizer.return_value = mock_tokenizer
        
        metric = CLIPScore(device="cpu")
        result = metric.compute(sample_images, sample_prompts)
        
        assert "clip_score" in result
        assert "clip_scores" in result
        assert isinstance(result["clip_score"], float)
        assert len(result["clip_scores"]) == len(sample_images)


class TestVLMScore:
    """Tests for VLMScore metric."""
    
    def test_init(self):
        """Test initialization."""
        metric = VLMScore(
            api_model="gpt-4-vision-preview",
            device="cpu",
        )
        
        assert metric.api_model == "gpt-4-vision-preview"
        
    def test_build_eval_prompt(self):
        """Test evaluation prompt building."""
        metric = VLMScore()
        prompt = metric._build_eval_prompt(
            "a red apple",
            ["alignment", "quality", "coherence"]
        )
        
        assert "a red apple" in prompt
        assert "alignment" in prompt
        assert "quality" in prompt
        assert "coherence" in prompt
        
    def test_parse_scores_valid_json(self):
        """Test parsing valid JSON response."""
        metric = VLMScore()
        response = '{"alignment": 8, "quality": 7, "coherence": 9}'
        criteria = ["alignment", "quality", "coherence"]
        
        scores = metric._parse_scores(response, criteria)
        
        assert scores["alignment"] == 0.8
        assert scores["quality"] == 0.7
        assert scores["coherence"] == 0.9
        
    def test_parse_scores_invalid(self):
        """Test parsing invalid response."""
        metric = VLMScore()
        response = "Invalid response"
        criteria = ["alignment", "quality"]
        
        scores = metric._parse_scores(response, criteria)
        
        # Should return defaults
        assert scores["alignment"] == 0.5
        assert scores["quality"] == 0.5


class TestCompositionScore:
    """Tests for CompositionScore metric."""
    
    def test_init(self):
        """Test initialization."""
        metric = CompositionScore(device="cpu")
        
        assert metric.device == "cpu"
        
    def test_parse_prompt_colors(self):
        """Test color extraction from prompt."""
        metric = CompositionScore()
        elements = metric._parse_prompt("a red apple and a green banana")
        
        assert len(elements["colors"]) >= 2
        assert ("red", "apple") in elements["colors"]
        assert ("green", "banana") in elements["colors"]
        
    def test_parse_prompt_counts(self):
        """Test count extraction from prompt."""
        metric = CompositionScore()
        elements = metric._parse_prompt("two cats playing with three balls")
        
        assert len(elements["counts"]) >= 2
        
    def test_parse_prompt_relations(self):
        """Test spatial relation extraction from prompt."""
        metric = CompositionScore()
        elements = metric._parse_prompt("a cat on a table")
        
        assert len(elements["relations"]) >= 1
        
    def test_compute(self, sample_images, sample_prompts):
        """Test composition score computation."""
        metric = CompositionScore(device="cpu")
        
        result = metric.compute(sample_images, sample_prompts)
        
        assert "object_presence" in result
        assert "attribute_binding" in result
        assert "counting" in result
        assert "spatial_relations" in result


# =============================================================================
# Benchmarks Tests
# =============================================================================

class TestT2ICompBench:
    """Tests for T2I-CompBench."""
    
    def test_init_default(self):
        """Test default initialization."""
        bench = T2ICompBench()
        
        assert bench.name == "T2I-CompBench"
        assert bench.data_dir is None
        
    def test_init_with_data_dir(self):
        """Test initialization with data directory."""
        bench = T2ICompBench(data_dir="/path/to/data")
        
        assert bench.data_dir == Path("/path/to/data")
        
    def test_get_prompts_default(self):
        """Test getting default prompts."""
        bench = T2ICompBench()
        prompts = bench.get_prompts()
        
        assert isinstance(prompts, dict)
        assert "color" in prompts
        assert "shape" in prompts
        assert "texture" in prompts
        assert "spatial" in prompts
        assert "non_spatial" in prompts
        assert "complex" in prompts
        
    def test_get_all_prompts(self):
        """Test getting all prompts as flat list."""
        bench = T2ICompBench()
        prompts = bench.get_all_prompts()
        
        assert isinstance(prompts, list)
        assert len(prompts) > 0
        assert all(isinstance(p, str) for p in prompts)
        
    def test_prompts_content(self):
        """Test that prompts contain expected content."""
        bench = T2ICompBench()
        prompts = bench.get_prompts()
        
        # Color prompts should contain color words
        for prompt in prompts["color"]:
            prompt_lower = prompt.lower()
            assert any(color in prompt_lower for color in 
                      ["red", "blue", "green", "yellow", "purple", "orange", "pink", "black", "white", "brown"])
            
        # Spatial prompts should contain spatial relations
        for prompt in prompts["spatial"]:
            prompt_lower = prompt.lower()
            assert any(rel in prompt_lower for rel in 
                      ["on", "under", "above", "below", "behind", "left", "right"])


class TestTIFABench:
    """Tests for TIFA Benchmark."""
    
    def test_init(self):
        """Test initialization."""
        bench = TIFABench()
        
        assert bench.name == "TIFA"
        
    def test_get_prompts(self):
        """Test getting prompts."""
        bench = TIFABench()
        prompts = bench.get_prompts()
        
        assert isinstance(prompts, list)
        assert len(prompts) > 0
        
    def test_get_prompts_with_questions(self):
        """Test getting prompts with VQA questions."""
        bench = TIFABench()
        data = bench.get_prompts_with_questions()
        
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Each item should have prompt and questions
        for item in data:
            assert "prompt" in item
            assert "questions" in item
            assert isinstance(item["questions"], list)
            
            for q in item["questions"]:
                assert "question" in q
                assert "expected_answer" in q


class TestGenEvalBench:
    """Tests for GenEval Benchmark."""
    
    def test_init(self):
        """Test initialization."""
        bench = GenEvalBench()
        
        assert bench.name == "GenEval-2"
        
    def test_get_prompts(self):
        """Test getting all prompts."""
        bench = GenEvalBench()
        prompts = bench.get_prompts()
        
        assert isinstance(prompts, list)
        assert len(prompts) > 0
        
    def test_get_prompts_by_category(self):
        """Test getting prompts by category."""
        bench = GenEvalBench()
        prompts = bench.get_prompts_by_category()
        
        assert isinstance(prompts, dict)
        assert "single_object" in prompts
        assert "two_objects" in prompts
        assert "counting" in prompts
        assert "colors" in prompts
        assert "position" in prompts


class TestGenAIBench:
    """Tests for GenAI Benchmark."""
    
    def test_init(self):
        """Test initialization."""
        bench = GenAIBench()
        
        assert bench.name == "GenAI-Bench"
        
    def test_get_prompts(self):
        """Test getting prompts."""
        bench = GenAIBench()
        prompts = bench.get_prompts()
        
        assert isinstance(prompts, list)
        assert len(prompts) > 0
        
    def test_prompts_diversity(self):
        """Test that prompts cover diverse topics."""
        bench = GenAIBench()
        prompts = bench.get_prompts()
        
        all_text = " ".join(prompts).lower()
        
        # Should cover various styles/topics
        assert any(term in all_text for term in ["photograph", "photo", "image"])
        assert any(term in all_text for term in ["painting", "illustration", "art"])


class TestBenchmarkWithCustomData:
    """Tests for loading custom benchmark data."""
    
    def test_t2i_compbench_custom_data(self):
        """Test T2I-CompBench with custom data file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create custom prompts file
            prompts_file = Path(tmpdir) / "prompts.json"
            custom_prompts = {
                "color": ["custom red prompt"],
                "spatial": ["custom spatial prompt"],
            }
            with open(prompts_file, 'w') as f:
                json.dump(custom_prompts, f)
                
            bench = T2ICompBench(data_dir=tmpdir)
            prompts = bench.get_prompts()
            
            assert prompts == custom_prompts
            
    def test_tifa_custom_data(self):
        """Test TIFA with custom data file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create custom data file
            data_file = Path(tmpdir) / "tifa_data.json"
            custom_data = [
                {
                    "prompt": "custom prompt",
                    "questions": [
                        {"question": "Custom question?", "expected_answer": "yes"}
                    ]
                }
            ]
            with open(data_file, 'w') as f:
                json.dump(custom_data, f)
                
            bench = TIFABench(data_dir=tmpdir)
            data = bench.get_prompts_with_questions()
            
            assert data == custom_data


class TestMetricComputeInterface:
    """Tests for metric compute interface."""
    
    def test_all_metrics_have_compute(self):
        """Test that all metrics implement compute method."""
        metrics = [CLIPScore, VLMScore, CompositionScore]
        
        for MetricClass in metrics:
            assert hasattr(MetricClass, 'compute')
            assert callable(getattr(MetricClass, 'compute'))
            
    def test_all_benchmarks_have_get_prompts(self):
        """Test that all benchmarks implement get_prompts method."""
        benchmarks = [T2ICompBench, TIFABench, GenEvalBench, GenAIBench]
        
        for BenchClass in benchmarks:
            assert hasattr(BenchClass, 'get_prompts')
            assert callable(getattr(BenchClass, 'get_prompts'))
            
    def test_all_benchmarks_have_name(self):
        """Test that all benchmarks have name property."""
        benchmarks = [T2ICompBench(), TIFABench(), GenEvalBench(), GenAIBench()]
        
        for bench in benchmarks:
            assert hasattr(bench, 'name')
            assert isinstance(bench.name, str)
