"""
Unit Tests for Reward Models
============================

Tests for CLIPRewardModel, VLMRewardModel, and CompositeRewardModel.
"""

import json
from unittest.mock import Mock, patch, MagicMock

import pytest
import torch
import numpy as np
from PIL import Image

from src.models.reward_models import (
    RewardOutput,
    RewardModel,
    CLIPRewardModel,
    VLMRewardModel,
    CompositeRewardModel,
)


class TestRewardOutput:
    """Tests for RewardOutput dataclass."""
    
    def test_init_with_rewards_only(self):
        """Test initialization with rewards only."""
        rewards = torch.tensor([0.5, 0.7, 0.9])
        output = RewardOutput(rewards=rewards)
        
        assert torch.equal(output.rewards, rewards)
        assert output.details is None
        
    def test_init_with_details(self):
        """Test initialization with rewards and details."""
        rewards = torch.tensor([0.5, 0.7])
        details = {"type": "clip", "scores": [0.5, 0.7]}
        output = RewardOutput(rewards=rewards, details=details)
        
        assert torch.equal(output.rewards, rewards)
        assert output.details == details


class TestCLIPRewardModel:
    """Tests for CLIPRewardModel class."""
    
    def test_init(self):
        """Test initialization."""
        model = CLIPRewardModel(
            model_name="ViT-B-32",
            pretrained="openai",
            device="cpu",
        )
        
        assert model.model_name == "ViT-B-32"
        assert model.pretrained == "openai"
        assert model.device == "cpu"
        assert model.model is None  # Not loaded yet
        
    def test_to_device(self):
        """Test device movement."""
        model = CLIPRewardModel(device="cpu")
        model.to("cuda")
        
        assert model.device == "cuda"
        
    def test_compute_reward_not_loaded(self, sample_images, sample_prompts):
        """Test that compute_reward raises error when model not loaded."""
        model = CLIPRewardModel(device="cpu")
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.compute_reward(sample_images, sample_prompts)
            
    @patch('src.models.reward_models.open_clip')
    def test_load_model(self, mock_open_clip):
        """Test model loading."""
        # Setup mocks
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_tokenizer = Mock()
        mock_open_clip.create_model_and_transforms.return_value = (
            mock_model, None, mock_preprocess
        )
        mock_open_clip.get_tokenizer.return_value = mock_tokenizer
        
        model = CLIPRewardModel(device="cpu")
        model.load_model()
        
        assert model.model is not None
        assert model.preprocess is not None
        assert model.tokenizer is not None
        mock_model.eval.assert_called_once()
        
    @patch('src.models.reward_models.open_clip')
    def test_compute_reward(self, mock_open_clip, sample_images, sample_prompts):
        """Test reward computation."""
        # Setup mocks
        mock_model = Mock()
        mock_preprocess = Mock(side_effect=lambda x: torch.randn(3, 224, 224))
        mock_tokenizer = Mock(return_value=torch.zeros(len(sample_prompts), 77, dtype=torch.long))
        
        # Mock encode functions to return normalized features
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
        
        model = CLIPRewardModel(device="cpu")
        model.load_model()
        
        output = model.compute_reward(sample_images, sample_prompts)
        
        assert isinstance(output, RewardOutput)
        assert output.rewards.shape == (len(sample_images),)
        # Rewards should be between -1 and 1 (cosine similarity)
        assert (output.rewards >= -1).all() and (output.rewards <= 1).all()
        
    @patch('src.models.reward_models.open_clip')
    def test_compute_reward_with_embeddings(self, mock_open_clip, sample_images, sample_prompts):
        """Test reward computation with embedding return."""
        # Setup mocks
        mock_model = Mock()
        mock_preprocess = Mock(side_effect=lambda x: torch.randn(3, 224, 224))
        mock_tokenizer = Mock(return_value=torch.zeros(len(sample_prompts), 77, dtype=torch.long))
        
        image_features = torch.randn(len(sample_images), 768)
        text_features = torch.randn(len(sample_prompts), 768)
        
        mock_model.encode_image.return_value = image_features
        mock_model.encode_text.return_value = text_features
        
        mock_open_clip.create_model_and_transforms.return_value = (
            mock_model, None, mock_preprocess
        )
        mock_open_clip.get_tokenizer.return_value = mock_tokenizer
        
        model = CLIPRewardModel(device="cpu")
        model.load_model()
        
        output = model.compute_reward(
            sample_images, sample_prompts, return_embeddings=True
        )
        
        assert "image_features" in output.details
        assert "text_features" in output.details


class TestVLMRewardModel:
    """Tests for VLMRewardModel class."""
    
    def test_init_local(self):
        """Test initialization for local model."""
        model = VLMRewardModel(
            model_name_or_path="llava-hf/llava-1.5-7b-hf",
            device="cpu",
            use_api=False,
        )
        
        assert model.model_name_or_path == "llava-hf/llava-1.5-7b-hf"
        assert model.use_api is False
        assert model.model is None
        
    def test_init_api(self):
        """Test initialization for API mode."""
        model = VLMRewardModel(
            use_api=True,
            api_model="gpt-4-vision-preview",
            device="cpu",
        )
        
        assert model.use_api is True
        assert model.api_model == "gpt-4-vision-preview"
        
    def test_get_default_eval_prompt(self):
        """Test default evaluation prompt generation."""
        model = VLMRewardModel(use_api=True, api_model="gpt-4v")
        prompt = model._get_default_eval_prompt("a red apple")
        
        assert "a red apple" in prompt
        assert "Object Presence" in prompt
        assert "Attribute Accuracy" in prompt
        assert "Spatial Relations" in prompt
        
    def test_parse_reward_response_valid_json(self):
        """Test parsing valid JSON response."""
        model = VLMRewardModel(use_api=True, api_model="gpt-4v")
        response = '{"object_score": 8, "attribute_score": 7, "spatial_score": 6, "quality_score": 8, "total_score": 7.25}'
        
        reward, details = model._parse_reward_response(response)
        
        assert reward == pytest.approx(0.725, rel=0.01)
        assert details["total_score"] == 7.25
        
    def test_parse_reward_response_embedded_json(self):
        """Test parsing JSON embedded in text."""
        model = VLMRewardModel(use_api=True, api_model="gpt-4v")
        response = 'Here is my evaluation: {"total_score": 8.5} based on the criteria.'
        
        reward, details = model._parse_reward_response(response)
        
        assert reward == pytest.approx(0.85, rel=0.01)
        
    def test_parse_reward_response_number_fallback(self):
        """Test parsing response with number fallback."""
        model = VLMRewardModel(use_api=True, api_model="gpt-4v")
        response = "The image scores 7.5 out of 10."
        
        reward, details = model._parse_reward_response(response)
        
        assert reward == pytest.approx(0.75, rel=0.01)
        
    def test_parse_reward_response_invalid(self):
        """Test parsing invalid response returns default."""
        model = VLMRewardModel(use_api=True, api_model="gpt-4v")
        response = "I cannot evaluate this image."
        
        reward, details = model._parse_reward_response(response)
        
        assert reward == 0.5  # Default
        assert "parse_error" in details or "raw_response" in details
        
    @patch('src.models.reward_models.VLMRewardModel._call_vlm_api')
    def test_compute_reward_api(self, mock_call_api, sample_images, sample_prompts):
        """Test API-based reward computation."""
        mock_call_api.return_value = '{"total_score": 8}'
        
        model = VLMRewardModel(use_api=True, api_model="gpt-4v", device="cpu")
        output = model.compute_reward(sample_images[:2], sample_prompts[:2])
        
        assert isinstance(output, RewardOutput)
        assert output.rewards.shape == (2,)
        assert output.details["type"] == "api_vlm"


class TestCompositeRewardModel:
    """Tests for CompositeRewardModel class."""
    
    def test_init_equal_weights(self, mock_reward_model):
        """Test initialization with equal weights."""
        models = {
            "clip": mock_reward_model,
            "vlm": mock_reward_model,
        }
        composite = CompositeRewardModel(models, device="cpu")
        
        assert len(composite.reward_models) == 2
        assert composite.weights["clip"] == pytest.approx(0.5)
        assert composite.weights["vlm"] == pytest.approx(0.5)
        
    def test_init_custom_weights(self, mock_reward_model):
        """Test initialization with custom weights."""
        models = {
            "clip": mock_reward_model,
            "vlm": mock_reward_model,
        }
        weights = {"clip": 0.3, "vlm": 0.7}
        composite = CompositeRewardModel(models, weights=weights, device="cpu")
        
        # Weights should be normalized
        assert composite.weights["clip"] == pytest.approx(0.3)
        assert composite.weights["vlm"] == pytest.approx(0.7)
        
    def test_compute_reward(self, sample_images, sample_prompts):
        """Test composite reward computation."""
        # Create mock models with different rewards
        mock1 = Mock()
        mock1.compute_reward.return_value = RewardOutput(
            rewards=torch.tensor([0.8, 0.6, 0.7, 0.9]),
            details={"type": "clip"}
        )
        
        mock2 = Mock()
        mock2.compute_reward.return_value = RewardOutput(
            rewards=torch.tensor([0.6, 0.8, 0.5, 0.7]),
            details={"type": "vlm"}
        )
        
        models = {"clip": mock1, "vlm": mock2}
        weights = {"clip": 0.6, "vlm": 0.4}
        composite = CompositeRewardModel(models, weights=weights, device="cpu")
        
        output = composite.compute_reward(sample_images, sample_prompts)
        
        assert isinstance(output, RewardOutput)
        assert output.rewards.shape == (4,)
        
        # Verify weighted combination
        expected = torch.tensor([0.8, 0.6, 0.7, 0.9]) * 0.6 + torch.tensor([0.6, 0.8, 0.5, 0.7]) * 0.4
        assert torch.allclose(output.rewards, expected, atol=1e-5)
        
    def test_compute_reward_details(self, sample_images, sample_prompts, mock_reward_model):
        """Test that details contain component rewards."""
        models = {"model1": mock_reward_model, "model2": mock_reward_model}
        composite = CompositeRewardModel(models, device="cpu")
        
        output = composite.compute_reward(sample_images, sample_prompts)
        
        assert "component_rewards" in output.details
        assert "weights" in output.details
        assert "model1" in output.details["component_rewards"]
        assert "model2" in output.details["component_rewards"]


class TestRewardModelEdgeCases:
    """Edge case tests for reward models."""
    
    def test_single_image_prompt(self, sample_image, sample_prompt):
        """Test with single image and prompt."""
        mock_model = Mock()
        mock_model.compute_reward.return_value = RewardOutput(
            rewards=torch.tensor([0.8]),
            details={}
        )
        
        output = mock_model.compute_reward([sample_image], [sample_prompt])
        
        assert output.rewards.shape == (1,)
        
    def test_empty_input(self):
        """Test with empty input."""
        model = CLIPRewardModel(device="cpu")
        
        # Should handle empty gracefully or raise appropriate error
        # Depends on implementation
        pass
        
    def test_mismatched_batch_sizes(self, sample_images, sample_prompts):
        """Test behavior with mismatched batch sizes."""
        # This should be handled by the implementation
        # Testing that mocks handle it correctly
        pass


class TestVLMAPIIntegration:
    """Integration tests for VLM API calls (requires API keys)."""
    
    @pytest.mark.api
    @pytest.mark.skipif(
        "OPENAI_API_KEY" not in __import__('os').environ,
        reason="OPENAI_API_KEY not set"
    )
    def test_openai_api_call(self, sample_image, sample_prompt):
        """Test actual OpenAI API call."""
        model = VLMRewardModel(
            use_api=True,
            api_model="gpt-4-vision-preview",
            device="cpu",
        )
        
        output = model.compute_reward([sample_image], [sample_prompt])
        
        assert isinstance(output, RewardOutput)
        assert 0 <= output.rewards[0].item() <= 1
        
    @pytest.mark.api
    @pytest.mark.skipif(
        "ANTHROPIC_API_KEY" not in __import__('os').environ,
        reason="ANTHROPIC_API_KEY not set"
    )
    def test_anthropic_api_call(self, sample_image, sample_prompt):
        """Test actual Anthropic API call."""
        model = VLMRewardModel(
            use_api=True,
            api_model="claude-3-opus-20240229",
            device="cpu",
        )
        
        output = model.compute_reward([sample_image], [sample_prompt])
        
        assert isinstance(output, RewardOutput)
        assert 0 <= output.rewards[0].item() <= 1
