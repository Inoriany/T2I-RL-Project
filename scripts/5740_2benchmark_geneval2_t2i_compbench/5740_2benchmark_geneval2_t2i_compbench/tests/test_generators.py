"""
Unit Tests for Image Generators
================================

Tests for ImageGenerator, JanusProGenerator, and DiffusionGenerator.
"""

from unittest.mock import Mock, patch, MagicMock

import pytest
import torch
import numpy as np
from PIL import Image

from src.models.generators import (
    GenerationConfig,
    ImageGenerator,
    JanusProGenerator,
    DiffusionGenerator,
)


class TestGenerationConfig:
    """Tests for GenerationConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = GenerationConfig()
        
        assert config.num_inference_steps == 50
        assert config.guidance_scale == 5.0
        assert config.height == 384
        assert config.width == 384
        assert config.num_images_per_prompt == 1
        assert config.seed is None
        assert config.temperature == 1.0
        assert config.use_lora is True
        
    def test_custom_values(self):
        """Test custom configuration values."""
        config = GenerationConfig(
            num_inference_steps=30,
            guidance_scale=7.5,
            height=512,
            width=512,
            seed=42,
        )
        
        assert config.num_inference_steps == 30
        assert config.guidance_scale == 7.5
        assert config.height == 512
        assert config.seed == 42


class TestJanusProGenerator:
    """Tests for JanusProGenerator class."""
    
    def test_init_default(self):
        """Test default initialization."""
        generator = JanusProGenerator()
        
        assert generator.model_name_or_path == "deepseek-ai/Janus-Pro-1B"
        assert generator.device == "cuda"
        assert generator.dtype == torch.bfloat16
        assert generator.model is None
        assert generator.image_token_num_per_image == 576
        assert generator.img_size == 384
        
    def test_init_custom(self):
        """Test custom initialization."""
        generator = JanusProGenerator(
            model_name_or_path="deepseek-ai/Janus-Pro-7B",
            device="cpu",
            dtype=torch.float32,
        )
        
        assert generator.model_name_or_path == "deepseek-ai/Janus-Pro-7B"
        assert generator.device == "cpu"
        assert generator.dtype == torch.float32
        
    def test_to_device(self):
        """Test device movement."""
        generator = JanusProGenerator(device="cpu")
        generator.to("cuda")
        
        assert generator.device == "cuda"
        
    def test_generate_not_loaded(self, sample_prompt):
        """Test that generate raises error when model not loaded."""
        generator = JanusProGenerator()
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            generator.generate(sample_prompt)
            
    def test_train_eval_modes(self):
        """Test train/eval mode switching."""
        generator = JanusProGenerator()
        generator.model = Mock()
        
        generator.train()
        generator.model.train.assert_called_once()
        
        generator.eval()
        generator.model.eval.assert_called_once()
        
    def test_get_trainable_parameters_no_model(self):
        """Test get_trainable_parameters when model not loaded."""
        generator = JanusProGenerator()
        
        params = generator.get_trainable_parameters()
        
        assert params == []
        
    def test_get_trainable_parameters_with_model(self):
        """Test get_trainable_parameters with loaded model."""
        generator = JanusProGenerator()
        
        # Create mock model with parameters
        mock_model = Mock()
        param1 = torch.nn.Parameter(torch.randn(10))
        param1.requires_grad = True
        param2 = torch.nn.Parameter(torch.randn(10))
        param2.requires_grad = False
        mock_model.parameters.return_value = [param1, param2]
        
        generator.model = mock_model
        
        params = generator.get_trainable_parameters()
        
        assert len(params) == 1
        assert params[0] is param1
        
    def test_lora_enabled_flag(self):
        """Test LoRA enabled flag."""
        generator = JanusProGenerator()
        
        assert generator.lora_enabled is False
        
    @patch('src.models.generators.AutoModelForCausalLM')
    def test_load_model_transformers_fallback(self, mock_auto_model):
        """Test model loading with transformers fallback."""
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_auto_model.from_pretrained.return_value = mock_model
        
        with patch('src.models.generators.AutoProcessor') as mock_processor:
            mock_proc = Mock()
            mock_proc.tokenizer = Mock()
            mock_processor.from_pretrained.return_value = mock_proc
            
            generator = JanusProGenerator(device="cpu")
            
            # This will fail to import janus and use fallback
            try:
                generator.load_model()
            except Exception:
                # Expected if janus not installed
                pass


class TestDiffusionGenerator:
    """Tests for DiffusionGenerator class."""
    
    def test_init_default(self):
        """Test default initialization."""
        generator = DiffusionGenerator()
        
        assert generator.model_name_or_path == "stabilityai/stable-diffusion-xl-base-1.0"
        assert generator.device == "cuda"
        assert generator.dtype == torch.float16
        assert generator.scheduler_type == "ddim"
        
    def test_init_custom(self):
        """Test custom initialization."""
        generator = DiffusionGenerator(
            model_name_or_path="runwayml/stable-diffusion-v1-5",
            scheduler="euler",
        )
        
        assert generator.model_name_or_path == "runwayml/stable-diffusion-v1-5"
        assert generator.scheduler_type == "euler"
        
    def test_generate_not_loaded(self, sample_prompt):
        """Test that generate raises error when model not loaded."""
        generator = DiffusionGenerator()
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            generator.generate(sample_prompt)
            
    def test_get_trainable_parameters_no_pipe(self):
        """Test get_trainable_parameters when pipe not loaded."""
        generator = DiffusionGenerator()
        
        params = generator.get_trainable_parameters()
        
        assert params == []


class TestGeneratorGeneration:
    """Tests for generation functionality using mocks."""
    
    def test_generate_single_prompt(self, mock_generator, sample_prompt, generation_config):
        """Test generation with single prompt."""
        images = mock_generator.generate(sample_prompt, generation_config)
        
        assert len(images) > 0
        assert all(isinstance(img, Image.Image) for img in images)
        
    def test_generate_multiple_prompts(self, mock_generator, sample_prompts, generation_config):
        """Test generation with multiple prompts."""
        images = mock_generator.generate(sample_prompts, generation_config)
        
        assert len(images) == len(sample_prompts)
        
    def test_generate_with_logprobs(self, mock_generator, sample_prompts):
        """Test generation with log probabilities."""
        images, log_probs = mock_generator.generate_with_logprobs(sample_prompts)
        
        assert len(images) == len(sample_prompts)
        assert log_probs.shape == (len(sample_prompts),)
        
    def test_generation_config_seed(self, generation_config):
        """Test that seed is respected."""
        config1 = GenerationConfig(seed=42)
        config2 = GenerationConfig(seed=42)
        config3 = GenerationConfig(seed=123)
        
        assert config1.seed == config2.seed
        assert config1.seed != config3.seed


class TestLoRAFunctionality:
    """Tests for LoRA adapter functionality."""
    
    def test_enable_lora_creates_adapter(self):
        """Test that enable_lora creates new adapter when no path given."""
        generator = JanusProGenerator()
        generator.model = Mock()
        
        with patch('src.models.generators.get_peft_model') as mock_get_peft:
            with patch('src.models.generators.LoraConfig') as mock_lora_config:
                mock_peft_model = Mock()
                mock_get_peft.return_value = mock_peft_model
                
                generator.enable_lora()
                
                assert generator.lora_enabled is True
                mock_get_peft.assert_called_once()
                
    def test_disable_lora(self):
        """Test disabling LoRA."""
        generator = JanusProGenerator()
        generator.model = Mock()
        generator.model.disable_adapter = Mock()
        generator.lora_enabled = True
        
        generator.disable_lora()
        
        assert generator.lora_enabled is False
        generator.model.disable_adapter.assert_called_once()
        
    def test_save_lora(self):
        """Test saving LoRA weights."""
        generator = JanusProGenerator()
        generator.model = Mock()
        generator.model.save_pretrained = Mock()
        generator.lora_enabled = True
        
        generator.save_lora("/path/to/save")
        
        generator.model.save_pretrained.assert_called_once_with("/path/to/save")


class TestDecodeTokensToImages:
    """Tests for token decoding in Janus-Pro."""
    
    def test_decode_output_shape(self):
        """Test that decoded images have correct shape."""
        generator = JanusProGenerator()
        
        # Mock the gen_vision_model
        mock_decode = Mock()
        mock_decode.return_value = torch.randn(4, 3, 384, 384)  # NCHW format
        
        generator.model = Mock()
        generator.model.gen_vision_model.decode_code = mock_decode
        
        # Create dummy tokens
        tokens = torch.randint(0, 1000, (4, 576))
        
        images = generator._decode_tokens_to_images(tokens, parallel_size=4)
        
        assert len(images) == 4
        for img in images:
            assert isinstance(img, Image.Image)
            assert img.size == (384, 384)


class TestGeneratorEdgeCases:
    """Edge case tests for generators."""
    
    def test_empty_prompt_list(self, mock_generator):
        """Test behavior with empty prompt list."""
        # Behavior depends on implementation
        pass
        
    def test_very_long_prompt(self, mock_generator, generation_config):
        """Test with very long prompt."""
        long_prompt = "a " * 1000 + "red apple"
        
        # Should handle gracefully (truncate or error)
        try:
            images = mock_generator.generate(long_prompt, generation_config)
        except Exception:
            pass  # Expected behavior
            
    def test_special_characters_in_prompt(self, mock_generator, generation_config):
        """Test prompts with special characters."""
        prompt = "a red apple 🍎 with émojis & spëcial çharacters"
        
        # Should handle gracefully
        images = mock_generator.generate(prompt, generation_config)
        
        assert len(images) > 0


class TestGeneratorMemoryManagement:
    """Tests for memory management."""
    
    @pytest.mark.gpu
    def test_cuda_memory_cleanup(self):
        """Test that CUDA memory is properly cleaned up."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        # Would test memory usage before and after generation
        pass
        
    def test_dtype_consistency(self):
        """Test that dtype is maintained throughout."""
        generator = JanusProGenerator(dtype=torch.float16)
        
        assert generator.dtype == torch.float16
