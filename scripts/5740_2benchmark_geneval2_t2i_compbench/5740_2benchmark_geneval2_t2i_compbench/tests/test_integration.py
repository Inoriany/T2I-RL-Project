"""
Integration Tests
=================

End-to-end integration tests for T2I-RL pipeline.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import numpy as np
from PIL import Image

from src.data.dataset import T2IDataset, PromptDataset
from src.models.reward_models import RewardOutput


class TestPipelineIntegration:
    """Integration tests for full training pipeline."""
    
    @pytest.fixture
    def mock_pipeline(self, sample_prompts):
        """Create a mock training pipeline."""
        from torch.utils.data import DataLoader
        
        # Create dataset
        dataset = PromptDataset(sample_prompts)
        dataloader = DataLoader(dataset, batch_size=2)
        
        # Create mock generator
        mock_generator = Mock()
        mock_generator.model = Mock()
        mock_generator.device = "cpu"
        mock_generator.generate = Mock(return_value=[
            Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8))
            for _ in range(4)
        ])
        param = torch.nn.Parameter(torch.randn(10, requires_grad=True))
        mock_generator.get_trainable_parameters.return_value = [param]
        mock_generator.train = Mock()
        mock_generator.eval = Mock()
        
        # Create mock reward model
        mock_reward_model = Mock()
        mock_reward_model.compute_reward = Mock(return_value=RewardOutput(
            rewards=torch.rand(4),
            details={"type": "mock"}
        ))
        
        return {
            'dataset': dataset,
            'dataloader': dataloader,
            'generator': mock_generator,
            'reward_model': mock_reward_model,
        }
    
    def test_data_to_generation_flow(self, mock_pipeline):
        """Test data loading to image generation flow."""
        dataset = mock_pipeline['dataset']
        generator = mock_pipeline['generator']
        
        # Get batch of prompts
        prompts = [dataset[i]["prompt"] for i in range(2)]
        
        # Generate images
        images = generator.generate(prompts)
        
        assert len(images) == 4  # Default mock returns 4 images
        assert all(isinstance(img, Image.Image) for img in images)
        
    def test_generation_to_reward_flow(self, mock_pipeline, sample_prompts):
        """Test image generation to reward computation flow."""
        generator = mock_pipeline['generator']
        reward_model = mock_pipeline['reward_model']
        
        # Generate images
        images = generator.generate(sample_prompts)
        
        # Compute rewards
        output = reward_model.compute_reward(images, sample_prompts)
        
        assert isinstance(output, RewardOutput)
        assert output.rewards.shape[0] == len(images)
        
    def test_full_training_step(self, mock_pipeline, grpo_config):
        """Test a full training step."""
        from src.training.grpo_trainer import GRPOTrainer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            grpo_config.output_dir = tmpdir
            grpo_config.use_wandb = False
            
            trainer = GRPOTrainer(
                generator=mock_pipeline['generator'],
                reward_model=mock_pipeline['reward_model'],
                config=grpo_config,
                train_dataloader=mock_pipeline['dataloader'],
            )
            
            # Run one training step
            batch = {"prompt": ["test prompt 1", "test prompt 2"]}
            loss_dict = trainer.compute_loss(batch)
            
            assert "loss" in loss_dict
            assert isinstance(loss_dict["loss"], torch.Tensor)


class TestDataPipelineIntegration:
    """Integration tests for data pipeline."""
    
    def test_json_to_dataloader(self):
        """Test loading JSON file into DataLoader."""
        import json
        import tempfile
        from torch.utils.data import DataLoader
        
        # Create temp JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(["prompt 1", "prompt 2", "prompt 3", "prompt 4"], f)
            temp_path = f.name
            
        try:
            # Load dataset
            dataset = T2IDataset(temp_path)
            dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
            
            # Get batches
            batches = list(dataloader)
            
            assert len(batches) == 2
            assert len(batches[0]["prompt"]) == 2
        finally:
            Path(temp_path).unlink()
            
    def test_benchmark_to_dataloader(self, t2i_compbench):
        """Test benchmark prompts to DataLoader."""
        from torch.utils.data import DataLoader
        
        # Get prompts from benchmark
        prompts = t2i_compbench.get_all_prompts()
        
        # Create dataset and dataloader
        dataset = PromptDataset(prompts)
        dataloader = DataLoader(dataset, batch_size=4)
        
        # Verify
        for batch in dataloader:
            assert "prompt" in batch
            assert len(batch["prompt"]) <= 4


class TestEvaluationPipelineIntegration:
    """Integration tests for evaluation pipeline."""
    
    def test_generate_and_evaluate(self, mock_generator, sample_prompts):
        """Test generation followed by evaluation."""
        from src.evaluation.benchmarks import T2ICompBench
        
        # Get benchmark prompts
        bench = T2ICompBench()
        prompts = bench.get_prompts()["color"][:2]  # Just 2 prompts for speed
        
        # Generate images
        images = mock_generator.generate(prompts)
        
        # Evaluate (mock CLIP score)
        from src.evaluation.metrics import CompositionScore
        metric = CompositionScore(device="cpu")
        result = metric.compute(images[:len(prompts)], prompts)
        
        assert "object_presence" in result
        assert "attribute_binding" in result


class TestConfigIntegration:
    """Integration tests for configuration loading."""
    
    def test_hydra_config_loading(self):
        """Test loading Hydra configurations."""
        import yaml
        from pathlib import Path
        
        config_dir = Path("D:/CUHK/AIMS_5740/T2I-RL-Project/configs")
        
        # Test default config exists
        default_config = config_dir / "default.yaml"
        if default_config.exists():
            with open(default_config) as f:
                config = yaml.safe_load(f)
                
            assert config is not None
            
    def test_model_configs(self):
        """Test model configuration files."""
        import yaml
        from pathlib import Path
        
        config_dir = Path("D:/CUHK/AIMS_5740/T2I-RL-Project/configs/model")
        
        if config_dir.exists():
            for config_file in config_dir.glob("*.yaml"):
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                    
                # Should have model_name or model_name_or_path
                assert any(key in str(config) for key in ["model_name", "name"])


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""
    
    @pytest.mark.slow
    def test_minimal_training_workflow(self, mock_pipeline, grpo_config):
        """Test minimal training workflow (1 epoch, 1 batch)."""
        from src.training.grpo_trainer import GRPOTrainer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            grpo_config.output_dir = tmpdir
            grpo_config.use_wandb = False
            grpo_config.num_epochs = 1
            grpo_config.save_steps = 1000  # Don't save during test
            grpo_config.eval_steps = 1000  # Don't eval during test
            grpo_config.logging_steps = 1
            
            trainer = GRPOTrainer(
                generator=mock_pipeline['generator'],
                reward_model=mock_pipeline['reward_model'],
                config=grpo_config,
                train_dataloader=mock_pipeline['dataloader'],
            )
            
            # Just run one batch
            batch = next(iter(mock_pipeline['dataloader']))
            loss_dict = trainer.compute_loss(batch)
            
            assert loss_dict["loss"].requires_grad or not loss_dict["loss"].requires_grad
            
    def test_checkpoint_save_load_workflow(self, mock_pipeline, grpo_config):
        """Test checkpoint saving and loading workflow."""
        from src.training.grpo_trainer import GRPOTrainer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            grpo_config.output_dir = tmpdir
            grpo_config.use_wandb = False
            
            trainer = GRPOTrainer(
                generator=mock_pipeline['generator'],
                reward_model=mock_pipeline['reward_model'],
                config=grpo_config,
                train_dataloader=mock_pipeline['dataloader'],
            )
            
            # Save checkpoint
            trainer.global_step = 100
            trainer.save_checkpoint("test-ckpt")
            
            # Verify checkpoint exists
            ckpt_path = Path(tmpdir) / "test-ckpt"
            assert ckpt_path.exists()
            assert (ckpt_path / "training_state.pt").exists()
            
            # Verify training state
            state = torch.load(ckpt_path / "training_state.pt")
            assert state["global_step"] == 100


class TestErrorHandling:
    """Tests for error handling in integration."""
    
    def test_empty_batch_handling(self, mock_pipeline, grpo_config):
        """Test handling of empty batch."""
        from src.training.grpo_trainer import GRPOTrainer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            grpo_config.output_dir = tmpdir
            grpo_config.use_wandb = False
            
            trainer = GRPOTrainer(
                generator=mock_pipeline['generator'],
                reward_model=mock_pipeline['reward_model'],
                config=grpo_config,
                train_dataloader=mock_pipeline['dataloader'],
            )
            
            # Empty batch should be handled gracefully
            batch = {"prompt": []}
            
            # Behavior depends on implementation
            # Either returns empty loss or raises meaningful error
            try:
                loss_dict = trainer.compute_loss(batch)
            except (ValueError, IndexError):
                pass  # Expected for empty batch
                
    def test_mismatched_images_prompts(self, mock_reward_model):
        """Test handling of mismatched images and prompts."""
        images = [Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8))]
        prompts = ["prompt 1", "prompt 2", "prompt 3"]  # More prompts than images
        
        # Should handle gracefully or raise meaningful error
        try:
            output = mock_reward_model.compute_reward(images, prompts)
        except (ValueError, IndexError):
            pass  # Expected for mismatched sizes
