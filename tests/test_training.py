"""
Unit Tests for Training Module
==============================

Tests for BaseTrainer, GRPOTrainer, and training configurations.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import torch
import numpy as np
from PIL import Image

from src.training.base_trainer import BaseTrainer, TrainingConfig
from src.training.grpo_trainer import GRPOTrainer, GRPOConfig


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = TrainingConfig()
        
        assert config.learning_rate == 1e-5
        assert config.num_epochs == 10
        assert config.batch_size == 4
        assert config.gradient_accumulation_steps == 4
        assert config.max_grad_norm == 1.0
        assert config.kl_coef == 0.1
        assert config.warmup_steps == 100
        assert config.use_wandb is True
        
    def test_custom_values(self):
        """Test custom configuration values."""
        config = TrainingConfig(
            learning_rate=5e-6,
            num_epochs=5,
            batch_size=8,
            use_wandb=False,
        )
        
        assert config.learning_rate == 5e-6
        assert config.num_epochs == 5
        assert config.batch_size == 8
        assert config.use_wandb is False


class TestGRPOConfig:
    """Tests for GRPOConfig dataclass."""
    
    def test_inherits_training_config(self):
        """Test that GRPOConfig inherits from TrainingConfig."""
        config = GRPOConfig()
        
        # Should have TrainingConfig attributes
        assert hasattr(config, 'learning_rate')
        assert hasattr(config, 'num_epochs')
        
    def test_grpo_specific_values(self):
        """Test GRPO-specific configuration values."""
        config = GRPOConfig()
        
        assert config.num_samples_per_prompt == 4
        assert config.temperature == 1.0
        assert config.clip_ratio == 0.2
        assert config.use_advantage_normalization is True
        assert config.baseline_type == "mean"
        assert config.ema_decay == 0.99
        
    def test_custom_grpo_values(self):
        """Test custom GRPO configuration."""
        config = GRPOConfig(
            num_samples_per_prompt=8,
            temperature=0.8,
            baseline_type="min",
            target_kl=0.05,
        )
        
        assert config.num_samples_per_prompt == 8
        assert config.temperature == 0.8
        assert config.baseline_type == "min"
        assert config.target_kl == 0.05


class TestGRPOTrainer:
    """Tests for GRPOTrainer class."""
    
    @pytest.fixture
    def trainer_setup(self, mock_generator, mock_reward_model, sample_dataloader, grpo_config):
        """Setup trainer with mocks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            grpo_config.output_dir = tmpdir
            grpo_config.use_wandb = False
            
            # Make generator return trainable parameters
            param = torch.nn.Parameter(torch.randn(10, requires_grad=True))
            mock_generator.get_trainable_parameters.return_value = [param]
            
            yield {
                'generator': mock_generator,
                'reward_model': mock_reward_model,
                'dataloader': sample_dataloader,
                'config': grpo_config,
                'output_dir': tmpdir,
            }
    
    def test_init(self, trainer_setup):
        """Test trainer initialization."""
        trainer = GRPOTrainer(
            generator=trainer_setup['generator'],
            reward_model=trainer_setup['reward_model'],
            config=trainer_setup['config'],
            train_dataloader=trainer_setup['dataloader'],
        )
        
        assert trainer.generator is trainer_setup['generator']
        assert trainer.reward_model is trainer_setup['reward_model']
        assert trainer.optimizer is not None
        assert trainer.global_step == 0
        
    def test_compute_advantages_mean_baseline(self, trainer_setup):
        """Test advantage computation with mean baseline."""
        trainer = GRPOTrainer(
            generator=trainer_setup['generator'],
            reward_model=trainer_setup['reward_model'],
            config=trainer_setup['config'],
            train_dataloader=trainer_setup['dataloader'],
        )
        
        # Test rewards: (batch_size=2, K=4)
        rewards = torch.tensor([
            [0.5, 0.7, 0.3, 0.9],  # mean = 0.6
            [0.4, 0.6, 0.8, 0.2],  # mean = 0.5
        ])
        
        advantages = trainer._compute_advantages(rewards)
        
        # Advantages should be reward - mean
        expected_row1 = rewards[0] - rewards[0].mean()
        expected_row2 = rewards[1] - rewards[1].mean()
        
        assert advantages.shape == (2, 4)
        assert torch.allclose(advantages[0], expected_row1, atol=1e-5)
        assert torch.allclose(advantages[1], expected_row2, atol=1e-5)
        
    def test_compute_advantages_min_baseline(self, trainer_setup):
        """Test advantage computation with min baseline."""
        config = trainer_setup['config']
        config.baseline_type = "min"
        
        trainer = GRPOTrainer(
            generator=trainer_setup['generator'],
            reward_model=trainer_setup['reward_model'],
            config=config,
            train_dataloader=trainer_setup['dataloader'],
        )
        
        rewards = torch.tensor([
            [0.5, 0.7, 0.3, 0.9],  # min = 0.3
            [0.4, 0.6, 0.8, 0.2],  # min = 0.2
        ])
        
        advantages = trainer._compute_advantages(rewards)
        
        # Advantages should be reward - min
        assert advantages[0, 2] == 0.0  # 0.3 - 0.3 = 0
        assert advantages[1, 3] == 0.0  # 0.2 - 0.2 = 0
        assert advantages[0, 3] == pytest.approx(0.6, rel=1e-5)  # 0.9 - 0.3
        
    def test_compute_advantages_ema_baseline(self, trainer_setup):
        """Test advantage computation with EMA baseline."""
        config = trainer_setup['config']
        config.baseline_type = "ema"
        config.ema_decay = 0.9
        
        trainer = GRPOTrainer(
            generator=trainer_setup['generator'],
            reward_model=trainer_setup['reward_model'],
            config=config,
            train_dataloader=trainer_setup['dataloader'],
        )
        
        # First call - EMA should be initialized
        rewards1 = torch.tensor([[0.5, 0.7, 0.3, 0.9]])
        advantages1 = trainer._compute_advantages(rewards1)
        
        assert trainer.reward_ema is not None
        
        # Second call - EMA should be updated
        rewards2 = torch.tensor([[0.8, 0.8, 0.8, 0.8]])
        advantages2 = trainer._compute_advantages(rewards2)
        
        # EMA should have moved toward new mean
        assert trainer.reward_ema != rewards1.mean().item()
        
    def test_reference_model_setup(self, trainer_setup):
        """Test that reference model is properly set up."""
        trainer = GRPOTrainer(
            generator=trainer_setup['generator'],
            reward_model=trainer_setup['reward_model'],
            config=trainer_setup['config'],
            train_dataloader=trainer_setup['dataloader'],
        )
        
        # Reference model should exist and be frozen
        assert trainer.ref_model is not None
        
    def test_log_without_wandb(self, trainer_setup, capsys):
        """Test logging without wandb."""
        trainer = GRPOTrainer(
            generator=trainer_setup['generator'],
            reward_model=trainer_setup['reward_model'],
            config=trainer_setup['config'],
            train_dataloader=trainer_setup['dataloader'],
        )
        
        metrics = {"loss": 0.5, "reward_mean": 0.7}
        trainer.log(metrics, step=100)
        
        captured = capsys.readouterr()
        assert "100" in captured.out
        assert "loss" in captured.out or "0.5" in captured.out


class TestBaseTrainer:
    """Tests for BaseTrainer class."""
    
    def test_training_config_output_dir_creation(self, mock_generator, mock_reward_model, sample_dataloader):
        """Test that output directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "new_output"
            
            config = TrainingConfig(
                output_dir=str(output_dir),
                use_wandb=False,
            )
            
            param = torch.nn.Parameter(torch.randn(10, requires_grad=True))
            mock_generator.get_trainable_parameters.return_value = [param]
            
            # Create a concrete implementation for testing
            class ConcreteTrainer(BaseTrainer):
                def compute_loss(self, batch):
                    return {"loss": torch.tensor(0.5)}
            
            trainer = ConcreteTrainer(
                generator=mock_generator,
                reward_model=mock_reward_model,
                config=config,
                train_dataloader=sample_dataloader,
            )
            
            assert output_dir.exists()
            
    def test_save_checkpoint(self, mock_generator, mock_reward_model, sample_dataloader):
        """Test checkpoint saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                output_dir=tmpdir,
                use_wandb=False,
            )
            
            param = torch.nn.Parameter(torch.randn(10, requires_grad=True))
            mock_generator.get_trainable_parameters.return_value = [param]
            mock_generator.model = Mock()
            mock_generator.model.save_pretrained = Mock()
            
            class ConcreteTrainer(BaseTrainer):
                def compute_loss(self, batch):
                    return {"loss": torch.tensor(0.5)}
            
            trainer = ConcreteTrainer(
                generator=mock_generator,
                reward_model=mock_reward_model,
                config=config,
                train_dataloader=sample_dataloader,
            )
            
            trainer.save_checkpoint("test-checkpoint")
            
            checkpoint_dir = Path(tmpdir) / "test-checkpoint"
            assert checkpoint_dir.exists()
            assert (checkpoint_dir / "training_state.pt").exists()


class TestTrainerEvaluation:
    """Tests for trainer evaluation functionality."""
    
    def test_evaluate(self, mock_generator, mock_reward_model, sample_dataloader):
        """Test evaluation method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                output_dir=tmpdir,
                use_wandb=False,
            )
            
            param = torch.nn.Parameter(torch.randn(10, requires_grad=True))
            mock_generator.get_trainable_parameters.return_value = [param]
            mock_generator.model = Mock()
            
            class ConcreteTrainer(BaseTrainer):
                def compute_loss(self, batch):
                    return {"loss": torch.tensor(0.5)}
            
            trainer = ConcreteTrainer(
                generator=mock_generator,
                reward_model=mock_reward_model,
                config=config,
                train_dataloader=sample_dataloader,
                eval_dataloader=sample_dataloader,
            )
            
            eval_metrics = trainer.evaluate()
            
            assert "avg_reward" in eval_metrics
            assert isinstance(eval_metrics["avg_reward"], float)


class TestGRPOLossComputation:
    """Tests for GRPO loss computation."""
    
    def test_compute_loss_returns_required_keys(self, mock_generator, mock_reward_model, sample_dataloader, grpo_config):
        """Test that compute_loss returns all required keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            grpo_config.output_dir = tmpdir
            grpo_config.use_wandb = False
            
            param = torch.nn.Parameter(torch.randn(10, requires_grad=True))
            mock_generator.get_trainable_parameters.return_value = [param]
            
            trainer = GRPOTrainer(
                generator=mock_generator,
                reward_model=mock_reward_model,
                config=grpo_config,
                train_dataloader=sample_dataloader,
            )
            
            batch = {"prompt": ["test prompt 1", "test prompt 2"]}
            loss_dict = trainer.compute_loss(batch)
            
            assert "loss" in loss_dict
            assert "policy_loss" in loss_dict
            assert "kl_div" in loss_dict
            assert "reward_mean" in loss_dict


class TestTrainerOptimizer:
    """Tests for optimizer setup."""
    
    def test_optimizer_setup(self, mock_generator, mock_reward_model, sample_dataloader):
        """Test optimizer is properly configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                output_dir=tmpdir,
                learning_rate=5e-6,
                weight_decay=0.01,
                use_wandb=False,
            )
            
            param = torch.nn.Parameter(torch.randn(10, requires_grad=True))
            mock_generator.get_trainable_parameters.return_value = [param]
            
            class ConcreteTrainer(BaseTrainer):
                def compute_loss(self, batch):
                    return {"loss": torch.tensor(0.5)}
            
            trainer = ConcreteTrainer(
                generator=mock_generator,
                reward_model=mock_reward_model,
                config=config,
                train_dataloader=sample_dataloader,
            )
            
            assert trainer.optimizer is not None
            assert trainer.scheduler is not None
            
            # Check learning rate
            for param_group in trainer.optimizer.param_groups:
                assert param_group['lr'] == 5e-6
