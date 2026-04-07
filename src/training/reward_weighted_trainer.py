"""
Reward-Weighted Trainer
========================

Simpler alternative to GRPO that uses reward-weighted maximum likelihood.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.training.base_trainer import BaseTrainer, TrainingConfig


@dataclass 
class RewardWeightedConfig(TrainingConfig):
    """Configuration for reward-weighted training."""
    temperature: float = 1.0
    reward_threshold: float = 0.0  # Only train on samples above threshold
    use_rejection_sampling: bool = False
    num_samples: int = 4


class RewardWeightedTrainer(BaseTrainer):
    """
    Reward-Weighted Maximum Likelihood Trainer.
    
    Simpler than GRPO - weights the MLE loss by the reward:
    L = -E[r(x) * log p(x|prompt)]
    
    This is equivalent to policy gradient with no baseline.
    """
    
    def __init__(self, *args, rw_config: Optional[RewardWeightedConfig] = None, **kwargs):
        if rw_config is not None:
            kwargs["config"] = rw_config
        super().__init__(*args, **kwargs)
        self.rw_config = rw_config or self.config
        
    def compute_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compute reward-weighted loss.
        
        Args:
            batch: Dictionary with 'prompt' and optionally 'image' keys
            
        Returns:
            Dictionary with loss and metrics
        """
        prompts = batch["prompt"]
        
        # Generate samples
        with torch.no_grad():
            images = self.generator.generate(prompts)
            
            # Compute rewards
            reward_output = self.reward_model.compute_reward(images, prompts)
            rewards = reward_output.rewards
            
        # Filter by threshold if enabled
        if self.rw_config.reward_threshold > 0:
            mask = rewards > self.rw_config.reward_threshold
            if mask.sum() == 0:
                # No samples above threshold, use all with low weight
                weights = F.softmax(rewards / self.rw_config.temperature, dim=0)
            else:
                # Zero out weights for samples below threshold
                weights = rewards * mask.float()
                weights = F.softmax(weights / self.rw_config.temperature, dim=0)
        else:
            # Softmax over rewards as weights
            weights = F.softmax(rewards / self.rw_config.temperature, dim=0)
            
        # Compute weighted MLE loss
        # Placeholder - actual implementation depends on model
        log_probs = self._compute_log_probs(prompts, images)
        loss = -(weights.detach() * log_probs).sum()
        
        return {
            "loss": loss,
            "reward_mean": rewards.mean(),
            "reward_max": rewards.max(),
            "effective_batch_size": (weights > 0.01).sum().float(),
        }
        
    def _compute_log_probs(
        self,
        prompts: List[str],
        images: List[Image.Image],
    ) -> torch.Tensor:
        """Compute log probabilities (model-specific)."""
        # Placeholder
        return torch.zeros(len(prompts), device=self.device)
