"""
GRPO Trainer (Group Relative Policy Optimization)
==================================================

Implementation of GRPO for T2I model training.
Based on: https://github.com/CaraJ7/T2I-R1

GRPO is an RL algorithm that:
1. Generates multiple samples per prompt
2. Computes relative rewards within each group
3. Updates policy to increase probability of high-reward samples
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.training.base_trainer import BaseTrainer, TrainingConfig


@dataclass
class GRPOConfig(TrainingConfig):
    """GRPO-specific configuration."""
    # GRPO params
    num_samples_per_prompt: int = 4
    temperature: float = 1.0
    clip_ratio: float = 0.2
    use_advantage_normalization: bool = True
    baseline_type: str = "mean"  # "mean", "min", "ema"
    ema_decay: float = 0.99
    
    # KL regularization
    kl_coef: float = 0.1
    target_kl: Optional[float] = None  # Early stopping if KL exceeds this


class GRPOTrainer(BaseTrainer):
    """
    GRPO (Group Relative Policy Optimization) Trainer.
    
    Algorithm:
    1. For each prompt, generate K samples from the current policy
    2. Compute rewards for all samples using the reward model
    3. Compute advantages as reward - baseline (group mean/min)
    4. Update policy to maximize advantage-weighted log probability
    5. Apply KL penalty to prevent policy from diverging too far
    """
    
    def __init__(self, *args, grpo_config: Optional[GRPOConfig] = None, **kwargs):
        # Use GRPO config if provided
        if grpo_config is not None:
            kwargs["config"] = grpo_config
        super().__init__(*args, **kwargs)
        
        self.grpo_config = grpo_config or self.config
        
        # EMA baseline tracking
        self.reward_ema = None
        
        # Reference model for KL computation (frozen copy of initial policy)
        self.ref_model = None
        if self.grpo_config.kl_coef > 0 or self.grpo_config.target_kl is not None:
            self._setup_reference_model()
        else:
            # Skip reference model to save GPU memory when KL is disabled
            self.ref_model = None
        
    def _setup_reference_model(self) -> None:
        """Create frozen reference model for KL computation."""
        import copy
        
        # Deep copy the model
        self.ref_model = copy.deepcopy(self.generator.model)
        
        # Freeze all parameters
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
        self.ref_model.eval()
        
    def compute_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compute GRPO loss for a batch.
        
        Args:
            batch: Dictionary with 'prompt' key
            
        Returns:
            Dictionary with loss and auxiliary metrics
        """
        prompts = batch["prompt"]
        batch_size = len(prompts)
        K = self.grpo_config.num_samples_per_prompt
        
        # Expand prompts for multiple samples
        expanded_prompts = [p for p in prompts for _ in range(K)]
        
        # Generate samples and compute log probabilities
        # NOTE: log_probs must keep gradients for policy optimization.
        images, log_probs = self._generate_with_logprobs(expanded_prompts)

        # Compute rewards without gradient tracking (reward model is fixed)
        with torch.no_grad():
            reward_output = self.reward_model.compute_reward(images, expanded_prompts)
            rewards = reward_output.rewards  # Shape: (batch_size * K,)
            
        # Reshape rewards to (batch_size, K)
        rewards = rewards.view(batch_size, K)
        log_probs = log_probs.view(batch_size, K)
        
        # Compute advantages
        advantages = self._compute_advantages(rewards)
        
        # Normalize advantages
        if self.grpo_config.use_advantage_normalization:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        # Compute policy loss
        # We want to increase log_prob for high-advantage samples
        policy_loss = -(advantages.detach() * log_probs).mean()
        
        # Compute KL divergence penalty
        kl_div = self._compute_kl_divergence(expanded_prompts, images)
        
        # Total loss
        total_loss = policy_loss + self.grpo_config.kl_coef * kl_div
        
        return {
            "loss": total_loss,
            "policy_loss": policy_loss,
            "kl_div": kl_div,
            "reward_mean": rewards.mean(),
            "reward_std": rewards.std(),
            "advantage_mean": advantages.mean(),
        }
        
    def _generate_with_logprobs(
        self,
        prompts: List[str],
    ) -> tuple:
        """
        Generate images and compute log probabilities.
        
        Returns:
            Tuple of (images, log_probs)
        """
        if hasattr(self.generator, "generate_with_logprobs"):
            images, log_probs = self.generator.generate_with_logprobs(prompt=prompts)
            return images, log_probs

        # Fallback path for generators without logprob support
        images = self.generator.generate(prompt=prompts)
        log_probs = self._compute_log_probs(prompts, images)
        return images, log_probs
    
    def _compute_log_probs(
        self,
        prompts: List[str],
        images: List[Image.Image],
    ) -> torch.Tensor:
        """
        Compute log probabilities of generated images.
        
        For autoregressive models: sum of log_softmax over generated tokens
        For diffusion models: negative denoising loss
        """
        # Placeholder - implementation depends on model architecture
        # Return dummy values for now
        return torch.zeros(len(prompts), device=self.device)
    
    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute advantages using the specified baseline.
        
        Args:
            rewards: Shape (batch_size, K) rewards for each sample group
            
        Returns:
            Shape (batch_size, K) advantages
        """
        if self.grpo_config.baseline_type == "mean":
            # Group mean baseline
            baseline = rewards.mean(dim=1, keepdim=True)
        elif self.grpo_config.baseline_type == "min":
            # Min baseline (more aggressive)
            baseline = rewards.min(dim=1, keepdim=True).values
        elif self.grpo_config.baseline_type == "ema":
            # Exponential moving average baseline
            if self.reward_ema is None:
                self.reward_ema = rewards.mean().item()
            else:
                self.reward_ema = (
                    self.grpo_config.ema_decay * self.reward_ema 
                    + (1 - self.grpo_config.ema_decay) * rewards.mean().item()
                )
            baseline = torch.full_like(rewards, self.reward_ema)
        else:
            raise ValueError(f"Unknown baseline type: {self.grpo_config.baseline_type}")
            
        return rewards - baseline
    
    def _compute_kl_divergence(
        self,
        prompts: List[str],
        images: List[Image.Image],
    ) -> torch.Tensor:
        """
        Compute KL divergence between current policy and reference policy.
        
        KL(π || π_ref) = E[log π(a|s) - log π_ref(a|s)]
        """
        if self.ref_model is None or self.grpo_config.kl_coef <= 0:
            return torch.tensor(0.0, device=self.device)

        # Current policy log probs
        current_log_probs = self._compute_log_probs(prompts, images)
        
        # Reference policy log probs (using frozen reference model)
        with torch.no_grad():
            ref_log_probs = self._compute_ref_log_probs(prompts, images)
            
        kl_div = (current_log_probs - ref_log_probs).mean()
        
        return kl_div
    
    def _compute_ref_log_probs(
        self,
        prompts: List[str],
        images: List[Image.Image],
    ) -> torch.Tensor:
        """Compute log probs using reference model."""
        if self.ref_model is None:
            return torch.zeros(len(prompts), device=self.device)
        # Placeholder - use reference model for computation
        return torch.zeros(len(prompts), device=self.device)


class FlowGRPOTrainer(GRPOTrainer):
    """
    GRPO Trainer specialized for Flow-based models (e.g., Flux).
    
    Adapts GRPO for continuous-time flow matching models.
    """
    
    def _compute_log_probs(
        self,
        prompts: List[str],
        images: List[Image.Image],
    ) -> torch.Tensor:
        """
        Compute log probs for flow models using flow matching loss.
        
        For flow models, we use the negative flow matching loss
        as a proxy for log probability.
        """
        # Convert images to tensors
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        image_tensors = torch.stack([transform(img) for img in images]).to(self.device)
        
        # Compute flow matching loss
        # This is model-specific and requires the flow model's forward pass
        # Placeholder implementation
        flow_loss = torch.zeros(len(prompts), device=self.device)
        
        return -flow_loss  # Negative loss as log prob proxy
