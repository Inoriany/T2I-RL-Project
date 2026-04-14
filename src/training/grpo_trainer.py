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
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.training.base_trainer import BaseTrainer, TrainingConfig


@dataclass
class GRPOConfig(TrainingConfig):
    """GRPO-specific configuration."""
    # GRPO params
    num_samples_per_prompt: int = 4
    temperature: float = 1.0
    clip_ratio: float = 0.2
    ppo_epochs: int = 1
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
        self._warned_missing_kl_support = False
        if self.grpo_config.kl_coef > 0 or self.grpo_config.target_kl is not None:
            self._setup_reference_model()
        else:
            # Skip reference model to save GPU memory when KL is disabled
            self.ref_model = None
        
    def _setup_reference_model(self) -> None:
        """Create frozen reference model for KL computation.
        
        Memory-aware strategy:
        - GPU VRAM >= 20 GB: save initial LoRA weights; for KL scoring,
          temporarily swap LoRA weights on the *same* base model (no deepcopy).
        - GPU VRAM < 20 GB: skip reference model entirely and force kl_coef=0
          to avoid OOM (deepcopy of a 4-bit model often loses quantization and
          doubles VRAM usage).
        """
        # Detect available VRAM
        vram_gb = 0.0
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

        if vram_gb < 20.0:
            # Low-VRAM path: disable KL entirely
            print(
                f"[GRPOTrainer] Low VRAM detected ({vram_gb:.1f} GB < 20 GB). "
                f"Disabling KL regularization (kl_coef forced to 0) to prevent OOM."
            )
            self.grpo_config.kl_coef = 0.0
            self.grpo_config.target_kl = None
            self.ref_model = None
            self._ref_lora_state = None
            return

        # High-VRAM path: save initial LoRA state dict instead of deepcopy.
        # This avoids duplicating the full base model and breaking 4-bit quantization.
        self._ref_lora_state = None
        try:
            from peft import PeftModel
            model = self.generator.model
            if isinstance(model, PeftModel):
                # Save a frozen copy of the initial LoRA adapter weights
                import copy
                self._ref_lora_state = copy.deepcopy(
                    {k: v.cpu() for k, v in model.state_dict().items()
                     if "lora_" in k}
                )
                print(
                    f"[GRPOTrainer] Saved initial LoRA state for KL reference "
                    f"({len(self._ref_lora_state)} tensors, VRAM={vram_gb:.1f} GB). "
                    f"No deepcopy of base model needed."
                )
                # ref_model stays None — we use weight-swap approach
                self.ref_model = None
                return
        except ImportError:
            pass

        # Fallback: try deepcopy (only for non-quantized models on high-VRAM GPUs)
        try:
            import copy
            self.ref_model = copy.deepcopy(self.generator.model)
            for param in self.ref_model.parameters():
                param.requires_grad = False
            self.ref_model.eval()
            print(f"[GRPOTrainer] Created deepcopy reference model (VRAM={vram_gb:.1f} GB).")
        except Exception as e:
            print(
                f"[GRPOTrainer] Failed to deepcopy reference model: {e}. "
                f"Disabling KL regularization."
            )
            self.grpo_config.kl_coef = 0.0
            self.ref_model = None
            self._ref_lora_state = None
        
    def compute_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compute GRPO loss for a batch.
        
        Args:
            batch: Dictionary with 'prompt' key
            
        Returns:
            Dictionary with loss and auxiliary metrics
        """
        rollout = self._prepare_rollout_batch(batch)
        return self._compute_replay_loss(rollout)

    def _train_epoch(self) -> None:
        """Train one epoch with rollout-once, replay-many PPO updates."""
        self.generator.model.train()

        total_loss = 0.0
        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.current_epoch}",
        )

        if self.config.gradient_accumulation_steps != 1:
            warnings.warn(
                "GRPO PPO replay uses one optimizer step per inner PPO epoch; "
                "forcing effective gradient_accumulation_steps=1 for correctness.",
                stacklevel=2,
            )

        for _, batch in enumerate(progress_bar):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            rollout = self._prepare_rollout_batch(batch)
            ppo_losses = []

            for inner_idx in range(self.grpo_config.ppo_epochs):
                loss_dict = self._compute_replay_loss(rollout)
                loss = loss_dict["loss"]

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.generator.get_trainable_parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                self.scheduler.step()

                self.global_step += 1
                total_loss += loss.item()
                ppo_losses.append(loss.item())

                for key, value in loss_dict.items():
                    if key == "loss":
                        continue
                    if isinstance(value, torch.Tensor):
                        if value.numel() == 1:
                            scalar = value.detach().item()
                        else:
                            continue
                    elif isinstance(value, (int, float)):
                        scalar = float(value)
                    else:
                        continue
                    metric_sums[key] = metric_sums.get(key, 0.0) + scalar
                    metric_counts[key] = metric_counts.get(key, 0) + 1

                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / self.config.logging_steps
                    avg_metrics = {
                        f"train/{k}": metric_sums[k] / max(metric_counts[k], 1)
                        for k in metric_sums
                    }
                    self.log({
                        "train/loss": avg_loss,
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                        **avg_metrics,
                    })
                    total_loss = 0.0
                    metric_sums = {}
                    metric_counts = {}

                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")

                if (
                    self.eval_dataloader is not None
                    and self.global_step % self.config.eval_steps == 0
                ):
                    eval_metrics = self.evaluate()
                    self.log({f"eval/{k}": v for k, v in eval_metrics.items()})
                    self.generator.model.train()

            progress_bar.set_postfix({
                "loss": round(ppo_losses[-1], 4),
                "ppo_epochs": self.grpo_config.ppo_epochs,
            })

    def _prepare_rollout_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Roll out once and cache rewards/advantages for PPO-style replays."""
        prompts = batch["prompt"]
        batch_size = len(prompts)
        K = self.grpo_config.num_samples_per_prompt
        expanded_prompts = [p for p in prompts for _ in range(K)]

        images, current_log_probs, generation_info = self._generate_with_logprobs(expanded_prompts)

        with torch.no_grad():
            reward_output = self.reward_model.compute_reward(images, expanded_prompts)
            rewards = reward_output.rewards.view(batch_size, K)

        advantages = self._compute_advantages(rewards)
        if self.grpo_config.use_advantage_normalization:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_log_probs = current_log_probs.detach().view(batch_size, K)
        if generation_info is not None:
            collected_old_log_probs = []
            for item in generation_info:
                if isinstance(item, dict) and "old_log_probs" in item:
                    collected_old_log_probs.append(item["old_log_probs"])
            if collected_old_log_probs:
                old_log_probs = torch.cat(collected_old_log_probs).view(batch_size, K)

        rollout = {
            "prompts": prompts,
            "expanded_prompts": expanded_prompts,
            "images": images,
            "generation_info": generation_info,
            "old_log_probs": old_log_probs.detach(),
            "rewards": rewards.detach(),
            "advantages": advantages.detach(),
            "reward_output": reward_output,
        }
        return rollout

    def _compute_replay_loss(self, rollout: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Re-score a cached rollout under current policy for PPO-style updates."""
        rewards = rollout["rewards"]
        advantages = rollout["advantages"]
        old_log_probs = rollout["old_log_probs"]
        generation_info = rollout["generation_info"]
        reward_output = rollout["reward_output"]

        if generation_info is not None and hasattr(self.generator, "score_from_generation_info"):
            current_log_probs = self.generator.score_from_generation_info(
                generation_info,
                use_grad=True,
            ).view_as(old_log_probs)
        else:
            current_log_probs = old_log_probs.clone().detach().requires_grad_(True)

        log_ratio = current_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.grpo_config.clip_ratio,
            1.0 + self.grpo_config.clip_ratio,
        )

        surrogate_unclipped = ratio * advantages
        surrogate_clipped = clipped_ratio * advantages
        surrogate = torch.minimum(surrogate_unclipped, surrogate_clipped)
        policy_loss = -surrogate.mean()
        clip_fraction = ((ratio - clipped_ratio).abs() > 1e-8).float().mean()

        kl_div = self._compute_kl_divergence(
            rollout["expanded_prompts"],
            rollout["images"],
            current_log_probs.reshape(-1),
            generation_info=generation_info,
        )

        total_loss = policy_loss + self.grpo_config.kl_coef * kl_div

        metrics = {
            "loss": total_loss,
            "policy_loss": policy_loss,
            "kl_div": kl_div,
            "reward_mean": rewards.mean(),
            "reward_std": rewards.std(),
            "advantage_mean": advantages.mean(),
            "ratio_mean": ratio.mean(),
            "ratio_std": ratio.std(),
            "clip_fraction": clip_fraction,
        }

        details = getattr(reward_output, "details", None) or {}
        component_rewards = details.get("component_rewards")
        if isinstance(component_rewards, dict):
            for name, values in component_rewards.items():
                if isinstance(values, torch.Tensor) and values.numel() > 0:
                    metrics[f"reward_{name}_mean"] = values.mean()
                    metrics[f"reward_{name}_std"] = (
                        values.std() if values.numel() > 1 else torch.tensor(0.0, device=values.device)
                    )

        responses = details.get("responses")
        if isinstance(responses, list) and responses:
            parse_error_count = sum(
                1 for r in responses if isinstance(r, dict) and r.get("parse_error")
            )
            metrics["vlm_parse_error_rate"] = torch.tensor(
                parse_error_count / len(responses),
                device=self.device,
            )

        return metrics
        
    def _generate_with_logprobs(
        self,
        prompts: List[str],
    ) -> tuple:
        """
        Generate images and compute log probabilities.
        
        Returns:
            Tuple of (images, log_probs, generation_info)
        """
        if hasattr(self.generator, "generate_with_logprobs"):
            try:
                output = self.generator.generate_with_logprobs(
                    prompt=prompts,
                    return_generation_info=True,
                )
            except TypeError:
                output = self.generator.generate_with_logprobs(prompt=prompts)

            if isinstance(output, tuple) and len(output) == 3:
                images, log_probs, generation_info = output
            else:
                images, log_probs = output
                generation_info = None

            return images, log_probs, generation_info

        # Fallback path for generators without logprob support
        images = self.generator.generate(prompt=prompts)
        log_probs = self._compute_log_probs(prompts, images)
        return images, log_probs, None
    
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
        current_log_probs: torch.Tensor,
        generation_info: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Compute KL divergence between current policy and reference policy.
        
        KL(π || π_ref) = E[log π(a|s) - log π_ref(a|s)]
        
        Supports three modes:
        1. ref_model exists (deepcopy path) — score with ref_model directly
        2. _ref_lora_state exists (LoRA weight-swap path) — temporarily swap
           LoRA weights to initial values, score, then swap back
        3. Neither exists — return 0 (KL disabled)
        """
        if self.grpo_config.kl_coef <= 0:
            return torch.tensor(0.0, device=self.device)

        has_ref_model = self.ref_model is not None
        has_ref_lora = getattr(self, "_ref_lora_state", None) is not None

        if not has_ref_model and not has_ref_lora:
            return torch.tensor(0.0, device=self.device)

        if generation_info is None or not hasattr(self.generator, "score_from_generation_info"):
            if not self._warned_missing_kl_support:
                warnings.warn(
                    "KL regularization requested, but generator does not expose "
                    "score_from_generation_info(); KL term will be zero.",
                    stacklevel=2,
                )
                self._warned_missing_kl_support = True
            return torch.tensor(0.0, device=self.device)

        if has_ref_model:
            # Standard path: score with a separate reference model
            with torch.no_grad():
                ref_log_probs = self.generator.score_from_generation_info(
                    generation_info,
                    model=self.ref_model,
                )
        else:
            # LoRA weight-swap path: temporarily load initial LoRA weights
            model = self.generator.model
            # Save current LoRA weights
            current_lora_state = {
                k: v.clone() for k, v in model.state_dict().items()
                if "lora_" in k
            }
            try:
                # Load reference (initial) LoRA weights
                ref_state = {k: v.to(self.device) for k, v in self._ref_lora_state.items()}
                model.load_state_dict(ref_state, strict=False)
                with torch.no_grad():
                    ref_log_probs = self.generator.score_from_generation_info(
                        generation_info,
                        use_grad=False,
                    )
            finally:
                # Restore current LoRA weights
                current_state = {k: v.to(self.device) for k, v in current_lora_state.items()}
                model.load_state_dict(current_state, strict=False)
                del current_lora_state, current_state

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
