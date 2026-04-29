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
    # Janus CFG (GenerationConfig.guidance_scale). Lower → more diverse rollouts.
    guidance_scale: float = 5.0
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

        # KL reference policy.  For LoRA training we treat the *base* model
        # (adapter disabled) as the reference policy π_ref, so there is no
        # need to allocate a separate frozen copy — we just temporarily turn
        # the LoRA adapter off while scoring with the current model.
        self.use_kl_penalty: bool = False
        self._setup_reference_model()

    def _setup_reference_model(self) -> None:
        """Check whether a KL reference policy is available.

        Strategy (LoRA-friendly, zero extra GPU memory):
            π_θ   = base + LoRA (current policy, grads on)
            π_ref = base only   (via PeftModel.disable_adapter())

        This avoids the deepcopy pitfalls of storing a second full model.
        """
        kl_requested = (
            self.grpo_config.kl_coef > 0
            or self.grpo_config.target_kl is not None
        )
        if not kl_requested:
            self.use_kl_penalty = False
            print("[GRPOTrainer] KL penalty disabled (kl_coef=0, target_kl=None)")
            return

        model = self.generator.model
        has_peft_adapter = hasattr(model, "disable_adapter") and callable(
            getattr(model, "disable_adapter")
        )
        has_scorer = hasattr(self.generator, "score_tokens_with_model")

        if has_peft_adapter and has_scorer:
            self.use_kl_penalty = True
            print(
                "[GRPOTrainer] KL reference policy = base model via "
                "PeftModel.disable_adapter()"
            )
        else:
            self.use_kl_penalty = False
            missing = []
            if not has_peft_adapter:
                missing.append("PEFT disable_adapter()")
            if not has_scorer:
                missing.append("generator.score_tokens_with_model()")
            print(
                f"[GRPOTrainer] KL penalty disabled — missing: {', '.join(missing)}"
            )
        
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

        # expanded_prompts is the per-image prompt list (one entry per
        # generated image, grouped by prompt):
        #   [p0, p0, ...(K times), p1, p1, ...(K times), ...]
        # It is still required by the reward model, which scores each
        # (image, prompt) pair independently.
        expanded_prompts = [p for p in prompts for _ in range(K)]

        # ------------------------------------------------------------------
        # Generation — algorithmic equivalent to the original, but batched.
        #
        # Original behaviour: hand the generator ``expanded_prompts`` (length
        # B*K) and let it loop B*K times with ``parallel_size=1``, meaning
        # every autoregressive forward pass had a CFG batch of 2.
        #
        # New behaviour: hand the generator the B unique prompts together
        # with ``num_images_per_prompt=K``. ``_generate_with_logprobs_single``
        # already supports ``parallel_size>1``: the K rows of its
        # ``torch.multinomial`` call are sampled independently, so the K
        # returned samples per prompt are i.i.d. draws from the *same*
        # policy distribution — identical sampling semantics, identical
        # log-prob formula, identical CFG combination, identical return
        # ordering:
        #     [p0_0, p0_1, ..., p0_{K-1}, p1_0, ..., p_{B-1}_{K-1}]
        # which matches ``expanded_prompts`` position-for-position, so
        # every downstream tensor operation (``rewards.view(B, K)``,
        # ``_compute_advantages``, KL, policy-gradient loss) is unchanged.
        # The only difference is that each forward pass now processes a
        # CFG batch of K*2 instead of 2, which is what actually saturates
        # the GPU. (The specific RNG trajectory from ``torch.multinomial``
        # differs — purely a random-seed-level effect on the realised
        # samples, not on the distribution they are drawn from.)
        # ------------------------------------------------------------------
        from src.models.generators import GenerationConfig

        gen_config = GenerationConfig(
            num_images_per_prompt=K,
            temperature=self.grpo_config.temperature,
            guidance_scale=self.grpo_config.guidance_scale,
        )

        # Generate samples and compute log probabilities.
        # NOTE: log_probs retains gradients for policy optimisation.
        # When KL is enabled we also need the raw token sequences to score
        # with the reference policy (= base model, LoRA disabled) — request
        # them via return_tokens.
        need_tokens = self.use_kl_penalty
        gen_result = self._generate_with_logprobs(
            prompts, return_tokens=need_tokens, config=gen_config
        )
        if need_tokens:
            images, log_probs, generated_tokens = gen_result
        else:
            images, log_probs = gen_result
            generated_tokens = None

        # Compute rewards without gradient tracking (reward model is fixed)
        with torch.no_grad():
            reward_output = self.reward_model.compute_reward(images, expanded_prompts)
            rewards = reward_output.rewards  # Shape: (batch_size * K,)

        # Reshape rewards & log_probs to (batch_size, K)
        rewards = rewards.view(batch_size, K)
        log_probs_flat = log_probs  # keep flat for KL computation
        log_probs = log_probs.view(batch_size, K)

        # Compute group-relative advantages
        advantages = self._compute_advantages(rewards)

        # Normalise advantages across the full batch
        if self.grpo_config.use_advantage_normalization:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy-gradient loss: L_pg = -E[Â · log π_θ(a|s)]
        policy_loss = -(advantages.detach() * log_probs).mean()

        # KL divergence penalty: KL(π_θ ∥ π_ref)
        # Pass current log_probs so we avoid a redundant scoring forward pass.
        kl_div = self._compute_kl_divergence(
            expanded_prompts,
            images,
            current_log_probs=log_probs_flat,
            generated_tokens=generated_tokens,
        )

        # Early-stopping based on KL budget
        if (
            self.grpo_config.target_kl is not None
            and kl_div.item() > self.grpo_config.target_kl
        ):
            # Return zero loss to skip this update (KL exceeded budget)
            zero = torch.tensor(0.0, device=self.device, requires_grad=True)
            return {
                "loss": zero,
                "policy_loss": policy_loss,
                "kl_div": kl_div,
                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),
                "advantage_mean": advantages.mean(),
                "kl_budget_exceeded": torch.tensor(1.0),
            }

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
        return_tokens: bool = False,
        config: Optional[Any] = None,
    ) -> tuple:
        """
        Generate images and compute log probabilities.

        Args:
            prompts: Text prompts to generate from.
            return_tokens: If True and the generator supports it, also return
                the raw generated token sequences (needed for KL computation
                with the frozen reference model).
            config: Optional ``GenerationConfig`` forwarded to the generator.
                Used by :meth:`compute_loss` to request ``parallel_size=K``
                via ``num_images_per_prompt`` so that K i.i.d. samples per
                prompt are produced in a single batched forward pass instead
                of K sequential calls.

        Returns:
            (images, log_probs)  OR  (images, log_probs, tokens)
        """
        if hasattr(self.generator, "generate_with_logprobs"):
            result = self.generator.generate_with_logprobs(
                prompt=prompts, config=config, return_tokens=return_tokens
            )
            # Generator returns (images, log_probs) or (images, log_probs, tokens)
            return result

        # Fallback path for generators without logprob support
        images = self.generator.generate(prompt=prompts, config=config)
        log_probs = self._compute_log_probs(prompts, images)
        if return_tokens:
            return images, log_probs, None
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
        current_log_probs: Optional[torch.Tensor] = None,
        generated_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute KL divergence between current policy and reference policy.

        KL(π_θ ∥ π_ref) = E[log π_θ(a|s) - log π_ref(a|s)]

        For LoRA training the reference policy is the base model with the
        adapter disabled (via PeftModel.disable_adapter()). This avoids
        keeping a deepcopy of the full model in GPU memory.

        Args:
            prompts: Text prompts (same order as images).
            images: Generated PIL images (kept for API compatibility).
            current_log_probs: Already-computed log-probs from the current
                policy (shape: (N,)). Reusing these avoids a duplicate
                forward pass.
            generated_tokens: Raw token sequences from the generator
                (shape: (N, T)).

        Returns:
            Scalar KL divergence tensor (with gradient w.r.t. current policy).
        """
        # Guard: KL disabled, or scorer unavailable, or tokens not captured.
        if (
            not self.use_kl_penalty
            or generated_tokens is None
            or current_log_probs is None
        ):
            return torch.tensor(0.0, device=self.device)

        # ------------------------------------------------------------------
        # Reference-policy log-probs — run the scorer with LoRA disabled.
        # `disable_adapter()` is a context manager provided by PeftModel that
        # temporarily routes forward passes through the un-adapted base
        # weights; it is cheap and has no GPU memory overhead.
        # ------------------------------------------------------------------
        try:
            with torch.no_grad(), self.generator.model.disable_adapter():
                was_training = self.generator.model.training
                self.generator.model.eval()
                ref_log_probs = self.generator.score_tokens_with_model(
                    model=self.generator.model,
                    prompts=prompts,
                    generated_tokens=generated_tokens,
                )
                if was_training:
                    self.generator.model.train()
        except Exception as e:
            print(f"[GRPO] ref policy scoring failed ({e}); KL = 0 this step")
            return torch.tensor(0.0, device=self.device)

        ref_log_probs = ref_log_probs.detach().to(current_log_probs.device)

        # KL(π_θ ∥ π_ref) ≈ mean(log π_θ − log π_ref).  KL ≥ 0 by definition,
        # clamp to suppress numerical noise from degenerate token log-probs.
        kl_div = (current_log_probs - ref_log_probs).mean()
        kl_div = kl_div.clamp(min=0.0)
        return kl_div


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
