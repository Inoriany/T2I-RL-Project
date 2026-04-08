"""
Base Trainer
=============

Abstract base class for all T2I-RL trainers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Basic training params
    learning_rate: float = 1e-5
    num_epochs: int = 10
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # RL-specific params
    num_samples_per_prompt: int = 4  # For GRPO
    kl_coef: float = 0.1
    reward_scale: float = 1.0
    
    # Optimization
    warmup_steps: int = 100
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 100
    output_dir: str = "./outputs"
    save_total_limit: int = 3
    
    # Logging
    logging_steps: int = 10
    use_wandb: bool = True
    wandb_project: str = "t2i-rl"
    wandb_run_name: Optional[str] = None
    
    # Hardware
    fp16: bool = False
    bf16: bool = True
    dataloader_num_workers: int = 4


class BaseTrainer(ABC):
    """
    Abstract base class for T2I-RL trainers.
    
    Provides common functionality for:
    - Training loop management
    - Logging and checkpointing
    - Gradient handling
    - Evaluation
    """
    
    def __init__(
        self,
        generator: Any,  # ImageGenerator
        reward_model: Any,  # RewardModel
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ):
        self.generator = generator
        self.reward_model = reward_model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_step = 0
        self.current_epoch = 0
        
        # Initialize optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        self._setup_optimizer()
        
        # Setup logging
        self.logger = None
        if config.use_wandb:
            self._setup_wandb()
            
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
    def _setup_optimizer(self) -> None:
        """Setup optimizer and learning rate scheduler."""
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup
        
        trainable_params = self.generator.get_trainable_parameters()
        
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay,
        )
        
        total_steps = (
            len(self.train_dataloader) 
            * self.config.num_epochs 
            // self.config.gradient_accumulation_steps
        )
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )
        
    def _setup_wandb(self) -> None:
        """Setup Weights & Biases logging."""
        try:
            import wandb
            
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=vars(self.config),
            )
            self.logger = wandb
        except ImportError:
            print("wandb not installed, skipping W&B logging")
            
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics."""
        step = step or self.global_step
        
        if self.logger is not None:
            self.logger.log(metrics, step=step)
            
        # Also print to console
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float))])
        print(f"Step {step} | {metrics_str}")
        
    @abstractmethod
    def compute_loss(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss for a batch.
        
        Args:
            batch: Dictionary containing batch data
            
        Returns:
            Dictionary with 'loss' and optional auxiliary losses
        """
        pass
    
    def train(self) -> None:
        """Main training loop."""
        start_epoch = self.current_epoch if self.global_step > 0 else 0
        remaining_epochs = max(self.config.num_epochs - start_epoch, 0)

        print(f"Starting training for {self.config.num_epochs} epochs")
        if start_epoch > 0:
            print(f"Resuming from epoch {start_epoch}, global step {self.global_step}")
        print(f"Total steps (remaining): {len(self.train_dataloader) * remaining_epochs}")

        for epoch in range(start_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            self._train_epoch()
            
            # Evaluation
            if self.eval_dataloader is not None:
                eval_metrics = self.evaluate()
                self.log({"epoch": epoch, **eval_metrics})
                
            # Save checkpoint
            self.save_checkpoint(f"checkpoint-epoch-{epoch}")
            
        print("Training complete!")
        
    def _train_epoch(self) -> None:
        """Train for one epoch."""
        self.generator.model.train()
        
        total_loss = 0
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.current_epoch}",
        )
        
        for step, batch in enumerate(progress_bar):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Compute loss
            loss_dict = self.compute_loss(batch)
            loss = loss_dict["loss"] / self.config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            total_loss += loss.item()
            
            # Update weights
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.generator.get_trainable_parameters(),
                    self.config.max_grad_norm,
                )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / self.config.logging_steps
                    self.log({
                        "train/loss": avg_loss,
                        "train/learning_rate": self.scheduler.get_last_lr()[0],
                        **{f"train/{k}": v.item() for k, v in loss_dict.items() if k != "loss"},
                    })
                    total_loss = 0
                    
                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")
                    
                # Evaluation
                if (
                    self.eval_dataloader is not None 
                    and self.global_step % self.config.eval_steps == 0
                ):
                    eval_metrics = self.evaluate()
                    self.log({f"eval/{k}": v for k, v in eval_metrics.items()})
                    self.generator.model.train()
                    
            progress_bar.set_postfix({"loss": loss.item()})
            
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation."""
        self.generator.model.eval()
        
        total_reward = 0
        num_samples = 0
        
        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            prompts = batch["prompt"]
            
            # Generate images
            images = self.generator.generate(prompts)
            
            # Compute rewards
            reward_output = self.reward_model.compute_reward(images, prompts)
            
            total_reward += reward_output.rewards.sum().item()
            num_samples += len(prompts)
            
        return {
            "avg_reward": total_reward / num_samples,
        }
        
    def save_checkpoint(self, name: str) -> None:
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model (LoRA weights if using LoRA)
        if hasattr(self.generator.model, "save_pretrained"):
            self.generator.model.save_pretrained(checkpoint_dir)
            
        # Save optimizer and scheduler
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.current_epoch + 1 if name.startswith("checkpoint-epoch-") else self.current_epoch,
            "config": vars(self.config),
        }, checkpoint_dir / "training_state.pt")
        
        print(f"Saved checkpoint to {checkpoint_dir}")
        
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint_dir = Path(checkpoint_path)
        
        # Load model
        if hasattr(self.generator, "load_model"):
            from peft import PeftModel
            try:
                self.generator.model = PeftModel.from_pretrained(
                    self.generator.model,
                    checkpoint_dir,
                    is_trainable=True,
                )
            except TypeError:
                # Backward compatibility for older PEFT versions
                self.generator.model = PeftModel.from_pretrained(
                    self.generator.model,
                    checkpoint_dir,
                )
                for name, param in self.generator.model.named_parameters():
                    if "lora_" in name:
                        param.requires_grad = True

            trainable_count = sum(
                1 for p in self.generator.model.parameters() if p.requires_grad
            )
            print(f"Trainable parameters after loading checkpoint: {trainable_count}")

            if hasattr(self.generator, "lora_enabled"):
                self.generator.lora_enabled = True

            # Rebuild optimizer/scheduler with current model parameters
            self._setup_optimizer()
            
        # Load training state
        state_path = checkpoint_dir / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path)
            try:
                self.optimizer.load_state_dict(state["optimizer"])
                self.scheduler.load_state_dict(state["scheduler"])
            except Exception as e:
                print(f"Warning: optimizer/scheduler state load failed, using fresh states: {e}")
            self.global_step = state["global_step"]
            self.current_epoch = state["epoch"]
            
        print(f"Loaded checkpoint from {checkpoint_dir}")
