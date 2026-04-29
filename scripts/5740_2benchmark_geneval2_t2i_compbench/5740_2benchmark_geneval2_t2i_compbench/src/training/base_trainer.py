"""
Base Trainer
=============

Abstract base class for all T2I-RL trainers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Callable, Tuple
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
    warmup_ratio: float = 0.1   # Fraction of total_steps used for warmup (overrides warmup_steps when > 0)
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

        # Logged train/loss, train/reward_mean, train/reward_std, train/kl_div points for plotting
        self.train_loss_history: List[Tuple[int, float]] = []
        self.train_reward_history: List[Tuple[int, float]] = []
        self.train_reward_std_history: List[Tuple[int, float]] = []
        self.train_kl_history: List[Tuple[int, float]] = []
        
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
        
        total_steps = max(
            1,
            len(self.train_dataloader)
            * self.config.num_epochs
            // self.config.gradient_accumulation_steps,
        )

        # Derive warmup_steps from ratio when warmup_ratio > 0; otherwise use the
        # static warmup_steps value from config.
        if self.config.warmup_ratio > 0:
            num_warmup_steps = max(1, int(total_steps * self.config.warmup_ratio))
        else:
            num_warmup_steps = self.config.warmup_steps
        print(
            f"[Optimizer] total_steps={total_steps}  "
            f"warmup_steps={num_warmup_steps} "
            f"(ratio={self.config.warmup_ratio:.2f})"
        )

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
        )
        
    def _setup_wandb(self) -> None:
        """Setup Weights & Biases logging."""
        try:
            import wandb
            from dataclasses import asdict

            # Convert config to dict, excluding non-serializable values (tensors, etc.)
            config_dict = {}
            for k, v in vars(self.config).items():
                # Skip private attributes and non-serializable types
                if k.startswith('_'):
                    continue
                if isinstance(v, (int, float, str, bool, type(None))):
                    config_dict[k] = v
                elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float, str, bool)) for x in v):
                    config_dict[k] = v
                else:
                    # Convert other types to string representation
                    config_dict[k] = str(v)

            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=config_dict,
            )
            self.logger = wandb
        except ImportError:
            print("wandb not installed, skipping W&B logging")
            
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics."""
        step = step or self.global_step

        if "train/loss" in metrics and isinstance(metrics["train/loss"], (int, float)):
            self.train_loss_history.append((step, float(metrics["train/loss"])))
        if "train/reward_mean" in metrics and isinstance(metrics["train/reward_mean"], (int, float)):
            self.train_reward_history.append((step, float(metrics["train/reward_mean"])))
        if "train/reward_std" in metrics and isinstance(metrics["train/reward_std"], (int, float)):
            self.train_reward_std_history.append((step, float(metrics["train/reward_std"])))
        if "train/kl_div" in metrics and isinstance(metrics["train/kl_div"], (int, float)):
            self.train_kl_history.append((step, float(metrics["train/kl_div"])))
        
        if self.logger is not None:
            self.logger.log(metrics, step=step)
            
        # Also print to console
        metrics_str = " | ".join([f"{k}: {v:.6g}" for k, v in metrics.items() if isinstance(v, (int, float))])
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

    def save_training_loss_plot(self, filename: str = "training_curves.png") -> None:
        """Save loss / reward_mean / reward_std / kl_div curves as a single multi-subplot figure."""
        import json
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir = Path(self.config.output_dir)
        has_loss       = bool(self.train_loss_history)
        has_reward     = bool(self.train_reward_history)
        has_reward_std = bool(self.train_reward_std_history)
        has_kl         = bool(self.train_kl_history)

        if not (has_loss or has_reward or has_reward_std or has_kl):
            print("No training metrics logged; skip curve plot.")
            return

        n_plots = int(has_loss) + int(has_reward) + int(has_reward_std) + int(has_kl)
        fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]

        ax_idx = 0
        if has_loss:
            steps, losses = zip(*self.train_loss_history)
            axes[ax_idx].plot(steps, losses, color="steelblue", label="train/loss")
            axes[ax_idx].set_xlabel("step")
            axes[ax_idx].set_ylabel("loss")
            axes[ax_idx].set_title("Training Loss")
            axes[ax_idx].legend()
            axes[ax_idx].grid(True, alpha=0.3)
            ax_idx += 1

        if has_reward:
            r_steps, rewards = zip(*self.train_reward_history)
            axes[ax_idx].plot(r_steps, rewards, color="darkorange", label="train/reward_mean")
            axes[ax_idx].set_xlabel("step")
            axes[ax_idx].set_ylabel("reward")
            axes[ax_idx].set_title("Training Reward Mean")
            axes[ax_idx].legend()
            axes[ax_idx].grid(True, alpha=0.3)
            ax_idx += 1

        if has_reward_std:
            rs_steps, reward_stds = zip(*self.train_reward_std_history)
            axes[ax_idx].plot(rs_steps, reward_stds, color="crimson", label="train/reward_std")
            axes[ax_idx].set_xlabel("step")
            axes[ax_idx].set_ylabel("reward std")
            axes[ax_idx].set_title("Training Reward Std")
            axes[ax_idx].legend()
            axes[ax_idx].grid(True, alpha=0.3)
            ax_idx += 1

        if has_kl:
            k_steps, kls = zip(*self.train_kl_history)
            axes[ax_idx].plot(k_steps, kls, color="seagreen", label="train/kl_div")
            axes[ax_idx].set_xlabel("step")
            axes[ax_idx].set_ylabel("KL divergence")
            axes[ax_idx].set_title("Training KL Divergence")
            axes[ax_idx].legend()
            axes[ax_idx].grid(True, alpha=0.3)

        plt.tight_layout()
        path = out_dir / filename
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved training curves to {path}")

        # Persist raw data for later analysis
        json_data: dict = {}
        if has_loss:
            steps, losses = zip(*self.train_loss_history)
            json_data["loss_steps"] = list(steps)
            json_data["losses"]     = list(losses)
        if has_reward:
            r_steps, rewards = zip(*self.train_reward_history)
            json_data["reward_steps"] = list(r_steps)
            json_data["reward_means"] = list(rewards)
        if has_reward_std:
            rs_steps, reward_stds = zip(*self.train_reward_std_history)
            json_data["reward_std_steps"] = list(rs_steps)
            json_data["reward_stds"]      = list(reward_stds)
        if has_kl:
            k_steps, kls = zip(*self.train_kl_history)
            json_data["kl_steps"] = list(k_steps)
            json_data["kl_divs"]  = list(kls)

        json_path = out_dir / "training_loss.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"Saved training data to {json_path}")
