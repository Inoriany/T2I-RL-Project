#!/usr/bin/env python3
"""
Training Script for T2I-RL
===========================

Main entry point for training text-to-image models with RL.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py model.name=sdxl training.algorithm=reward_weighted
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from src.models.generators import JanusProGenerator, DiffusionGenerator
from src.models.reward_models import CLIPRewardModel, VLMRewardModel, CompositeRewardModel
from src.training.grpo_trainer import GRPOTrainer, GRPOConfig
from src.training.reward_weighted_trainer import RewardWeightedTrainer, RewardWeightedConfig
from src.data.dataset import T2IDataset


def setup_generator(cfg: DictConfig):
    """Initialize the image generator based on config."""
    if cfg.model.name == "janus-pro":
        generator = JanusProGenerator(
            model_name_or_path=cfg.model.model_path,
            dtype=getattr(torch, cfg.model.dtype),
            use_flash_attention=cfg.model.use_flash_attention,
        )
    elif cfg.model.name in ["sdxl", "stable-diffusion"]:
        generator = DiffusionGenerator(
            model_name_or_path=cfg.model.model_path,
            dtype=getattr(torch, cfg.model.dtype),
        )
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")
        
    # Load model
    generator.load_model()
    
    # Enable LoRA if configured
    if cfg.model.lora.enabled:
        generator.enable_lora(
            lora_config={
                "r": cfg.model.lora.r,
                "lora_alpha": cfg.model.lora.alpha,
                "lora_dropout": cfg.model.lora.dropout,
                "target_modules": list(cfg.model.lora.target_modules),
            }
        )
        
    return generator


def setup_reward_model(cfg: DictConfig):
    """Initialize reward model(s) based on config."""
    reward_models = {}
    
    if cfg.reward.type in ["clip", "composite"]:
        clip_model = CLIPRewardModel(
            model_name=cfg.reward.clip.model_name,
            pretrained=cfg.reward.clip.pretrained,
        )
        clip_model.load_model()
        reward_models["clip"] = clip_model
        
    if cfg.reward.type in ["vlm", "composite"]:
        vlm_model = VLMRewardModel(
            use_api=cfg.reward.vlm.use_api,
            api_model=cfg.reward.vlm.api_model,
        )
        vlm_model.load_model()
        reward_models["vlm"] = vlm_model
        
    if cfg.reward.type == "composite":
        weights = {
            "clip": cfg.reward.clip.weight,
            "vlm": cfg.reward.vlm.weight,
        }
        return CompositeRewardModel(reward_models, weights)
    elif cfg.reward.type == "clip":
        return reward_models["clip"]
    elif cfg.reward.type == "vlm":
        return reward_models["vlm"]
    else:
        raise ValueError(f"Unknown reward type: {cfg.reward.type}")


def setup_dataloaders(cfg: DictConfig):
    """Setup training and evaluation dataloaders."""
    # Training data
    if cfg.data.train_file:
        train_dataset = T2IDataset(
            data_path=cfg.data.train_file,
            max_samples=cfg.data.max_train_samples,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.hardware.dataloader_num_workers,
        )
    else:
        # Use default prompts for demo
        train_dataloader = get_demo_dataloader(cfg)
        
    # Evaluation data
    eval_dataloader = None
    if cfg.data.eval_file:
        eval_dataset = T2IDataset(
            data_path=cfg.data.eval_file,
            max_samples=cfg.data.max_eval_samples,
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.hardware.dataloader_num_workers,
        )
        
    return train_dataloader, eval_dataloader


def get_demo_dataloader(cfg: DictConfig):
    """Get a demo dataloader with sample prompts."""
    from src.evaluation.benchmarks import T2ICompBench
    
    benchmark = T2ICompBench()
    prompts = benchmark.get_all_prompts()
    
    # Simple dataset wrapper
    class SimpleDataset:
        def __init__(self, prompts):
            self.prompts = prompts
            
        def __len__(self):
            return len(self.prompts)
            
        def __getitem__(self, idx):
            return {"prompt": self.prompts[idx]}
            
    dataset = SimpleDataset(prompts)
    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=lambda x: {"prompt": [item["prompt"] for item in x]},
    )


def setup_trainer(cfg: DictConfig, generator, reward_model, train_dataloader, eval_dataloader):
    """Initialize the trainer based on config."""
    if cfg.training.algorithm == "grpo":
        grpo_cfg = cfg.training.grpo
        num_samples_per_prompt = getattr(grpo_cfg, "num_samples_per_prompt", None)
        if num_samples_per_prompt is None:
            num_samples_per_prompt = getattr(grpo_cfg, "group_size")

        clip_ratio = getattr(grpo_cfg, "clip_ratio", None)
        if clip_ratio is None:
            clip_ratio = getattr(grpo_cfg, "clip_range", 0.2)

        baseline_type = getattr(grpo_cfg, "baseline_type", "mean")
        baseline_aliases = {
            "group_mean": "mean",
            "none": "mean",
        }
        baseline_type = baseline_aliases.get(baseline_type, baseline_type)

        config = GRPOConfig(
            learning_rate=cfg.training.learning_rate,
            num_epochs=cfg.training.num_epochs,
            batch_size=cfg.training.batch_size,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            max_grad_norm=cfg.training.max_grad_norm,
            num_samples_per_prompt=num_samples_per_prompt,
            temperature=grpo_cfg.temperature,
            kl_coef=grpo_cfg.kl_coef,
            clip_ratio=clip_ratio,
            use_advantage_normalization=grpo_cfg.use_advantage_normalization,
            baseline_type=baseline_type,
            warmup_steps=cfg.training.warmup_steps,
            weight_decay=cfg.training.weight_decay,
            save_steps=cfg.training.save_steps,
            eval_steps=cfg.training.eval_steps,
            output_dir=cfg.training.output_dir,
            use_wandb=cfg.logging.use_wandb,
            wandb_project=cfg.logging.wandb_project,
            wandb_run_name=cfg.logging.wandb_run_name,
            logging_steps=cfg.logging.logging_steps,
            bf16=cfg.hardware.bf16,
            fp16=cfg.hardware.fp16,
        )
        
        return GRPOTrainer(
            generator=generator,
            reward_model=reward_model,
            config=config,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            grpo_config=config,
        )
        
    elif cfg.training.algorithm == "reward_weighted":
        config = RewardWeightedConfig(
            learning_rate=cfg.training.learning_rate,
            num_epochs=cfg.training.num_epochs,
            batch_size=cfg.training.batch_size,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            output_dir=cfg.training.output_dir,
        )
        
        return RewardWeightedTrainer(
            generator=generator,
            reward_model=reward_model,
            config=config,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            rw_config=config,
        )
    else:
        raise ValueError(f"Unknown training algorithm: {cfg.training.algorithm}")


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    """Main training function."""
    print("=" * 60)
    print("T2I-RL Training")
    print("=" * 60)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set seed
    torch.manual_seed(cfg.hardware.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.hardware.seed)
        
    # Setup components
    print("\n[1/4] Setting up image generator...")
    generator = setup_generator(cfg)
    
    print("\n[2/4] Setting up reward model...")
    reward_model = setup_reward_model(cfg)
    
    print("\n[3/4] Setting up dataloaders...")
    train_dataloader, eval_dataloader = setup_dataloaders(cfg)
    
    print("\n[4/4] Setting up trainer...")
    trainer = setup_trainer(
        cfg, generator, reward_model, train_dataloader, eval_dataloader
    )
    
    # Start training
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    trainer.train()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Checkpoints saved to: {cfg.training.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
