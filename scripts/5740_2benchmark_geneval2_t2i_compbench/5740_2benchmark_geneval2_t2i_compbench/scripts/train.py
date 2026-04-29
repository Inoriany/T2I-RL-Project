#!/usr/bin/env python3
"""
Training Script for T2I-RL
===========================

Main entry point for training text-to-image models with RL.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py model.name=sdxl training.algorithm=reward_weighted

Environment Variables:
    USE_MODELSCOPE: Set to "true" to use ModelScope instead of HuggingFace
    MODELSCOPE_CACHE: Cache directory for ModelScope downloads (default: ./modelscope_models)
"""

import os
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup ModelScope support BEFORE importing transformers
# This will redirect HuggingFace downloads to ModelScope mirrors if USE_MODELSCOPE=true
if os.environ.get("USE_MODELSCOPE", "false").lower() == "true":
    from src.utils import modelscope_helper
    modelscope_helper.setup_modelscope()

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

# ── CUDA performance knobs (must be set before model init) ──
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

from src.models.generators import JanusProGenerator, DiffusionGenerator
from src.models.reward_models import CLIPRewardModel, VLMRewardModel, CompositeRewardModel
from src.training.grpo_trainer import GRPOTrainer, GRPOConfig
from src.training.reward_weighted_trainer import RewardWeightedTrainer, RewardWeightedConfig
from src.data.dataset import T2IDataset


def build_run_tag(cfg: DictConfig) -> str:
    """Suffix folder name: k, LoRA r/alpha, reward type."""
    k = cfg.data.get("max_prompts_per_category")
    k_part = f"k{k}" if k is not None else "kfull"
    r = cfg.model.lora.r
    alpha = cfg.model.lora.alpha
    rt = cfg.reward.type
    return f"{k_part}_r{r}_a{alpha}_{rt}"


def apply_run_output_dir(cfg: DictConfig) -> None:
    """Set cfg.training.output_dir to a subfolder tagged with k, r, alpha, reward type."""
    if not cfg.training.get("append_run_tag_to_output_dir", True):
        return
    base = Path(cfg.training.output_dir)
    tag = build_run_tag(cfg)
    cfg.training.output_dir = str(base / tag)


def setup_generator(cfg: DictConfig):
    """Initialize the image generator based on config."""
    if cfg.model.name == "janus-pro":
        generator = JanusProGenerator(
            model_name_or_path=cfg.model.model_path,
            dtype=getattr(torch, cfg.model.dtype),
            use_flash_attention=cfg.model.use_flash_attention,
            prefer_local_files=cfg.model.get("prefer_local_files", True),
        )
    elif cfg.model.name in ["sdxl", "stable-diffusion"]:
        generator = DiffusionGenerator(
            model_name_or_path=cfg.model.model_path,
            dtype=getattr(torch, cfg.model.dtype),
            prefer_local_files=cfg.model.get("prefer_local_files", True),
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
            prefer_local_files=cfg.reward.clip.get("prefer_local_files", True),
        )
        clip_model.load_model()
        reward_models["clip"] = clip_model
        
    if cfg.reward.type in ["vlm", "composite"]:
        vlm_cfg = cfg.reward.vlm
        use_api = getattr(vlm_cfg, "use_api", False)
        api_model = getattr(vlm_cfg, "api_model", None) or getattr(
            vlm_cfg, "model_name", None
        )
        # Default local model: Qwen2.5-VL-3B-Instruct (fits on 1×H200 alongside Janus-Pro)
        model_name_or_path = getattr(
            vlm_cfg, "model_name_or_path", "Qwen/Qwen2.5-VL-3B-Instruct"
        )
        vlm_model = VLMRewardModel(
            model_name_or_path=model_name_or_path,
            use_api=use_api,
            api_model=api_model,
            dtype=torch.bfloat16,
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
            max_prompts_per_category=cfg.data.get("max_prompts_per_category"),
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.hardware.dataloader_num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=lambda x: {"prompt": [item["prompt"] for item in x]},
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
            pin_memory=torch.cuda.is_available(),
            collate_fn=lambda x: {"prompt": [item["prompt"] for item in x]},
        )
        
    return train_dataloader, eval_dataloader


def get_demo_dataloader(cfg: DictConfig):
    """Get a demo dataloader with sample prompts."""
    from src.evaluation.benchmarks import T2ICompBench
    
    benchmark = T2ICompBench()
    k = cfg.data.get("max_prompts_per_category")
    if k is not None:
        by_cat = benchmark.get_prompts()
        prompts = []
        for _cat, plist in by_cat.items():
            prompts.extend(plist[:k])
    else:
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
        config = GRPOConfig(
            learning_rate=cfg.training.learning_rate,
            num_epochs=cfg.training.num_epochs,
            batch_size=cfg.training.batch_size,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            max_grad_norm=cfg.training.max_grad_norm,
            num_samples_per_prompt=cfg.training.grpo.num_samples_per_prompt,
            temperature=cfg.training.grpo.temperature,
            guidance_scale=cfg.training.grpo.get("guidance_scale", 5.0),
            kl_coef=cfg.training.grpo.kl_coef,
            use_advantage_normalization=cfg.training.grpo.use_advantage_normalization,
            baseline_type=cfg.training.grpo.baseline_type,
            warmup_steps=cfg.training.warmup_steps,
            warmup_ratio=cfg.training.get("warmup_ratio", 0.1),
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
    
    # Apply per-run output directory tag (reward_type, LoRA r/alpha, k).
    # MUST be called before any component setup so all outputs land in the
    # correct subfolder (e.g. outputs/kfull_r16_a32_clip/).
    apply_run_output_dir(cfg)

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
    trainer.save_training_loss_plot()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Checkpoints saved to: {cfg.training.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
