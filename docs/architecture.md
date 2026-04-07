# System Architecture

This document describes the architecture of the T2I-RL framework, including component design, data flow, and extension points.

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Component Details](#2-component-details)
3. [Data Flow](#3-data-flow)
4. [Extension Points](#4-extension-points)
5. [Configuration System](#5-configuration-system)

---

## 1. High-Level Architecture

### 1.1 System Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           T2I-RL Framework                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│  │    Data      │    │    Models    │    │   Training   │                 │
│  │   Module     │    │    Module    │    │    Module    │                 │
│  ├──────────────┤    ├──────────────┤    ├──────────────┤                 │
│  │ • Dataset    │    │ • Generators │    │ • BaseTrainer│                 │
│  │ • Prompts    │    │ • Rewards    │    │ • GRPO       │                 │
│  │ • Loaders    │    │ • LoRA       │    │ • RW Trainer │                 │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                 │
│         │                   │                   │                          │
│         └───────────────────┼───────────────────┘                          │
│                             │                                              │
│                             ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                        Evaluation Module                              │ │
│  ├──────────────────────────────────────────────────────────────────────┤ │
│  │  • Benchmarks (T2I-CompBench, TIFA, GenEval)                         │ │
│  │  • Metrics (CLIP Score, VLM Score, FID)                              │ │
│  │  • Evaluator (orchestration)                                          │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Module Dependencies

```
                    ┌─────────────┐
                    │   configs/  │
                    │  (Hydra)    │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │  scripts/   │ │    src/     │ │ notebooks/  │
    │  train.py   │ │  (core)     │ │ quickstart  │
    │ evaluate.py │ │             │ │             │
    └─────────────┘ └─────────────┘ └─────────────┘
```

---

## 2. Component Details

### 2.1 Models Module (`src/models/`)

#### 2.1.1 Generator Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      BaseGenerator (Abstract)                        │
├─────────────────────────────────────────────────────────────────────┤
│  + generate(prompts, **kwargs) -> List[Image]                       │
│  + generate_with_logprobs(prompts, **kwargs) -> (images, logp, ids) │
│  + get_trainable_parameters() -> Iterator[Parameter]                │
│  + save_pretrained(path) / load_pretrained(path)                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
┌─────────────────────────────────┐  ┌─────────────────────────────────┐
│       JanusProGenerator         │  │       SDXLGenerator             │
├─────────────────────────────────┤  ├─────────────────────────────────┤
│ • Autoregressive generation     │  │ • Diffusion-based generation    │
│ • 576 visual tokens (24×24)     │  │ • Latent diffusion              │
│ • CFG support                   │  │ • ControlNet support            │
│ • Native LoRA integration       │  │ • LoRA/DreamBooth               │
└─────────────────────────────────┘  └─────────────────────────────────┘
```

**JanusProGenerator Details:**

```python
class JanusProGenerator:
    """
    Janus-Pro-1B: Unified multimodal model for T2I generation.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Janus-Pro Model                           │
    ├─────────────────────────────────────────────────────────────┤
    │  Text Encoder    │  Language Model  │  Vision Decoder       │
    │  (LLM backbone)  │  (Autoregressive)│  (VQ-VAE)            │
    │                  │                  │                       │
    │  prompt ──────▶  │  ──▶ tokens ──▶  │  ──▶ image           │
    │                  │  (576 tokens)    │  (384×384)           │
    └─────────────────────────────────────────────────────────────┘
    
    Generation Process:
    1. Format prompt with conversation template
    2. Encode text with LLM
    3. Autoregressively generate 576 visual tokens
    4. Apply CFG: logits = logits_cond + cfg_weight * (logits_cond - logits_uncond)
    5. Decode tokens to image via VQ-VAE decoder
    """
```

#### 2.1.2 Reward Model Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      BaseRewardModel (Abstract)                      │
├─────────────────────────────────────────────────────────────────────┤
│  + compute_reward(images, prompts) -> List[float]                   │
│  + compute_reward_batch(images, prompts) -> Tensor                  │
│  + get_reward_breakdown(image, prompt) -> Dict[str, float]          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            ▼                       ▼                       ▼
┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
│  CLIPRewardModel  │  │  BLIPRewardModel  │  │  VLMRewardModel   │
├───────────────────┤  ├───────────────────┤  ├───────────────────┤
│ • Cosine sim      │  │ • ITM score       │  │ • API-based       │
│ • Fast (~10ms)    │  │ • ITC score       │  │ • GPT-4V/Claude   │
│ • Global align    │  │ • Medium speed    │  │ • Compositional   │
└───────────────────┘  └───────────────────┘  └───────────────────┘
```

### 2.2 Training Module (`src/training/`)

#### 2.2.1 Trainer Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BaseTrainer (Abstract)                        │
├─────────────────────────────────────────────────────────────────────┤
│  Core Methods:                                                       │
│  + train() -> Dict[str, float]                                      │
│  + training_step(batch) -> Tuple[Tensor, Dict]                      │
│  + validation_step(batch) -> Dict                                   │
│  + save_checkpoint(path) / load_checkpoint(path)                    │
│                                                                      │
│  Hooks (override in subclasses):                                    │
│  + on_train_start() / on_train_end()                                │
│  + on_epoch_start() / on_epoch_end()                                │
│  + on_step_start() / on_step_end()                                  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
┌─────────────────────────────────┐  ┌─────────────────────────────────┐
│         GRPOTrainer             │  │    RewardWeightedTrainer        │
├─────────────────────────────────┤  ├─────────────────────────────────┤
│ • Group sampling (K per prompt) │  │ • Single sample per prompt      │
│ • Relative reward normalization │  │ • Weighted MLE objective        │
│ • KL regularization             │  │ • Simpler, faster               │
│ • PPO-style clipping (optional) │  │ • Higher variance               │
└─────────────────────────────────┘  └─────────────────────────────────┘
```

#### 2.2.2 GRPO Training Loop

```
┌─────────────────────────────────────────────────────────────────────┐
│                     GRPO Training Step                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input: Batch of prompts [p₁, p₂, ..., pB]                          │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Step 1: Generate K samples per prompt                        │    │
│  │                                                              │    │
│  │  for each pᵢ:                                                │    │
│  │    {x¹ᵢ, x²ᵢ, ..., xᴷᵢ} ~ π_θ(·|pᵢ)                         │    │
│  │    record log π_θ(xᵏᵢ|pᵢ) for each k                        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Step 2: Compute rewards                                      │    │
│  │                                                              │    │
│  │  for each xᵏᵢ:                                               │    │
│  │    rᵏᵢ = R(xᵏᵢ, pᵢ)  // VLM or CLIP reward                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Step 3: Normalize rewards within groups                      │    │
│  │                                                              │    │
│  │  for each pᵢ:                                                │    │
│  │    μᵢ = mean({r¹ᵢ, ..., rᴷᵢ})                                │    │
│  │    σᵢ = std({r¹ᵢ, ..., rᴷᵢ})                                 │    │
│  │    r̂ᵏᵢ = (rᵏᵢ - μᵢ) / (σᵢ + ε)                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Step 4: Compute loss and update                              │    │
│  │                                                              │    │
│  │  L_GRPO = -Σᵢ Σₖ r̂ᵏᵢ · log π_θ(xᵏᵢ|pᵢ)                      │    │
│  │  L_KL = Σᵢ Σₖ [log π_θ(xᵏᵢ|pᵢ) - log π_ref(xᵏᵢ|pᵢ)]         │    │
│  │  L_total = L_GRPO + β · L_KL                                 │    │
│  │                                                              │    │
│  │  θ ← θ - α · ∇_θ L_total                                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Output: loss, {reward_mean, reward_std, kl_div, ...}               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Evaluation Module (`src/evaluation/`)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Evaluation Pipeline                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │  Benchmark  │    │  Generator  │    │   Metrics   │             │
│  │  Prompts    │───▶│  Inference  │───▶│  Compute    │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│                                                │                     │
│                                                ▼                     │
│  Supported Benchmarks:                   ┌─────────────┐            │
│  • T2I-CompBench                         │   Results   │            │
│  • TIFA                                  │   Report    │            │
│  • GenEval-2                             └─────────────┘            │
│  • GenAI-Bench                                                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow

### 3.1 Training Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Training Data Flow                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  data/prompts/          src/data/           src/training/                  │
│  ┌─────────────┐       ┌─────────────┐     ┌─────────────┐                 │
│  │train_prompts│──────▶│PromptDataset│────▶│ DataLoader  │                 │
│  │   .json     │       │             │     │             │                 │
│  └─────────────┘       └─────────────┘     └──────┬──────┘                 │
│                                                   │                         │
│                                                   │ batch of prompts        │
│                                                   ▼                         │
│                                            ┌─────────────┐                 │
│                                            │  GRPOTrainer│                 │
│                                            │             │                 │
│                                            └──────┬──────┘                 │
│                                                   │                         │
│                        ┌──────────────────────────┼──────────────────────┐ │
│                        │                          │                      │ │
│                        ▼                          ▼                      ▼ │
│                 ┌─────────────┐           ┌─────────────┐        ┌──────┐ │
│                 │  Generator  │           │RewardModel  │        │ LoRA │ │
│                 │  (forward)  │──images──▶│  (score)    │        │Update│ │
│                 └─────────────┘           └─────────────┘        └──────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Evaluation Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Evaluation Data Flow                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Benchmark               Generator              Metrics                     │
│  ┌─────────────┐        ┌─────────────┐       ┌─────────────┐             │
│  │T2I-CompBench│───────▶│  Generate   │──────▶│  Compute    │             │
│  │   prompts   │        │   images    │       │   scores    │             │
│  └─────────────┘        └─────────────┘       └──────┬──────┘             │
│                                                      │                     │
│                                                      ▼                     │
│                              ┌────────────────────────────────────────┐   │
│                              │           Results Aggregation          │   │
│                              ├────────────────────────────────────────┤   │
│                              │  Category        │  Score              │   │
│                              │  ───────────────────────────────────   │   │
│                              │  color_binding   │  0.85               │   │
│                              │  spatial_rel     │  0.72               │   │
│                              │  counting        │  0.68               │   │
│                              │  overall         │  0.76               │   │
│                              └────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Extension Points

### 4.1 Adding a New Generator

```python
# src/models/generators.py

from src.models.generators import BaseGenerator

class MyCustomGenerator(BaseGenerator):
    """Custom T2I generator implementation."""
    
    def __init__(self, model_name: str, device: str, **kwargs):
        super().__init__()
        # Initialize your model here
        self.model = load_my_model(model_name)
        self.device = device
    
    def generate(
        self,
        prompts: List[str],
        num_inference_steps: int = 50,
        **kwargs
    ) -> List[Image.Image]:
        """Generate images from prompts."""
        # Your generation logic
        pass
    
    def generate_with_logprobs(
        self,
        prompts: List[str],
        **kwargs
    ) -> Tuple[List[Image.Image], torch.Tensor, torch.Tensor]:
        """Generate with log probabilities for RL training."""
        # Required for GRPO training
        pass
```

### 4.2 Adding a New Reward Model

```python
# src/models/reward_models.py

from src.models.reward_models import BaseRewardModel

class MyCustomReward(BaseRewardModel):
    """Custom reward model implementation."""
    
    def __init__(self, **kwargs):
        super().__init__()
        # Initialize your reward model
    
    def compute_reward(
        self,
        images: List[Image.Image],
        prompts: List[str]
    ) -> List[float]:
        """Compute rewards for image-prompt pairs."""
        rewards = []
        for image, prompt in zip(images, prompts):
            score = self._evaluate(image, prompt)
            rewards.append(score)
        return rewards
```

### 4.3 Adding a New Trainer

```python
# src/training/my_trainer.py

from src.training.base_trainer import BaseTrainer

class MyCustomTrainer(BaseTrainer):
    """Custom training algorithm."""
    
    def training_step(
        self,
        batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Single training step."""
        prompts = batch["prompts"]
        
        # Your training logic here
        loss = self._compute_loss(prompts)
        metrics = {"loss": loss.item()}
        
        return loss, metrics
```

### 4.4 Adding a New Benchmark

```python
# src/evaluation/benchmarks.py

from src.evaluation.benchmarks import BaseBenchmark

class MyBenchmark(BaseBenchmark):
    """Custom evaluation benchmark."""
    
    def load_prompts(self) -> List[Dict]:
        """Load benchmark prompts."""
        # Return list of {"prompt": str, "category": str, ...}
        pass
    
    def evaluate(
        self,
        images: List[Image.Image],
        prompts: List[Dict]
    ) -> Dict[str, float]:
        """Evaluate generated images."""
        # Return {"metric1": score, "metric2": score, ...}
        pass
```

---

## 5. Configuration System

### 5.1 Hydra Configuration

We use [Hydra](https://hydra.cc/) for configuration management:

```
configs/
├── default.yaml          # Main config with defaults
├── model/
│   ├── janus_pro.yaml    # Janus-Pro specific config
│   └── sdxl.yaml         # SDXL specific config
├── training/
│   ├── grpo.yaml         # GRPO algorithm config
│   └── reward_weighted.yaml
├── reward/
│   ├── clip.yaml         # CLIP reward config
│   └── vlm.yaml          # VLM reward config
└── experiment/
    ├── baseline.yaml     # Baseline experiment
    └── full_vlm.yaml     # Full VLM training
```

### 5.2 Configuration Structure

```yaml
# configs/default.yaml

defaults:
  - model: janus_pro
  - training: grpo
  - reward: clip
  - _self_

project:
  name: "t2i-rl"
  seed: 42
  output_dir: "outputs"

model:
  generator:
    name: "deepseek-ai/Janus-Pro-1B"
    use_lora: true
    lora_rank: 16
    torch_dtype: "bfloat16"

training:
  num_epochs: 10
  batch_size: 4
  learning_rate: 1e-5
  gradient_accumulation_steps: 4
  
  grpo:
    group_size: 4
    kl_coef: 0.01
    normalize_rewards: true

reward:
  type: "clip"  # or "vlm"
  clip:
    model_name: "openai/clip-vit-large-patch14"

evaluation:
  benchmarks:
    - "t2i_compbench"
  metrics:
    - "clip_score"
    - "vlm_score"
```

### 5.3 Override Examples

```bash
# Override single values
python scripts/train.py training.batch_size=8

# Override nested values
python scripts/train.py model.generator.lora_rank=32

# Use different config files
python scripts/train.py --config-name=experiment/full_vlm

# Multi-run with different seeds
python scripts/train.py -m project.seed=1,2,3,4,5
```

---

## File Structure Summary

```
T2I-RL-Project/
├── configs/
│   └── default.yaml              # Hydra configuration
├── data/
│   └── prompts/
│       ├── train_prompts.json    # Training prompts
│       └── val_prompts.json      # Validation prompts
├── docs/
│   ├── architecture.md           # This file
│   └── methodology.md            # Algorithm details
├── notebooks/
│   └── quickstart.ipynb          # Usage tutorial
├── scripts/
│   ├── train.py                  # Training entry point
│   └── evaluate.py               # Evaluation entry point
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py            # Dataset classes
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── benchmarks.py         # Benchmark loaders
│   │   ├── evaluator.py          # Evaluation orchestration
│   │   └── metrics.py            # Metric implementations
│   ├── models/
│   │   ├── __init__.py
│   │   ├── generators.py         # T2I generators
│   │   └── reward_models.py      # Reward models
│   ├── training/
│   │   ├── __init__.py
│   │   ├── base_trainer.py       # Base trainer class
│   │   ├── grpo_trainer.py       # GRPO implementation
│   │   └── reward_weighted_trainer.py
│   └── utils/
│       └── __init__.py           # Utilities
├── .gitignore
├── environment.yml               # Conda environment
├── requirements.txt              # Pip requirements
├── setup.py                      # Package setup
└── README.md                     # Project overview
```

---

*Last updated: April 2026*
