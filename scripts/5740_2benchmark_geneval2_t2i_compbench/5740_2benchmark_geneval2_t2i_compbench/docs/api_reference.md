# API Reference

This document provides detailed API documentation for all public classes and methods in the T2I-RL framework.

## Table of Contents

1. [Models](#1-models)
   - [Generators](#11-generators)
   - [Reward Models](#12-reward-models)
2. [Training](#2-training)
   - [Base Trainer](#21-base-trainer)
   - [GRPO Trainer](#22-grpo-trainer)
   - [Reward Weighted Trainer](#23-reward-weighted-trainer)
3. [Evaluation](#3-evaluation)
   - [Evaluator](#31-evaluator)
   - [Metrics](#32-metrics)
   - [Benchmarks](#33-benchmarks)
4. [Data](#4-data)
   - [Datasets](#41-datasets)
5. [Configuration](#5-configuration)

---

## 1. Models

### 1.1 Generators

#### `BaseGenerator`

Abstract base class for all T2I generators.

```python
from src.models.generators import BaseGenerator

class BaseGenerator(ABC):
    """Abstract base class for text-to-image generators."""
    
    @abstractmethod
    def generate(
        self,
        prompts: List[str],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        **kwargs
    ) -> List[Image.Image]:
        """
        Generate images from text prompts.
        
        Args:
            prompts: List of text prompts
            num_inference_steps: Number of denoising steps (for diffusion models)
            guidance_scale: Classifier-free guidance scale
            **kwargs: Additional model-specific arguments
            
        Returns:
            List of PIL Images
        """
        pass
    
    @abstractmethod
    def generate_with_logprobs(
        self,
        prompts: List[str],
        **kwargs
    ) -> Tuple[List[Image.Image], torch.Tensor, torch.Tensor]:
        """
        Generate images with log probabilities for RL training.
        
        Args:
            prompts: List of text prompts
            **kwargs: Additional arguments
            
        Returns:
            Tuple of:
                - List of PIL Images
                - Log probabilities tensor [batch, seq_len]
                - Generated token IDs [batch, seq_len]
        """
        pass
    
    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Return iterator over trainable parameters."""
        pass
    
    def save_pretrained(self, path: str) -> None:
        """Save model weights to path."""
        pass
    
    def load_pretrained(self, path: str) -> None:
        """Load model weights from path."""
        pass
```

---

#### `JanusProGenerator`

Janus-Pro-1B generator with autoregressive image generation.

```python
from src.models.generators import JanusProGenerator

generator = JanusProGenerator(
    model_name: str = "deepseek-ai/Janus-Pro-1B",
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    use_lora: bool = False,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_target_modules: List[str] = ["q_proj", "v_proj", "k_proj", "o_proj"]
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | `"deepseek-ai/Janus-Pro-1B"` | HuggingFace model name or path |
| `device` | str | `"cuda"` | Device to load model on |
| `torch_dtype` | dtype | `torch.bfloat16` | Model precision |
| `use_lora` | bool | `False` | Enable LoRA fine-tuning |
| `lora_rank` | int | `16` | LoRA rank |
| `lora_alpha` | int | `32` | LoRA alpha scaling |
| `lora_target_modules` | List[str] | `["q_proj", ...]` | Modules to apply LoRA |

**Methods:**

```python
# Basic generation
images = generator.generate(
    prompts=["a red apple on a blue plate"],
    cfg_weight=5.0,           # CFG guidance weight
    temperature=1.0,          # Sampling temperature
    top_p=None,               # Nucleus sampling (optional)
    num_inference_steps=1     # Always 1 for autoregressive
)

# Generation with log probabilities (for GRPO)
images, log_probs, token_ids = generator.generate_with_logprobs(
    prompts=["a red apple on a blue plate"],
    cfg_weight=5.0,
    temperature=1.0
)
# log_probs shape: [batch_size, 576]  (576 visual tokens)
# token_ids shape: [batch_size, 576]

# Get reference log probs (for KL computation)
ref_log_probs = generator.compute_reference_logprobs(
    prompts=["a red apple on a blue plate"],
    generated_ids=token_ids
)
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `image_size` | int | Output image size (384) |
| `num_image_tokens` | int | Number of visual tokens (576) |
| `model` | nn.Module | The underlying model |
| `processor` | VLChatProcessor | Text/image processor |

---

#### `SDXLGenerator`

Stable Diffusion XL generator (diffusion-based).

```python
from src.models.generators import SDXLGenerator

generator = SDXLGenerator(
    model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
    use_lora: bool = False,
    lora_rank: int = 4
)
```

**Note:** SDXL uses diffusion process, so `generate_with_logprobs` returns an approximation suitable for reward-weighted training but not exact GRPO.

---

### 1.2 Reward Models

#### `BaseRewardModel`

Abstract base class for reward models.

```python
from src.models.reward_models import BaseRewardModel

class BaseRewardModel(ABC):
    """Abstract base class for reward models."""
    
    @abstractmethod
    def compute_reward(
        self,
        images: List[Image.Image],
        prompts: List[str]
    ) -> List[float]:
        """
        Compute rewards for image-prompt pairs.
        
        Args:
            images: List of generated images
            prompts: List of corresponding prompts
            
        Returns:
            List of reward scores (higher is better)
        """
        pass
    
    def compute_reward_batch(
        self,
        images: List[Image.Image],
        prompts: List[str]
    ) -> torch.Tensor:
        """Compute rewards and return as tensor."""
        rewards = self.compute_reward(images, prompts)
        return torch.tensor(rewards)
```

---

#### `CLIPRewardModel`

CLIP-based image-text similarity reward.

```python
from src.models.reward_models import CLIPRewardModel

reward_model = CLIPRewardModel(
    model_name: str = "openai/clip-vit-large-patch14",
    device: str = "cuda"
)

# Compute rewards
rewards = reward_model.compute_reward(
    images=[img1, img2],
    prompts=["a red apple", "a blue car"]
)
# rewards: [0.82, 0.75] (cosine similarities)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | `"openai/clip-vit-large-patch14"` | CLIP model to use |
| `device` | str | `"cuda"` | Device for computation |

---

#### `OpenCLIPRewardModel`

OpenCLIP-based reward with more model options.

```python
from src.models.reward_models import OpenCLIPRewardModel

reward_model = OpenCLIPRewardModel(
    model_name: str = "ViT-bigG-14",
    pretrained: str = "laion2b_s39b_b160k",
    device: str = "cuda"
)
```

---

#### `BLIPRewardModel`

BLIP-based reward using ITM (Image-Text Matching) scores.

```python
from src.models.reward_models import BLIPRewardModel

reward_model = BLIPRewardModel(
    model_name: str = "Salesforce/blip2-itm-vit-g",
    device: str = "cuda"
)

# Returns ITM probability scores
rewards = reward_model.compute_reward(images, prompts)
```

---

#### `VLMRewardModel`

VLM-based reward using GPT-4V or Claude for semantic evaluation.

```python
from src.models.reward_models import VLMRewardModel

reward_model = VLMRewardModel(
    model_name: str = "gpt-4-vision-preview",  # or "claude-3-opus-20240229"
    provider: str = "openai",                   # or "anthropic"
    api_key: str = None,                        # Uses env var if None
    evaluation_prompt: str = None,              # Custom prompt template
    max_retries: int = 3,
    timeout: float = 30.0
)

# Compute rewards with detailed breakdown
rewards = reward_model.compute_reward(images, prompts)

# Get detailed breakdown
breakdown = reward_model.get_reward_breakdown(image, prompt)
# breakdown: {
#     "presence": 0.9,
#     "attributes": 0.8,
#     "spatial": 0.7,
#     "quality": 0.85,
#     "reasoning": "All objects present, colors correct..."
# }
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | `"gpt-4-vision-preview"` | VLM model name |
| `provider` | str | `"openai"` | API provider |
| `api_key` | str | `None` | API key (uses env var if None) |
| `evaluation_prompt` | str | `None` | Custom evaluation prompt |
| `max_retries` | int | `3` | Retry count on API failure |
| `timeout` | float | `30.0` | API timeout in seconds |

**Environment Variables:**
- `OPENAI_API_KEY` for OpenAI
- `ANTHROPIC_API_KEY` for Anthropic

---

## 2. Training

### 2.1 Base Trainer

```python
from src.training.base_trainer import BaseTrainer, TrainingConfig

@dataclass
class TrainingConfig:
    """Training configuration."""
    learning_rate: float = 1e-5
    batch_size: int = 4
    num_epochs: int = 10
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 1000
    output_dir: str = "outputs"
    use_wandb: bool = False
    wandb_project: str = "t2i-rl"
```

```python
class BaseTrainer(ABC):
    """Abstract base trainer class."""
    
    def __init__(
        self,
        config: TrainingConfig,
        generator: BaseGenerator,
        reward_model: BaseRewardModel,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        device: str = "cuda"
    ):
        pass
    
    def train(self) -> Dict[str, float]:
        """
        Run full training loop.
        
        Returns:
            Dictionary of final metrics
        """
        pass
    
    @abstractmethod
    def training_step(
        self,
        batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single training step (must be implemented by subclasses).
        
        Args:
            batch: Dictionary containing "prompts" key
            
        Returns:
            Tuple of (loss tensor, metrics dict)
        """
        pass
    
    def validation_step(
        self,
        batch: Dict[str, Any]
    ) -> Dict[str, float]:
        """Run validation on a batch."""
        pass
    
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        pass
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        pass
```

---

### 2.2 GRPO Trainer

```python
from src.training.grpo_trainer import GRPOTrainer, GRPOConfig

@dataclass
class GRPOConfig(TrainingConfig):
    """GRPO-specific configuration."""
    group_size: int = 4           # Samples per prompt
    kl_coef: float = 0.01         # KL penalty coefficient
    clip_range: float = 0.2       # PPO-style clipping (optional)
    normalize_rewards: bool = True # Normalize within groups
    use_advantage_normalization: bool = True
    baseline_type: str = "group_mean"  # "group_mean", "ema", "none"
    ema_decay: float = 0.99       # For EMA baseline
```

```python
trainer = GRPOTrainer(
    config=grpo_config,
    generator=generator,
    reward_model=reward_model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    device="cuda"
)

# Run training
metrics = trainer.train()

# Or run single step
loss, step_metrics = trainer.training_step({"prompts": ["a red apple"]})
```

**Training Step Details:**

```python
def training_step(self, batch):
    """
    GRPO training step.
    
    1. Generate K images per prompt
    2. Compute rewards for all images
    3. Normalize rewards within each group
    4. Compute policy gradient loss
    5. Add KL penalty
    
    Returns:
        loss: Total loss (GRPO + KL)
        metrics: {
            "loss": float,
            "reward_mean": float,
            "reward_std": float,
            "kl_div": float,
            "entropy": float
        }
    """
```

---

### 2.3 Reward Weighted Trainer

Simpler alternative to GRPO using reward-weighted MLE.

```python
from src.training.reward_weighted_trainer import RewardWeightedTrainer

trainer = RewardWeightedTrainer(
    config=config,
    generator=generator,
    reward_model=reward_model,
    train_dataset=train_dataset
)
```

**Objective:**
$$\mathcal{L} = -\mathbb{E}_{x \sim \pi_\theta}[w(r) \cdot \log \pi_\theta(x|p)]$$

where $w(r)$ is a reward weighting function (e.g., softmax temperature).

---

## 3. Evaluation

### 3.1 Evaluator

```python
from src.evaluation.evaluator import Evaluator

evaluator = Evaluator(
    generator: BaseGenerator,
    metrics: List[str] = ["clip_score", "vlm_score"],
    device: str = "cuda"
)

# Evaluate on prompts
results = evaluator.evaluate(
    prompts=["a red apple", "a blue car"],
    num_samples_per_prompt: int = 1,
    save_images: bool = True,
    output_dir: str = "outputs/eval"
)
# results: {
#     "clip_score": 0.78,
#     "vlm_score": 0.82,
#     "per_prompt": [...]
# }

# Evaluate on benchmark
benchmark_results = evaluator.evaluate_benchmark(
    benchmark_name="t2i_compbench",
    split="test"
)
```

---

### 3.2 Metrics

```python
from src.evaluation.metrics import (
    compute_clip_score,
    compute_fid,
    compute_inception_score,
    compute_vlm_score
)

# CLIP Score
clip_scores = compute_clip_score(
    images: List[Image.Image],
    prompts: List[str],
    model_name: str = "openai/clip-vit-large-patch14"
) -> List[float]

# FID (requires reference images)
fid = compute_fid(
    generated_images: List[Image.Image],
    reference_images: List[Image.Image]
) -> float

# Inception Score
is_mean, is_std = compute_inception_score(
    images: List[Image.Image],
    splits: int = 10
) -> Tuple[float, float]

# VLM Score
vlm_scores = compute_vlm_score(
    images: List[Image.Image],
    prompts: List[str],
    model_name: str = "gpt-4-vision-preview"
) -> List[float]
```

---

### 3.3 Benchmarks

```python
from src.evaluation.benchmarks import (
    T2ICompBenchEvaluator,
    TIFAEvaluator,
    GenEvalEvaluator
)

# T2I-CompBench
t2i_eval = T2ICompBenchEvaluator(
    data_dir: str = "data/benchmarks/t2i_compbench"
)

prompts = t2i_eval.load_prompts(
    category: str = "color_binding"  # or "spatial", "counting", etc.
)

results = t2i_eval.evaluate(
    generator=generator,
    categories=["color_binding", "spatial_relations"]
)
# results: {
#     "color_binding": {"accuracy": 0.85, "blip_score": 0.78},
#     "spatial_relations": {"accuracy": 0.72, "blip_score": 0.65}
# }
```

**Supported Benchmarks:**

| Benchmark | Categories | Metrics |
|-----------|------------|---------|
| T2I-CompBench | color, shape, texture, spatial, non-spatial, complex | BLIP-VQA accuracy |
| TIFA | object, count, color, spatial, other | VQA accuracy |
| GenEval-2 | single object, two objects, counting, colors, position | Detection-based |
| GenAI-Bench | basic, advanced, complex | VLM-based |

---

## 4. Data

### 4.1 Datasets

```python
from src.data.dataset import PromptDataset, CompositionalPromptDataset

# Basic prompt dataset
dataset = PromptDataset(
    prompts_file: str = "data/prompts/train_prompts.json",
    transform: Optional[Callable] = None
)

# Access prompts
prompt = dataset[0]  # Returns {"prompt": "...", "category": "..."}

# Iterate
for batch in DataLoader(dataset, batch_size=4):
    prompts = batch["prompts"]
```

```python
# Compositional prompt dataset with templates
dataset = CompositionalPromptDataset(
    templates_file: str = "data/templates.json",
    vocab_file: str = "data/vocab.json",
    num_samples: int = 10000
)
```

**Template Format:**

```json
{
  "templates": {
    "color_binding": [
      "a {color1} {object1} and a {color2} {object2}",
      "a {color1} {object1} next to a {color2} {object2}"
    ]
  },
  "vocab": {
    "colors": ["red", "blue", "green", "yellow"],
    "objects": ["apple", "car", "cat", "house"]
  }
}
```

---

## 5. Configuration

### 5.1 Hydra Config Loading

```python
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    # Access config values
    lr = cfg.training.learning_rate
    model_name = cfg.model.generator.name
    
    # Create objects from config
    generator = hydra.utils.instantiate(cfg.model.generator)
```

### 5.2 Programmatic Config

```python
from omegaconf import OmegaConf

# Create config
config = OmegaConf.create({
    "training": {
        "learning_rate": 1e-5,
        "batch_size": 4
    },
    "model": {
        "generator": {
            "name": "deepseek-ai/Janus-Pro-1B"
        }
    }
})

# Load from file
config = OmegaConf.load("configs/default.yaml")

# Merge configs
config = OmegaConf.merge(base_config, override_config)

# Convert to dict
config_dict = OmegaConf.to_container(config, resolve=True)
```

---

## Quick Reference

### Common Imports

```python
# Models
from src.models.generators import JanusProGenerator, SDXLGenerator
from src.models.reward_models import CLIPRewardModel, VLMRewardModel

# Training
from src.training.grpo_trainer import GRPOTrainer, GRPOConfig
from src.training.base_trainer import TrainingConfig

# Evaluation
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import compute_clip_score
from src.evaluation.benchmarks import T2ICompBenchEvaluator

# Data
from src.data.dataset import PromptDataset
```

### Minimal Training Example

```python
from src.models.generators import JanusProGenerator
from src.models.reward_models import CLIPRewardModel
from src.training.grpo_trainer import GRPOTrainer, GRPOConfig
from src.data.dataset import PromptDataset

# Setup
generator = JanusProGenerator(use_lora=True)
reward_model = CLIPRewardModel()
dataset = PromptDataset("data/prompts/train_prompts.json")

# Config
config = GRPOConfig(
    learning_rate=1e-5,
    batch_size=2,
    group_size=4,
    num_epochs=5
)

# Train
trainer = GRPOTrainer(config, generator, reward_model, dataset)
metrics = trainer.train()
```

---

*Last updated: April 2026*
