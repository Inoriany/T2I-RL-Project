# T2I-RL: Text-to-Image Generation with Reinforcement Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-available-green.svg)](docs/)

> **AIMS 5740 Final Project** - Text-to-Image Generation with Understanding-based Reward

A research framework for training text-to-image models using reinforcement learning with vision-language model (VLM) based rewards. This project focuses on improving compositional generation capabilities including object presence, attribute binding, counting accuracy, and spatial relationships.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Team Workflow](#team-workflow)
- [Documentation](#documentation)
- [Training](#training-algorithms)
- [Evaluation](#evaluation-benchmarks)
- [Results](#results)
- [References](#references)

## Overview

Text-to-image models often generate realistic images but **hallucinate** objects, attributes, or spatial relations. This project aligns generation using **image understanding as the reward signal**.

### The Problem

| Prompt | Common Failures |
|--------|----------------|
| "a **red** apple and a **blue** cup" | Colors swapped, wrong colors |
| "**three** cats on a sofa" | Wrong count (2 or 4 cats) |
| "a bird **above** the tree" | Bird beside/below tree |
| "a **wooden** table with **metal** legs" | Wrong textures/materials |

### Our Solution

We use reinforcement learning to fine-tune T2I models with **understanding-based rewards**:

1. **Generate** images from compositional prompts
2. **Evaluate** using VLMs that understand object presence, attributes, and relations
3. **Optimize** the generator to increase reward (better compositional accuracy)

### Key Features

- **Multiple Generator Support**: Janus-Pro-1B (unified MLLM), Stable Diffusion XL
- **Flexible Reward Models**: CLIP-based similarity, VLM-based evaluation (GPT-4V, Claude)
- **RL Training Algorithms**: GRPO (Group Relative Policy Optimization), Reward-Weighted MLE
- **Comprehensive Evaluation**: T2I-CompBench, TIFA, GenEval-2, GenAI-Bench
- **Efficient Fine-tuning**: LoRA support for memory-efficient training

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        T2I-RL Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Text Prompt                                                   │
│       │                                                         │
│       ▼                                                         │
│   ┌─────────────────┐                                          │
│   │ Image Generator │  ← Janus-Pro / SDXL / Flow Models        │
│   │   (with LoRA)   │                                          │
│   └────────┬────────┘                                          │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────┐     ┌─────────────────┐                  │
│   │  Generated      │────▶│  Reward Model   │                  │
│   │    Image        │     │  (CLIP + VLM)   │                  │
│   └─────────────────┘     └────────┬────────┘                  │
│                                    │                            │
│                                    ▼                            │
│                           ┌─────────────────┐                  │
│                           │  Scalar Reward  │                  │
│                           └────────┬────────┘                  │
│                                    │                            │
│                                    ▼                            │
│                           ┌─────────────────┐                  │
│                           │   RL Update     │                  │
│                           │  (GRPO / RW)    │                  │
│                           └─────────────────┘                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Reward Model Architectures

| CLIP-based RM | VLM-based RM |
|---------------|--------------|
| Image → Image Encoder → Image Emb | Image + Text + Instruction |
| Text → Text Encoder → Text Emb | ↓ |
| ↓ | Vision-Language Model |
| Cosine Similarity | ↓ |
| ↓ | Hidden State → Regression Head |
| **Reward Score** | **Reward Score** |

## Project Structure

```
T2I-RL-Project/
├── configs/                    # Hydra configuration files
│   └── default.yaml           # Default training config
├── src/                       # Source code
│   ├── models/               
│   │   ├── generators.py      # Image generators (Janus-Pro, SDXL)
│   │   └── reward_models.py   # Reward models (CLIP, VLM)
│   ├── training/             
│   │   ├── base_trainer.py    # Base trainer class
│   │   ├── grpo_trainer.py    # GRPO algorithm
│   │   └── reward_weighted_trainer.py
│   ├── evaluation/           
│   │   ├── evaluator.py       # Main evaluator
│   │   ├── metrics.py         # Evaluation metrics
│   │   └── benchmarks.py      # Benchmark implementations
│   └── data/                 
│       └── dataset.py         # Dataset classes
├── scripts/                   
│   ├── train.py               # Training script
│   └── evaluate.py            # Evaluation script
├── experiments/               # Experiment outputs
├── docs/                      # Documentation
│   ├── methodology.md         # Detailed algorithm explanation
│   ├── architecture.md        # System architecture
│   └── api_reference.md       # API documentation
├── notebooks/                 # Jupyter notebooks
│   └── quickstart.ipynb       # Getting started tutorial
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment
└── README.md
```

## Team Workflow

> **Quick reference for all team members.** Full training is run on Google Colab (free T4/A100).  
> All notebooks are self-contained — just fill in your API key in Step 0 and run all cells.

### Member A — Full Training (Main Run)

| Item | Value |
|------|-------|
| Notebook | `notebooks/colab_full_training.ipynb` |
| Config | `configs/experiment/grpo_siliconflow.yaml` |
| Data | `data/prompts/train_prompts_large.json` (1029 prompts) |
| Reward | CLIP (0.3) + SiliconFlow Qwen2.5-VL-32B (0.7) |
| Est. time | ~10-14h on 24GB GPU |
| Est. cost | ~¥15-20 API |

**Required:** Set `SILICONFLOW_API_KEY` in Step 0.  
Get your key at [cloud.siliconflow.cn](https://cloud.siliconflow.cn/).

---

### Member B — Ablation Experiments

Run the same `colab_full_training.ipynb` notebook, but **change `OUTPUT_DIR` and `--config-name`** in Step 0:

| Ablation | Config | Change vs. Baseline |
|----------|--------|---------------------|
| `grpo_full_lora_r8` | `ablation_lora_r8` | LoRA rank 8 (fewer params) |
| `grpo_full_lora_r32` | `ablation_lora_r32` | LoRA rank 32 (more params) |
| `grpo_full_clip_only` | `ablation_clip_only` | CLIP reward only (no API cost) |
| `grpo_full_vlm_only` | `ablation_vlm_only` | VLM reward only (no CLIP gate) |
| `grpo_full_5ep` | `ablation_5ep` | 5 epochs instead of 10 |

In Step 0 of the notebook, change:
```python
OUTPUT_DIR = "/content/drive/MyDrive/T2I-RL/outputs/grpo_full_lora_r8"  # match ablation name
```
And in Step 6 (training cell), change `--config-name`:
```bash
python scripts/train.py --config-name experiment/ablation_lora_r8
```

After each run, collect benchmark scores using `notebooks/evaluation_and_analysis.ipynb`.

---

### Member C — Qualitative Analysis

Use `notebooks/evaluation_and_analysis.ipynb` with checkpoints from Member A's run.

Deliverables:
- Before/after image grids (baseline Janus-Pro vs. trained checkpoint)
- Error taxonomy table: missing objects / wrong count / wrong attribute / wrong relation
- Failure mode analysis (cases where reward is high but image is wrong)
- Reward signal case study (show reward curve vs. visual quality)

---

### Member D — TIFA & GenAI-Bench Evaluation

Reuse Member B's Colab environment. Run:
```bash
python scripts/evaluate.py \
    --checkpoint /path/to/checkpoint \
    --benchmark tifa \
    --output_dir outputs/eval_tifa

python scripts/evaluate.py \
    --checkpoint /path/to/checkpoint \
    --benchmark genai_bench \
    --output_dir outputs/eval_genai
```

Also write **Background & Related Work** sections of the final report.

---

### SiliconFlow API Notes

- **Recommended model:** `Qwen/Qwen2.5-VL-32B-Instruct` (~¥1.89/M tokens)
- **Free alternative:** `THUDM/GLM-4.1V-9B-Thinking` (free tier, lower quality)
- Switch models by changing `VLM_MODEL` in Step 0 of any notebook
- Monitor usage at [cloud.siliconflow.cn/account/billing](https://cloud.siliconflow.cn/account/billing)

---



| Document | Description |
|----------|-------------|
| [Methodology](docs/methodology.md) | Detailed explanation of GRPO algorithm, reward design, theoretical analysis |
| [Architecture](docs/architecture.md) | System architecture, component design, extension guide |
| [API Reference](docs/api_reference.md) | Complete API documentation for all classes and methods |
| [Quickstart Notebook](notebooks/quickstart.ipynb) | Interactive tutorial with code examples |

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (for GPU training)
- 24GB+ GPU memory (for Janus-Pro-1B with LoRA)

### Setup

```bash
# Clone the repository
git clone https://github.com/Inoriany/T2I-RL-Project.git
cd T2I-RL-Project

# Option 1: Using conda (recommended)
conda env create -f environment.yml
conda activate t2i-rl

# Option 2: Using pip
pip install -r requirements.txt
```

### API Keys (for VLM-based rewards)

```bash
# For GPT-4V based rewards
export OPENAI_API_KEY="your-api-key"

# For Claude based rewards
export ANTHROPIC_API_KEY="your-api-key"
```

## Quick Start

### Basic Usage

```python
from src.models.generators import JanusProGenerator
from src.models.reward_models import CLIPRewardModel

# Initialize generator
generator = JanusProGenerator(
    model_name="deepseek-ai/Janus-Pro-1B",
    use_lora=True,
    device="cuda"
)

# Generate images
images = generator.generate(
    prompts=["a red apple on a blue plate"],
    cfg_weight=5.0
)

# Compute rewards
reward_model = CLIPRewardModel()
rewards = reward_model.compute_reward(images, prompts)
```

### Training

```bash
# Train with default config (Janus-Pro + GRPO)
python scripts/train.py

# Train with custom config
python scripts/train.py \
    model.generator.name=deepseek-ai/Janus-Pro-1B \
    training.algorithm=grpo \
    training.learning_rate=1e-5 \
    training.num_epochs=10 \
    reward.type=vlm
```

### Evaluation

```bash
# Evaluate on all benchmarks
python scripts/evaluate.py --checkpoint outputs/checkpoint-1000

# Evaluate on specific benchmark
python scripts/evaluate.py \
    --benchmark t2i_compbench \
    --save_images \
    --output_dir outputs/eval
```

## Training Algorithms

### GRPO (Group Relative Policy Optimization)

Our primary training algorithm, adapted from [DeepSeekMath](https://arxiv.org/abs/2402.03300) for image generation:

```
For each prompt p:
  1. Generate K images: {x₁, ..., xₖ} ~ πθ(·|p)
  2. Compute rewards: {r₁, ..., rₖ}
  3. Normalize within group: r̂ᵢ = (rᵢ - mean) / std
  4. Update: L = -Σ r̂ᵢ · log πθ(xᵢ|p) + β · KL(πθ || πref)
```

**Key Advantages:**
- No separate value network needed
- All K samples contribute to learning
- Group normalization reduces variance
- KL regularization prevents reward hacking

```yaml
# Key hyperparameters (configs/default.yaml)
training:
  grpo:
    group_size: 4              # Samples per prompt
    kl_coef: 0.01              # KL penalty weight
    normalize_rewards: true    # Group normalization
    clip_range: 0.2            # Optional PPO-style clipping
```

### Reward-Weighted MLE

Simpler alternative that weights MLE loss by rewards:

```
L = -E[w(r) · log p(x|prompt)]
```

Where w(r) is a weighting function (e.g., softmax with temperature).

See [docs/methodology.md](docs/methodology.md) for detailed algorithm explanations.

## Evaluation Benchmarks

| Benchmark | Categories | Metrics | Description |
|-----------|------------|---------|-------------|
| **T2I-CompBench** | Color, Shape, Texture, Spatial, Non-spatial | BLIP-VQA accuracy | Compositional generation evaluation |
| **TIFA** | Object, Count, Color, Spatial, Other | VQA accuracy | Text-image faithfulness |
| **GenEval-2** | Single/Two objects, Counting, Colors, Position | Detection-based | General T2I evaluation |
| **GenAI-Bench** | Basic, Advanced, Complex | VLM-based | Comprehensive AI benchmark |

### Evaluation Categories

```
Compositional Challenges:
├── Attribute Binding
│   ├── Color: "a red apple and a blue cup"
│   ├── Shape: "a round table and a square chair"
│   └── Texture: "a wooden desk and a metal lamp"
├── Spatial Relations
│   ├── On/Under: "a cat on top of a box"
│   ├── Left/Right: "a dog to the left of a tree"
│   └── Front/Behind: "a car in front of a house"
├── Counting
│   └── "three birds flying in the sky"
└── Complex Scenes
    └── "a chef cooking in a kitchen with pots on a stove"
```

### Error Taxonomy

The evaluation automatically categorizes failures:

| Error Type | Example | Detection Method |
|------------|---------|------------------|
| Missing Objects | "cat and dog" → only cat visible | Object detection |
| Wrong Count | "three apples" → 2 apples | Counting |
| Wrong Attribute | "red car" → blue car | Attribute classification |
| Wrong Relation | "cat on box" → cat beside box | Spatial reasoning |

## Results

*Results will be updated after training experiments.*

### Baseline Comparison

| Model | Color Binding | Spatial Rel. | Counting | Overall |
|-------|---------------|--------------|----------|---------|
| Janus-Pro (baseline) | - | - | - | - |
| + GRPO (CLIP) | - | - | - | - |
| + GRPO (VLM) | - | - | - | - |

### Training Progress

Training metrics tracked:
- Reward mean/std per epoch
- KL divergence from reference model
- Validation scores on held-out prompts
- Benchmark scores at checkpoints

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 16GB VRAM | 24GB+ VRAM |
| RAM | 32GB | 64GB |
| Storage | 50GB | 100GB+ |

**Tested Configurations:**
- NVIDIA RTX 3090 (24GB) - Full training
- NVIDIA RTX 4090 (24GB) - Full training
- NVIDIA A100 (40/80GB) - Multi-GPU training

## References

### Papers

- **GRPO**: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300) - Our RL algorithm foundation
- **T2I-CompBench**: [A Comprehensive Benchmark for Compositional T2I Generation](https://arxiv.org/abs/2307.06350) - Evaluation benchmark
- **TIFA**: [Text-to-Image Faithfulness Evaluation](https://arxiv.org/abs/2303.11897) - VQA-based evaluation
- **Training Diffusion Models with RL**: [Black et al., 2023](https://arxiv.org/abs/2305.13301) - RL for T2I

### Related Projects

- [Janus-Pro](https://huggingface.co/deepseek-ai/Janus-Pro-1B) - Unified multimodal LLM (our base generator)
- [T2I-R1](https://github.com/CaraJ7/T2I-R1) - RL for T2I with reasoning
- [ReasonGen-R1](https://github.com/Franklin-Zhang0/ReasonGen-R1) - Unified multimodal reasoning

## Citation

```bibtex
@misc{t2i-rl-2026,
  title={T2I-RL: Text-to-Image Generation with Understanding-based Reward},
  author={AIMS 5740 Team},
  year={2026},
  url={https://github.com/Inoriany/T2I-RL-Project}
}
```

## Contributing

We welcome contributions! Please see our guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- AIMS 5740 Course Staff for project guidance
- DeepSeek AI for the Janus-Pro model
- Open source community for evaluation benchmarks
- Hugging Face for model hosting and transformers library
