# T2I-RL: Text-to-Image Generation with Reinforcement Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **AIMS 5740 Final Project** - Text-to-Image Generation with Understanding-based Reward

A research framework for training text-to-image models using reinforcement learning with vision-language model (VLM) based rewards. This project focuses on improving compositional generation capabilities including object presence, attribute binding, counting accuracy, and spatial relationships.

## Overview

Text-to-image models often generate realistic images but **hallucinate** objects, attributes, or spatial relations. This project aligns generation using **image understanding as the reward signal**.

### Key Features

- **Multiple Generator Support**: Janus-Pro-1B (unified MLLM), Stable Diffusion XL, Flow-based models
- **Flexible Reward Models**: CLIP-based similarity, VLM-based evaluation (GPT-4V, Claude), composite rewards
- **RL Training Algorithms**: GRPO (Group Relative Policy Optimization), Reward-Weighted MLE
- **Comprehensive Evaluation**: T2I-CompBench, TIFA, GenEval-2, GenAI-Bench
- **Error Analysis**: Automatic taxonomy of failures (missing objects, wrong count, wrong attribute, wrong relation)

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
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment
└── README.md
```

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

### Training

```bash
# Train with default config (Janus-Pro + GRPO)
python scripts/train.py

# Train with custom config
python scripts/train.py \
    model.name=janus-pro \
    training.algorithm=grpo \
    training.learning_rate=1e-5 \
    training.num_epochs=10

# Train with SDXL + Reward-Weighted
python scripts/train.py \
    model.name=sdxl \
    training.algorithm=reward_weighted
```

### Evaluation

```bash
# Evaluate on all benchmarks
python scripts/evaluate.py --checkpoint outputs/checkpoint-1000

# Evaluate on specific benchmark
python scripts/evaluate.py \
    --benchmark t2i_compbench tifa \
    --save_images \
    --use_vlm
```

## Training Algorithms

### GRPO (Group Relative Policy Optimization)

Based on [T2I-R1](https://github.com/CaraJ7/T2I-R1):

1. Generate K samples per prompt from current policy
2. Compute rewards using reward model
3. Compute advantages relative to group baseline
4. Update policy to increase probability of high-reward samples

```python
# Key hyperparameters
training:
  grpo:
    num_samples_per_prompt: 4
    temperature: 1.0
    kl_coef: 0.1
    baseline_type: "mean"  # mean, min, or ema
```

### Reward-Weighted MLE

Simpler alternative that weights MLE loss by rewards:

```
L = -E[r(x) * log p(x|prompt)]
```

## Evaluation Benchmarks

| Benchmark | Description | Metrics |
|-----------|-------------|---------|
| **T2I-CompBench** | Compositional generation | Color, Shape, Texture, Spatial, Non-spatial |
| **TIFA** | Text-Image Faithfulness | VQA accuracy |
| **GenEval-2** | General evaluation | Single/Two objects, Counting, Colors, Position |
| **GenAI-Bench** | Comprehensive AI benchmark | Overall quality |

### Error Taxonomy

The evaluation automatically categorizes errors:

- **Missing Objects**: Objects mentioned but not present
- **Wrong Count**: Incorrect number of objects
- **Wrong Attribute**: Incorrect color, size, texture
- **Wrong Relation**: Incorrect spatial relationships

## Results

*Results will be added after training experiments.*

| Model | T2I-CompBench | TIFA | GenEval-2 |
|-------|---------------|------|-----------|
| Baseline | - | - | - |
| + GRPO | - | - | - |
| + VLM Reward | - | - | - |

## References

### Related Work

- [T2I-R1](https://github.com/CaraJ7/T2I-R1) - RL for T2I with reasoning
- [ReasonGen-R1](https://github.com/Franklin-Zhang0/ReasonGen-R1) - Unified multimodal reasoning with GRPO
- [Janus-Pro](https://huggingface.co/deepseek-ai/Janus-Pro-1B) - Unified multimodal LLM

### Benchmarks

- [T2I-CompBench](https://github.com/Karine-Huang/T2I-CompBench) (NeurIPS 2023)
- [TIFA](https://tifa-benchmark.github.io/)
- [GenEval-2](https://arxiv.org/abs/2512.16853)

## Citation

```bibtex
@misc{t2i-rl-2024,
  title={T2I-RL: Text-to-Image Generation with Understanding-based Reward},
  author={AIMS 5740 Team},
  year={2024},
  url={https://github.com/Inoriany/T2I-RL-Project}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- AIMS 5740 Course Staff for project guidance
- DeepSeek AI for Janus-Pro model
- Open source community for evaluation benchmarks
