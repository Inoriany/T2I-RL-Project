# Methodology: Text-to-Image Generation with Reinforcement Learning

This document provides a detailed explanation of the methodology used in this project, including theoretical foundations, algorithm details, and implementation considerations.

## Table of Contents

1. [Problem Formulation](#1-problem-formulation)
2. [Reinforcement Learning Framework](#2-reinforcement-learning-framework)
3. [GRPO Algorithm](#3-grpo-algorithm)
4. [Reward Models](#4-reward-models)
5. [Training Pipeline](#5-training-pipeline)
6. [Theoretical Analysis](#6-theoretical-analysis)

---

## 1. Problem Formulation

### 1.1 Compositional Text-to-Image Generation

Given a text prompt $p$ describing a scene with multiple objects, attributes, and spatial relationships, the goal is to generate an image $x$ that accurately reflects all compositional elements mentioned in $p$.

**Challenges in Compositional Generation:**

| Challenge | Example | Failure Mode |
|-----------|---------|--------------|
| **Attribute Binding** | "a red apple and a blue cup" | Colors swapped between objects |
| **Object Presence** | "three cats playing" | Wrong number of objects |
| **Spatial Relations** | "a cat on top of a box" | Incorrect spatial arrangement |
| **Complex Scenes** | "a chef cooking in a kitchen with..." | Missing or hallucinated elements |

### 1.2 Limitations of Current Approaches

Standard text-to-image models (DALL-E, Stable Diffusion, etc.) are trained with:

$$\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t, c)\|^2\right]$$

This objective optimizes for reconstruction quality but does not explicitly optimize for semantic alignment or compositional accuracy.

### 1.3 Our Approach: RL with Understanding-Based Rewards

We reformulate T2I generation as a reinforcement learning problem:

- **State**: Text prompt $p$
- **Action**: Generated image $x$
- **Policy**: Generator $\pi_\theta(x|p)$
- **Reward**: VLM-based evaluation $R(x, p)$

The objective becomes:

$$\max_\theta \mathbb{E}_{p \sim \mathcal{D}, x \sim \pi_\theta(\cdot|p)}[R(x, p)] - \beta \cdot \text{KL}(\pi_\theta || \pi_{\text{ref}})$$

---

## 2. Reinforcement Learning Framework

### 2.1 Policy Gradient Methods

For autoregressive generators like Janus-Pro, the image generation process can be viewed as sequential token generation:

$$\pi_\theta(x|p) = \prod_{t=1}^{T} \pi_\theta(x_t | x_{<t}, p)$$

where $T = 576$ visual tokens for Janus-Pro (24×24 grid).

The policy gradient is:

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(x_t | x_{<t}, p) \cdot A(x, p)\right]$$

where $A(x, p)$ is the advantage function.

### 2.2 Variance Reduction

Raw policy gradients suffer from high variance. We employ several techniques:

1. **Baseline Subtraction**: $A(x, p) = R(x, p) - b(p)$
2. **Reward Normalization**: Standardize rewards within batches
3. **Group Relative Rewards**: Compare samples within the same prompt group

### 2.3 KL Regularization

To prevent the policy from diverging too far from the pretrained model:

$$\mathcal{L}_{\text{KL}} = \text{KL}(\pi_\theta || \pi_{\text{ref}}) = \mathbb{E}_{x \sim \pi_\theta}\left[\log \frac{\pi_\theta(x|p)}{\pi_{\text{ref}}(x|p)}\right]$$

This preserves generation quality while improving compositional accuracy.

---

## 3. GRPO Algorithm

### 3.1 Group Relative Policy Optimization

GRPO (Group Relative Policy Optimization) is our primary training algorithm, inspired by [DeepSeekMath](https://arxiv.org/abs/2402.03300) and adapted for image generation.

**Key Insight**: Instead of requiring a separate value network, GRPO uses the relative performance within a group of samples from the same prompt as the baseline.

### 3.2 Algorithm Details

```
Algorithm: GRPO for Text-to-Image Generation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: 
  - Policy π_θ (Janus-Pro with LoRA)
  - Reference policy π_ref (frozen Janus-Pro)
  - Reward model R (VLM-based)
  - Prompt dataset D
  - Group size K, KL coefficient β, clip range ε

For each training iteration:
  1. Sample batch of prompts {p_1, ..., p_B} from D
  
  2. For each prompt p_i:
     a. Generate K images: {x_i^1, ..., x_i^K} ~ π_θ(·|p_i)
     b. Record log probabilities: log π_θ(x_i^k|p_i)
     c. Compute rewards: r_i^k = R(x_i^k, p_i)
  
  3. Normalize rewards within each group:
     r̂_i^k = (r_i^k - mean_k(r_i)) / (std_k(r_i) + ε)
  
  4. Compute GRPO loss:
     L_GRPO = -1/(B·K) Σ_i Σ_k [r̂_i^k · log π_θ(x_i^k|p_i)]
  
  5. Compute KL penalty:
     L_KL = 1/(B·K) Σ_i Σ_k [log π_θ(x_i^k|p_i) - log π_ref(x_i^k|p_i)]
  
  6. Total loss: L = L_GRPO + β · L_KL
  
  7. Update θ using gradient descent

Output: Fine-tuned policy π_θ
```

### 3.3 Hyperparameter Choices

| Parameter | Recommended Value | Description |
|-----------|------------------|-------------|
| Group size $K$ | 4-8 | Number of samples per prompt |
| KL coefficient $\beta$ | 0.01-0.1 | Regularization strength |
| Learning rate | 1e-5 to 5e-5 | For LoRA parameters |
| LoRA rank | 16-64 | Low-rank adaptation dimension |
| Clip range $\epsilon$ | 0.2 | For PPO-style clipping (optional) |

### 3.4 Why GRPO for T2I?

| Advantage | Explanation |
|-----------|-------------|
| **No value network** | Eliminates need for critic, reducing complexity |
| **Sample efficient** | Multiple samples per prompt are all used for learning |
| **Stable training** | Group normalization reduces variance |
| **Applicable to VLM rewards** | Works with non-differentiable reward signals |

---

## 4. Reward Models

### 4.1 Reward Model Hierarchy

We support multiple reward models with increasing semantic understanding:

```
┌─────────────────────────────────────────────────────────────┐
│                    Reward Model Hierarchy                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Level 3: VLM-Based Rewards (GPT-4V, Claude)               │
│  ├── Deep semantic understanding                            │
│  ├── Explicit reasoning about composition                   │
│  └── Detailed attribute and relation checking               │
│                                                             │
│  Level 2: Specialized Vision Models                         │
│  ├── Object detection (presence checking)                   │
│  ├── Attribute classifiers (color, shape)                   │
│  └── Spatial relation predictors                            │
│                                                             │
│  Level 1: CLIP-Based Similarity                             │
│  ├── Fast computation                                       │
│  ├── Global image-text alignment                            │
│  └── Limited compositional understanding                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 VLM-Based Reward Design

Our VLM reward decomposes evaluation into multiple aspects:

```python
evaluation_prompt = """
Evaluate this image against the prompt: "{prompt}"

Score each aspect from 0 to 10:

1. OBJECT PRESENCE (weight: 0.3)
   - Are ALL mentioned objects present in the image?
   - Are there any extra objects not mentioned?

2. ATTRIBUTE ACCURACY (weight: 0.3)
   - Do objects have the correct colors?
   - Do objects have the correct sizes/shapes?
   - Are textures and materials accurate?

3. SPATIAL RELATIONS (weight: 0.25)
   - Are objects positioned correctly relative to each other?
   - "on top of", "next to", "behind", "under", etc.

4. OVERALL QUALITY (weight: 0.15)
   - Image quality and coherence
   - Natural appearance
   - Artistic quality

Return JSON: {
  "presence": <score>,
  "attributes": <score>,
  "spatial": <score>,
  "quality": <score>,
  "reasoning": "<brief explanation>"
}
"""
```

**Final Reward Calculation:**

$$R(x, p) = 0.3 \cdot s_{\text{presence}} + 0.3 \cdot s_{\text{attributes}} + 0.25 \cdot s_{\text{spatial}} + 0.15 \cdot s_{\text{quality}}$$

### 4.3 CLIP-Based Reward

For faster iteration, we also support CLIP-based rewards:

$$R_{\text{CLIP}}(x, p) = \cos(E_{\text{image}}(x), E_{\text{text}}(p))$$

**Limitations:**
- CLIP embeddings are global, not compositionally aware
- "a red apple and blue cup" ≈ "a blue apple and red cup" in CLIP space
- Suitable for warm-up training or when API costs are a concern

### 4.4 Hybrid Reward Strategy

For practical training, we recommend:

```
Phase 1 (Warm-up):     CLIP rewards only
Phase 2 (Main):        VLM rewards with CLIP as fast filter
Phase 3 (Refinement):  VLM rewards on hard examples
```

---

## 5. Training Pipeline

### 5.1 Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Prompts   │────▶│  Generator  │────▶│   Images    │
│   Dataset   │     │ (Janus-Pro) │     │  (K per p)  │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                   │
                           │                   ▼
                           │            ┌─────────────┐
                           │            │   Reward    │
                           │            │   Model     │
                           │            └─────────────┘
                           │                   │
                           ▼                   ▼
                    ┌─────────────────────────────────┐
                    │         GRPO Loss               │
                    │  L = -r̂ · log π + β · KL       │
                    └─────────────────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────┐
                    │      LoRA Parameter Update      │
                    └─────────────────────────────────┘
```

### 5.2 Data Pipeline

**Training Prompts:**
- Source: Curated compositional prompts (provided in `data/prompts/`)
- Categories: Color binding, spatial relations, counting, attributes, complex scenes
- Augmentation: Template-based generation for scaling

**Prompt Template Examples:**
```python
color_template = "a {color1} {object1} and a {color2} {object2}"
spatial_template = "a {object1} {relation} a {object2}"
counting_template = "{number} {color} {objects}"
```

### 5.3 Efficient Training Strategies

1. **LoRA Fine-tuning**: Only train low-rank adapters (~0.1% of parameters)
2. **Gradient Accumulation**: Simulate larger batch sizes
3. **Mixed Precision**: Use bfloat16 for memory efficiency
4. **Reward Caching**: Cache VLM responses for repeated prompts

### 5.4 Checkpointing and Evaluation

- Save checkpoints every N steps
- Evaluate on validation prompts periodically
- Track metrics: reward mean, reward std, KL divergence, generation quality

---

## 6. Theoretical Analysis

### 6.1 Convergence Properties

Under standard assumptions (bounded rewards, Lipschitz policy), GRPO converges to a local optimum of:

$$J(\theta) = \mathbb{E}_{p, x}[R(x, p)] - \beta \cdot \text{KL}(\pi_\theta || \pi_{\text{ref}})$$

The group-relative baseline provides an unbiased estimate of the advantage while reducing variance compared to single-sample estimators.

### 6.2 Sample Complexity

With group size $K$ and batch size $B$:
- Total samples per iteration: $B \times K$
- Effective gradient estimates: $B \times K$ (all samples contribute)
- Variance reduction factor: $\approx \sqrt{K}$ compared to single-sample

### 6.3 Reward Hacking Considerations

**Potential Issues:**
- Model may exploit reward model weaknesses
- VLMs can have systematic biases

**Mitigations:**
- KL regularization prevents extreme deviations
- Multi-aspect rewards harder to hack than single scores
- Periodic human evaluation for quality assurance

### 6.4 Comparison with Alternatives

| Method | Pros | Cons |
|--------|------|------|
| **GRPO (Ours)** | Simple, stable, no critic needed | Requires multiple samples |
| **PPO** | Well-studied, clipping stabilizes | Needs value network |
| **DPO** | No reward model needed | Requires preference data |
| **REINFORCE** | Simplest | High variance |
| **Reward-weighted regression** | Single sample OK | Biased gradient estimate |

---

## References

1. DeepSeek-AI. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300

2. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347

3. Black, K., et al. (2023). Training Diffusion Models with Reinforcement Learning. arXiv:2305.13301

4. Wu, X., et al. (2023). Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis. arXiv:2306.09341

5. Huang, K., et al. (2023). T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation. NeurIPS 2023

---

*Last updated: April 2026*
