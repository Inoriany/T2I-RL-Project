#!/bin/bash
# Training script for T2I-RL with different LoRA configurations.
# Bug fixes applied:
#   - Job 3 was using r=32/alpha=64 instead of the intended r=4/alpha=8
#   - Job 4 was using r=32/alpha=64 instead of the intended r=16/alpha=32
#   - "exit 0" after Job 2 prevented Jobs 3 & 4 from running

set -e  # abort on first failure

run_job() {
    local tag="$1"; shift
    echo ""
    echo "=========================================="
    echo "Starting Training: $tag"
    echo "=========================================="
    python scripts/train.py "$@" \
        training.append_run_tag_to_output_dir=false \
        logging.use_wandb=false
    echo ""
    echo "  >>> $tag completed successfully!"
    echo "=========================================="
}

# ── Job 1: LoRA r=8, alpha=16 ─────────────────────────────────────────────
run_job "LoRA r=8, alpha=16" \
    model.name=janus-pro \
    model.model_path=deepseek-ai/Janus-Pro-1B \
    model.dtype=bfloat16 \
    model.lora.enabled=true \
    model.lora.r=8 \
    model.lora.alpha=16 \
    data.max_prompts_per_category=10 \
    training.output_dir=./outputs/r8_a16 \
    reward.type=composite \
    training.batch_size=4 \
    training.grpo.num_samples_per_prompt=4 \
    training.grpo.kl_coef=0.0

# ── Job 2: LoRA r=32, alpha=64 ────────────────────────────────────────────
run_job "LoRA r=32, alpha=64" \
    model.name=janus-pro \
    model.model_path=deepseek-ai/Janus-Pro-1B \
    model.dtype=bfloat16 \
    model.lora.enabled=true \
    model.lora.r=32 \
    model.lora.alpha=64 \
    data.max_prompts_per_category=10 \
    training.output_dir=./outputs/r32_a64 \
    reward.type=composite \
    training.batch_size=4 \
    training.grpo.num_samples_per_prompt=4 \
    training.grpo.kl_coef=0.0

# ── Job 3: LoRA r=4, alpha=8  (was broken — used r=32/alpha=64) ───────────
run_job "LoRA r=4, alpha=8" \
    model.name=janus-pro \
    model.model_path=deepseek-ai/Janus-Pro-1B \
    model.dtype=bfloat16 \
    model.lora.enabled=true \
    model.lora.r=4 \
    model.lora.alpha=8 \
    data.max_prompts_per_category=10 \
    training.output_dir=./outputs/r4_a8 \
    reward.type=composite \
    training.batch_size=4 \
    training.grpo.num_samples_per_prompt=4 \
    training.grpo.kl_coef=0.0

# ── Job 4: LoRA r=16, alpha=32  (was broken — used r=32/alpha=64) ─────────
run_job "LoRA r=16, alpha=32" \
    model.name=janus-pro \
    model.model_path=deepseek-ai/Janus-Pro-1B \
    model.dtype=bfloat16 \
    model.lora.enabled=true \
    model.lora.r=16 \
    model.lora.alpha=32 \
    data.max_prompts_per_category=10 \
    training.output_dir=./outputs/r16_a32 \
    reward.type=composite \
    training.batch_size=4 \
    training.grpo.num_samples_per_prompt=4 \
    training.grpo.kl_coef=0.0

echo ""
echo "=========================================="
echo "All training jobs completed successfully!"
echo "=========================================="
