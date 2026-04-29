#!/usr/bin/env bash
# =============================================================================
# run_ablation.sh  —  One-click ablation study
# =============================================================================
#
# Trains Janus-Pro with every combination of:
#   REWARD_TYPES   × LORA_CONFIGS   × TRAIN_STEPS
#
# For each run:
#   1. Trains with the specified config → unique output dir, separate checkpoint
#   2. Saves training loss plot (PNG + JSON)
#   3. Runs T2I-CompBench and GenEval-2 evaluation immediately after
#   4. Saves per-run eval_summary.json into the same output dir
#   5. At the end, writes a combined comparison table to results/ablation_summary.json
#
# Usage:
#   bash scripts/run_ablation.sh
#   bash scripts/run_ablation.sh --dry-run     # print commands, no execution
#
# Environment variables (all optional, override defaults):
#   MODEL_PATH          — HF model id or local path  (default: deepseek-ai/Janus-Pro-1B)
#   BASE_OUTPUT_DIR     — root for all run outputs    (default: ./ablation_outputs)
#   DATA_ROOT           — benchmark data directory    (default: ./data)
#   MAX_SAMPLES         — max prompts per benchmark   (default: 50)
#   MAX_PROMPTS_CAT     — prompts per T2I-CB category (default: 5)
#   USE_WANDB           — "true"/"false"              (default: false)
#   WANDB_PROJECT       — W&B project name            (default: t2i-rl-ablation)
#   SKIP_DOWNLOAD       — "true" to skip data dl      (default: false)
#   USE_MODELSCOPE      — "true" to use ModelScope  (default: false)
#                         Set to "true" if HuggingFace is not accessible
#   MODELSCOPE_CACHE_DIR — ModelScope hub root       (default: ~/.cache/modelscope/hub)
#   EVAL_BLIP_MODEL      — BLIP-VQA dir for benchmarks (default: $MODELSCOPE_CACHE_DIR/models/Salesforce/blip-vqa-base)
#   EVAL_OWL_MODEL       — OwlViT dir for benchmarks   (default: $MODELSCOPE_CACHE_DIR/models/google/owlv2-base-patch16-ensemble)
#
#   GRPO sampling (optional — more diverse rollouts / stronger within-group spread):
#   GRPO_TEMPERATURE              — softmax temp for visual tokens (default: 1.0; try 1.1–1.3)
#   GRPO_GUIDANCE_SCALE           — Janus CFG (default: 5.0; try 3.5–4.5 for more diversity)
#   GRPO_NUM_SAMPLES_PER_PROMPT   — group size K (default: 6; larger K → more independent draws)
#
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Download model from ModelScope if needed
# ─────────────────────────────────────────────────────────────────────────────

# Default ModelScope cache directory (same as modelscope CLI)
DEFAULT_MODELSCOPE_CACHE="${HOME}/.cache/modelscope/hub"

download_model_from_modelscope() {
    local model_id="$1"
    local cache_dir="${2:-$DEFAULT_MODELSCOPE_CACHE}"
    local temp_file="/tmp/modelscope_download_path_$$.txt"

    echo "  → Checking ModelScope for model: $model_id" >&2
    echo "  → Cache directory: $cache_dir" >&2

    python3 - <<PYEOF
import os
import sys
import io

# Redirect ALL logging to stderr
import logging
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

try:
    from modelscope import snapshot_download
except ImportError:
    print("Error: modelscope package not installed.", file=sys.stderr)
    print("Please install it with: pip install modelscope", file=sys.stderr)
    sys.exit(1)

model_id = "${model_id}"
cache_dir = "${cache_dir}"
os.makedirs(cache_dir, exist_ok=True)

try:
    modelscope_id = model_id

    # Redirect stdout to stderr during snapshot_download so that
    # progress bars / log messages don't pollute the captured output.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        local_path = snapshot_download(modelscope_id, cache_dir=cache_dir)
    finally:
        sys.stdout = real_stdout

    # Only the path goes to stdout — this is what the shell captures.
    print(local_path, flush=True)
except Exception as e:
    print(f"Error downloading model: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
PYEOF
}

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Default model path (will be overridden if USE_MODELSCOPE=true)
DEFAULT_MODEL_PATH="deepseek-ai/Janus-Pro-1B"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"

# Default to ModelScope on this machine (no local HuggingFace cache)
USE_MODELSCOPE="${USE_MODELSCOPE:-true}"
# Use default ModelScope cache directory (same as `modelscope download` CLI)
# This ensures we can find models already downloaded by the CLI
MODELSCOPE_CACHE_DIR="${MODELSCOPE_CACHE_DIR:-$DEFAULT_MODELSCOPE_CACHE}"
export MODELSCOPE_CACHE_DIR

# Local benchmark aux models (BLIP-VQA, OwlViT) — avoids Hub download when pre-fetched via modelscope download
EVAL_BLIP_MODEL="${EVAL_BLIP_MODEL:-${MODELSCOPE_CACHE_DIR}/models/Salesforce/blip-vqa-base}"
EVAL_OWL_MODEL="${EVAL_OWL_MODEL:-${MODELSCOPE_CACHE_DIR}/models/google/owlv2-base-patch16-ensemble}"

# If USE_MODELSCOPE is true, download the model and update MODEL_PATH
if [[ "$USE_MODELSCOPE" == "true" ]]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Using ModelScope for model download"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Original MODEL_PATH: $MODEL_PATH"
    
    # Download from ModelScope and get local path
    echo "  → Calling ModelScope to get model path..."
    LOCAL_MODEL_PATH=$(download_model_from_modelscope "$MODEL_PATH" "$MODELSCOPE_CACHE_DIR")
    
    # Debug output
    if [[ -z "$LOCAL_MODEL_PATH" ]]; then
        echo "  ✗ ModelScope returned empty path"
        exit 1
    fi
    
    echo "  → ModelScope returned path: $LOCAL_MODEL_PATH"
    
    # Check if path exists
    if [[ -d "$LOCAL_MODEL_PATH" ]]; then
        MODEL_PATH="$LOCAL_MODEL_PATH"
        echo "  ✓ Updated MODEL_PATH: $MODEL_PATH"
        echo ""
        echo "  Note: Model will be loaded from local ModelScope cache"
    else
        # Try alternative path (models/ subdirectory - used by CLI)
        ALT_PATH="${MODELSCOPE_CACHE_DIR}/models/${MODEL_PATH}"
        echo "  → Trying alternative path: $ALT_PATH"
        
        if [[ -d "$ALT_PATH" ]]; then
            MODEL_PATH="$ALT_PATH"
            echo "  ✓ Found model at alternative path: $MODEL_PATH"
            echo ""
            echo "  Note: Model will be loaded from local ModelScope cache"
        else
            echo "  ✗ Path does not exist: $LOCAL_MODEL_PATH"
            echo "  ✗ Alternative path also does not exist: $ALT_PATH"
            exit 1
        fi
    fi
    echo ""
fi

# Export ModelScope settings for child Python processes
# This ensures auxiliary models (CLIP, BLIP, OwlViT, etc.) also use ModelScope
if [[ "$USE_MODELSCOPE" == "true" ]]; then
    export USE_MODELSCOPE="true"
    export MODELSCOPE_CACHE="${MODELSCOPE_CACHE_DIR}"
    export HF_ENDPOINT="https://www.modelscope.cn/hf"
    echo "  Exported environment variables for child processes:"
    echo "    USE_MODELSCOPE=$USE_MODELSCOPE"
    echo "    HF_ENDPOINT=$HF_ENDPOINT"
    echo ""
fi

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-./ablation_outputs}"
DATA_ROOT="${DATA_ROOT:-./data}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
MAX_PROMPTS_CAT="${MAX_PROMPTS_CAT:-50}"
USE_WANDB="${USE_WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-t2i-rl-ablation}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-false}"

# GRPO rollout diversity (passed through to Hydra → GenerationConfig)
GRPO_TEMPERATURE="${GRPO_TEMPERATURE:-1.0}"
GRPO_GUIDANCE_SCALE="${GRPO_GUIDANCE_SCALE:-5.0}"
GRPO_NUM_SAMPLES_PER_PROMPT="${GRPO_NUM_SAMPLES_PER_PROMPT:-6}"

# ─────────────────────────────────────────────────────────────────────────────
# Pre-download VLM reward model (Qwen2.5-VL-7B-Instruct) from ModelScope
# so training does not stall waiting for a large download mid-run.
# ─────────────────────────────────────────────────────────────────────────────
VLM_MODEL_NAME_PRE="Qwen/Qwen2.5-VL-7B-Instruct"
VLM_LOCAL_CHECK="${MODELSCOPE_CACHE_DIR}/models/${VLM_MODEL_NAME_PRE}"

if [[ ! -d "$VLM_LOCAL_CHECK" ]]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Pre-downloading VLM reward model: $VLM_MODEL_NAME_PRE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    download_model_from_modelscope "$VLM_MODEL_NAME_PRE" "$MODELSCOPE_CACHE_DIR"
    echo "  ✓ VLM reward model pre-downloaded"
    echo ""
else
    echo "  ✓ VLM reward model already cached: $VLM_LOCAL_CHECK"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Output logging: tee all stdout+stderr to a timestamped log file
# ─────────────────────────────────────────────────────────────────────────────
mkdir -p "$BASE_OUTPUT_DIR"
LOG_FILE="${BASE_OUTPUT_DIR}/ablation_run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "  All output is being logged to: $LOG_FILE"
echo ""

DRY_RUN=false
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        DRY_RUN=true
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# Ablation axes
# ─────────────────────────────────────────────────────────────────────────────

# Reward types to test
# NOTE: "vlm" and "composite" use a local Qwen2.5-VL-3B-Instruct model
#       downloaded from ModelScope. No API key needed.
# REWARD_TYPES=("clip" "composite" "vlm")
REWARD_TYPES=("vlm")

# LoRA configs: "r alpha" pairs
LORA_CONFIGS=("16 32" "32 64" "64 128")
# LORA_CONFIGS=("4 8")

# Training step counts  (mapped to num_epochs for simplicity)
# Adjust for your GPU: each epoch ≈ (num_prompts / batch_size) steps.
TRAIN_STEPS_LIST=("90000000000")
# TRAIN_STEPS_LIST=("100")
# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

log() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $*"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

run_cmd() {
    if $DRY_RUN; then
        echo "[DRY-RUN] $*"
    else
        eval "$@"
    fi
}

# Check if a reward type is available
reward_available() {
    local rt="$1"
    # All reward types are available since VLM runs locally now
    return 0
}

# ─────────────────────────────────────────────────────────────────────────────
# 0. Setup
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$BASE_OUTPUT_DIR"
RESULTS_DIR="$BASE_OUTPUT_DIR/results"
mkdir -p "$RESULTS_DIR"

SUMMARY_FILE="$RESULTS_DIR/ablation_summary.json"

echo "[]" > "$SUMMARY_FILE"   # initialise empty JSON array

log "T2I-RL Ablation Study"
echo "  MODEL_PATH    : $MODEL_PATH"
echo "  BASE_OUTPUT   : $BASE_OUTPUT_DIR"
echo "  DATA_ROOT     : $DATA_ROOT"
echo "  MAX_SAMPLES   : $MAX_SAMPLES"
echo "  REWARD_TYPES  : ${REWARD_TYPES[*]}"
echo "  LORA_CONFIGS  : ${LORA_CONFIGS[*]}"
echo "  TRAIN_STEPS   : ${TRAIN_STEPS_LIST[*]}"
echo "  DRY_RUN       : $DRY_RUN"
if [[ "$USE_MODELSCOPE" == "true" ]]; then
    echo "  USE_MODELSCOPE: true (using ModelScope instead of HuggingFace)"
    echo "  HF_ENDPOINT   : $HF_ENDPOINT"
fi
echo "  EVAL_BLIP_MODEL: $EVAL_BLIP_MODEL"
echo "  EVAL_OWL_MODEL : $EVAL_OWL_MODEL"
echo "  GRPO_TEMPERATURE            : $GRPO_TEMPERATURE"
echo "  GRPO_GUIDANCE_SCALE         : $GRPO_GUIDANCE_SCALE"
echo "  GRPO_NUM_SAMPLES_PER_PROMPT : $GRPO_NUM_SAMPLES_PER_PROMPT"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Download benchmark data
# ─────────────────────────────────────────────────────────────────────────────

# if [[ "$SKIP_DOWNLOAD" != "true" ]]; then
#     log "Downloading benchmark data"
#     run_cmd python scripts/download_eval_data.py --data_root "$DATA_ROOT"
# fi

# ─────────────────────────────────────────────────────────────────────────────
# 2. Main ablation loop
# ─────────────────────────────────────────────────────────────────────────────

TOTAL_RUNS=0
COMPLETED_RUNS=0
FAILED_RUNS=0

# Compute total runs
for rt in "${REWARD_TYPES[@]}"; do
    for lc in "${LORA_CONFIGS[@]}"; do
        for steps in "${TRAIN_STEPS_LIST[@]}"; do
            TOTAL_RUNS=$((TOTAL_RUNS + 1))
        done
    done
done

echo ""
echo "  Total runs planned: $TOTAL_RUNS"
echo ""

RUN_IDX=0

for REWARD_TYPE in "${REWARD_TYPES[@]}"; do

    if ! reward_available "$REWARD_TYPE"; then
        continue
    fi

    for LORA_CFG in "${LORA_CONFIGS[@]}"; do
        LORA_R=$(echo "$LORA_CFG" | awk '{print $1}')
        LORA_A=$(echo "$LORA_CFG" | awk '{print $2}')

        for TRAIN_STEPS in "${TRAIN_STEPS_LIST[@]}"; do

            RUN_IDX=$((RUN_IDX + 1))

            # Unique run tag: no spaces, fully descriptive
            RUN_TAG="reward_${REWARD_TYPE}_r${LORA_R}_a${LORA_A}_steps${TRAIN_STEPS}"
            RUN_DIR="$BASE_OUTPUT_DIR/$RUN_TAG"

            log "Run $RUN_IDX / $TOTAL_RUNS : $RUN_TAG"
            echo "  Output dir: $RUN_DIR"

            # ─────────────────────────────────────────────────────────────
            # 2a. Train
            # ─────────────────────────────────────────────────────────────

            # Local VLM model for reward (downloaded from ModelScope)
            VLM_MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

            WANDB_RUN_NAME="${RUN_TAG}"
            WANDB_FLAG="false"
            if [[ "$USE_WANDB" == "true" ]]; then
                WANDB_FLAG="true"
            fi

            TRAIN_CMD="python scripts/train.py \
                model.name=janus-pro \
                model.model_path=${MODEL_PATH} \
                model.dtype=bfloat16 \
                model.lora.enabled=true \
                model.lora.r=${LORA_R} \
                model.lora.alpha=${LORA_A} \
                model.lora.dropout=0.05 \
                reward.type=${REWARD_TYPE} \
                reward.clip.model_name=ViT-L-14 \
                reward.clip.pretrained=openai \
                reward.clip.weight=0.5 \
                reward.vlm.use_api=false \
                ++reward.vlm.model_name_or_path=${VLM_MODEL_NAME} \
                reward.vlm.weight=0.5 \
                training.algorithm=grpo \
                training.grpo.num_samples_per_prompt=${GRPO_NUM_SAMPLES_PER_PROMPT} \
                training.grpo.kl_coef=0.01 \
                training.grpo.temperature=${GRPO_TEMPERATURE} \
                training.grpo.guidance_scale=${GRPO_GUIDANCE_SCALE} \
                training.grpo.use_advantage_normalization=true \
                training.grpo.baseline_type=mean \
                training.max_grad_norm=1.0 \
                training.learning_rate=1e-5 \
                training.batch_size=4 \
                training.gradient_accumulation_steps=2 \
                training.warmup_ratio=0.1 \
                training.save_steps=${TRAIN_STEPS} \
                training.eval_steps=99999 \
                training.output_dir=${RUN_DIR} \
                training.append_run_tag_to_output_dir=false \
                training.num_epochs=5 \
                data.train_file=./data/prompts/train_prompts.json \
                data.max_prompts_per_category=${MAX_PROMPTS_CAT} \
                data.max_train_samples=${TRAIN_STEPS} \
                logging.use_wandb=${WANDB_FLAG} \
                logging.wandb_project=${WANDB_PROJECT} \
                logging.wandb_run_name=${WANDB_RUN_NAME} \
                logging.logging_steps=1"

            echo "  → Training..."
            if run_cmd "$TRAIN_CMD"; then
                echo "  ✓ Training completed."
            else
                echo "  ✗ Training FAILED for $RUN_TAG"
                FAILED_RUNS=$((FAILED_RUNS + 1))
                continue
            fi

            # ─────────────────────────────────────────────────────────────
            # 2b. Find best checkpoint
            # ─────────────────────────────────────────────────────────────

            LORA_CKPT=""
            # Prefer epoch checkpoint (end-of-training), then step checkpoint
            for ckpt_name in "checkpoint-epoch-0" "checkpoint-${TRAIN_STEPS}"; do
                CANDIDATE="${RUN_DIR}/${ckpt_name}"
                if [[ -d "$CANDIDATE" ]]; then
                    LORA_CKPT="$CANDIDATE"
                    break
                fi
            done
            # If no named checkpoint found, look for any checkpoint-* dir
            if [[ -z "$LORA_CKPT" ]]; then
                LORA_CKPT=$(find "$RUN_DIR" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | sort -t- -k2 -rn | head -1 || true)
            fi

            LORA_ARG=""
            if [[ -n "$LORA_CKPT" ]]; then
                echo "  → Using checkpoint: $LORA_CKPT"
                LORA_ARG="--lora_checkpoint $LORA_CKPT"
            else
                echo "  → No checkpoint found; evaluating with trained model weights."
            fi

            # ─────────────────────────────────────────────────────────────
            # 2c. Evaluate
            # ─────────────────────────────────────────────────────────────

            EVAL_OUTPUT="${RUN_DIR}/evaluation"

            EVAL_CMD="python scripts/evaluate_benchmarks.py \
                --model_path ${MODEL_PATH} \
                ${LORA_ARG} \
                --benchmarks all \
                --data_root ${DATA_ROOT} \
                --output_dir ${EVAL_OUTPUT} \
                --blip_model ${EVAL_BLIP_MODEL} \
                --owl_model ${EVAL_OWL_MODEL} \
                --no_save_images"

            echo "  → Evaluating..."
            if run_cmd "$EVAL_CMD"; then
                echo "  ✓ Evaluation completed."
            else
                echo "  ✗ Evaluation FAILED for $RUN_TAG"
                FAILED_RUNS=$((FAILED_RUNS + 1))
                continue
            fi

            # ─────────────────────────────────────────────────────────────
            # 2d. Append to summary JSON (pure Python one-liner for portability)
            # ─────────────────────────────────────────────────────────────

            if ! $DRY_RUN; then
                python3 - <<PYEOF
import json, pathlib, sys

summary_path = pathlib.Path("${SUMMARY_FILE}")
eval_path    = pathlib.Path("${EVAL_OUTPUT}/eval_summary.json")

try:
    with open(summary_path) as f:
        summary = json.load(f)
except Exception:
    summary = []

entry = {
    "run_tag":      "${RUN_TAG}",
    "reward_type":  "${REWARD_TYPE}",
    "lora_r":       ${LORA_R},
    "lora_alpha":   ${LORA_A},
    "train_steps":  ${TRAIN_STEPS},
    "output_dir":   "${RUN_DIR}",
}

if eval_path.exists():
    with open(eval_path) as f:
        eval_data = json.load(f)
    benchmarks = eval_data.get("benchmarks", {})
    if "t2i_compbench" in benchmarks:
        entry["t2i_compbench_overall"] = benchmarks["t2i_compbench"].get("overall_mean", 0.0)
        entry["t2i_compbench_categories"] = benchmarks["t2i_compbench"].get("category_means", {})
    if "geneval2" in benchmarks:
        entry["geneval2_overall"] = benchmarks["geneval2"].get("overall_score", 0.0)
        entry["geneval2_per_skill"] = benchmarks["geneval2"].get("per_skill_scores", {})
else:
    print(f"  Warning: eval_summary.json not found at {eval_path}", file=sys.stderr)

# Merge loss/reward/kl history if available
loss_path = pathlib.Path("${RUN_DIR}/training_loss.json")
if loss_path.exists():
    with open(loss_path) as f:
        loss_data = json.load(f)
    losses  = loss_data.get("losses", [])
    rewards = loss_data.get("reward_means", [])
    kls     = loss_data.get("kl_divs", [])
    entry["final_loss"]        = losses[-1]  if losses  else None
    entry["final_reward_mean"] = rewards[-1] if rewards else None
    entry["final_kl_div"]      = kls[-1]     if kls     else None

summary.append(entry)
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"  Updated ablation summary: {summary_path}")
PYEOF
            fi

            COMPLETED_RUNS=$((COMPLETED_RUNS + 1))
            echo "  ✓ Run $RUN_IDX complete: $RUN_TAG"

        done  # TRAIN_STEPS
    done  # LORA_CONFIGS
done  # REWARD_TYPES

# ─────────────────────────────────────────────────────────────────────────────
# 3. Generate comparison plot + final report
# ─────────────────────────────────────────────────────────────────────────────

log "Generating ablation comparison report"

if ! $DRY_RUN; then
    python3 - <<'PYEOF'
import json, pathlib, sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

results_dir = pathlib.Path("${RESULTS_DIR}")
summary_path = results_dir / "ablation_summary.json"

if not summary_path.exists():
    print("No ablation summary found — skipping report generation.")
    sys.exit(0)

with open(summary_path) as f:
    runs = json.load(f)

if not runs:
    print("No runs recorded.")
    sys.exit(0)

# ── Text report ──────────────────────────────────────────────────────────────
lines = [
    "=" * 80,
    "Ablation Study Summary",
    "=" * 80,
    f"{'Run Tag':<45} {'T2I-CB':>8} {'GE-2':>8} {'Steps':>7} {'LoRA r':>7} {'Reward':<12}",
    "-" * 80,
]
for r in runs:
    t2i  = r.get("t2i_compbench_overall", float("nan"))
    ge2  = r.get("geneval2_overall",      float("nan"))
    tag  = r.get("run_tag", "?")[:44]
    lines.append(
        f"{tag:<45} {t2i:>8.4f} {ge2:>8.4f} {r.get('train_steps', 0):>7} "
        f"{r.get('lora_r', 0):>7} {r.get('reward_type', '?'):<12}"
    )

report = "\n".join(lines)
print(report)
report_path = results_dir / "ablation_report.txt"
with open(report_path, "w") as f:
    f.write(report)
print(f"\nText report saved to: {report_path}")

if not HAS_MATPLOTLIB or not runs:
    sys.exit(0)

# ── Matplotlib comparison plots ───────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# (A) LoRA rank vs scores (for each reward type)
reward_types = sorted({r.get("reward_type", "?") for r in runs})
colors_palette = plt.cm.Set1.colors

for ax, metric, label in [
    (axes[0], "t2i_compbench_overall", "T2I-CompBench"),
    (axes[1], "geneval2_overall",      "GenEval-2"),
]:
    for rt, color in zip(reward_types, colors_palette):
        subset = [r for r in runs if r.get("reward_type") == rt]
        # Group by lora_r and average over steps
        lora_rs = sorted({r.get("lora_r", 0) for r in subset})
        xs, ys = [], []
        for lr in lora_rs:
            vals = [r.get(metric, 0.0) for r in subset if r.get("lora_r") == lr]
            if vals:
                xs.append(lr)
                ys.append(np.mean(vals))
        if xs:
            ax.plot(xs, ys, marker="o", label=f"reward={rt}", color=color)
    ax.set_xlabel("LoRA rank (r)")
    ax.set_ylabel("Score")
    ax.set_title(f"{label} vs LoRA rank")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# (C) Training steps vs T2I-CompBench (for each LoRA config)
ax = axes[2]
lora_configs = sorted({(r.get("lora_r", 0), r.get("lora_alpha", 0)) for r in runs})
for (lr, la), color in zip(lora_configs, colors_palette):
    subset = [r for r in runs if r.get("lora_r") == lr and r.get("lora_alpha") == la]
    steps_vals = sorted({r.get("train_steps", 0) for r in subset})
    xs, ys = [], []
    for s in steps_vals:
        vals = [r.get("t2i_compbench_overall", 0.0) for r in subset if r.get("train_steps") == s]
        if vals:
            xs.append(s)
            ys.append(np.mean(vals))
    if xs:
        ax.plot(xs, ys, marker="s", label=f"r={lr}/a={la}", color=color)

ax.set_xlabel("Training steps")
ax.set_ylabel("T2I-CompBench score")
ax.set_title("T2I-CompBench vs Training steps")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle("T2I-RL Ablation Study", fontsize=14, fontweight="bold")
plt.tight_layout()

plot_path = results_dir / "ablation_comparison.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Comparison plot saved to: {plot_path}")
PYEOF
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. Final status
# ─────────────────────────────────────────────────────────────────────────────

log "Ablation Study Finished"
echo "  Total planned  : $TOTAL_RUNS"
echo "  Completed      : $COMPLETED_RUNS"
echo "  Failed         : $FAILED_RUNS"
echo "  Summary JSON   : $SUMMARY_FILE"
echo "  Outputs root   : $BASE_OUTPUT_DIR"
echo ""
