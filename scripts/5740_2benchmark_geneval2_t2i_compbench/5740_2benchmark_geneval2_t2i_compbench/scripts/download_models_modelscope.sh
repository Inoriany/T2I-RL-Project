#!/bin/bash
# Download all required models from ModelScope
# This script downloads:
#   1. Janus-Pro-1B (main generation model)
#   2. CLIP models (for reward calculation)
#   3. Other auxiliary models

set -euo pipefail

echo "================================================"
echo "Downloading models from ModelScope"
echo "================================================"
echo ""

# Check if modelscope is installed
if ! command -v modelscope &> /dev/null; then
    echo "Error: modelscope CLI not found"
    echo "Please install it: pip install modelscope"
    exit 1
fi

# Create cache directories
mkdir -p ~/.cache/modelscope/hub
mkdir -p ~/.cache/huggingface/hub

echo "Step 1/3: Downloading Janus-Pro-1B (main model)..."
echo "-----------------------------------------------"
modelscope download --model deepseek-ai/Janus-Pro-1B
echo "✓ Janus-Pro-1B downloaded"
echo ""

echo "Step 2/3: Downloading CLIP models (for reward calculation)..."
echo "-----------------------------------------------"
# ViT-L-14 (most commonly used)
echo "→ Downloading CLIP ViT-L-14 (openai)..."
modelscope download --model timm/vit_large_patch14_clip_224.openai || echo "Warning: Failed to download CLIP ViT-L-14"

# ViT-B-32 (alternative)
echo "→ Downloading CLIP ViT-B-32 (openai)..."
modelscope download --model timm/vit_base_patch32_clip_224.openai || echo "Warning: Failed to download CLIP ViT-B-32"

echo "✓ CLIP models downloaded (or already exist)"
echo ""

echo "Step 2.5/3: Setting up HuggingFace cache structure..."
echo "-----------------------------------------------"
# Run the setup script to create proper HF cache structure
if [[ -f "setup_hf_cache.py" ]]; then
    python3 setup_hf_cache.py
else
    echo "Warning: setup_hf_cache.py not found, skipping cache setup"
fi
echo ""

echo "Step 3/4: Downloading Qwen2.5-VL-3B-Instruct (VLM reward model)..."
echo "-----------------------------------------------"
modelscope download --model Qwen/Qwen2.5-VL-3B-Instruct
echo "✓ Qwen2.5-VL-3B-Instruct downloaded"
echo ""

echo "Step 4/4: Other auxiliary models (optional)..."
echo "-----------------------------------------------"
# These are downloaded on-demand by the scripts
echo "→ Other models (BLIP, OwlViT, etc.) will be downloaded automatically when needed"
echo ""

echo "================================================"
echo "All required models downloaded!"
echo "================================================"
echo ""
echo "You can now run:"
echo "  export USE_MODELSCOPE=true"
echo "  bash scripts/run_ablation.sh"
echo ""
echo "Model cache locations:"
echo "  - Main models: ~/.cache/modelscope/hub/"
echo "  - CLIP models: ~/.cache/huggingface/hub/"
echo ""
