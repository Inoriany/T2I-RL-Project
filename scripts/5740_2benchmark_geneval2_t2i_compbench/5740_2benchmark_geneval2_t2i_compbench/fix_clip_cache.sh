#!/bin/bash
# Fix CLIP model cache - create proper HuggingFace cache structure from ModelScope download

set -euo pipefail

echo "================================================"
echo "Fixing CLIP Model Cache"
echo "================================================"
echo ""

# Check if Python script exists
if [[ ! -f "setup_hf_cache.py" ]]; then
    echo "✗ setup_hf_cache.py not found"
    exit 1
fi

# Run the Python setup script
echo "→ Setting up HuggingFace cache structure..."
python3 setup_hf_cache.py

echo ""
echo "================================================"
echo "Done!"
echo "================================================"
