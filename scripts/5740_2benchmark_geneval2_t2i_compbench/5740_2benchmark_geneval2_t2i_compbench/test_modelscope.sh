#!/bin/bash
# Test ModelScope integration

set -euo pipefail

echo "Testing ModelScope integration..."
echo ""

# Test 1: Check if modelscope is installed
echo "Test 1: Checking if modelscope is installed..."
if python3 -c "from modelscope import snapshot_download" 2>/dev/null; then
    echo "  ✓ modelscope is installed"
else
    echo "  ✗ modelscope is not installed"
    echo "  Run: pip install modelscope"
    exit 1
fi

# Test 2: Test getting model path from ModelScope
echo ""
echo "Test 2: Testing model path retrieval..."
MODEL_ID="deepseek-ai/Janus-Pro-1B"
CACHE_DIR="${HOME}/.cache/modelscope/hub"

echo "  → Getting path for model: $MODEL_ID"
echo "  → Cache directory: $CACHE_DIR"

# Create temp file for output
TEMP_OUTPUT="/tmp/modelscope_test_output_$$.txt"

python3 - "$MODEL_ID" "$CACHE_DIR" "$TEMP_OUTPUT" <<'PYEOF'
import os
import sys
import logging

# Redirect all logging to stderr
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
os.environ['MODELSCOPE_LOG_LEVEL'] = '40'  # ERROR level

# Temporarily redirect stdout to suppress modelscope's prints
original_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')

try:
    from modelscope import snapshot_download
except ImportError:
    sys.stdout = original_stdout
    print("Error: modelscope not installed", file=sys.stderr)
    sys.exit(1)

# Restore stdout
sys.stdout = original_stdout

model_id = sys.argv[1]
cache_dir = sys.argv[2]
temp_output = sys.argv[3]

os.makedirs(cache_dir, exist_ok=True)

try:
    local_path = snapshot_download(model_id, cache_dir=cache_dir)
    with open(temp_output, 'w') as f:
        f.write(local_path)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
PYEOF

if [[ -f "$TEMP_OUTPUT" ]]; then
    LOCAL_PATH=$(cat "$TEMP_OUTPUT")
    rm -f "$TEMP_OUTPUT"
else
    echo "  ✗ Failed to get model path"
    exit 1
fi

if [[ -n "$LOCAL_PATH" ]]; then
    echo "  ✓ Got model path: $LOCAL_PATH"

    if [[ -d "$LOCAL_PATH" ]]; then
        echo "  ✓ Path exists and is a directory"
        echo ""
        echo "Test 3: Checking for required files..."

        # Check for key files
        found_count=0
        for file in "config.json" "pytorch_model.bin" "tokenizer.json"; do
            if [[ -f "$LOCAL_PATH/$file" ]]; then
                echo "  ✓ Found $file"
                ((found_count++)) || true
            else
                echo "  ✗ Missing $file"
            fi
        done
        
        if [[ $found_count -ge 2 ]]; then
            echo ""
            echo "=========================================="
            echo "All tests passed!"
            echo "You can now run: USE_MODELSCOPE=true bash scripts/run_ablation.sh"
            echo "=========================================="
        else
            echo ""
            echo "  ⚠ Warning: Some model files are missing"
            echo "  The model may need to be fully downloaded first"
        fi
    else
        echo "  ✗ Path does not exist: $LOCAL_PATH"
        
        # Try alternative path (models/ subdirectory)
        ALT_PATH="${CACHE_DIR}/models/${MODEL_ID}"
        echo "  → Trying alternative path: $ALT_PATH"
        if [[ -d "$ALT_PATH" ]]; then
            echo "  ✓ Found model at alternative path!"
            echo "  You may need to set: export MODELSCOPE_CACHE_DIR=\"${CACHE_DIR}/models\""
        else
            exit 1
        fi
    fi
else
    echo "  ✗ Got empty path"
    exit 1
fi
