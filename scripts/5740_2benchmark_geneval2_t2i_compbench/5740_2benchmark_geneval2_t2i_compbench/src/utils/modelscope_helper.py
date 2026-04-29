"""
ModelScope Helper
=================

Helper functions for downloading and using models from ModelScope
instead of HuggingFace. Useful when HuggingFace is not accessible.

Usage:
    1. Set environment variable: USE_MODELSCOPE=true
    2. Import this module early in your script (before transformers)
    3. Or use the provided download functions

Example:
    import os
    os.environ['USE_MODELSCOPE'] = 'true'
    from src.utils import modelscope_helper
    modelscope_helper.setup_modelscope()

    # Now use transformers as usual - it will use ModelScope mirrors
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Some shells / images set HF_ENDPOINT to huggingface.modelscope.cn, which often
# fails DNS. huggingface_hub and open_clip then cannot download timm/CLIP weights.
# The official ModelScope mirror for HuggingFace-compatible URLs is www.modelscope.cn/hf .
_BAD_HF_HOST = "huggingface.modelscope.cn"
_GOOD_MS_HF_MIRROR = "https://www.modelscope.cn/hf"
if _BAD_HF_HOST in (os.environ.get("HF_ENDPOINT") or ""):
    os.environ["HF_ENDPOINT"] = _GOOD_MS_HF_MIRROR


def setup_modelscope():
    """
    Setup ModelScope integration with transformers.

    This function patches the HuggingFace transformers library to use
    ModelScope mirrors for model downloads. Should be called before
    importing transformers.

    It works by setting HF_ENDPOINT to ModelScope's HuggingFace mirror.
    """
    # Method 1: Use ModelScope's HF mirror endpoint
    os.environ.setdefault("HF_ENDPOINT", "https://www.modelscope.cn/hf")
    os.environ.setdefault("HF_HUB_OFFLINE", "0")

    print("[ModelScope] Enabled ModelScope mirror for HuggingFace models")
    print(f"[ModelScope] HF_ENDPOINT set to: {os.environ.get('HF_ENDPOINT')}")


def download_model(
    model_id: str,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    """
    Download a model from ModelScope.

    Args:
        model_id: Model ID on ModelScope (e.g., "deepseek-ai/Janus-Pro-1B")
                  Most models use the same ID as HuggingFace
        cache_dir: Directory to cache downloaded models
        revision: Specific revision to download

    Returns:
        Local path to the downloaded model

    Raises:
        ImportError: If modelscope package is not installed
        RuntimeError: If download fails
    """
    try:
        from modelscope import snapshot_download
    except ImportError:
        raise ImportError(
            "modelscope package is required. Install it with:\n"
            "  pip install modelscope"
        )

    if cache_dir is None:
        cache_dir = os.environ.get("MODELSCOPE_CACHE", "./modelscope_models")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    print(f"[ModelScope] Downloading model: {model_id}")
    print(f"[ModelScope] Cache directory: {cache_dir}")

    try:
        local_path = snapshot_download(
            model_id,
            cache_dir=str(cache_path),
            revision=revision,
        )
        print(f"[ModelScope] Model downloaded to: {local_path}")
        return local_path
    except Exception as e:
        raise RuntimeError(f"Failed to download model from ModelScope: {e}")


def get_model_path(
    model_id: str,
    use_modelscope: Optional[bool] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """
    Get the local path to a model, downloading from ModelScope if needed.

    This is a convenience function that:
    1. Checks if USE_MODELSCOPE environment variable is set
    2. Downloads from ModelScope if enabled
    3. Returns the original model_id otherwise (for HuggingFace)

    Args:
        model_id: Model ID (e.g., "deepseek-ai/Janus-Pro-1B")
        use_modelscope: Override USE_MODELSCOPE env var
        cache_dir: Cache directory for ModelScope downloads

    Returns:
        Path to the model (local path if downloaded, original ID otherwise)
    """
    if use_modelscope is None:
        use_modelscope = os.environ.get("USE_MODELSCOPE", "false").lower() == "true"

    if use_modelscope:
        return download_model(model_id, cache_dir)

    return model_id


def setup_open_clip_modelscope():
    """
    Setup ModelScope for open_clip models.

    open_clip uses timm models that are hosted on HuggingFace.
    This function configures open_clip to use ModelScope mirrors.

    It also pre-downloads common CLIP models from ModelScope.
    """
    import warnings

    # Set HF endpoint for timm/huggingface_hub
    os.environ.setdefault("HF_ENDPOINT", "https://www.modelscope.cn/hf")
    os.environ.setdefault("HF_HUB_OFFLINE", "0")

    # Disable SSL verification if needed (some environments have SSL issues)
    # os.environ.setdefault("CURL_CA_BUNDLE", "")
    # os.environ.setdefault("REQUESTS_CA_BUNDLE", "")

    print("[ModelScope] Configured for open_clip models")


def download_open_clip_model(
    model_name: str = "ViT-L-14",
    pretrained: str = "openai",
    cache_dir: Optional[str] = None,
) -> str:
    """
    Download a CLIP model from ModelScope for use with open_clip.

    Common models:
        - ViT-L-14 / openai
        - ViT-B-32 / openai
        - ViT-B-16 / openai

    Args:
        model_name: CLIP model architecture name
        pretrained: Pretrained weights source
        cache_dir: Cache directory

    Returns:
        Local path to the model directory
    """
    # Map open_clip model names to ModelScope/HF model IDs
    # These models are typically under the timm organization on HF
    model_mapping = {
        ("ViT-L-14", "openai"): "timm/vit_large_patch14_clip_224.openai",
        ("ViT-B-32", "openai"): "timm/vit_base_patch32_clip_224.openai",
        ("ViT-B-16", "openai"): "timm/vit_base_patch16_clip_224.openai",
    }

    model_id = model_mapping.get((model_name, pretrained))

    if model_id is None:
        # Try to construct the model ID
        print(f"[ModelScope] Warning: Unknown model {model_name}/{pretrained}, "
              f"will try to use as-is from HuggingFace")
        return f"{model_name}/{pretrained}"

    print(f"[ModelScope] Downloading open_clip model: {model_id}")

    try:
        local_path = download_model(model_id, cache_dir)
        print(f"[ModelScope] open_clip model ready at: {local_path}")
        return local_path
    except Exception as e:
        print(f"[ModelScope] Warning: Failed to download {model_id}: {e}")
        print(f"[ModelScope] Will try to use default HuggingFace download")
        raise


# Auto-setup if USE_MODELSCOPE is set in environment
if os.environ.get("USE_MODELSCOPE", "false").lower() == "true":
    try:
        setup_modelscope()
        setup_open_clip_modelscope()
    except Exception as e:
        print(f"[ModelScope] Warning: Failed to setup ModelScope: {e}", file=sys.stderr)
