#!/usr/bin/env python3
"""
Setup HuggingFace cache structure for models downloaded via ModelScope.

This script creates the proper HuggingFace cache structure that open_clip expects:
  ~/.cache/huggingface/hub/models--<org>--<model>/snapshots/main/<files>

Usage:
  python setup_hf_cache.py
"""

import os
import sys
import json
import hashlib
from pathlib import Path


def create_hf_cache_structure(source_dir: Path, model_id: str):
    """
    Create HuggingFace cache structure from ModelScope download.

    HuggingFace structure:
      models--{org}--{model}/
        snapshots/
          {commit_hash}/
            <model_files>
        refs/
          main -> {commit_hash}

    Args:
        source_dir: Path to ModelScope downloaded model
        model_id: Model ID (e.g., "timm/vit_large_patch14_clip_224.openai")
    """

    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    safe_model_id = model_id.replace("/", "--")
    model_cache_dir = hf_cache / f"models--{safe_model_id}"

    print(f"Creating HF cache structure for: {model_id}")
    print(f"  Source: {source_dir}")
    print(f"  Target: {model_cache_dir}")

    # Create directories
    snapshots_dir = model_cache_dir / "snapshots"
    refs_dir = model_cache_dir / "refs"

    snapshots_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)

    # Generate a fake commit hash (HuggingFace uses SHA, we use MD5 of model_id)
    commit_hash = hashlib.md5(model_id.encode()).hexdigest()
    snapshot_dir = snapshots_dir / commit_hash
    snapshot_dir.mkdir(exist_ok=True)

    print(f"  Snapshot: {snapshot_dir}")

    # Copy or symlink files from source to snapshot directory
    for item in source_dir.iterdir():
        target = snapshot_dir / item.name

        if target.exists() or target.is_symlink():
            print(f"  ✓ Already exists: {item.name}")
            continue

        try:
            # Create symlink (preferred - saves space)
            os.symlink(item.absolute(), target)
            print(f"  → Symlinked: {item.name}")
        except Exception as e:
            # Fallback: copy file
            import shutil
            if item.is_file():
                shutil.copy2(item, target)
                print(f"  → Copied: {item.name}")
            elif item.is_dir():
                shutil.copytree(item, target)
                print(f"  → Copied dir: {item.name}")

    # Create refs/main pointing to commit hash
    refs_main = refs_dir / "main"
    refs_main.write_text(commit_hash)
    print(f"  → Created refs/main -> {commit_hash}")

    # Create index.json (HuggingFace uses this)
    index_file = model_cache_dir / "snapshots" / commit_hash / "index.json"
    if not index_file.exists():
        index_data = {"version": "0.1", "files": {}}
        for item in source_dir.iterdir():
            if item.is_file():
                index_data["files"][item.name] = {"size": item.stat().st_size}
        index_file.write_text(json.dumps(index_data, indent=2))
        print(f"  → Created index.json")

    print(f"\n✓ HF cache structure created at: {model_cache_dir}")
    return model_cache_dir


def find_modelscope_models():
    """Find models downloaded via ModelScope."""

    ms_cache = Path.home() / ".cache" / "modelscope" / "hub"

    if not ms_cache.exists():
        print(f"ModelScope cache not found: {ms_cache}")
        return []

    models = []

    # Check models/ subdirectory
    models_dir = ms_cache / "models"
    if models_dir.exists():
        for org_dir in models_dir.iterdir():
            if org_dir.is_dir():
                for model_dir in org_dir.iterdir():
                    if model_dir.is_dir():
                        model_id = f"{org_dir.name}/{model_dir.name}"
                        models.append((model_id, model_dir))

    # Also check direct downloads
    for org_dir in ms_cache.iterdir():
        if org_dir.is_dir() and org_dir.name != "models":
            for model_dir in org_dir.iterdir():
                if model_dir.is_dir():
                    model_id = f"{org_dir.name}/{model_dir.name}"
                    # Check if already found
                    if not any(m[0] == model_id for m in models):
                        models.append((model_id, model_dir))

    return models


def main():
    print("=" * 70)
    print("Setting up HuggingFace Cache Structure")
    print("=" * 70)
    print()

    # Find ModelScope models
    ms_models = find_modelscope_models()

    if not ms_models:
        print("✗ No ModelScope models found")
        print()
        print("Please download models first:")
        print("  modelscope download --model timm/vit_large_patch14_clip_224.openai")
        return 1

    print(f"Found {len(ms_models)} model(s) in ModelScope cache:")
    for model_id, path in ms_models:
        print(f"  - {model_id}")
        print(f"    Location: {path}")
    print()

    # Process each model
    processed = []
    for model_id, source_dir in ms_models:
        # Check if it's a CLIP model (needed by open_clip)
        if "clip" in model_id.lower() or "timm/vit" in model_id:
            print(f"Processing: {model_id}")
            try:
                cache_dir = create_hf_cache_structure(source_dir, model_id)
                processed.append((model_id, cache_dir))
                print()
            except Exception as e:
                print(f"✗ Failed to process {model_id}: {e}")
                import traceback
                traceback.print_exc()
                print()

    if processed:
        print("=" * 70)
        print("✓ Setup complete!")
        print("=" * 70)
        print()
        print("Processed models:")
        for model_id, cache_dir in processed:
            print(f"  - {model_id}")
            print(f"    HF cache: {cache_dir}")
        print()
        print("You can now run:")
        print("  USE_MODELSCOPE=true bash scripts/run_ablation.sh")
        return 0
    else:
        print("✗ No models were processed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
