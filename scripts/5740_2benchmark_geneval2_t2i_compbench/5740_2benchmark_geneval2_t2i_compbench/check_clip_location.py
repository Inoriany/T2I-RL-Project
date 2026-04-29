#!/usr/bin/env python3
"""Check where CLIP model is located and create symlink if needed"""

import os
from pathlib import Path

def find_clip_model():
    """Find CLIP model in various locations"""

    model_id = "timm/vit_large_patch14_clip_224.openai"
    model_filename = "open_clip_pytorch_model.bin"

    print("=" * 70)
    print("Checking CLIP Model Location")
    print("=" * 70)
    print()

    # Possible locations
    locations = [
        ("HuggingFace cache (standard)",
         Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}"),
        ("ModelScope cache (standard)",
         Path.home() / ".cache" / "modelscope" / "hub" / model_id),
        ("ModelScope cache (models subdir)",
         Path.home() / ".cache" / "modelscope" / "hub" / "models" / model_id),
        ("ModelScope cache (alternate)",
         Path.home() / ".cache" / "modelscope" / "hub"),
    ]

    found_paths = []

    for name, path in locations:
        print(f"Checking: {name}")
        print(f"  Path: {path}")

        if not path.exists():
            print(f"  Status: ✗ Does not exist")
        else:
            # Look for the model file
            if path.is_dir():
                # Check for model file directly
                model_file = path / model_filename
                if model_file.exists():
                    print(f"  Status: ✓ Found model file!")
                    print(f"  File: {model_file}")
                    found_paths.append((name, path, model_file))
                else:
                    # Check subdirectories
                    found = False
                    for subdir in path.rglob("*"):
                        if subdir.name == model_filename:
                            print(f"  Status: ✓ Found model file in subdir!")
                            print(f"  File: {subdir}")
                            found_paths.append((name, subdir.parent, subdir))
                            found = True
                            break
                    if not found:
                        print(f"  Status: ✓ Directory exists, but model file not found")
                        print(f"  Contents:")
                        for item in path.iterdir():
                            print(f"    - {item.name}")
            else:
                print(f"  Status: ? Path exists but is not a directory")
        print()

    return found_paths

def create_symlink_if_needed(source_path, target_path):
    """Create symlink from source to target if needed"""

    print("=" * 70)
    print("Creating Symlink (if needed)")
    print("=" * 70)
    print()

    # Check if target already exists
    if target_path.exists():
        if target_path.is_symlink():
            print(f"Target already exists as symlink: {target_path}")
            print(f"  -> Points to: {target_path.resolve()}")
            return True
        else:
            print(f"Target already exists (not a symlink): {target_path}")
            return True

    # Create parent directories
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Create symlink
    try:
        os.symlink(source_path, target_path, target_is_directory=True)
        print(f"✓ Created symlink:")
        print(f"  Source: {source_path}")
        print(f"  Target: {target_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to create symlink: {e}")
        return False

def main():
    found = find_clip_model()

    if not found:
        print("=" * 70)
        print("✗ CLIP Model Not Found!")
        print("=" * 70)
        print()
        print("The model doesn't seem to be downloaded. Please run:")
        print()
        print("  modelscope download --model timm/vit_large_patch14_clip_224.openai")
        print()
        print("Or if you have the model elsewhere, please provide the path manually.")
        return 1

    print("=" * 70)
    print(f"✓ Found {len(found)} location(s) with CLIP model")
    print("=" * 70)
    print()

    # Use the first found location
    name, model_dir, model_file = found[0]

    # Check if it's in the HuggingFace cache (where open_clip looks)
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    model_id = "timm/vit_large_patch14_clip_224.openai"
    expected_hf_path = hf_cache / f"models--{model_id.replace('/', '--')}"

    if expected_hf_path.exists() or str(model_dir) == str(expected_hf_path):
        print(f"✓ Model is in HuggingFace cache (where open_clip expects it)")
        print(f"  Location: {model_dir}")
        return 0

    # Model is in ModelScope cache, need to create symlink or copy
    print(f"Model is in: {name}")
    print(f"  {model_dir}")
    print()
    print(f"But open_clip expects it in HuggingFace cache:")
    print(f"  {expected_hf_path}")
    print()

    # Ask user if they want to create symlink
    print("Options:")
    print("1. Create symlink (fast, saves space)")
    print("2. Copy files (slower, uses more space)")
    print("3. Do nothing (model won't be found)")
    print()

    response = input("Choose option (1/2/3): ").strip()

    if response == "1":
        if create_symlink_if_needed(model_dir, expected_hf_path):
            print()
            print("✓ Symlink created successfully!")
            print("You can now run the training script.")
            return 0
        else:
            print()
            print("✗ Failed to create symlink")
            return 1
    elif response == "2":
        print()
        print("To copy files, run:")
        print(f"  cp -r {model_dir} {expected_hf_path}")
        print()
        print("After copying, you can run the training script.")
        return 0
    else:
        print()
        print("No action taken. The model may not be found by open_clip.")
        return 0

if __name__ == "__main__":
    exit(main())
