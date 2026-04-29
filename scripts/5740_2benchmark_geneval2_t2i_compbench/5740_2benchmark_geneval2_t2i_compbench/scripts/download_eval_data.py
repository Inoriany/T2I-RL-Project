#!/usr/bin/env python3
"""
Download Official Evaluation Benchmark Data
============================================

Downloads the **official** benchmark data from their original sources:

  • T2I-CompBench  — HuggingFace dataset  ``limingcv/T2I-CompBench``
                     (paper: https://karine-h.github.io/T2I-CompBench/)
                     (repo:  https://github.com/Karine-Huang/T2I-CompBench)

  • GenEval-2      — GitHub raw JSONL
                     (paper: https://arxiv.org/abs/2404.08087)
                     (repo:  https://github.com/facebookresearch/GenEval2)

Run this script on a machine with internet access BEFORE training / evaluation:

    python scripts/download_eval_data.py
    python scripts/download_eval_data.py --benchmarks t2i_compbench
    python scripts/download_eval_data.py --benchmarks geneval2
    python scripts/download_eval_data.py --data_root /path/to/data

Requirements (install once):
    pip install datasets huggingface_hub tqdm

The script writes data to:
    <data_root>/t2i_compbench/   — 6 category JSON files (val split)
    <data_root>/geneval2/        — geneval2_data.jsonl  +  geneval2_data.json
"""

import argparse
import json
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# T2I-CompBench
# ─────────────────────────────────────────────────────────────────────────────

# Official categories in the limingcv/T2I-CompBench HuggingFace dataset
T2I_COMPBENCH_CATEGORIES = [
    "color",
    "shape",
    "texture",
    "spatial",
    "non_spatial",
    "complex",
]

# HuggingFace dataset configuration names that map to each category.
# Run:  from datasets import get_dataset_config_names
#       get_dataset_config_names("limingcv/T2I-CompBench")
# to verify these names if the dataset is ever restructured.
T2I_COMPBENCH_CONFIGS = {
    "color":       "color_attr",
    "shape":       "shape_attr",
    "texture":     "texture_attr",
    "spatial":     "spatial_rel",
    "non_spatial": "non_spatial_rel",
    "complex":     "complex",
}


def download_t2i_compbench(data_root: Path) -> None:
    """Download T2I-CompBench from HuggingFace (limingcv/T2I-CompBench)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "[ERROR] 'datasets' package not found.\n"
            "Install with:  pip install datasets huggingface_hub\n"
            "Then re-run this script."
        )
        sys.exit(1)

    save_dir = data_root / "t2i_compbench"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Downloading T2I-CompBench  →  {save_dir}")
    print(f"{'='*60}")

    all_prompts: dict[str, list[dict]] = {}

    for category, config_name in T2I_COMPBENCH_CONFIGS.items():
        print(f"\n  [{category}]  loading config '{config_name}' …")
        try:
            ds = load_dataset("limingcv/T2I-CompBench", config_name, split="validation")
        except Exception:
            # Some configs may only have a 'train' split
            try:
                ds = load_dataset("limingcv/T2I-CompBench", config_name, split="train")
                print(f"    (no 'validation' split — using 'train')")
            except Exception as exc:
                print(f"    [WARN] Could not load config '{config_name}': {exc}")
                continue

        records = []
        for item in ds:
            record = {"prompt": item.get("prompt", item.get("caption", ""))}
            # Preserve all metadata fields the original dataset provides
            for k, v in item.items():
                if k not in ("image",):      # skip raw PIL images
                    record[k] = v
            records.append(record)

        # Save per-category file
        out_file = save_dir / f"{category}_val.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        all_prompts[category] = records
        print(f"    Saved {len(records)} records  →  {out_file}")

    # Save unified prompts file used by T2ICompBenchEvaluator
    prompts_file = save_dir / "prompts.json"
    with open(prompts_file, "w", encoding="utf-8") as f:
        json.dump(all_prompts, f, indent=2, ensure_ascii=False)
    total = sum(len(v) for v in all_prompts.values())
    print(f"\n  Combined prompts file  →  {prompts_file}  ({total} total)")
    print("  T2I-CompBench download complete ✓")


# ─────────────────────────────────────────────────────────────────────────────
# GenEval-2
# ─────────────────────────────────────────────────────────────────────────────

# All known URLs for the official GenEval-2 JSONL file.
# The script tries them in order and stops at the first success.
GENEVAL2_URLS = [
    # GitHub raw — primary source
    "https://raw.githubusercontent.com/facebookresearch/GenEval2/main/geneval2_data.jsonl",
    # HuggingFace mirror (if the repo ever moves there)
    "https://huggingface.co/datasets/facebook/GenEval2/resolve/main/geneval2_data.jsonl",
]


def _http_download(url: str, dest: Path) -> bool:
    """Download *url* to *dest*. Returns True on success."""
    import urllib.request
    import urllib.error

    print(f"    Trying  {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 64 * 1024
            with open(dest, "wb") as f:
                while True:
                    data = resp.read(chunk)
                    if not data:
                        break
                    f.write(data)
                    downloaded += len(data)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r    {downloaded/1024:.0f} KB / {total/1024:.0f} KB  ({pct:.0f}%)", end="", flush=True)
        print()  # newline after progress
        return True
    except (urllib.error.URLError, OSError) as exc:
        print(f"    Failed: {exc}")
        if dest.exists():
            dest.unlink()
        return False


def download_geneval2(data_root: Path) -> None:
    """Download GenEval-2 from the official GitHub repo."""
    save_dir = data_root / "geneval2"
    save_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = save_dir / "geneval2_data.jsonl"

    print(f"\n{'='*60}")
    print(f"Downloading GenEval-2  →  {save_dir}")
    print(f"{'='*60}")

    success = False
    for url in GENEVAL2_URLS:
        if _http_download(url, jsonl_path):
            success = True
            break

    if not success:
        print(
            "\n[ERROR] Could not download GenEval-2 data from any known URL.\n"
            "Please download it manually:\n"
            "  1. Visit https://github.com/facebookresearch/GenEval2\n"
            "  2. Download geneval2_data.jsonl\n"
            f"  3. Place it at:  {jsonl_path}"
        )
        sys.exit(1)

    # Validate + count records
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] Line {line_no} is not valid JSON: {exc}")

    print(f"  Loaded {len(records)} records from JSONL")

    # Also write a .json version (list-of-dicts) for tools that prefer it
    json_path = save_dir / "geneval2_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"  JSON mirror  →  {json_path}")

    # Print breakdown by atom_count
    from collections import Counter
    counts = Counter(r.get("atom_count", r.get("num_objects", "?")) for r in records)
    print("  Breakdown by atom_count:")
    for k in sorted(counts):
        print(f"    atom_count={k}: {counts[k]} prompts")

    print("  GenEval-2 download complete ✓")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download official T2I evaluation benchmark data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["t2i_compbench", "geneval2", "all"],
        default=["all"],
        help="Which benchmarks to download (default: all)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory for benchmark data (default: ./data)",
    )
    args = parser.parse_args()

    benchmarks = args.benchmarks
    if "all" in benchmarks:
        benchmarks = ["t2i_compbench", "geneval2"]

    data_root = Path(args.data_root).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    print(f"Data root: {data_root}")
    print(f"Benchmarks: {benchmarks}")

    if "t2i_compbench" in benchmarks:
        download_t2i_compbench(data_root)

    if "geneval2" in benchmarks:
        download_geneval2(data_root)

    print("\n" + "=" * 60)
    print("All benchmark data downloaded successfully.")
    print("=" * 60)
    print(f"\nYou can now run evaluation with:")
    print(f"  python scripts/evaluate_benchmarks.py \\")
    print(f"      --model_path deepseek-ai/Janus-Pro-1B \\")
    print(f"      --data_root {data_root} \\")
    print(f"      --output_dir ./evaluation_results/base")


if __name__ == "__main__":
    main()
