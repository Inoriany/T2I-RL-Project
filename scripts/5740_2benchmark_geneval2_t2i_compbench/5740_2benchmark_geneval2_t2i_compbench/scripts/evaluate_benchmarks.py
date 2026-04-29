#!/usr/bin/env python3
"""
Post-Training Benchmark Evaluation Script
==========================================

Evaluates a trained (or base) Janus-Pro checkpoint on:
  • T2I-CompBench  (non-MLLM: BLIP-VQA, OwlViT, CLIPScore)
  • GenEval-2      (non-MLLM: OwlViT, CLIP colour)

Usage::

    # Evaluate the base model (no checkpoint)
    python scripts/evaluate_benchmarks.py \\
        --model_path deepseek-ai/Janus-Pro-1B \\
        --output_dir evaluation_results/base

    # Evaluate a LoRA checkpoint
    python scripts/evaluate_benchmarks.py \\
        --model_path deepseek-ai/Janus-Pro-1B \\
        --lora_checkpoint ./outputs/r16_a32_clip/checkpoint-epoch-0 \\
        --output_dir evaluation_results/r16_a32_clip

    # Run only specific benchmarks
    python scripts/evaluate_benchmarks.py \\
        --model_path deepseek-ai/Janus-Pro-1B \\
        --benchmarks t2i_compbench \\
        --output_dir evaluation_results/t2i_only

    # Limit samples for a quick test
    python scripts/evaluate_benchmarks.py \\
        --model_path deepseek-ai/Janus-Pro-1B \\
        --max_samples 20 \\
        --output_dir evaluation_results/quick_test

Environment Variables:
    USE_MODELSCOPE: Set to "true" to use ModelScope instead of HuggingFace
    MODELSCOPE_CACHE / MODELSCOPE_CACHE_DIR: ModelScope hub root (default: ~/.cache/modelscope/hub).
        If BLIP/OwlViT are downloaded there, local paths are used automatically (no Hub download).

    Janus base model: if you pass a HuggingFace-style id (e.g. deepseek-ai/Janus-Pro-1B) but the
    weights live under the ModelScope cache (from an earlier snapshot_download), this script
    resolves to that local directory so transformers loads from disk instead of the HF hub.
"""

import os
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# open_clip/timm use huggingface_hub; a bad HF_ENDPOINT breaks CLIP downloads.
# huggingface.modelscope.cn often does not resolve — replace with the official mirror.
_hf_ep = os.environ.get("HF_ENDPOINT") or ""
if "huggingface.modelscope.cn" in _hf_ep:
    os.environ["HF_ENDPOINT"] = "https://www.modelscope.cn/hf"

# Setup ModelScope support BEFORE importing transformers
# This will redirect HuggingFace downloads to ModelScope mirrors if USE_MODELSCOPE=true
if os.environ.get("USE_MODELSCOPE", "false").lower() == "true":
    from src.utils import modelscope_helper
    modelscope_helper.setup_modelscope()

import torch


def _modelscope_hub_root_dirs() -> List[Path]:
    """Ordered list of ModelScope hub roots to search for already-downloaded models."""
    seen = set()
    out: List[Path] = []
    for key in ("MODELSCOPE_CACHE_DIR", "MODELSCOPE_CACHE"):
        v = os.environ.get(key)
        if not v:
            continue
        p = Path(v).expanduser().resolve()
        if p not in seen:
            seen.add(p)
            out.append(p)
    default = (Path.home() / ".cache/modelscope/hub").resolve()
    if default not in seen:
        out.append(default)
    return out


def _is_pretrained_dir(p: Path) -> bool:
    return p.is_dir() and (p / "config.json").is_file()


def find_pretrained_dir_under_modelscope(model_id: str) -> Optional[Path]:
    """
    Locate snapshot_download layout for model_id under the ModelScope hub:
      <hub>/models/<model_id>/ or .../snapshots/<hash>/
    """
    if "/" not in model_id:
        return None
    for root in _modelscope_hub_root_dirs():
        for rel in (Path("models") / model_id, Path(model_id)):
            base = root / rel
            if _is_pretrained_dir(base):
                return base
            snaps = base / "snapshots"
            if snaps.is_dir():
                candidates = [
                    d for d in snaps.iterdir()
                    if d.is_dir() and _is_pretrained_dir(d)
                ]
                if candidates:
                    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    return candidates[0]
    return None


def resolve_base_model_path(model_path: str) -> str:
    """
    Prefer a local directory for the Janus base model when weights live under
    ModelScope cache (different layout from HuggingFace hub cache).
    """
    p = Path(model_path).expanduser()
    if p.is_dir() and (p / "config.json").is_file():
        return str(p.resolve())

    if "/" not in model_path:
        return model_path

    found = find_pretrained_dir_under_modelscope(model_path)
    if found is not None:
        return str(found)

    if os.environ.get("USE_MODELSCOPE", "false").lower() == "true":
        try:
            from src.utils.modelscope_helper import get_model_path

            cache = os.environ.get("MODELSCOPE_CACHE_DIR") or os.environ.get("MODELSCOPE_CACHE")
            out = get_model_path(model_path, use_modelscope=True, cache_dir=cache)
            if Path(out).is_dir():
                return out
        except Exception as e:
            print(f"[Eval] ModelScope get_model_path failed: {e}", file=sys.stderr)

    return model_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate T2I model on benchmarks")

    # Model
    parser.add_argument("--model_path",  type=str, default="deepseek-ai/Janus-Pro-1B")
    parser.add_argument("--lora_checkpoint", type=str, default=None,
                        help="Path to LoRA checkpoint directory (optional)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cuda")

    # Benchmarks
    parser.add_argument("--benchmarks", nargs="+",
                        choices=["t2i_compbench", "geneval2", "all"],
                        default=["all"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per benchmark (None = use all)")
    parser.add_argument("--max_samples_per_category", type=int, default=None,
                        help="Max samples per T2I-CompBench category")

    # OwlViT / BLIP / CLIP
    parser.add_argument("--owl_model",   type=str, default="google/owlv2-base-patch16-ensemble")
    parser.add_argument("--blip_model",  type=str, default="Salesforce/blip-vqa-base")
    parser.add_argument("--clip_model",  type=str, default="ViT-L-14")
    parser.add_argument("--clip_pretrained", type=str, default="openai")
    parser.add_argument("--owl_threshold", type=float, default=0.10)

    # Output
    parser.add_argument("--output_dir", type=str, default="./evaluation_results")
    parser.add_argument("--save_images", action="store_true", default=True)
    parser.add_argument("--no_save_images", dest="save_images", action="store_false")

    return parser.parse_args()


def load_generator(args):
    """Load Janus-Pro generator, optionally with a LoRA checkpoint."""
    from src.models.generators import JanusProGenerator

    dtype = getattr(torch, args.dtype)
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"[Eval] Loading generator: {args.model_path} ({args.dtype})")
    generator = JanusProGenerator(
        model_name_or_path=args.model_path,
        dtype=dtype,
        device=device,
    )
    generator.load_model()

    if args.lora_checkpoint:
        print(f"[Eval] Loading LoRA checkpoint: {args.lora_checkpoint}")
        generator.enable_lora(lora_path=args.lora_checkpoint)

    return generator


def run_t2i_compbench(generator, args) -> dict:
    """Run T2I-CompBench evaluation."""
    from src.evaluation.t2i_compbench_eval import T2ICompBenchEvaluator

    data_dir = Path(args.data_root) / "t2i_compbench"
    output_dir = Path(args.output_dir) / "t2i_compbench"
    device = args.device if torch.cuda.is_available() else "cpu"

    evaluator = T2ICompBenchEvaluator(
        data_dir=str(data_dir),
        device=device,
        blip_model=args.blip_model,
        owl_model=args.owl_model,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
    )
    evaluator.load_models()

    results = evaluator.evaluate(
        generator,
        max_samples_per_category=args.max_samples_per_category or args.max_samples,
        output_dir=str(output_dir) if args.save_images else None,
    )
    return results


def run_geneval2(generator, args) -> dict:
    """Run GenEval-2 evaluation."""
    from src.evaluation.geneval2_eval import GenEval2Evaluator

    data_dir   = Path(args.data_root) / "geneval2"
    output_dir = Path(args.output_dir) / "geneval2"
    device = args.device if torch.cuda.is_available() else "cpu"

    evaluator = GenEval2Evaluator(
        data_dir=str(data_dir),
        device=device,
        owl_model=args.owl_model,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        owl_threshold=args.owl_threshold,
    )
    evaluator.load_models()

    results = evaluator.evaluate(
        generator,
        max_samples=args.max_samples,
        output_dir=str(output_dir) if args.save_images else None,
    )
    return results


def apply_local_benchmark_models(args):
    """
    Align with run_ablation.sh: honor EVAL_BLIP_MODEL / EVAL_OWL_MODEL, and if
    args still use HuggingFace-style IDs, prefer a local ModelScope cache dir
    when it exists (avoids redundant Hub downloads).
    """
    ev_blip = os.environ.get("EVAL_BLIP_MODEL")
    ev_owl = os.environ.get("EVAL_OWL_MODEL")
    if ev_blip:
        args.blip_model = ev_blip
    if ev_owl:
        args.owl_model = ev_owl

    ms_root = os.environ.get("MODELSCOPE_CACHE_DIR") or os.environ.get("MODELSCOPE_CACHE")
    if not ms_root:
        return

    def _resolve(hf_or_path: str) -> str:
        p = Path(hf_or_path)
        if p.is_dir():
            return str(p)
        if "/" not in hf_or_path:
            return hf_or_path
        root = Path(ms_root)
        for candidate in (root / "models" / hf_or_path, root / hf_or_path):
            if candidate.is_dir():
                return str(candidate)
        return hf_or_path

    args.blip_model = _resolve(args.blip_model)
    args.owl_model = _resolve(args.owl_model)


def main():
    args = parse_args()
    apply_local_benchmark_models(args)

    resolved = resolve_base_model_path(args.model_path)
    if resolved != args.model_path:
        print(f"[Eval] Resolved --model_path to local cache: {resolved}")
        args.model_path = resolved

    benchmarks = args.benchmarks
    if "all" in benchmarks:
        benchmarks = ["t2i_compbench", "geneval2"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("T2I-RL Benchmark Evaluation")
    print("=" * 60)
    print(f"  Model:      {args.model_path}")
    print(f"  LoRA ckpt:  {args.lora_checkpoint or '(none)'}")
    print(f"  Benchmarks: {benchmarks}")
    print(f"  Output dir: {output_dir}")
    print()

    # Load generator once (shared across benchmarks)
    generator = load_generator(args)

    all_results = {
        "model_path": args.model_path,
        "lora_checkpoint": args.lora_checkpoint,
        "benchmarks": {},
    }

    if "t2i_compbench" in benchmarks:
        print("\n" + "=" * 60)
        print("Running T2I-CompBench...")
        print("=" * 60)
        t2i_results = run_t2i_compbench(generator, args)
        all_results["benchmarks"]["t2i_compbench"] = {
            "overall_mean": t2i_results["overall_mean"],
            "category_means": t2i_results["category_means"],
        }
        print(f"\n→ T2I-CompBench overall: {t2i_results['overall_mean']:.4f}")

    if "geneval2" in benchmarks:
        print("\n" + "=" * 60)
        print("Running GenEval-2...")
        print("=" * 60)
        ge2_results = run_geneval2(generator, args)
        all_results["benchmarks"]["geneval2"] = {
            "overall_score": ge2_results["overall_score"],
            "per_skill_scores": ge2_results["per_skill_scores"],
        }
        print(f"\n→ GenEval-2 overall: {ge2_results['overall_score']:.4f}")

    # Save combined summary
    summary_path = output_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print final summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    for bench, res in all_results["benchmarks"].items():
        if bench == "t2i_compbench":
            print(f"  T2I-CompBench overall: {res['overall_mean']:.4f}")
            for cat, score in res["category_means"].items():
                print(f"    {cat:<16}: {score:.4f}")
        elif bench == "geneval2":
            print(f"  GenEval-2 overall:     {res['overall_score']:.4f}")
            for skill, score in res["per_skill_scores"].items():
                print(f"    {skill:<20}: {score:.4f}")

    print(f"\nFull results saved to: {summary_path}")
    return all_results


if __name__ == "__main__":
    main()
