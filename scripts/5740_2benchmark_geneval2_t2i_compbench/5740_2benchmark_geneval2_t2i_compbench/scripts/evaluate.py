#!/usr/bin/env python3
"""
Evaluation Script for T2I-RL
=============================

Evaluate trained T2I models on standard benchmarks.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoint-1000
    python scripts/evaluate.py --benchmark tifa --save_images

Environment Variables:
    USE_MODELSCOPE: Set to "true" to use ModelScope instead of HuggingFace
    MODELSCOPE_CACHE: Cache directory for ModelScope downloads (default: ./modelscope_models)
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup ModelScope support BEFORE importing transformers
# This will redirect HuggingFace downloads to ModelScope mirrors if USE_MODELSCOPE=true
if os.environ.get("USE_MODELSCOPE", "false").lower() == "true":
    from src.utils import modelscope_helper
    modelscope_helper.setup_modelscope()

import torch

from src.models.generators import JanusProGenerator, DiffusionGenerator
from src.models.reward_models import CLIPRewardModel, VLMRewardModel
from src.evaluation.evaluator import T2IEvaluator, EvaluationConfig
from src.evaluation.benchmarks import T2ICompBench, TIFABench, GenEvalBench


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate T2I-RL models")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="janus-pro",
        choices=["janus-pro", "sdxl"],
        help="Base model type",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        nargs="+",
        default=["t2i_compbench", "tifa", "geneval"],
        help="Benchmarks to evaluate on",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save generated images",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--use_vlm",
        action="store_true",
        help="Use VLM-based evaluation (requires API key)",
    )
    parser.add_argument(
        "--vlm_model",
        type=str,
        default="gpt-4-vision-preview",
        help="VLM model for evaluation",
    )
    
    return parser.parse_args()


def load_generator(args):
    """Load the image generator."""
    print(f"Loading {args.model} generator...")
    
    if args.model == "janus-pro":
        generator = JanusProGenerator(
            model_name_or_path="deepseek-ai/Janus-Pro-1B",
            dtype=torch.bfloat16,
        )
    else:
        generator = DiffusionGenerator(
            model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
            dtype=torch.float16,
        )
        
    generator.load_model()
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        generator.enable_lora(args.checkpoint)
        
    return generator


def load_reward_models(args):
    """Load reward models for evaluation."""
    reward_models = {}
    
    # Always load CLIP
    print("Loading CLIP reward model...")
    clip_model = CLIPRewardModel()
    clip_model.load_model()
    reward_models["clip"] = clip_model
    
    # Optionally load VLM
    if args.use_vlm:
        print(f"Loading VLM reward model ({args.vlm_model})...")
        vlm_model = VLMRewardModel(
            use_api=True,
            api_model=args.vlm_model,
        )
        vlm_model.load_model()
        reward_models["vlm"] = vlm_model
        
    return reward_models


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("=" * 60)
    print("T2I-RL Evaluation")
    print("=" * 60)
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    # Load components
    generator = load_generator(args)
    reward_models = load_reward_models(args)
    
    # Setup evaluator
    config = EvaluationConfig(
        benchmarks=args.benchmark,
        compute_clip_score=True,
        compute_vlm_score=args.use_vlm,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
        save_images=args.save_images,
    )
    
    evaluator = T2IEvaluator(
        generator=generator,
        config=config,
        reward_models=reward_models,
    )
    
    # Run evaluation
    print("\nStarting evaluation...")
    results = evaluator.evaluate()
    
    # Generate report
    report = evaluator.generate_report()
    print("\n" + report)
    
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
