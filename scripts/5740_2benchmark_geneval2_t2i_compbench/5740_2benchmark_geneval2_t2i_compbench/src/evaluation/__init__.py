"""
Evaluation Module
=================

Comprehensive evaluation suite for T2I models:
- T2I-CompBench (non-MLLM: BLIP-VQA, OwlViT, CLIPScore)
- GenEval-2 (non-MLLM: OwlViT, CLIP colour)
- Legacy evaluator / benchmarks
- Custom VLM-based evaluation
"""

from src.evaluation.evaluator import T2IEvaluator
from src.evaluation.metrics import CLIPScore, VLMScore, CompositionScore
from src.evaluation.benchmarks import T2ICompBench, TIFABench, GenEvalBench
from src.evaluation.t2i_compbench_eval import T2ICompBenchEvaluator
from src.evaluation.geneval2_eval import GenEval2Evaluator

__all__ = [
    "T2IEvaluator",
    "CLIPScore",
    "VLMScore",
    "CompositionScore",
    "T2ICompBench",
    "TIFABench",
    "GenEvalBench",
    # Non-MLLM evaluators
    "T2ICompBenchEvaluator",
    "GenEval2Evaluator",
]
