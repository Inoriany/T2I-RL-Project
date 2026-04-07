"""
Evaluation Module
=================

Comprehensive evaluation suite for T2I models:
- T2I-CompBench metrics
- TIFA evaluation
- GenEval-2 
- Custom VLM-based evaluation
"""

from src.evaluation.evaluator import T2IEvaluator
from src.evaluation.metrics import CLIPScore, VLMScore, CompositionScore
from src.evaluation.benchmarks import T2ICompBench, TIFABench, GenEvalBench

__all__ = [
    "T2IEvaluator",
    "CLIPScore",
    "VLMScore", 
    "CompositionScore",
    "T2ICompBench",
    "TIFABench",
    "GenEvalBench",
]
