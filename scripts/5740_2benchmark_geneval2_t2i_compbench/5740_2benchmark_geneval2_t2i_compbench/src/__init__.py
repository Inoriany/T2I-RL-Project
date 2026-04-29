"""
T2I-RL: Text-to-Image Generation with Reinforcement Learning
============================================================

A research framework for training text-to-image models using 
understanding-based rewards from Vision-Language Models.
"""

__version__ = "0.1.0"
__author__ = "T2I-RL Team"

from src.models import ImageGenerator, RewardModel
from src.training import GRPOTrainer, RewardWeightedTrainer
from src.evaluation import T2IEvaluator

__all__ = [
    "ImageGenerator",
    "RewardModel", 
    "GRPOTrainer",
    "RewardWeightedTrainer",
    "T2IEvaluator",
]
