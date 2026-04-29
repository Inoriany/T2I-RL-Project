"""
Training Module
===============

RL-based training algorithms for T2I models:
- GRPO (Group Relative Policy Optimization)
- Reward-Weighted Regression
- PPO (Proximal Policy Optimization)
"""

from src.training.grpo_trainer import GRPOTrainer
from src.training.reward_weighted_trainer import RewardWeightedTrainer
from src.training.base_trainer import BaseTrainer

__all__ = [
    "BaseTrainer",
    "GRPOTrainer",
    "RewardWeightedTrainer",
]
