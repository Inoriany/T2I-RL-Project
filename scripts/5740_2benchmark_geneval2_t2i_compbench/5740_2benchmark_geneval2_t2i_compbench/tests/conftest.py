"""
Pytest Configuration and Fixtures
==================================

Shared fixtures for T2I-RL tests.
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, MagicMock

import pytest
import torch
from PIL import Image
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Device Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def cpu_device():
    """Force CPU device for tests that don't need GPU."""
    return "cpu"


# =============================================================================
# Image Fixtures
# =============================================================================

@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing."""
    return Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8))


@pytest.fixture
def sample_images():
    """Create multiple sample PIL Images for testing."""
    return [
        Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8))
        for _ in range(4)
    ]


@pytest.fixture
def sample_image_tensor():
    """Create a sample image tensor."""
    return torch.randn(3, 384, 384)


@pytest.fixture
def batch_image_tensors():
    """Create a batch of image tensors."""
    return torch.randn(4, 3, 384, 384)


# =============================================================================
# Prompt Fixtures
# =============================================================================

@pytest.fixture
def sample_prompt():
    """Single sample prompt."""
    return "a red apple on a wooden table"


@pytest.fixture
def sample_prompts():
    """Multiple sample prompts."""
    return [
        "a red apple on a wooden table",
        "a blue car next to a yellow house",
        "two cats playing with a ball",
        "a chef cooking in a modern kitchen",
    ]


@pytest.fixture
def compositional_prompts():
    """Prompts testing compositional abilities."""
    return {
        "color": [
            "a red apple and a green banana",
            "a blue car next to a yellow house",
        ],
        "spatial": [
            "a cat sitting on top of a dog",
            "a book under the table",
        ],
        "counting": [
            "two birds flying",
            "three apples on a plate",
        ],
    }


# =============================================================================
# Data Fixtures
# =============================================================================

@pytest.fixture
def temp_json_file():
    """Create a temporary JSON file with prompts."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        data = {"prompts": ["prompt 1", "prompt 2", "prompt 3"]}
        json.dump(data, f)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def temp_jsonl_file():
    """Create a temporary JSONL file with prompts."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for i in range(3):
            f.write(json.dumps({"prompt": f"prompt {i+1}", "id": i}) + "\n")
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def temp_txt_file():
    """Create a temporary TXT file with prompts."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("prompt 1\n")
        f.write("prompt 2\n")
        f.write("prompt 3\n")
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file with prompts."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("prompt,category\n")
        f.write("prompt 1,color\n")
        f.write("prompt 2,spatial\n")
        f.write("prompt 3,counting\n")
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def temp_prompts_dir():
    """Create a temporary directory with prompt files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create JSON file
        json_path = Path(tmpdir) / "prompts.json"
        with open(json_path, 'w') as f:
            json.dump(["prompt 1", "prompt 2", "prompt 3"], f)
        
        yield tmpdir


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_generator():
    """Create a mock image generator."""
    mock = Mock()
    mock.model = Mock()
    mock.device = "cpu"
    mock.generate = Mock(return_value=[
        Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8))
        for _ in range(4)
    ])
    mock.generate_with_logprobs = Mock(return_value=(
        [Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)) for _ in range(4)],
        torch.randn(4)
    ))
    mock.get_trainable_parameters = Mock(return_value=[torch.nn.Parameter(torch.randn(10))])
    mock.train = Mock()
    mock.eval = Mock()
    return mock


@pytest.fixture
def mock_reward_model():
    """Create a mock reward model."""
    from src.models.reward_models import RewardOutput
    
    mock = Mock()
    mock.device = "cpu"
    mock.compute_reward = Mock(return_value=RewardOutput(
        rewards=torch.rand(4),
        details={"type": "mock"}
    ))
    return mock


@pytest.fixture
def mock_clip_model():
    """Create a mock CLIP model."""
    mock = Mock()
    mock.encode_image = Mock(return_value=torch.randn(4, 768))
    mock.encode_text = Mock(return_value=torch.randn(4, 768))
    mock.eval = Mock()
    return mock


@pytest.fixture
def mock_vlm_response():
    """Mock VLM API response."""
    return '{"object_score": 8, "attribute_score": 7, "spatial_score": 6, "quality_score": 8, "total_score": 7.25}'


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def training_config():
    """Create a training configuration."""
    from src.training.base_trainer import TrainingConfig
    
    return TrainingConfig(
        learning_rate=1e-5,
        num_epochs=1,
        batch_size=2,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        warmup_steps=10,
        save_steps=100,
        eval_steps=50,
        logging_steps=10,
        use_wandb=False,
    )


@pytest.fixture
def grpo_config():
    """Create a GRPO configuration."""
    from src.training.grpo_trainer import GRPOConfig
    
    return GRPOConfig(
        learning_rate=1e-5,
        num_epochs=1,
        batch_size=2,
        num_samples_per_prompt=4,
        temperature=1.0,
        clip_ratio=0.2,
        use_advantage_normalization=True,
        kl_coef=0.1,
        use_wandb=False,
    )


@pytest.fixture
def generation_config():
    """Create a generation configuration."""
    from src.models.generators import GenerationConfig
    
    return GenerationConfig(
        num_inference_steps=10,  # Small for testing
        guidance_scale=5.0,
        height=384,
        width=384,
        num_images_per_prompt=1,
        seed=42,
        temperature=1.0,
    )


# =============================================================================
# DataLoader Fixtures
# =============================================================================

@pytest.fixture
def sample_dataloader(sample_prompts):
    """Create a sample DataLoader."""
    from torch.utils.data import DataLoader
    from src.data.dataset import PromptDataset
    
    dataset = PromptDataset(sample_prompts)
    return DataLoader(dataset, batch_size=2, shuffle=False)


# =============================================================================
# Benchmark Fixtures
# =============================================================================

@pytest.fixture
def t2i_compbench():
    """Create T2I-CompBench instance."""
    from src.evaluation.benchmarks import T2ICompBench
    return T2ICompBench()


@pytest.fixture
def tifa_bench():
    """Create TIFA benchmark instance."""
    from src.evaluation.benchmarks import TIFABench
    return TIFABench()


@pytest.fixture
def geneval_bench():
    """Create GenEval benchmark instance."""
    from src.evaluation.benchmarks import GenEvalBench
    return GenEvalBench()


# =============================================================================
# Utility Functions
# =============================================================================

def requires_gpu(func):
    """Decorator to skip tests that require GPU if not available."""
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )(func)


def requires_api_key(env_var: str):
    """Decorator to skip tests that require API keys."""
    def decorator(func):
        return pytest.mark.skipif(
            os.environ.get(env_var) is None,
            reason=f"{env_var} not set"
        )(func)
    return decorator
