"""
Evaluation Benchmarks
=====================

Implementations of standard T2I evaluation benchmarks.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
import json


class BaseBenchmark(ABC):
    """Base class for evaluation benchmarks."""
    
    @abstractmethod
    def get_prompts(self) -> List[str]:
        """Get benchmark prompts."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Benchmark name."""
        pass


class T2ICompBench(BaseBenchmark):
    """
    T2I-CompBench: Compositional Text-to-Image Generation Benchmark.
    
    Reference: NeurIPS 2023
    https://github.com/Karine-Huang/T2I-CompBench
    
    Categories:
    - Color attribution
    - Shape attribution  
    - Texture attribution
    - Spatial relationships
    - Non-spatial relationships
    - Complex compositions
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else None
        
        # Default prompts for each category (samples)
        self._default_prompts = {
            "color": [
                "a red apple and a green banana",
                "a blue car next to a yellow house",
                "a purple flower in a white vase",
                "a black cat with orange eyes",
                "a pink dress on a brown chair",
            ],
            "shape": [
                "a triangular pizza on a round plate",
                "a square window in a rectangular building",
                "a circular mirror above an oval table",
                "a cylindrical vase next to a cubic box",
                "a heart-shaped balloon with a star pattern",
            ],
            "texture": [
                "a fluffy cat on a smooth marble floor",
                "a rough wooden table with a glossy apple",
                "a shiny metal robot on a fuzzy carpet",
                "a matte ceramic bowl with bumpy oranges",
                "a silky dress draped over a woven basket",
            ],
            "spatial": [
                "a cat sitting on top of a dog",
                "a book under the table",
                "a bird flying above the clouds",
                "a car parked behind the house",
                "a lamp to the left of the computer",
            ],
            "non_spatial": [
                "a dog wearing a hat",
                "a robot holding a flower",
                "a child riding a bicycle",
                "an artist painting a landscape",
                "a chef cooking in a kitchen",
            ],
            "complex": [
                "a red apple and a green banana on a blue plate next to a yellow cup",
                "a small white cat sitting on a large brown dog in front of a red house",
                "two birds flying above three trees beside a lake",
                "a chef in a white hat cooking pasta while a waiter serves wine",
                "a vintage red car parked under a green tree next to a blue bench",
            ],
        }
        
    @property
    def name(self) -> str:
        return "T2I-CompBench"
        
    def get_prompts(self) -> Dict[str, List[str]]:
        """Get prompts organized by category."""
        if self.data_dir and (self.data_dir / "prompts.json").exists():
            with open(self.data_dir / "prompts.json") as f:
                return json.load(f)
        return self._default_prompts
        
    def get_all_prompts(self) -> List[str]:
        """Get all prompts as a flat list."""
        all_prompts = []
        for prompts in self.get_prompts().values():
            all_prompts.extend(prompts)
        return all_prompts


class TIFABench(BaseBenchmark):
    """
    TIFA: Text-to-Image Faithfulness Evaluation with Question Answering.
    
    Reference: https://tifa-benchmark.github.io/
    
    Uses VQA to evaluate whether generated images faithfully represent
    the text prompts.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else None
        
        # Default prompts with questions
        self._default_data = [
            {
                "prompt": "a red apple on a wooden table",
                "questions": [
                    {"question": "What color is the apple?", "expected_answer": "red"},
                    {"question": "What is the apple on?", "expected_answer": "table"},
                    {"question": "What material is the table?", "expected_answer": "wood"},
                ]
            },
            {
                "prompt": "two cats playing with a ball",
                "questions": [
                    {"question": "How many cats are there?", "expected_answer": "two"},
                    {"question": "What are the cats doing?", "expected_answer": "playing"},
                    {"question": "What are they playing with?", "expected_answer": "ball"},
                ]
            },
            {
                "prompt": "a blue bird sitting on a branch",
                "questions": [
                    {"question": "What color is the bird?", "expected_answer": "blue"},
                    {"question": "What is the bird doing?", "expected_answer": "sitting"},
                    {"question": "Where is the bird?", "expected_answer": "branch"},
                ]
            },
            {
                "prompt": "a chef cooking in a modern kitchen",
                "questions": [
                    {"question": "Who is in the image?", "expected_answer": "chef"},
                    {"question": "What is the chef doing?", "expected_answer": "cooking"},
                    {"question": "What style is the kitchen?", "expected_answer": "modern"},
                ]
            },
            {
                "prompt": "three dogs running in a green field",
                "questions": [
                    {"question": "How many dogs are there?", "expected_answer": "three"},
                    {"question": "What are the dogs doing?", "expected_answer": "running"},
                    {"question": "What color is the field?", "expected_answer": "green"},
                ]
            },
        ]
        
    @property
    def name(self) -> str:
        return "TIFA"
        
    def get_prompts(self) -> List[str]:
        """Get prompts."""
        return [item["prompt"] for item in self._default_data]
        
    def get_prompts_with_questions(self) -> List[Dict[str, Any]]:
        """Get prompts with associated VQA questions."""
        if self.data_dir and (self.data_dir / "tifa_data.json").exists():
            with open(self.data_dir / "tifa_data.json") as f:
                return json.load(f)
        return self._default_data


class GenEvalBench(BaseBenchmark):
    """
    GenEval-2: Comprehensive T2I Evaluation Benchmark.
    
    Reference: https://arxiv.org/abs/2512.16853
    
    Evaluates:
    - Single object generation
    - Two object composition
    - Counting
    - Color attribution
    - Position/spatial relations
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else None
        
        # Default prompts covering different categories
        self._default_prompts = {
            "single_object": [
                "a photo of a cat",
                "a realistic image of a dog",
                "a picture of a red rose",
                "an illustration of a mountain",
                "a photograph of a sunset",
            ],
            "two_objects": [
                "a cat and a dog",
                "an apple next to an orange",
                "a book on a shelf",
                "a cup beside a plate",
                "a bird and a flower",
            ],
            "counting": [
                "three apples",
                "two birds flying",
                "five red balloons",
                "four trees in a row",
                "one cat and two dogs",
            ],
            "colors": [
                "a red car",
                "a blue house",
                "a yellow sunflower",
                "a green frog",
                "a purple butterfly",
            ],
            "position": [
                "a cat on the left and a dog on the right",
                "a bird above a tree",
                "a ball under the table",
                "a car in front of a house",
                "a person behind a fence",
            ],
        }
        
    @property
    def name(self) -> str:
        return "GenEval-2"
        
    def get_prompts(self) -> List[str]:
        """Get all prompts."""
        all_prompts = []
        for prompts in self._default_prompts.values():
            all_prompts.extend(prompts)
        return all_prompts
        
    def get_prompts_by_category(self) -> Dict[str, List[str]]:
        """Get prompts organized by category."""
        if self.data_dir and (self.data_dir / "geneval_prompts.json").exists():
            with open(self.data_dir / "geneval_prompts.json") as f:
                return json.load(f)
        return self._default_prompts


class GenAIBench(BaseBenchmark):
    """
    GenAI-Bench: Comprehensive AI Generation Benchmark.
    
    Reference: https://openreview.net/pdf?id=d1b464d7c538923ddbca638886afdc54d57ebae7
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else None
        
        self._default_prompts = [
            "A professional photograph of an astronaut riding a horse on Mars",
            "A watercolor painting of a sunset over the ocean",
            "A cyberpunk cityscape at night with neon lights",
            "A cozy cottage in a snowy forest during winter",
            "An abstract representation of music and emotion",
            "A detailed illustration of a steampunk machine",
            "A serene Japanese garden with cherry blossoms",
            "A dramatic scene of a volcano erupting",
            "A whimsical illustration of a fairy tale castle",
            "A photorealistic image of a futuristic car",
        ]
        
    @property
    def name(self) -> str:
        return "GenAI-Bench"
        
    def get_prompts(self) -> List[str]:
        """Get prompts."""
        if self.data_dir and (self.data_dir / "genai_prompts.json").exists():
            with open(self.data_dir / "genai_prompts.json") as f:
                return json.load(f)
        return self._default_prompts
