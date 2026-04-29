"""Normalized records used by the evaluation workflow."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class GeneratedSampleRecord:
    benchmark: str
    sample_id: str
    prompt: str
    variant: str
    seed: int
    model_name: str
    checkpoint_or_lora: str
    image_path: str
    generation_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ScoredSampleRecord:
    benchmark: str
    sample_id: str
    variant: str
    prompt: str
    score: float
    subscores: Dict[str, Any] = field(default_factory=dict)
    error_types: List[str] = field(default_factory=list)
    judge_metadata: Dict[str, Any] = field(default_factory=dict)
    image_path: str = ""
    category: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
