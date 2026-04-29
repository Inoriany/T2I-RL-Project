"""Repository-local benchmark manifest loaders."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.evaluation.io import read_jsonl


class BaseBenchmark:
    """Load and validate benchmark samples from a JSONL manifest."""

    benchmark_name = "base"
    required_fields: Sequence[str] = ()
    default_manifest_relative_path: Sequence[str] = ()

    def __init__(self, manifest_path: Optional[Path] = None):
        self.manifest_path = Path(manifest_path) if manifest_path else self.default_manifest_path()

    @classmethod
    def default_manifest_path(cls) -> Path:
        if not cls.default_manifest_relative_path:
            raise NotImplementedError("Subclasses must define default_manifest_relative_path")
        return Path(__file__).resolve().parents[2].joinpath(*cls.default_manifest_relative_path)

    def load_samples(self) -> List[Dict[str, Any]]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Benchmark manifest not found: {self.manifest_path}")
        return read_jsonl(self.manifest_path)

    def validate_sample(self, sample: Dict[str, Any]) -> None:
        missing = [field for field in self.required_fields if field not in sample]
        if missing:
            raise ValueError(
                f"{self.benchmark_name} sample missing required fields: {', '.join(sorted(missing))}"
            )

    def iter_samples(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        samples = self.load_samples()
        if limit is not None:
            samples = samples[:limit]
        seen_ids = set()
        validated: List[Dict[str, Any]] = []
        for sample in samples:
            self.validate_sample(sample)
            sample_id = sample["sample_id"]
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample_id detected in {self.benchmark_name}: {sample_id}")
            seen_ids.add(sample_id)
            validated.append(sample)
        return validated


class TIFABenchmark(BaseBenchmark):
    benchmark_name = "tifa"
    required_fields = ("sample_id", "prompt", "category", "source", "questions")
    default_manifest_relative_path = ("data", "evaluation", "tifa", "samples.jsonl")


class GenAIBenchmark(BaseBenchmark):
    benchmark_name = "genai_bench"
    required_fields = ("sample_id", "prompt", "category", "skills", "source")
    default_manifest_relative_path = ("data", "evaluation", "genai_bench", "samples.jsonl")
