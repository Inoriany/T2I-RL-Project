"""
T2I Evaluator
=============

Main evaluator class for comprehensive T2I model evaluation.
"""

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json

import torch
from PIL import Image
from tqdm import tqdm


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    # Benchmarks to run
    benchmarks: List[str] = field(default_factory=lambda: [
        "t2i_compbench",
        "tifa", 
        "geneval",
    ])
    
    # Metrics
    compute_clip_score: bool = True
    compute_fid: bool = False  # Requires reference images
    compute_vlm_score: bool = True
    
    # Generation
    num_images_per_prompt: int = 1
    batch_size: int = 8
    seed: int = 42
    
    # Output
    output_dir: str = "./evaluation_results"
    save_images: bool = True
    save_detailed_results: bool = True


class T2IEvaluator:
    """
    Comprehensive T2I Model Evaluator.
    
    Evaluates text-to-image models on multiple benchmarks and metrics:
    
    Benchmarks:
    - T2I-CompBench: Compositional generation (attributes, relations, counting)
    - TIFA: Text-Image Faithfulness Assessment
    - GenEval-2: General evaluation with VQA
    - GenAI-Bench: Comprehensive AI generation benchmark
    
    Metrics:
    - CLIP Score: Image-text alignment
    - VLM Score: VLM-based semantic evaluation
    - Composition Score: Object, attribute, relation accuracy
    
    Error Taxonomy:
    - Missing objects
    - Wrong count
    - Wrong attribute (color, size, texture)
    - Wrong spatial relation
    """
    
    def __init__(
        self,
        generator: Any,  # ImageGenerator
        config: Optional[EvaluationConfig] = None,
        reward_models: Optional[Dict[str, Any]] = None,
    ):
        self.generator = generator
        self.config = config or EvaluationConfig()
        self.reward_models = reward_models or {}
        
        # Setup output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = {}
        
    def evaluate(
        self,
        prompts: Optional[List[str]] = None,
        benchmark: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run full evaluation.
        
        Args:
            prompts: Custom prompts to evaluate (if None, use benchmark prompts)
            benchmark: Specific benchmark to run (if None, run all)
            
        Returns:
            Dictionary with all evaluation results
        """
        if benchmark:
            benchmarks = [benchmark]
        else:
            benchmarks = self.config.benchmarks
            
        all_results = {}
        
        for bench_name in benchmarks:
            print(f"\n{'='*50}")
            print(f"Running {bench_name} evaluation...")
            print(f"{'='*50}")
            
            if bench_name == "t2i_compbench":
                results = self._evaluate_t2i_compbench()
            elif bench_name == "tifa":
                results = self._evaluate_tifa()
            elif bench_name == "geneval":
                results = self._evaluate_geneval()
            elif bench_name == "custom" and prompts:
                results = self._evaluate_custom(prompts)
            else:
                print(f"Unknown benchmark: {bench_name}")
                continue
                
            all_results[bench_name] = results
            
        # Aggregate results
        self.results = all_results
        self._save_results()
        
        return all_results
    
    def _evaluate_t2i_compbench(self) -> Dict[str, Any]:
        """
        Evaluate on T2I-CompBench.
        
        Categories:
        - Color attribution
        - Shape attribution
        - Texture attribution
        - Spatial relationships
        - Non-spatial relationships
        - Complex compositions
        """
        from src.evaluation.benchmarks import T2ICompBench
        
        benchmark = T2ICompBench()
        prompts = benchmark.get_prompts()
        
        results = {
            "color": [],
            "shape": [],
            "texture": [],
            "spatial": [],
            "non_spatial": [],
            "complex": [],
        }
        
        for category, category_prompts in prompts.items():
            print(f"\nEvaluating {category} ({len(category_prompts)} prompts)...")
            
            category_results = self._generate_and_score(
                category_prompts,
                category=category,
            )
            results[category] = category_results
            
        # Compute aggregate metrics
        aggregate = {}
        for category, cat_results in results.items():
            if cat_results:
                scores = [r["score"] for r in cat_results]
                aggregate[f"{category}_mean"] = sum(scores) / len(scores)
                
        aggregate["overall_mean"] = sum(aggregate.values()) / len(aggregate)
        results["aggregate"] = aggregate
        
        return results
    
    def _evaluate_tifa(self) -> Dict[str, Any]:
        """
        Evaluate on TIFA benchmark.
        
        Uses VQA to assess text-image faithfulness.
        """
        from src.evaluation.benchmarks import TIFABench
        
        benchmark = TIFABench()
        prompts_with_questions = benchmark.get_prompts_with_questions()
        
        results = []
        
        for item in tqdm(prompts_with_questions, desc="TIFA Evaluation"):
            prompt = item["prompt"]
            questions = item["questions"]
            
            # Generate image
            images = self.generator.generate([prompt])
            image = images[0]
            
            # Answer questions using VQA
            question_results = []
            for q in questions:
                answer = self._answer_vqa(image, q["question"])
                correct = answer.lower() == q["expected_answer"].lower()
                question_results.append({
                    "question": q["question"],
                    "expected": q["expected_answer"],
                    "predicted": answer,
                    "correct": correct,
                })
                
            accuracy = sum(r["correct"] for r in question_results) / len(question_results)
            
            results.append({
                "prompt": prompt,
                "accuracy": accuracy,
                "questions": question_results,
            })
            
        # Aggregate
        overall_accuracy = sum(r["accuracy"] for r in results) / len(results)
        
        return {
            "results": results,
            "overall_accuracy": overall_accuracy,
        }
        
    def _evaluate_geneval(self) -> Dict[str, Any]:
        """
        Evaluate on GenEval-2 benchmark.
        """
        from src.evaluation.benchmarks import GenEvalBench
        
        benchmark = GenEvalBench()
        prompts = benchmark.get_prompts()
        
        results = self._generate_and_score(prompts)
        
        # Compute metrics
        scores = [r["score"] for r in results]
        
        return {
            "results": results,
            "mean_score": sum(scores) / len(scores),
            "std_score": torch.tensor(scores).std().item(),
        }
        
    def _evaluate_custom(self, prompts: List[str]) -> Dict[str, Any]:
        """Evaluate on custom prompts."""
        results = self._generate_and_score(prompts)
        
        scores = [r["score"] for r in results]
        
        return {
            "results": results,
            "mean_score": sum(scores) / len(scores),
        }
        
    def _generate_and_score(
        self,
        prompts: List[str],
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate images and compute scores."""
        results = []
        
        for i in range(0, len(prompts), self.config.batch_size):
            batch_prompts = prompts[i:i + self.config.batch_size]
            
            # Generate images
            images = self.generator.generate(batch_prompts)
            
            # Compute scores
            for prompt, image in zip(batch_prompts, images):
                scores = {}
                
                # CLIP score
                if self.config.compute_clip_score and "clip" in self.reward_models:
                    clip_output = self.reward_models["clip"].compute_reward([image], [prompt])
                    scores["clip_score"] = clip_output.rewards[0].item()
                    
                # VLM score
                if self.config.compute_vlm_score and "vlm" in self.reward_models:
                    vlm_output = self.reward_models["vlm"].compute_reward([image], [prompt])
                    scores["vlm_score"] = vlm_output.rewards[0].item()
                    
                # Aggregate score
                if scores:
                    score = sum(scores.values()) / len(scores)
                else:
                    score = 0.0
                    
                # Error analysis
                errors = self._analyze_errors(image, prompt)
                
                result = {
                    "prompt": prompt,
                    "score": score,
                    "scores": scores,
                    "errors": errors,
                    "category": category,
                }
                
                # Save image if configured
                if self.config.save_images:
                    img_path = self.output_dir / "images" / f"{len(results)}.png"
                    img_path.parent.mkdir(exist_ok=True)
                    image.save(img_path)
                    result["image_path"] = str(img_path)
                    
                results.append(result)
                
        return results
    
    def _analyze_errors(
        self,
        image: Image.Image,
        prompt: str,
    ) -> Dict[str, Any]:
        """
        Analyze errors in generated image.
        
        Error taxonomy:
        - missing_objects: Objects mentioned but not present
        - wrong_count: Incorrect number of objects
        - wrong_attribute: Wrong color, size, texture, etc.
        - wrong_relation: Incorrect spatial relationships
        """
        # Use VLM for error analysis
        if "vlm" not in self.reward_models:
            return {}
            
        analysis_prompt = f"""Analyze this image against the description: "{prompt}"

Identify any errors in these categories:
1. Missing Objects: List any objects mentioned in the description but not in the image
2. Wrong Count: Note if object counts don't match (e.g., "two cats" but only one shown)
3. Wrong Attributes: List any incorrect colors, sizes, textures, or other attributes
4. Wrong Relations: Note any incorrect spatial relationships (e.g., "cat on table" but cat is under table)

Respond in JSON format:
{{
    "missing_objects": [],
    "wrong_count": [],
    "wrong_attributes": [],
    "wrong_relations": [],
    "overall_fidelity": 0-10
}}"""

        # Get VLM analysis
        try:
            vlm = self.reward_models["vlm"]
            response = vlm._call_vlm_api(
                self._image_to_base64(image),
                analysis_prompt,
            )
            
            # Parse JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"Error analysis failed: {e}")
            
        return {}
    
    def _answer_vqa(self, image: Image.Image, question: str) -> str:
        """Answer a VQA question about the image."""
        if "vlm" not in self.reward_models:
            return ""
            
        try:
            vlm = self.reward_models["vlm"]
            response = vlm._call_vlm_api(
                self._image_to_base64(image),
                f"Answer this question about the image with a single word or short phrase: {question}",
            )
            return response.strip()
        except Exception as e:
            print(f"VQA failed: {e}")
            return ""
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        import base64
        from io import BytesIO
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def _save_results(self) -> None:
        """Save evaluation results to disk."""
        results_path = self.output_dir / "results.json"
        
        # Convert non-serializable objects
        def serialize(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, Path):
                return str(obj)
            return obj
            
        with open(results_path, "w") as f:
            json.dump(self.results, f, default=serialize, indent=2)
            
        print(f"\nResults saved to {results_path}")
        
    def generate_report(self) -> str:
        """Generate a human-readable evaluation report."""
        lines = [
            "=" * 60,
            "T2I-RL Evaluation Report",
            "=" * 60,
            "",
        ]
        
        for benchmark, results in self.results.items():
            lines.append(f"\n## {benchmark.upper()}")
            lines.append("-" * 40)
            
            if "aggregate" in results:
                lines.append("\nAggregate Metrics:")
                for metric, value in results["aggregate"].items():
                    lines.append(f"  {metric}: {value:.4f}")
                    
            if "overall_accuracy" in results:
                lines.append(f"\nOverall Accuracy: {results['overall_accuracy']:.4f}")
                
            if "mean_score" in results:
                lines.append(f"\nMean Score: {results['mean_score']:.4f}")
                
        report = "\n".join(lines)
        
        # Save report
        report_path = self.output_dir / "report.txt"
        with open(report_path, "w") as f:
            f.write(report)
            
        return report
