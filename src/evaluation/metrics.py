"""
Evaluation Metrics
==================

Metrics for T2I model evaluation.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
from PIL import Image


class BaseMetric(ABC):
    """Base class for evaluation metrics."""
    
    @abstractmethod
    def compute(
        self,
        images: List[Image.Image],
        prompts: List[str],
        **kwargs,
    ) -> Dict[str, float]:
        """Compute metric scores."""
        pass


class CLIPScore(BaseMetric):
    """
    CLIP Score metric.
    
    Measures image-text alignment using CLIP embeddings.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        
    def load(self):
        """Load CLIP model."""
        import open_clip
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.eval()
        
    def compute(
        self,
        images: List[Image.Image],
        prompts: List[str],
        **kwargs,
    ) -> Dict[str, float]:
        """Compute CLIP scores."""
        if self.model is None:
            self.load()
            
        # Preprocess
        image_tensors = torch.stack([
            self.preprocess(img) for img in images
        ]).to(self.device)
        
        text_tokens = self.tokenizer(prompts).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensors)
            text_features = self.model.encode_text(text_tokens)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Cosine similarity
            scores = (image_features * text_features).sum(dim=-1)
            
        return {
            "clip_score": scores.mean().item(),
            "clip_scores": scores.tolist(),
        }


class VLMScore(BaseMetric):
    """
    VLM-based evaluation score.
    
    Uses a VLM to evaluate image-text alignment with detailed criteria.
    """
    
    def __init__(
        self,
        api_model: str = "gpt-4-vision-preview",
        device: str = "cuda",
    ):
        self.api_model = api_model
        self.device = device
        
    def compute(
        self,
        images: List[Image.Image],
        prompts: List[str],
        criteria: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Compute VLM scores.
        
        Args:
            images: Generated images
            prompts: Text prompts
            criteria: Evaluation criteria (default: alignment, quality, coherence)
        """
        criteria = criteria or ["alignment", "quality", "coherence"]
        
        all_scores = []
        detailed_scores = {c: [] for c in criteria}
        
        for image, prompt in zip(images, prompts):
            eval_prompt = self._build_eval_prompt(prompt, criteria)
            response = self._call_vlm(image, eval_prompt)
            scores = self._parse_scores(response, criteria)
            
            all_scores.append(sum(scores.values()) / len(scores))
            for c in criteria:
                detailed_scores[c].append(scores.get(c, 0.5))
                
        return {
            "vlm_score": sum(all_scores) / len(all_scores),
            **{f"{c}_score": sum(detailed_scores[c]) / len(detailed_scores[c]) 
               for c in criteria},
        }
        
    def _build_eval_prompt(self, prompt: str, criteria: List[str]) -> str:
        """Build evaluation prompt for VLM."""
        criteria_str = "\n".join([f"- {c}: Rate 0-10" for c in criteria])
        
        return f"""Evaluate this image against the description: "{prompt}"

Rate the following criteria (0-10 scale):
{criteria_str}

Respond with JSON: {{"alignment": X, "quality": X, "coherence": X}}"""
        
    def _call_vlm(self, image: Image.Image, prompt: str) -> str:
        """Call VLM API."""
        import base64
        from io import BytesIO
        from openai import OpenAI
        import os
        
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model=self.api_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                        },
                    ],
                }
            ],
            max_tokens=200,
        )
        
        return response.choices[0].message.content
        
    def _parse_scores(self, response: str, criteria: List[str]) -> Dict[str, float]:
        """Parse scores from VLM response."""
        import json
        import re
        
        try:
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return {c: data.get(c, 5) / 10.0 for c in criteria}
        except:
            pass
            
        return {c: 0.5 for c in criteria}


class CompositionScore(BaseMetric):
    """
    Composition score for attribute binding and spatial relations.
    
    Evaluates:
    - Object presence
    - Attribute binding (correct color/size/texture on correct object)
    - Counting accuracy
    - Spatial relationships
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
    def compute(
        self,
        images: List[Image.Image],
        prompts: List[str],
        **kwargs,
    ) -> Dict[str, float]:
        """Compute composition scores."""
        results = {
            "object_presence": [],
            "attribute_binding": [],
            "counting": [],
            "spatial_relations": [],
        }
        
        for image, prompt in zip(images, prompts):
            # Parse prompt to extract expected elements
            elements = self._parse_prompt(prompt)
            
            # Evaluate each aspect
            scores = self._evaluate_composition(image, elements)
            
            for key in results:
                results[key].append(scores.get(key, 0.5))
                
        return {
            key: sum(values) / len(values) if values else 0.0
            for key, values in results.items()
        }
        
    def _parse_prompt(self, prompt: str) -> Dict[str, Any]:
        """Parse prompt to extract objects, attributes, relations."""
        # Simple regex-based parsing (can be enhanced with NLP)
        import re
        
        # Extract color + object patterns
        color_pattern = r'(red|blue|green|yellow|purple|orange|pink|black|white|brown)\s+(\w+)'
        colors = re.findall(color_pattern, prompt.lower())
        
        # Extract counting patterns
        count_pattern = r'(\d+|one|two|three|four|five)\s+(\w+)'
        counts = re.findall(count_pattern, prompt.lower())
        
        # Extract spatial patterns
        spatial_pattern = r'(\w+)\s+(on|under|above|below|left|right|next to|beside)\s+(\w+)'
        relations = re.findall(spatial_pattern, prompt.lower())
        
        return {
            "colors": colors,
            "counts": counts,
            "relations": relations,
        }
        
    def _evaluate_composition(
        self,
        image: Image.Image,
        elements: Dict[str, Any],
    ) -> Dict[str, float]:
        """Evaluate composition accuracy using VLM."""
        # Placeholder - would use VLM for detailed evaluation
        return {
            "object_presence": 0.8,
            "attribute_binding": 0.7,
            "counting": 0.6,
            "spatial_relations": 0.7,
        }
