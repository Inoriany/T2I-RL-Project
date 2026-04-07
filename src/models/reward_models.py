"""
Reward Models
=============

Reward models for RL-based T2I training:
- CLIP-based: Cosine similarity between image and text embeddings
- VLM-based: Scalar reward from Vision-Language Models
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass
import json

import torch
import torch.nn as nn
from PIL import Image


@dataclass
class RewardOutput:
    """Output from reward model."""
    rewards: torch.Tensor  # Shape: (batch_size,)
    details: Optional[Dict[str, Any]] = None  # Additional info (e.g., per-attribute scores)


class RewardModel(ABC):
    """
    Abstract base class for reward models.
    
    Reward models evaluate the quality of generated images with respect to
    the input text prompt, providing scalar rewards for RL training.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
    @abstractmethod
    def compute_reward(
        self,
        images: List[Image.Image],
        prompts: List[str],
        **kwargs: Any,
    ) -> RewardOutput:
        """
        Compute rewards for image-prompt pairs.
        
        Args:
            images: List of generated images
            prompts: List of text prompts
            **kwargs: Additional model-specific arguments
            
        Returns:
            RewardOutput with scalar rewards and optional details
        """
        pass
    
    def to(self, device: str) -> "RewardModel":
        """Move model to device."""
        self.device = device
        return self


class CLIPRewardModel(RewardModel):
    """
    CLIP-based Reward Model.
    
    Computes reward as the cosine similarity between CLIP image
    and text embeddings.
    
    Architecture:
        Image -> Image Encoder -> Image Embedding
        Text  -> Text Encoder  -> Text Embedding
        Reward = CosineSimilarity(Image Emb, Text Emb)
    """
    
    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str = "cuda",
    ):
        super().__init__(device)
        self.model_name = model_name
        self.pretrained = pretrained
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        
    def load_model(self) -> None:
        """Load CLIP model."""
        import open_clip
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.eval()
        
        print(f"Loaded CLIP model: {self.model_name} ({self.pretrained})")
        
    def compute_reward(
        self,
        images: List[Image.Image],
        prompts: List[str],
        return_embeddings: bool = False,
        **kwargs: Any,
    ) -> RewardOutput:
        """
        Compute CLIP similarity reward.
        
        Args:
            images: Generated images
            prompts: Text prompts
            return_embeddings: Whether to return embeddings in details
            
        Returns:
            RewardOutput with cosine similarity scores
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        # Preprocess images
        image_tensors = torch.stack([
            self.preprocess(img) for img in images
        ]).to(self.device)
        
        # Tokenize prompts
        text_tokens = self.tokenizer(prompts).to(self.device)
        
        with torch.no_grad():
            # Encode
            image_features = self.model.encode_image(image_tensors)
            text_features = self.model.encode_text(text_tokens)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity
            rewards = (image_features * text_features).sum(dim=-1)
            
        details = {}
        if return_embeddings:
            details["image_features"] = image_features
            details["text_features"] = text_features
            
        return RewardOutput(rewards=rewards, details=details)


class VLMRewardModel(RewardModel):
    """
    VLM-based Reward Model.
    
    Uses a Vision-Language Model (e.g., LLaVA, GPT-4V, Claude) to evaluate
    image-text alignment and produce scalar rewards.
    
    Architecture:
        (Image, Text, Instruction) -> VLM -> Hidden State -> Regression Head -> Reward
    
    This model can evaluate:
    - Object presence/absence
    - Attribute correctness (color, size, etc.)
    - Spatial relationships
    - Overall semantic alignment
    """
    
    def __init__(
        self,
        model_name_or_path: str = "llava-hf/llava-1.5-7b-hf",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        use_api: bool = False,
        api_model: Optional[str] = None,
    ):
        super().__init__(device)
        self.model_name_or_path = model_name_or_path
        self.dtype = dtype
        self.use_api = use_api
        self.api_model = api_model
        self.model = None
        self.processor = None
        self.reward_head = None
        
    def load_model(self) -> None:
        """Load VLM and reward head."""
        if self.use_api:
            print(f"Using API-based VLM: {self.api_model}")
            return
            
        from transformers import AutoModelForVision2Seq, AutoProcessor
        
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.dtype,
            device_map="auto",
        )
        
        # Initialize reward head (projects hidden states to scalar)
        hidden_size = self.model.config.text_config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        ).to(self.device)
        
        print(f"Loaded VLM: {self.model_name_or_path}")
        
    def compute_reward(
        self,
        images: List[Image.Image],
        prompts: List[str],
        evaluation_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> RewardOutput:
        """
        Compute VLM-based reward.
        
        Args:
            images: Generated images
            prompts: Original text prompts
            evaluation_prompt: Custom evaluation instruction
            
        Returns:
            RewardOutput with VLM-based reward scores
        """
        if self.use_api:
            return self._compute_reward_api(images, prompts, evaluation_prompt, **kwargs)
        else:
            return self._compute_reward_local(images, prompts, evaluation_prompt, **kwargs)
            
    def _compute_reward_local(
        self,
        images: List[Image.Image],
        prompts: List[str],
        evaluation_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> RewardOutput:
        """Compute reward using local VLM."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        default_eval_prompt = (
            "Evaluate how well this image matches the description: '{prompt}'. "
            "Consider object presence, attributes, spatial relationships, and overall quality."
        )
        eval_template = evaluation_prompt or default_eval_prompt
        
        all_rewards = []
        all_details = []
        
        for image, prompt in zip(images, prompts):
            eval_text = eval_template.format(prompt=prompt)
            
            inputs = self.processor(
                images=image,
                text=eval_text,
                return_tensors="pt",
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Get last hidden state of the last token
                last_hidden = outputs.hidden_states[-1][:, -1, :]
                
                # Compute reward
                reward = self.reward_head(last_hidden.float())
                all_rewards.append(reward.squeeze())
                
        rewards = torch.stack(all_rewards)
        return RewardOutput(rewards=rewards, details={"type": "local_vlm"})
    
    def _compute_reward_api(
        self,
        images: List[Image.Image],
        prompts: List[str],
        evaluation_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> RewardOutput:
        """Compute reward using API-based VLM (GPT-4V, Claude, etc.)."""
        import base64
        from io import BytesIO
        
        rewards = []
        details = []
        
        for image, prompt in zip(images, prompts):
            # Convert image to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Build evaluation prompt
            eval_prompt = evaluation_prompt or self._get_default_eval_prompt(prompt)
            
            # Call API
            response = self._call_vlm_api(img_base64, eval_prompt)
            
            # Parse reward from response
            reward, detail = self._parse_reward_response(response)
            rewards.append(reward)
            details.append(detail)
            
        return RewardOutput(
            rewards=torch.tensor(rewards, device=self.device),
            details={"responses": details, "type": "api_vlm"},
        )
    
    def _get_default_eval_prompt(self, prompt: str) -> str:
        """Get default evaluation prompt for API-based VLM."""
        return f"""Evaluate the image based on the following description:
"{prompt}"

Rate the image on a scale of 0-10 based on:
1. Object Presence: Are all mentioned objects present? (0-3 points)
2. Attribute Accuracy: Are colors, sizes, and other attributes correct? (0-3 points)
3. Spatial Relations: Are spatial relationships correct? (0-2 points)
4. Overall Quality: Is the image realistic and coherent? (0-2 points)

Respond with ONLY a JSON object in this format:
{{"object_score": X, "attribute_score": X, "spatial_score": X, "quality_score": X, "total_score": X}}
"""
    
    def _call_vlm_api(self, img_base64: str, prompt: str) -> str:
        """Call VLM API (implement based on chosen provider)."""
        if self.api_model and "gpt" in self.api_model.lower():
            return self._call_openai_api(img_base64, prompt)
        elif self.api_model and "claude" in self.api_model.lower():
            return self._call_anthropic_api(img_base64, prompt)
        elif self.api_model and "gemini" in self.api_model.lower():
            return self._call_gemini_api(img_base64, prompt)
        elif self.api_model and ("qwen" in self.api_model.lower() or "glm" in self.api_model.lower() or "silicon" in self.api_model.lower()):
            return self._call_siliconflow_api(img_base64, prompt)
        else:
            raise ValueError(f"Unknown API model: {self.api_model}. Supported: gpt, claude, gemini, qwen/glm (siliconflow)")
            
    def _call_openai_api(self, img_base64: str, prompt: str) -> str:
        """Call OpenAI GPT-4V API."""
        from openai import OpenAI
        import os
        
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model=self.api_model or "gpt-4-vision-preview",
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
            max_tokens=500,
        )
        
        return response.choices[0].message.content
        
    def _call_anthropic_api(self, img_base64: str, prompt: str) -> str:
        """Call Anthropic Claude API."""
        from anthropic import Anthropic
        import os
        
        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        
        response = client.messages.create(
            model=self.api_model or "claude-3-opus-20240229",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_base64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        
        return response.content[0].text
    
    def _call_gemini_api(self, img_base64: str, prompt: str) -> str:
        """Call Google Gemini API (FREE tier available)."""
        import google.generativeai as genai
        import os
        
        # Configure API key
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
        genai.configure(api_key=api_key)
        
        # Create model - use gemini-1.5-flash for free tier (fast & cheap)
        model_name = self.api_model if "gemini" in self.api_model else "gemini-1.5-flash"
        model = genai.GenerativeModel(model_name)
        
        # Prepare image
        import base64
        image_data = base64.b64decode(img_base64)
        
        # Create content with image and prompt
        response = model.generate_content([
            {
                "mime_type": "image/png",
                "data": image_data
            },
            prompt
        ])
        
        return response.text
    
    def _call_siliconflow_api(self, img_base64: str, prompt: str) -> str:
        """
        Call SiliconFlow API (国产免费VLM).
        
        Supports models like:
        - Qwen/Qwen2.5-VL-32B-Instruct (视觉语言模型)
        - Qwen/Qwen2-VL-72B-Instruct (视觉语言模型)
        - THUDM/GLM-4.1V-9B-Thinking (免费)
        
        Get API key at: https://cloud.siliconflow.cn/
        """
        from openai import OpenAI
        import os
        
        api_key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("SILICON_API_KEY")
        if not api_key:
            raise ValueError("Please set SILICONFLOW_API_KEY environment variable. Get it at https://cloud.siliconflow.cn/")
        
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.siliconflow.cn/v1"
        )
        
        # Default to Qwen2.5-VL-32B if not specified
        model_name = self.api_model
        if "silicon" in model_name.lower():
            model_name = "Qwen/Qwen2.5-VL-32B-Instruct"  # 默认使用这个视觉模型
        
        response = client.chat.completions.create(
            model=model_name,
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
            max_tokens=500,
        )
        
        return response.choices[0].message.content
        
    def _parse_reward_response(self, response: str) -> Tuple[float, Dict]:
        """Parse reward from VLM response."""
        try:
            # Try to extract JSON
            import re
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                data = json.loads(json_match.group())
                total = data.get("total_score", 5.0)
                return total / 10.0, data  # Normalize to [0, 1]
        except (json.JSONDecodeError, AttributeError):
            pass
            
        # Fallback: extract any number
        import re
        numbers = re.findall(r'(\d+(?:\.\d+)?)', response)
        if numbers:
            score = float(numbers[-1])
            if score > 1:
                score = score / 10.0  # Normalize if needed
            return min(max(score, 0.0), 1.0), {"raw_response": response}
            
        return 0.5, {"raw_response": response, "parse_error": True}


class CompositeRewardModel(RewardModel):
    """
    Composite Reward Model that combines multiple reward signals.
    
    Useful for combining CLIP similarity with VLM-based rewards,
    or for multi-aspect evaluation (objects, attributes, relations).
    """
    
    def __init__(
        self,
        reward_models: Dict[str, RewardModel],
        weights: Optional[Dict[str, float]] = None,
        device: str = "cuda",
    ):
        super().__init__(device)
        self.reward_models = reward_models
        self.weights = weights or {name: 1.0 for name in reward_models}
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
        
    def compute_reward(
        self,
        images: List[Image.Image],
        prompts: List[str],
        **kwargs: Any,
    ) -> RewardOutput:
        """Compute weighted combination of rewards."""
        all_outputs = {}
        weighted_rewards = None
        
        for name, model in self.reward_models.items():
            output = model.compute_reward(images, prompts, **kwargs)
            all_outputs[name] = output
            
            weighted = output.rewards * self.weights[name]
            if weighted_rewards is None:
                weighted_rewards = weighted
            else:
                weighted_rewards = weighted_rewards + weighted
                
        return RewardOutput(
            rewards=weighted_rewards,
            details={
                "component_rewards": {
                    name: out.rewards for name, out in all_outputs.items()
                },
                "weights": self.weights,
            },
        )
