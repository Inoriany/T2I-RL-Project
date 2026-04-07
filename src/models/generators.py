"""
Image Generators
================

Base classes and implementations for text-to-image generation models.
Supports unified multimodal models (Janus-Pro) and diffusion-based generators.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from PIL import Image


@dataclass
class GenerationConfig:
    """Configuration for image generation."""
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    num_images_per_prompt: int = 1
    seed: Optional[int] = None
    use_lora: bool = True
    lora_scale: float = 1.0


class ImageGenerator(ABC):
    """
    Abstract base class for all image generators.
    
    This provides a unified interface for different T2I architectures:
    - Unified Multimodal LLMs (e.g., Janus-Pro)
    - Diffusion Models (e.g., Stable Diffusion, SDXL)
    - Flow-based Models (e.g., Flux)
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.dtype = dtype
        self.model = None
        
    @abstractmethod
    def load_model(self) -> None:
        """Load the model and its components."""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: Union[str, List[str]],
        config: Optional[GenerationConfig] = None,
        **kwargs: Any,
    ) -> List[Image.Image]:
        """
        Generate images from text prompts.
        
        Args:
            prompt: Text prompt(s) for generation
            config: Generation configuration
            **kwargs: Additional model-specific arguments
            
        Returns:
            List of generated PIL Images
        """
        pass
    
    @abstractmethod
    def enable_lora(self, lora_path: str, scale: float = 1.0) -> None:
        """Enable LoRA adapter for fine-tuning."""
        pass
    
    @abstractmethod
    def disable_lora(self) -> None:
        """Disable LoRA adapter."""
        pass
    
    def to(self, device: str) -> "ImageGenerator":
        """Move model to specified device."""
        self.device = device
        if self.model is not None:
            self.model.to(device)
        return self
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get parameters that should be trained (e.g., LoRA params)."""
        if self.model is None:
            return []
        return [p for p in self.model.parameters() if p.requires_grad]


class JanusProGenerator(ImageGenerator):
    """
    Janus-Pro-1B: Unified Multimodal LLM for Text-to-Image Generation.
    
    Reference: https://huggingface.co/deepseek-ai/Janus-Pro-1B
    
    Janus-Pro is a unified multimodal model that can both understand
    and generate images, making it ideal for RL-based training with
    understanding-based rewards.
    """
    
    def __init__(
        self,
        model_name_or_path: str = "deepseek-ai/Janus-Pro-1B",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_flash_attention: bool = True,
    ):
        super().__init__(model_name_or_path, device, dtype)
        self.use_flash_attention = use_flash_attention
        self.processor = None
        self.lora_enabled = False
        
    def load_model(self) -> None:
        """Load Janus-Pro model and processor."""
        from transformers import AutoModelForCausalLM, AutoProcessor
        from peft import PeftModel, get_peft_model, LoraConfig
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )
        
        # Load model with optimizations
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": self.dtype,
            "device_map": "auto" if self.device == "cuda" else None,
        }
        
        if self.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            **model_kwargs,
        )
        
        print(f"Loaded Janus-Pro from {self.model_name_or_path}")
        print(f"Model dtype: {self.dtype}, Device: {self.device}")
        
    def generate(
        self,
        prompt: Union[str, List[str]],
        config: Optional[GenerationConfig] = None,
        **kwargs: Any,
    ) -> List[Image.Image]:
        """Generate images using Janus-Pro."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        config = config or GenerationConfig()
        
        if isinstance(prompt, str):
            prompt = [prompt]
            
        # Set random seed for reproducibility
        if config.seed is not None:
            torch.manual_seed(config.seed)
            
        # Prepare inputs
        images = []
        for p in prompt:
            # Format prompt for image generation
            formatted_prompt = f"Generate an image: {p}"
            
            inputs = self.processor(
                text=formatted_prompt,
                return_tensors="pt",
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get("max_new_tokens", 4096),
                    do_sample=True,
                    temperature=kwargs.get("temperature", 1.0),
                )
            
            # Decode image from outputs
            # Note: Actual implementation depends on Janus-Pro's output format
            generated_image = self._decode_image(outputs)
            images.append(generated_image)
            
        return images
    
    def _decode_image(self, outputs: torch.Tensor) -> Image.Image:
        """Decode model outputs to PIL Image."""
        # Placeholder - actual implementation depends on model architecture
        # This would decode the visual tokens back to an image
        raise NotImplementedError(
            "Image decoding implementation depends on Janus-Pro architecture. "
            "See the official Janus-Pro repository for details."
        )
    
    def enable_lora(
        self,
        lora_path: Optional[str] = None,
        scale: float = 1.0,
        lora_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Enable LoRA for parameter-efficient fine-tuning.
        
        Args:
            lora_path: Path to pre-trained LoRA weights (optional)
            scale: LoRA scaling factor
            lora_config: LoRA configuration dict (if creating new adapter)
        """
        from peft import PeftModel, get_peft_model, LoraConfig, TaskType
        
        if lora_path is not None:
            # Load pre-trained LoRA
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_path,
            )
        else:
            # Create new LoRA adapter
            default_config = {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                "task_type": TaskType.CAUSAL_LM,
            }
            if lora_config:
                default_config.update(lora_config)
                
            peft_config = LoraConfig(**default_config)
            self.model = get_peft_model(self.model, peft_config)
            
        self.lora_enabled = True
        self.model.print_trainable_parameters()
        
    def disable_lora(self) -> None:
        """Disable LoRA adapter."""
        if hasattr(self.model, "disable_adapter"):
            self.model.disable_adapter()
        self.lora_enabled = False


class DiffusionGenerator(ImageGenerator):
    """
    Diffusion-based Image Generator (e.g., Stable Diffusion, SDXL).
    
    Uses the diffusers library for flexible diffusion model support.
    """
    
    def __init__(
        self,
        model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        scheduler: str = "ddim",
    ):
        super().__init__(model_name_or_path, device, dtype)
        self.scheduler_type = scheduler
        self.pipe = None
        
    def load_model(self) -> None:
        """Load diffusion pipeline."""
        from diffusers import (
            StableDiffusionXLPipeline,
            DDIMScheduler,
            EulerAncestralDiscreteScheduler,
        )
        
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.dtype,
            use_safetensors=True,
        ).to(self.device)
        
        # Set scheduler
        if self.scheduler_type == "ddim":
            self.pipe.scheduler = DDIMScheduler.from_config(
                self.pipe.scheduler.config
            )
        elif self.scheduler_type == "euler":
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipe.scheduler.config
            )
            
        # Enable optimizations
        self.pipe.enable_xformers_memory_efficient_attention()
        
        print(f"Loaded diffusion model from {self.model_name_or_path}")
        
    def generate(
        self,
        prompt: Union[str, List[str]],
        config: Optional[GenerationConfig] = None,
        negative_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Image.Image]:
        """Generate images using diffusion model."""
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        config = config or GenerationConfig()
        
        generator = None
        if config.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(config.seed)
            
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            height=config.height,
            width=config.width,
            num_images_per_prompt=config.num_images_per_prompt,
            generator=generator,
        )
        
        return output.images
    
    def enable_lora(self, lora_path: str, scale: float = 1.0) -> None:
        """Load LoRA weights for diffusion model."""
        self.pipe.load_lora_weights(lora_path)
        self.pipe.fuse_lora(lora_scale=scale)
        
    def disable_lora(self) -> None:
        """Unload LoRA weights."""
        self.pipe.unfuse_lora()
        self.pipe.unload_lora_weights()
