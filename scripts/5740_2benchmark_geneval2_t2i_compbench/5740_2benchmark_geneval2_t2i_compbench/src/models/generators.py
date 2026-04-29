      
"""
Image Generators
================

Base classes and implementations for text-to-image generation models.
Supports unified multimodal models (Janus-Pro) and diffusion-based generators.

Based on official implementations:
- Janus-Pro: https://github.com/deepseek-ai/Janus
- Diffusers: https://github.com/huggingface/diffusers
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass
from contextlib import nullcontext
import types
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


@dataclass
class GenerationConfig:
    """Configuration for image generation."""
    num_inference_steps: int = 50
    guidance_scale: float = 5.0  # CFG weight
    height: int = 384
    width: int = 384
    num_images_per_prompt: int = 1
    seed: Optional[int] = None
    temperature: float = 1.0
    use_lora: bool = True
    lora_scale: float = 1.0


class ImageGenerator(ABC):
    """
    Abstract base class for all image generators.
    
    This provides a unified interface for different T2I architectures:
    - Unified Multimodal LLMs (e.g., Janus-Pro)
    - Diffusion Models (e.g., Stable Diffusion, SDXL)
    - Flow-based Models (e.g., Flux, JanusFlow)
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        prefer_local_files: bool = True,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.dtype = dtype
        self.prefer_local_files = prefer_local_files
        self.model = None

    def _from_pretrained_local_first(self, loader: Any, **kwargs: Any) -> Any:
        """Try local cache first; fallback to remote if needed."""
        if self.prefer_local_files:
            try:
                return loader(
                    self.model_name_or_path,
                    local_files_only=True,
                    **kwargs,
                )
            except TypeError:
                # Some loaders may not accept local_files_only
                pass
            except Exception as e:
                print(
                    "Local cache load failed, falling back to remote download: "
                    f"{e.__class__.__name__}: {e}"
                )
        return loader(self.model_name_or_path, **kwargs)
        
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
    
    def train(self) -> None:
        """Set model to training mode."""
        if self.model is not None:
            self.model.train()
            
    def eval(self) -> None:
        """Set model to evaluation mode."""
        if self.model is not None:
            self.model.eval()


class JanusProGenerator(ImageGenerator):
    """
    Janus-Pro: Unified Multimodal LLM for Text-to-Image Generation.
    
    Reference: https://github.com/deepseek-ai/Janus
    Paper: https://arxiv.org/abs/2501.17811
    
    Janus-Pro is a unified multimodal model that can both understand
    and generate images, making it ideal for RL-based training with
    understanding-based rewards.
    
    Key features:
    - Autoregressive image generation using visual tokens
    - Supports CFG (Classifier-Free Guidance)
    - 384x384 image output with 576 tokens per image
    - Supports 4-bit/8-bit quantization for memory efficiency
    """
    
    def __init__(
        self,
        model_name_or_path: str = "deepseek-ai/Janus-Pro-1B",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_flash_attention: bool = True,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        prefer_local_files: bool = True,
    ):
        super().__init__(
            model_name_or_path,
            device,
            dtype,
            prefer_local_files=prefer_local_files,
        )
        self.use_flash_attention = use_flash_attention
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.vl_chat_processor = None
        self.tokenizer = None
        self.lora_enabled = False
        
        # Janus-Pro specific parameters
        self.image_token_num_per_image = 576  # 24x24 tokens
        self.img_size = 384
        self.patch_size = 16
        
    def load_model(self) -> None:
        """Load Janus-Pro model and processor."""
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        
        # Configure quantization if requested
        quantization_config = None
        if self.load_in_4bit:
            # Use float16 for 4-bit to avoid dtype mismatch in conv layers
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print("Using 4-bit quantization (NF4)")
        elif self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            print("Using 8-bit quantization")
        
        # Try to import Janus-specific modules
        try:
            from janus.models import MultiModalityCausalLM, VLChatProcessor
            
            # Load processor
            self.vl_chat_processor = self._from_pretrained_local_first(
                VLChatProcessor.from_pretrained,
            )
            self.tokenizer = self.vl_chat_processor.tokenizer
            
            # Load model with optional quantization
            if quantization_config is not None:
                # Use float16 for quantized models to avoid dtype mismatches
                self.model = self._from_pretrained_local_first(
                    AutoModelForCausalLM.from_pretrained,
                    trust_remote_code=True,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
            else:
                self.model = self._from_pretrained_local_first(
                    AutoModelForCausalLM.from_pretrained,
                    trust_remote_code=True,
                )
            
        except ImportError:
            # Fallback: Load with transformers only
            print("Warning: janus package not found. Using transformers fallback.")
            print("Install janus: pip install git+https://github.com/deepseek-ai/Janus.git")
            
            from transformers import AutoProcessor
            
            self.vl_chat_processor = self._from_pretrained_local_first(
                AutoProcessor.from_pretrained,
                trust_remote_code=True,
            )
            self.tokenizer = self.vl_chat_processor.tokenizer
            
            # Load model with optional quantization
            if quantization_config is not None:
                # Use float16 for quantized models to avoid dtype mismatches
                self.model = self._from_pretrained_local_first(
                    AutoModelForCausalLM.from_pretrained,
                    trust_remote_code=True,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
            else:
                self.model = self._from_pretrained_local_first(
                    AutoModelForCausalLM.from_pretrained,
                    trust_remote_code=True,
                    torch_dtype=self.dtype,
                )
        
        # Move to device (skip if using quantization with device_map)
        if quantization_config is None:
            self.model = self.model.to(self.dtype).to(self.device).eval()
        else:
            self.model = self.model.eval()
            # Fix dtype mismatch: convert vision decoder to float16
            # The gen_vision_model is used for decoding and may have inconsistent dtypes
            if hasattr(self.model, 'gen_vision_model'):
                self.model.gen_vision_model = self.model.gen_vision_model.to(torch.float16)

        if hasattr(self.model, 'gen_vision_model'):
            self._patch_janus_upsample_dtype()
        
        # Update dtype for quantized models
        actual_dtype = torch.float16 if (self.load_in_4bit or self.load_in_8bit) else self.dtype
        
        print(f"Loaded Janus-Pro from {self.model_name_or_path}")
        print(f"Model dtype: {actual_dtype}, Device: {self.device}")
        if self.load_in_4bit:
            print("Quantization: 4-bit (memory efficient)")
        elif self.load_in_8bit:
            print("Quantization: 8-bit")

    def _get_compute_dtype(self) -> torch.dtype:
        """Return stable compute dtype for generation/decode."""
        if self.load_in_4bit or self.load_in_8bit:
            return torch.float16
        return self.dtype

    def _patch_janus_upsample_dtype(self) -> None:
        """Patch Janus Upsample to preserve input dtype.

        Janus VQ Upsample hard-casts interpolated tensors to bfloat16, which can
        mismatch decoder conv bias dtype (often float16) and crash with:
        "Input type (BFloat16) and bias type (Half) should be the same".
        """

        def _patched_forward(module_self, x):
            target_dtype = x.dtype
            if target_dtype != torch.float32:
                x = F.interpolate(
                    x.to(torch.float32),
                    scale_factor=2.0,
                    mode="nearest",
                ).to(target_dtype)
            else:
                x = F.interpolate(x, scale_factor=2.0, mode="nearest")

            if module_self.with_conv:
                x = module_self.conv(x)
            return x

        patched_count = 0
        for module in self.model.gen_vision_model.modules():
            if module.__class__.__name__ == "Upsample" and hasattr(module, "with_conv"):
                module.forward = types.MethodType(_patched_forward, module)
                patched_count += 1

        if patched_count > 0:
            print(f"Patched {patched_count} Janus Upsample layer(s) for dtype safety")
        
    def generate(
        self,
        prompt: Union[str, List[str]],
        config: Optional[GenerationConfig] = None,
        **kwargs: Any,
    ) -> List[Image.Image]:
        """
        Generate images using Janus-Pro autoregressive generation.
        
        Args:
            prompt: Text prompt(s) for generation
            config: Generation configuration
            **kwargs: Additional arguments (parallel_size, etc.)
            
        Returns:
            List of generated PIL Images
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        config = config or GenerationConfig()
        
        if isinstance(prompt, str):
            prompt = [prompt]
        
        # Set random seed for reproducibility
        if config.seed is not None:
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)
            
        return self._generate_batch(
            prompts=prompt,
            temperature=config.temperature,
            cfg_weight=config.guidance_scale,
            parallel_size=config.num_images_per_prompt,
            **kwargs,
        )
    
    @torch.inference_mode()
    def _generate_single(
        self,
        prompt: str,
        temperature: float = 1.0,
        parallel_size: int = 1,
        cfg_weight: float = 5.0,
        **kwargs: Any,
    ) -> List[Image.Image]:
        """
        Generate images for a single prompt using Janus-Pro.
        
        Based on official implementation from:
        https://github.com/deepseek-ai/Janus/blob/main/generation_inference.py
        """
        # Build conversation format
        conversation = [
            {"role": "<|User|>", "content": prompt},
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        # Apply SFT template
        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        formatted_prompt = sft_format + self.vl_chat_processor.image_start_tag
        
        # Encode prompt
        input_ids = self.tokenizer.encode(formatted_prompt)
        input_ids = torch.LongTensor(input_ids)
        
        # Create tokens for CFG (conditional and unconditional)
        tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).to(self.device)
        for i in range(parallel_size * 2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                # Unconditional: mask the prompt tokens
                tokens[i, 1:-1] = self.vl_chat_processor.pad_id
                
        # Get input embeddings
        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)
        compute_dtype = self._get_compute_dtype()
        inputs_embeds = inputs_embeds.to(compute_dtype)
        
        # Autoregressive generation
        generated_tokens = torch.zeros(
            (parallel_size, self.image_token_num_per_image), 
            dtype=torch.int
        ).to(self.device)
        
        past_key_values = None
        
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=compute_dtype)
            if self.device.startswith("cuda")
            else nullcontext()
        )

        with autocast_ctx:
            for i in range(self.image_token_num_per_image):
                outputs = self.model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    past_key_values=past_key_values if i != 0 else None,
                )
                past_key_values = outputs.past_key_values
                hidden_states = outputs.last_hidden_state

                # Get logits from generation head
                logits = self.model.gen_head(hidden_states[:, -1, :])

                # Apply CFG
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]
                logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

                # Sample next token
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens[:, i] = next_token.squeeze(dim=-1)

                # Prepare next input
                next_token = torch.cat(
                    [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)],
                    dim=1
                ).view(-1)
                img_embeds = self.model.prepare_gen_img_embeds(next_token)
                inputs_embeds = img_embeds.unsqueeze(dim=1).to(compute_dtype)
            
        # Decode generated tokens to images
        images = self._decode_tokens_to_images(generated_tokens, parallel_size)
        
        return images

    @torch.inference_mode()
    def _generate_batch(
        self,
        prompts: List[str],
        temperature: float = 1.0,
        parallel_size: int = 1,
        cfg_weight: float = 5.0,
        **kwargs: Any,
    ) -> List[Image.Image]:
        """Batched inference-only generation for multiple prompts.

        Processes all ``B`` prompts in a single autoregressive loop instead of
        looping one-by-one.  Variable-length prompts are left-padded and
        isolated via attention masks.  Sampling semantics are identical to
        ``_generate_single`` — only wall-clock time changes.
        """
        B = len(prompts)
        K = parallel_size
        N_cfg = B * K * 2
        N_img = B * K

        # ----- tokenize all prompts -----
        all_ids: List[torch.Tensor] = []
        for p in prompts:
            conv = [
                {"role": "<|User|>", "content": p},
                {"role": "<|Assistant|>", "content": ""},
            ]
            sft = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conv,
                sft_format=self.vl_chat_processor.sft_format,
                system_prompt="",
            )
            fmt = sft + self.vl_chat_processor.image_start_tag
            all_ids.append(torch.LongTensor(self.tokenizer.encode(fmt)))

        prompt_lens = [len(ids) for ids in all_ids]
        max_plen = max(prompt_lens)
        pad_id = self.vl_chat_processor.pad_id

        # ----- left-pad & build CFG pairs -----
        tokens = torch.full(
            (N_cfg, max_plen), pad_id, dtype=torch.int, device=self.device
        )
        attn_mask = torch.zeros(
            (N_cfg, max_plen), dtype=torch.long, device=self.device
        )
        row_plen = torch.zeros(N_cfg, dtype=torch.long, device=self.device)

        for b in range(B):
            ids = all_ids[b]
            L = prompt_lens[b]
            pad = max_plen - L
            for k in range(K):
                rc = (b * K + k) * 2
                ru = rc + 1
                tokens[rc, pad:] = ids
                tokens[ru, pad:] = ids
                if L > 2:
                    tokens[ru, pad + 1 : max_plen - 1] = pad_id
                attn_mask[rc, pad:] = 1
                attn_mask[ru, pad:] = 1
                row_plen[rc] = L
                row_plen[ru] = L

        compute_dtype = self._get_compute_dtype()
        T = self.image_token_num_per_image

        full_am = torch.zeros(
            (N_cfg, max_plen + T), dtype=torch.long, device=self.device
        )
        full_am[:, :max_plen] = attn_mask
        full_am[:, max_plen:] = 1

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=compute_dtype)
            if self.device.startswith("cuda")
            else nullcontext()
        )

        gen_tokens = torch.zeros(
            (N_img, T), dtype=torch.int, device=self.device
        )

        with autocast_ctx:
            inputs_embeds = self.model.language_model.get_input_embeddings()(
                tokens
            )
            inputs_embeds = inputs_embeds.to(compute_dtype)

            prompt_pos = attn_mask.long().cumsum(-1) - 1
            prompt_pos.masked_fill_(attn_mask == 0, 0)

            pkv = None
            for i in range(T):
                if i == 0:
                    cur_am = attn_mask
                    cur_pos = prompt_pos
                else:
                    cur_am = full_am[:, : max_plen + i]
                    cur_pos = (row_plen + i - 1).unsqueeze(1)

                outputs = self.model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=cur_am,
                    position_ids=cur_pos,
                    use_cache=True,
                    past_key_values=pkv if i != 0 else None,
                )
                pkv = outputs.past_key_values
                hidden = outputs.last_hidden_state

                logits = self.model.gen_head(hidden[:, -1, :])

                logit_c = logits[0::2, :]
                logit_u = logits[1::2, :]
                logits = logit_u + cfg_weight * (logit_c - logit_u)

                probs = torch.softmax(logits / temperature, dim=-1)
                nt = torch.multinomial(probs, num_samples=1)
                gen_tokens[:, i] = nt.squeeze(-1)

                nt_cfg = torch.cat(
                    [nt.unsqueeze(1), nt.unsqueeze(1)], dim=1
                ).view(-1)
                img_emb = self.model.prepare_gen_img_embeds(nt_cfg)
                inputs_embeds = img_emb.unsqueeze(1).to(compute_dtype)

            del pkv, outputs, hidden, inputs_embeds

        images = self._decode_tokens_to_images(gen_tokens, N_img)
        return images
    
    def _decode_tokens_to_images(
        self, 
        generated_tokens: torch.Tensor,
        parallel_size: int,
    ) -> List[Image.Image]:
        """Decode visual tokens back to PIL Images."""
        # Use the generation vision model to decode
        compute_dtype = self._get_compute_dtype()
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=compute_dtype)
            if self.device.startswith("cuda")
            else nullcontext()
        )

        with autocast_ctx:
            dec = self.model.gen_vision_model.decode_code(
                generated_tokens.to(dtype=torch.int),
                shape=[
                    parallel_size,
                    8,
                    self.img_size // self.patch_size,
                    self.img_size // self.patch_size
                ]
            )
        
        # Convert to numpy
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
        
        # Convert to PIL Images
        images = []
        for i in range(parallel_size):
            img = Image.fromarray(dec[i])
            images.append(img)
            
        return images
    
    def generate_with_logprobs(
        self,
        prompt: Union[str, List[str]],
        config: Optional[GenerationConfig] = None,
        return_tokens: bool = False,
        **kwargs: Any,
    ) -> Union[Tuple[List[Image.Image], torch.Tensor],
               Tuple[List[Image.Image], torch.Tensor, torch.Tensor]]:
        """
        Generate images and return log probabilities for RL training.

        Args:
            prompt: Text prompt(s).
            config: Generation config.
            return_tokens: If True, also return the generated token sequences
                (shape: (N, T)) in addition to images and log_probs.  These
                can be used to compute KL divergence with a reference model.

        Returns:
            (images, log_probs)  when ``return_tokens=False``  (default)
            (images, log_probs, generated_tokens)  when ``return_tokens=True``
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        config = config or GenerationConfig()

        if isinstance(prompt, str):
            prompt = [prompt]

        images, log_probs, gen_tokens = self._generate_with_logprobs_batch(
            prompts=prompt,
            temperature=config.temperature,
            cfg_weight=config.guidance_scale,
            parallel_size=config.num_images_per_prompt,
            **kwargs,
        )
        if return_tokens:
            return images, log_probs, gen_tokens
        return images, log_probs
    
    def _generate_with_logprobs_single(
        self,
        prompt: str,
        temperature: float = 1.0,
        parallel_size: int = 1,
        cfg_weight: float = 5.0,
        **kwargs: Any,
    ) -> Tuple[List[Image.Image], torch.Tensor]:
        """
        Generate images with log-probability tracking for RL training.

        Uses a two-phase REINFORCE-style approach to avoid OOM:

        Phase 1 (Generation) — ``torch.no_grad()``:
            Run the full 576-step autoregressive loop *without* gradients.
            This is identical to ``_generate_single`` but we also save the
            sampled token ids.

        Phase 2 (Scoring) — with gradients, single forward pass:
            Re-embed the *already-sampled* token sequence and do a single
            forward pass through the language model (with the prompt KV
            cached) to recompute the log-probabilities of those tokens.
            Only this pass builds a computation graph, so peak VRAM is
            ~1 forward pass worth of activations instead of 576x.

        The policy-gradient estimator  ``∇J = E[A * ∇ log π(a|s)]``  only
        needs ``log π`` to carry gradients — it does *not* need the sampling
        step itself to be differentiable.  So this decomposition is
        mathematically equivalent to the old single-loop implementation.
        """
        # -----------------------------------------------------------
        # Shared: build prompt tokens & CFG pair
        # -----------------------------------------------------------
        conversation = [
            {"role": "<|User|>", "content": prompt},
            {"role": "<|Assistant|>", "content": ""},
        ]

        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        formatted_prompt = sft_format + self.vl_chat_processor.image_start_tag

        input_ids = self.tokenizer.encode(formatted_prompt)
        input_ids = torch.LongTensor(input_ids)

        tokens = torch.zeros(
            (parallel_size * 2, len(input_ids)), dtype=torch.int
        ).to(self.device)
        for i in range(parallel_size * 2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = self.vl_chat_processor.pad_id

        compute_dtype = self._get_compute_dtype()

        # =========================================================
        # PHASE 1 — Generation (no gradients, low VRAM)
        # =========================================================
        generated_tokens = torch.zeros(
            (parallel_size, self.image_token_num_per_image),
            dtype=torch.int,
        ).to(self.device)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=compute_dtype)
            if self.device.startswith("cuda")
            else nullcontext()
        )

        prompt_kv = None  # saved from step 0 for Phase 2 reuse

        with torch.no_grad(), autocast_ctx:
            inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)
            inputs_embeds = inputs_embeds.to(compute_dtype)
            past_key_values = None

            for i in range(self.image_token_num_per_image):
                outputs = self.model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    past_key_values=past_key_values if i != 0 else None,
                )
                past_key_values = outputs.past_key_values

                if i == 0:
                    from transformers.cache_utils import DynamicCache
                    if isinstance(past_key_values, DynamicCache):
                        prompt_kv = DynamicCache()
                        for layer_idx in range(len(past_key_values)):
                            prompt_kv.update(
                                past_key_values.key_cache[layer_idx].clone(),
                                past_key_values.value_cache[layer_idx].clone(),
                                layer_idx,
                            )
                    else:
                        prompt_kv = tuple(
                            tuple(t.clone() for t in layer)
                            for layer in past_key_values
                        )

                hidden_states = outputs.last_hidden_state

                logits = self.model.gen_head(hidden_states[:, -1, :])

                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]
                logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens[:, i] = next_token.squeeze(dim=-1)

                next_token_cfg = next_token.repeat_interleave(2, dim=0).squeeze(-1)
                img_embeds = self.model.prepare_gen_img_embeds(next_token_cfg)
                inputs_embeds = img_embeds.unsqueeze(dim=1).to(compute_dtype)

            del past_key_values, outputs, hidden_states, inputs_embeds

        with torch.no_grad():
            images = self._decode_tokens_to_images(generated_tokens, parallel_size)

        # =========================================================
        # PHASE 2 — Scoring (single forward pass with gradients)
        # =========================================================
        # prompt_kv was saved at Phase 1 step 0 — no recomputation needed.

        # 2-b  Build CFG-paired image-token ids
        # generated_tokens: (parallel_size, 576) int
        # We need CFG interleaving: [cond_0, uncond_0, cond_1, uncond_1, ...]
        gen_tok_flat_cfg = torch.cat(
            [generated_tokens, generated_tokens], dim=0
        )  # (parallel_size*2, 576)
        interleaved_idx = torch.zeros(
            parallel_size * 2, dtype=torch.long, device=self.device
        )
        for s in range(parallel_size):
            interleaved_idx[2 * s] = s
            interleaved_idx[2 * s + 1] = s + parallel_size
        gen_tok_cfg = gen_tok_flat_cfg[interleaved_idx]  # (P*2, 576)

        total_img_tokens = self.image_token_num_per_image  # 576

        # 2-c  Single forward pass with gradients
        with autocast_ctx:
            all_img_embeds = self.model.prepare_gen_img_embeds(
                gen_tok_cfg.reshape(-1)
            )
            embed_dim = all_img_embeds.shape[-1]
            all_img_embeds = all_img_embeds.view(
                parallel_size * 2, total_img_tokens, embed_dim
            ).to(compute_dtype)

            scoring_out = self.model.language_model.model(
                inputs_embeds=all_img_embeds,
                use_cache=False,
                past_key_values=prompt_kv,
            )
            hidden = scoring_out.last_hidden_state  # (P*2, 576, dim)

            # gen_head at position j predicts token j+1.
            # hidden[:, 0..574, :] -> predictions for tokens 1..575.
            # We skip token 0 (losing 1/576 ~ 0.2% — negligible).
            pred_hidden = hidden[:, :-1, :]       # (P*2, 575, dim)
            logits = self.model.gen_head(pred_hidden)  # (P*2, 575, vocab)

            # Apply CFG to logits
            logit_cond = logits[0::2, :, :]       # (P, 575, vocab)
            logit_uncond = logits[1::2, :, :]     # (P, 575, vocab)
            cfg_logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

            # Log-probs of the actually-sampled tokens
            log_probs_all = torch.log_softmax(
                cfg_logits / temperature, dim=-1
            )  # (P, 575, vocab)

            target_tokens = generated_tokens[:, 1:]  # (P, 575)
            selected_log_probs = log_probs_all.gather(
                2, target_tokens.unsqueeze(-1).long()
            ).squeeze(-1)  # (P, 575)

            total_log_probs = selected_log_probs.sum(dim=1)  # (P,)

        del scoring_out, hidden, pred_hidden, logits, log_probs_all
        del prompt_kv, all_img_embeds, gen_tok_cfg, gen_tok_flat_cfg

        return images, total_log_probs, generated_tokens.detach().cpu()

    def _generate_with_logprobs_batch(
        self,
        prompts: List[str],
        temperature: float = 1.0,
        parallel_size: int = 1,
        cfg_weight: float = 5.0,
        **kwargs: Any,
    ) -> Tuple[List[Image.Image], torch.Tensor, torch.Tensor]:
        """Batched version of ``_generate_with_logprobs_single``.

        Processes **all** B prompts in a single set of forward passes
        instead of looping one-by-one.  Each prompt still produces
        ``parallel_size`` i.i.d. samples (same CFG interleaving), so the
        sampling semantics are identical — only wall-clock time changes.

        Variable-length prompts are handled via left-padding + attention
        masks so that no prompt attends to another's padding.
        """
        B = len(prompts)
        K = parallel_size
        N_cfg = B * K * 2          # CFG rows (cond + uncond per sample)
        N_img = B * K              # total images

        # ----- tokenize all prompts -----
        all_ids: List[torch.Tensor] = []
        for p in prompts:
            conv = [
                {"role": "<|User|>", "content": p},
                {"role": "<|Assistant|>", "content": ""},
            ]
            sft = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conv,
                sft_format=self.vl_chat_processor.sft_format,
                system_prompt="",
            )
            fmt = sft + self.vl_chat_processor.image_start_tag
            all_ids.append(torch.LongTensor(self.tokenizer.encode(fmt)))

        prompt_lens = [len(ids) for ids in all_ids]
        max_plen = max(prompt_lens)
        pad_id = self.vl_chat_processor.pad_id

        # ----- left-pad & build CFG pairs -----
        tokens = torch.full(
            (N_cfg, max_plen), pad_id, dtype=torch.int, device=self.device
        )
        attn_mask = torch.zeros(
            (N_cfg, max_plen), dtype=torch.long, device=self.device
        )
        row_plen = torch.zeros(N_cfg, dtype=torch.long, device=self.device)

        for b in range(B):
            ids = all_ids[b]
            L = prompt_lens[b]
            pad = max_plen - L
            for k in range(K):
                rc = (b * K + k) * 2       # cond row
                ru = rc + 1                 # uncond row
                tokens[rc, pad:] = ids
                tokens[ru, pad:] = ids
                if L > 2:
                    tokens[ru, pad + 1 : max_plen - 1] = pad_id
                attn_mask[rc, pad:] = 1
                attn_mask[ru, pad:] = 1
                row_plen[rc] = L
                row_plen[ru] = L

        compute_dtype = self._get_compute_dtype()
        T = self.image_token_num_per_image  # 576

        # full attention mask covering prompt + all generated tokens
        full_am = torch.zeros(
            (N_cfg, max_plen + T), dtype=torch.long, device=self.device
        )
        full_am[:, :max_plen] = attn_mask
        full_am[:, max_plen:] = 1

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=compute_dtype)
            if self.device.startswith("cuda")
            else nullcontext()
        )

        # ===========================================================
        # PHASE 1 — Generation (no gradients)
        # ===========================================================
        gen_tokens = torch.zeros(
            (N_img, T), dtype=torch.int, device=self.device
        )

        prompt_pos = attn_mask.long().cumsum(-1) - 1
        prompt_pos.masked_fill_(attn_mask == 0, 0)

        prompt_kv = None  # saved from step 0 for Phase 2 reuse

        with torch.no_grad(), autocast_ctx:
            inputs_embeds = self.model.language_model.get_input_embeddings()(
                tokens
            )
            inputs_embeds = inputs_embeds.to(compute_dtype)

            pkv = None
            for i in range(T):
                if i == 0:
                    cur_am = attn_mask
                    cur_pos = prompt_pos
                else:
                    cur_am = full_am[:, : max_plen + i]
                    cur_pos = (row_plen + i - 1).unsqueeze(1)

                outputs = self.model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=cur_am,
                    position_ids=cur_pos,
                    use_cache=True,
                    past_key_values=pkv if i != 0 else None,
                )
                pkv = outputs.past_key_values

                # Save prompt-only KV at step 0 to skip recomputation in Phase 2
                if i == 0:
                    from transformers.cache_utils import DynamicCache
                    if isinstance(pkv, DynamicCache):
                        prompt_kv = DynamicCache()
                        for layer_idx in range(len(pkv)):
                            prompt_kv.update(
                                pkv.key_cache[layer_idx].clone(),
                                pkv.value_cache[layer_idx].clone(),
                                layer_idx,
                            )
                    else:
                        prompt_kv = tuple(
                            tuple(t.clone() for t in layer)
                            for layer in pkv
                        )

                hidden = outputs.last_hidden_state

                logits = self.model.gen_head(hidden[:, -1, :])

                logit_c = logits[0::2, :]
                logit_u = logits[1::2, :]
                logits = logit_u + cfg_weight * (logit_c - logit_u)

                probs = torch.softmax(logits / temperature, dim=-1)
                nt = torch.multinomial(probs, num_samples=1)
                gen_tokens[:, i] = nt.squeeze(-1)

                nt_cfg = nt.repeat_interleave(2, dim=0).squeeze(-1)
                img_emb = self.model.prepare_gen_img_embeds(nt_cfg)
                inputs_embeds = img_emb.unsqueeze(1).to(compute_dtype)

            del pkv, outputs, hidden, inputs_embeds

        with torch.no_grad():
            images = self._decode_tokens_to_images(gen_tokens, N_img)

        # ===========================================================
        # PHASE 2 — Scoring (single forward pass with gradients)
        # ===========================================================
        # prompt_kv was saved at Phase 1 step 0 — no recomputation needed.

        # CFG-interleaved image tokens
        gt_dup = torch.cat([gen_tokens, gen_tokens], dim=0)  # (2*N_img, T)
        interleave = torch.stack([
            torch.arange(N_img, device=self.device),
            torch.arange(N_img, device=self.device) + N_img,
        ], dim=1).reshape(-1)
        gt_cfg = gt_dup[interleave]  # (N_cfg, T)

        score_am = full_am   # (N_cfg, max_plen + T)
        score_pos = row_plen.unsqueeze(1) + torch.arange(
            T, device=self.device
        ).unsqueeze(0)  # (N_cfg, T)

        with autocast_ctx:
            all_img_emb = self.model.prepare_gen_img_embeds(gt_cfg.reshape(-1))
            edim = all_img_emb.shape[-1]
            all_img_emb = all_img_emb.view(N_cfg, T, edim).to(compute_dtype)

            s_out = self.model.language_model.model(
                inputs_embeds=all_img_emb,
                attention_mask=score_am,
                position_ids=score_pos,
                use_cache=False,
                past_key_values=prompt_kv,
            )
            hidden = s_out.last_hidden_state   # (N_cfg, T, dim)

            pred_h = hidden[:, :-1, :]                          # (N_cfg, T-1, dim)
            logits = self.model.gen_head(pred_h)                # (N_cfg, T-1, V)

            lc = logits[0::2, :, :]
            lu = logits[1::2, :, :]
            cfg_logits = lu + cfg_weight * (lc - lu)            # (N_img, T-1, V)

            lp_all = torch.log_softmax(cfg_logits / temperature, dim=-1)
            tgt = gen_tokens[:, 1:].long()                      # (N_img, T-1)
            sel = lp_all.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
            total_lp = sel.sum(dim=1)                            # (N_img,)

        del s_out, hidden, pred_h, logits, lp_all
        del prompt_kv, all_img_emb, gt_cfg, gt_dup

        return images, total_lp, gen_tokens.detach().cpu()

    @torch.no_grad()
    def score_tokens_with_model(
        self,
        model: nn.Module,
        prompts: List[str],
        generated_tokens: torch.Tensor,
        temperature: float = 1.0,
        cfg_weight: float = 5.0,
    ) -> torch.Tensor:
        """
        Score generated token sequences with an arbitrary (frozen) model.

        Used by GRPOTrainer to compute reference-policy log-probs for the KL
        penalty without re-running the full autoregressive generation loop.

        Args:
            model: A frozen copy of the language model (e.g. ref_model).
            prompts: List of N text prompts.
            generated_tokens: Int tensor of shape (N, T) — the tokens that
                were sampled by the *current* policy.
            temperature: Sampling temperature (must match generation).
            cfg_weight: CFG guidance scale (must match generation).

        Returns:
            log_probs: Float tensor of shape (N,) — sum of token log-probs.
        """
        compute_dtype = self._get_compute_dtype()
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=compute_dtype)
            if self.device.startswith("cuda")
            else nullcontext()
        )

        generated_tokens = generated_tokens.to(self.device)
        N, T = generated_tokens.shape

        # ----- tokenize all prompts & left-pad -----
        all_ids: List[torch.Tensor] = []
        for prompt in prompts:
            conv = [
                {"role": "<|User|>", "content": prompt},
                {"role": "<|Assistant|>", "content": ""},
            ]
            sft = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conv,
                sft_format=self.vl_chat_processor.sft_format,
                system_prompt="",
            )
            fmt = sft + self.vl_chat_processor.image_start_tag
            all_ids.append(torch.LongTensor(self.tokenizer.encode(fmt)))

        prompt_lens = [len(ids) for ids in all_ids]
        max_plen = max(prompt_lens)
        pad_id = self.vl_chat_processor.pad_id

        tokens_cfg = torch.full(
            (N * 2, max_plen), pad_id, dtype=torch.int, device=self.device
        )
        attn_mask = torch.zeros(
            (N * 2, max_plen), dtype=torch.long, device=self.device
        )
        row_plen = torch.zeros(N * 2, dtype=torch.long, device=self.device)

        for b in range(N):
            ids = all_ids[b]
            L = prompt_lens[b]
            pad = max_plen - L
            rc, ru = b * 2, b * 2 + 1
            tokens_cfg[rc, pad:] = ids
            tokens_cfg[ru, pad:] = ids
            if L > 2:
                tokens_cfg[ru, pad + 1 : max_plen - 1] = pad_id
            attn_mask[rc, pad:] = 1
            attn_mask[ru, pad:] = 1
            row_plen[rc] = L
            row_plen[ru] = L

        with autocast_ctx:
            # Cache prompt KV
            p_emb = model.language_model.get_input_embeddings()(tokens_cfg)
            p_emb = p_emb.to(compute_dtype)
            p_pos = attn_mask.long().cumsum(-1) - 1
            p_pos.masked_fill_(attn_mask == 0, 0)
            p_out = model.language_model.model(
                inputs_embeds=p_emb,
                attention_mask=attn_mask,
                position_ids=p_pos,
                use_cache=True,
            )
            prompt_kv = p_out.past_key_values
            del p_out, p_emb

            # CFG-interleaved image tokens
            tok_dup = torch.cat(
                [generated_tokens, generated_tokens], dim=0
            )  # (2N, T)
            interleave = torch.stack([
                torch.arange(N, device=self.device),
                torch.arange(N, device=self.device) + N,
            ], dim=1).reshape(-1)
            tok_cfg = tok_dup[interleave]  # (2N, T)

            score_am = torch.zeros(
                (N * 2, max_plen + T), dtype=torch.long, device=self.device
            )
            score_am[:, :max_plen] = attn_mask
            score_am[:, max_plen:] = 1
            score_pos = row_plen.unsqueeze(1) + torch.arange(
                T, device=self.device
            ).unsqueeze(0)

            all_img_embeds = model.prepare_gen_img_embeds(tok_cfg.reshape(-1))
            embed_dim = all_img_embeds.shape[-1]
            all_img_embeds = all_img_embeds.view(N * 2, T, embed_dim).to(
                compute_dtype
            )

            scoring_out = model.language_model.model(
                inputs_embeds=all_img_embeds,
                attention_mask=score_am,
                position_ids=score_pos,
                use_cache=False,
                past_key_values=prompt_kv,
            )
            hidden = scoring_out.last_hidden_state  # (2N, T, dim)

            pred_hidden = hidden[:, :-1, :]  # (2N, T-1, dim)
            logits = model.gen_head(pred_hidden)  # (2N, T-1, vocab)

            logit_cond = logits[0::2, :, :]
            logit_uncond = logits[1::2, :, :]
            cfg_logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

            lp = torch.log_softmax(cfg_logits / temperature, dim=-1)
            target = generated_tokens[:, 1:].long()  # (N, T-1)
            sel = lp.gather(2, target.unsqueeze(-1)).squeeze(-1)  # (N, T-1)

            del prompt_kv, scoring_out, hidden, pred_hidden, logits

        return sel.sum(dim=1)  # (N,)

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
            print(f"Loaded LoRA weights from {lora_path}")
        else:
            # Create new LoRA adapter
            default_config = {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": [
                    "q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                "task_type": TaskType.CAUSAL_LM,
            }
            if lora_config:
                default_config.update(lora_config)
                
            peft_config = LoraConfig(**default_config)
            self.model = get_peft_model(self.model, peft_config)
            print("Created new LoRA adapter")
            
        self.lora_enabled = True
        self.model.print_trainable_parameters()
        
    def disable_lora(self) -> None:
        """Disable LoRA adapter."""
        if hasattr(self.model, "disable_adapter"):
            self.model.disable_adapter()
        self.lora_enabled = False
        
    def save_lora(self, save_path: str) -> None:
        """Save LoRA weights."""
        if self.lora_enabled and hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(save_path)
            print(f"Saved LoRA weights to {save_path}")


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
        prefer_local_files: bool = True,
    ):
        super().__init__(
            model_name_or_path,
            device,
            dtype,
            prefer_local_files=prefer_local_files,
        )
        self.scheduler_type = scheduler
        self.pipe = None
        
    def load_model(self) -> None:
        """Load diffusion pipeline."""
        from diffusers import (
            StableDiffusionXLPipeline,
            DDIMScheduler,
            EulerAncestralDiscreteScheduler,
        )
        
        self.pipe = self._from_pretrained_local_first(
            StableDiffusionXLPipeline.from_pretrained,
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
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            print("xformers not available, using default attention")
        
        # Store reference to model
        self.model = self.pipe.unet
        
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
        
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get trainable UNet parameters."""
        if self.pipe is None:
            return []
        return [p for p in self.pipe.unet.parameters() if p.requires_grad]

    