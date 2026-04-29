#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
import types
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmarks import GenAIBenchmark, TIFABenchmark
from src.evaluation.io import append_jsonl
from src.evaluation.janus_compat import (
    build_janus_model_load_kwargs,
    build_janus_retry_load_kwargs,
    import_janus_vlchatprocessor,
    is_meta_tensor_item_error,
)
from src.evaluation.schemas import GeneratedSampleRecord


@dataclass
class GenerationConfig:
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    height: int = 384
    width: int = 384
    num_images_per_prompt: int = 1
    prompt_batch_size: int = 1
    seed: Optional[int] = None
    temperature: float = 1.0


def default_lora_path() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "lora" / "grpo_siliconflow_quick_final"


def build_image_path(
    output_dir: Path,
    benchmark: str,
    variant: str,
    sample_id: str,
    image_index: int = 0,
) -> Path:
    suffix = "" if image_index == 0 else f"_{image_index:02d}"
    return Path(output_dir) / "images" / benchmark / variant / f"{sample_id}{suffix}.png"


def should_skip_sample(target_path: Path, resume: bool) -> bool:
    return resume and target_path.exists()


class JanusProRunner:
    def __init__(
        self,
        model_name_or_path: str = "deepseek-ai/Janus-Pro-1B",
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, dtype)
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.model = None
        self.vl_chat_processor = None
        self.tokenizer = None
        self.image_token_num_per_image = 576
        self.img_size = 384
        self.patch_size = 16

    def load_model(self) -> None:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        try:
            VLChatProcessor = import_janus_vlchatprocessor()
        except Exception as exc:
            raise RuntimeError(
                "Failed to import Janus VLChatProcessor. Janus text-to-image generation requires "
                "the official DeepSeek Janus package; AutoProcessor is not a valid fallback here. "
                "If the Janus package still fails on Python 3.10+ / newer transformers, reinstall "
                "Janus from the official DeepSeek GitHub repo and restart the runtime."
            ) from exc

        try:
            self.vl_chat_processor = VLChatProcessor.from_pretrained(self.model_name_or_path)
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize Janus VLChatProcessor from the pretrained model. "
                "This usually means the Colab environment has an incompatible Janus/protobuf/transformers setup. "
                "Reinstall Janus from the official GitHub repo, then restart the runtime before rerunning generation."
            ) from exc

        if not hasattr(self.vl_chat_processor, "tokenizer"):
            raise RuntimeError(
                f"Loaded processor type {type(self.vl_chat_processor).__name__} does not expose a tokenizer. "
                "Janus generation expects the official VLChatProcessor."
            )
        if not hasattr(self.vl_chat_processor, "apply_sft_template_for_multi_turn_prompts"):
            raise RuntimeError(
                f"Loaded processor type {type(self.vl_chat_processor).__name__} does not expose Janus prompt templating methods. "
                "Janus generation expects the official VLChatProcessor."
            )

        self.tokenizer = self.vl_chat_processor.tokenizer
        load_kwargs = build_janus_model_load_kwargs(
            torch_dtype=torch.float16 if quantization_config is not None else self.dtype,
            quantization_config=quantization_config,
        )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, **load_kwargs)
        except RuntimeError as exc:
            if not is_meta_tensor_item_error(exc):
                raise
            retry_kwargs = build_janus_retry_load_kwargs(load_kwargs)
            print(
                "[generate] detected Janus meta-tensor init failure; "
                "retrying model load with low_cpu_mem_usage=False"
            )
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, **retry_kwargs)
        if quantization_config is None:
            self.model = self.model.to(self.dtype).to(self.device).eval()
        else:
            self.model = self.model.eval()
            if hasattr(self.model, "gen_vision_model"):
                self.model.gen_vision_model = self.model.gen_vision_model.to(torch.float16)
        if hasattr(self.model, "gen_vision_model"):
            self._patch_janus_upsample_dtype()

    def _patch_janus_upsample_dtype(self) -> None:
        def _patched_forward(module_self, x):
            target_dtype = x.dtype
            if target_dtype != torch.float32:
                x = F.interpolate(x.to(torch.float32), scale_factor=2.0, mode="nearest").to(target_dtype)
            else:
                x = F.interpolate(x, scale_factor=2.0, mode="nearest")
            if module_self.with_conv:
                x = module_self.conv(x)
            return x

        for module in self.model.gen_vision_model.modules():
            if module.__class__.__name__ == "Upsample" and hasattr(module, "with_conv"):
                module.forward = _patched_forward.__get__(module, module.__class__)

    def _ensure_prepare_inputs_for_generation(self) -> None:
        if self.model is None or hasattr(self.model, "prepare_inputs_for_generation"):
            return

        def _prepare_inputs_for_generation(module_self, input_ids=None, **kwargs):
            payload = dict(kwargs)
            if input_ids is not None:
                payload["input_ids"] = input_ids
            return payload

        self.model.prepare_inputs_for_generation = types.MethodType(
            _prepare_inputs_for_generation,
            self.model,
        )

    def enable_lora(self, lora_path: str, scale: float = 1.0) -> None:
        from peft import PeftModel

        if self.model is None:
            raise RuntimeError("Model not loaded")
        self._ensure_prepare_inputs_for_generation()
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        if hasattr(self.model, "set_adapter"):
            try:
                self.model.set_adapter(scale)
            except Exception:
                pass

    def disable_lora(self) -> None:
        if hasattr(self.model, "disable_adapter"):
            self.model.disable_adapter()

    def generate(self, prompt: Union[str, List[str]], config: GenerationConfig) -> List[Image.Image]:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        if isinstance(prompt, str):
            prompt = [prompt]
        if config.seed is not None:
            torch.manual_seed(config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed)
        images: List[Image.Image] = []
        batch_size = max(1, config.prompt_batch_size)
        for start_idx in range(0, len(prompt), batch_size):
            prompt_batch = prompt[start_idx : start_idx + batch_size]
            images.extend(
                self._generate_batch(
                    prompts=prompt_batch,
                    temperature=config.temperature,
                    cfg_weight=config.guidance_scale,
                )
            )
        return images

    @torch.inference_mode()
    def _generate_batch(
        self,
        prompts: List[str],
        temperature: float = 1.0,
        cfg_weight: float = 5.0,
    ) -> List[Image.Image]:
        encoded_prompts: List[torch.Tensor] = []
        for prompt in prompts:
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
            encoded_prompts.append(torch.LongTensor(self.tokenizer.encode(formatted_prompt)))

        batch_size = len(encoded_prompts)
        max_len = max(ids.shape[0] for ids in encoded_prompts)
        pad_id = self.vl_chat_processor.pad_id
        tokens = torch.full((batch_size * 2, max_len), pad_id, dtype=torch.int, device=self.device)

        for prompt_idx, input_ids in enumerate(encoded_prompts):
            seq_len = input_ids.shape[0]
            start_col = max_len - seq_len
            cond_row = prompt_idx * 2
            uncond_row = cond_row + 1
            tokens[cond_row, start_col:] = input_ids.to(self.device)
            tokens[uncond_row, start_col:] = input_ids.to(self.device)
            if seq_len > 2:
                tokens[uncond_row, start_col + 1 : max_len - 1] = pad_id

        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)
        compute_dtype = torch.float16 if self.device.startswith("cuda") else self.dtype
        inputs_embeds = inputs_embeds.to(compute_dtype)
        generated_tokens = torch.zeros((batch_size, self.image_token_num_per_image), dtype=torch.int).to(self.device)
        past_key_values = None
        autocast_ctx = torch.autocast(device_type="cuda", dtype=compute_dtype) if self.device.startswith("cuda") else nullcontext()
        with autocast_ctx:
            for step in range(self.image_token_num_per_image):
                outputs = self.model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    past_key_values=past_key_values if step != 0 else None,
                )
                past_key_values = outputs.past_key_values
                hidden_states = outputs.last_hidden_state
                logits = self.model.gen_head(hidden_states[:, -1, :])
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]
                logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens[:, step] = next_token.squeeze(dim=-1)
                next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
                img_embeds = self.model.prepare_gen_img_embeds(next_token)
                inputs_embeds = img_embeds.unsqueeze(dim=1).to(compute_dtype)
        return self._decode_tokens_to_images(generated_tokens, batch_size)

    def _decode_tokens_to_images(self, generated_tokens: torch.Tensor, parallel_size: int) -> List[Image.Image]:
        compute_dtype = torch.float16 if self.device.startswith("cuda") else self.dtype
        autocast_ctx = torch.autocast(device_type="cuda", dtype=compute_dtype) if self.device.startswith("cuda") else nullcontext()
        with autocast_ctx:
            dec = self.model.gen_vision_model.decode_code(
                generated_tokens.to(dtype=torch.int),
                shape=[parallel_size, 8, self.img_size // self.patch_size, self.img_size // self.patch_size],
            )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
        return [Image.fromarray(dec[idx]) for idx in range(parallel_size)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark images for TIFA and GenAI-Bench")
    parser.add_argument("--benchmark", choices=["tifa", "genai_bench"], required=True)
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--variant", choices=["before", "after"], required=True)
    parser.add_argument("--base_model", type=str, default="deepseek-ai/Janus-Pro-1B")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--prompt_batch_size", type=int, default=1)
    return parser.parse_args()


def load_manifest(benchmark: str, manifest_path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    if benchmark == "tifa":
        return TIFABenchmark(manifest_path).iter_samples(limit=limit)
    return GenAIBenchmark(manifest_path).iter_samples(limit=limit)


def save_generated_sample(
    output_dir: Path,
    benchmark: str,
    variant: str,
    sample: Dict[str, Any],
    image: Image.Image,
    generation_config: GenerationConfig,
    model_name: str,
    checkpoint_or_lora: str,
    image_index: int = 0,
) -> Path:
    image_path = build_image_path(
        output_dir,
        benchmark,
        variant,
        sample["sample_id"],
        image_index=image_index,
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    record = GeneratedSampleRecord(
        benchmark=benchmark,
        sample_id=sample["sample_id"],
        prompt=sample["prompt"],
        variant=variant,
        seed=generation_config.seed if generation_config.seed is not None else -1,
        model_name=model_name,
        checkpoint_or_lora=checkpoint_or_lora,
        image_path=str(image_path),
        generation_config={
            **asdict(generation_config),
            "image_index": image_index,
        },
    )
    append_jsonl(output_dir / "results" / "generated_samples.jsonl", record.to_dict())
    return image_path


def run() -> None:
    args = parse_args()
    job_start_time = time.perf_counter()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest_path)
    samples = load_manifest(args.benchmark, manifest_path, args.limit)
    print(
        f"[generate] benchmark={args.benchmark} variant={args.variant} "
        f"samples={len(samples)} prompt_batch_size={args.prompt_batch_size} "
        f"dtype={args.dtype} device={args.device or ('cuda' if torch.cuda.is_available() else 'cpu')}"
    )
    results_path = output_dir / "results" / "generated_samples.jsonl"
    if results_path.exists() and not args.resume:
        results_path.unlink()
    generator = JanusProRunner(
        model_name_or_path=args.base_model,
        device=args.device,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    print(f"[generate] loading model={args.base_model}")
    generator.load_model()
    if args.variant == "after":
        lora_path = Path(args.lora_path) if args.lora_path else default_lora_path()
        print(f"[generate] enabling lora={lora_path}")
        generator.enable_lora(str(lora_path))
    else:
        generator.disable_lora()
        print("[generate] running baseline model without lora")

    generation_config = GenerationConfig(
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=1,
        prompt_batch_size=args.prompt_batch_size,
        seed=args.seed,
    )
    pending_samples = []
    skipped_count = 0
    for sample in samples:
        image_path = build_image_path(output_dir, args.benchmark, args.variant, sample["sample_id"])
        if should_skip_sample(image_path, args.resume):
            skipped_count += 1
            continue
        pending_samples.append(sample)
    print(
        f"[generate] pending={len(pending_samples)} skipped={skipped_count} "
        f"output_dir={output_dir}"
    )

    batch_size = max(1, args.prompt_batch_size)
    completed_count = 0
    for start_idx in range(0, len(pending_samples), batch_size):
        sample_batch = pending_samples[start_idx : start_idx + batch_size]
        prompts = [sample["prompt"] for sample in sample_batch]
        batch_start = time.perf_counter()
        batch_number = start_idx // batch_size + 1
        total_batches = (len(pending_samples) + batch_size - 1) // batch_size
        batch_ids = [sample["sample_id"] for sample in sample_batch]
        print(
            f"[generate] batch {batch_number}/{total_batches} "
            f"samples={start_idx + 1}-{start_idx + len(sample_batch)} "
            f"ids={batch_ids[0]}..{batch_ids[-1]}"
        )
        images = generator.generate(prompts, generation_config)
        for sample, image in zip(sample_batch, images):
            save_generated_sample(
                output_dir=output_dir,
                benchmark=args.benchmark,
                variant=args.variant,
                sample=sample,
                image=image,
                generation_config=generation_config,
                model_name=args.base_model,
                checkpoint_or_lora=str(args.lora_path or default_lora_path()) if args.variant == "after" else "base",
            )
            completed_count += 1
        batch_elapsed = time.perf_counter() - batch_start
        rate = len(sample_batch) / batch_elapsed if batch_elapsed > 0 else 0.0
        print(
            f"[generate] batch {batch_number}/{total_batches} done "
            f"in {batch_elapsed:.1f}s, rate={rate:.2f} prompts/s, "
            f"completed={completed_count}/{len(pending_samples)}"
        )

    total_elapsed = time.perf_counter() - job_start_time
    print(
        f"[generate] finished benchmark={args.benchmark} variant={args.variant} "
        f"generated={completed_count} skipped={skipped_count} "
        f"elapsed={total_elapsed/60:.1f} min"
    )


if __name__ == "__main__":
    run()
