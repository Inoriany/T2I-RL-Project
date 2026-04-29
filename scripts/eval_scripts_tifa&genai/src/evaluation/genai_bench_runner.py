import base64
import concurrent.futures
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from src.evaluation.io import append_jsonl, read_jsonl
from src.evaluation.judge_utils import request_json_chat_completion

DEFAULT_RUBRIC_DIMENSIONS = [
    "alignment",
    "instruction_fidelity",
    "compositionality",
    "visual_quality",
]


def compute_overall_score(subscores: Dict[str, float]) -> float:
    values = list(subscores.values())
    return round(sum(values) / len(values), 6) if values else 0.0


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def image_to_data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def build_rubric_prompt(prompt: str, category: str, skills: List[str]) -> str:
    rubric = "\n".join(f'- "{name}": score from 0 to 1' for name in DEFAULT_RUBRIC_DIMENSIONS)
    skills_text = ", ".join(skills) if skills else "none"
    return (
        "You are judging a text-to-image sample.\n"
        f"Prompt: {prompt}\n"
        f"Category: {category}\n"
        f"Skills: {skills_text}\n"
        "Return JSON only with numeric subscores.\n"
        f"Rubric:\n{rubric}\n"
        'Format: {"alignment": 0.0, "instruction_fidelity": 0.0, "compositionality": 0.0, "visual_quality": 0.0}'
    )


def build_openai_client(api_key: Optional[str], base_url: Optional[str] = None):
    from openai import OpenAI

    kwargs: Dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def judge_image(client: Any, model: str, image: Image.Image, prompt: str, max_tokens: int = 96) -> Dict[str, Any]:
    return request_json_chat_completion(
        client=client,
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_uri(image)}},
                ],
            }
        ],
        max_tokens=max_tokens,
        max_attempts=3,
    )


def _coerce_subscores(payload: Dict[str, Any]) -> Dict[str, float]:
    subscores = {}
    for name in DEFAULT_RUBRIC_DIMENSIONS:
        value = payload.get(name, 0.0)
        try:
            subscores[name] = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            subscores[name] = 0.0
    return subscores


def derive_error_types(subscores: Dict[str, float]) -> List[str]:
    error_types: List[str] = []
    if subscores.get("alignment", 1.0) < 0.5:
        error_types.append("low_alignment")
    if subscores.get("instruction_fidelity", 1.0) < 0.5:
        error_types.append("missed_instruction")
    if subscores.get("compositionality", 1.0) < 0.5:
        error_types.append("weak_composition")
    if subscores.get("visual_quality", 1.0) < 0.5:
        error_types.append("low_visual_quality")
    return error_types


def score_genai_sample(
    sample: Dict[str, Any],
    judge_client: Any,
    judge_model: str,
    image: Optional[Image.Image] = None,
    image_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if image is None:
        if image_path is None:
            raise ValueError("Either image or image_path must be provided")
        image = Image.open(image_path).convert("RGB")
    prompt = build_rubric_prompt(sample["prompt"], sample.get("category", ""), sample.get("skills", []))
    judge_result = judge_image(judge_client, judge_model, image, prompt)
    response_text = judge_result["raw_text"]
    payload = judge_result["parsed_json"]
    subscores = _coerce_subscores(payload)
    return {
        "benchmark": "genai_bench",
        "sample_id": sample["sample_id"],
        "prompt": sample["prompt"],
        "category": sample.get("category", ""),
        "skills": sample.get("skills", []),
        "score": compute_overall_score(subscores),
        "subscores": subscores,
        "error_types": derive_error_types(subscores),
        "judge_metadata": {
            "judge_model": judge_model,
            "judge_raw": response_text,
            "judge_finish_reason": judge_result["finish_reason"],
            "judge_attempts": judge_result["attempts"],
            "max_tokens": 96,
            "max_attempts": 3,
        },
    }


def _evaluate_single_sample(
    sample: Dict[str, Any],
    images_root: Path,
    variant: str,
    judge_client: Any,
    judge_model: str,
) -> Dict[str, Any]:
    image_path = Path(images_root) / "genai_bench" / variant / f"{sample['sample_id']}.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Missing generated image: {image_path}")
    result = score_genai_sample(
        sample=sample,
        image_path=image_path,
        judge_client=judge_client,
        judge_model=judge_model,
    )
    result["variant"] = variant
    result["image_path"] = str(image_path)
    return result


def evaluate_manifest(
    manifest_path: Path,
    images_root: Path,
    variant: str,
    judge_client: Any,
    judge_model: str,
    output_path: Path,
    resume: bool = False,
    max_workers: int = 1,
    log_every: int = 10,
) -> List[Dict[str, Any]]:
    output_path = Path(output_path)
    if output_path.exists() and not resume:
        output_path.unlink()
    existing_ids = set()
    if resume and output_path.exists():
        existing_ids = {row["sample_id"] for row in read_jsonl(output_path)}

    manifest_rows = read_jsonl(manifest_path)
    pending_samples = [sample for sample in manifest_rows if sample["sample_id"] not in existing_ids]
    print(
        f"[genai] variant={variant} total={len(manifest_rows)} pending={len(pending_samples)} "
        f"skipped={len(existing_ids)} workers={max_workers}"
    )
    results: List[Dict[str, Any]] = []
    start_time = time.perf_counter()
    if not pending_samples:
        print(f"[genai] nothing to do for variant={variant}")
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        future_to_sample = {
            executor.submit(
                _evaluate_single_sample,
                sample,
                Path(images_root),
                variant,
                judge_client,
                judge_model,
            ): sample
            for sample in pending_samples
        }
        for completed_idx, future in enumerate(concurrent.futures.as_completed(future_to_sample), start=1):
            sample = future_to_sample[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"[genai] failed sample_id={sample['sample_id']} error={exc}")
                raise
            append_jsonl(output_path, result)
            results.append(result)
            if completed_idx == 1 or completed_idx % max(1, log_every) == 0 or completed_idx == len(pending_samples):
                elapsed = time.perf_counter() - start_time
                rate = completed_idx / elapsed if elapsed > 0 else 0.0
                print(
                    f"[genai] completed={completed_idx}/{len(pending_samples)} "
                    f"last_sample={sample['sample_id']} rate={rate:.2f} samples/s elapsed={elapsed/60:.1f} min"
                )
    total_elapsed = time.perf_counter() - start_time
    print(
        f"[genai] finished variant={variant} scored={len(results)} skipped={len(existing_ids)} "
        f"elapsed={total_elapsed/60:.1f} min"
    )
    return results
