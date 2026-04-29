import base64
import concurrent.futures
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from src.evaluation.io import append_jsonl, read_jsonl
from src.evaluation.judge_utils import extract_json_object, request_json_chat_completion

ERROR_BY_QUESTION_TYPE = {
    "object": "missing_object",
    "attribute": "wrong_attribute",
    "count": "wrong_count",
    "relation": "wrong_relation",
}


def map_question_type_to_error(question_type: str) -> str:
    return ERROR_BY_QUESTION_TYPE.get(question_type.lower(), "other")


def compute_question_accuracy(question_results: List[Dict[str, Any]]) -> float:
    if not question_results:
        return 0.0
    return sum(1 for row in question_results if row.get("correct")) / len(question_results)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def answers_match(predicted: str, expected: str) -> bool:
    predicted_norm = normalize_text(predicted)
    expected_norm = normalize_text(expected)
    if not predicted_norm or not expected_norm:
        return False
    if predicted_norm == expected_norm:
        return True
    return expected_norm in predicted_norm or predicted_norm in expected_norm


def image_to_data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def build_question_prompt(prompt: str, question: str, expected_answer: str, question_type: str) -> str:
    return (
        "Judge the image against the prompt and answer the question.\n"
        f"Prompt: {prompt}\n"
        f"Question type: {question_type}\n"
        f"Question: {question}\n"
        f"Expected answer: {expected_answer}\n"
        "Return JSON only.\n"
        'Format: {"answer": "..."}\n'
        "Use the shortest valid answer possible. Do not include explanation."
    )


def build_openai_client(api_key: Optional[str], base_url: Optional[str] = None):
    from openai import OpenAI

    kwargs: Dict[str, Any] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def judge_question(
    client: Any,
    model: str,
    image: Image.Image,
    prompt: str,
    max_tokens: int = 64,
) -> Dict[str, Any]:
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


def score_tifa_sample(
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

    question_results: List[Dict[str, Any]] = []
    error_types: List[str] = []
    for question in sample["questions"]:
        prompt = build_question_prompt(
            sample["prompt"],
            question["question"],
            question["expected_answer"],
            question["question_type"],
        )
        judge_result = judge_question(judge_client, judge_model, image, prompt)
        response_text = judge_result["raw_text"]
        response_json = judge_result["parsed_json"] or extract_json_object(response_text)
        predicted = str(response_json.get("answer", response_text)).strip()
        correct = answers_match(predicted, question["expected_answer"])
        question_results.append(
            {
                "question": question["question"],
                "expected_answer": question["expected_answer"],
                "predicted_answer": predicted,
                "question_type": question["question_type"],
                "correct": correct,
                "judge_raw": response_text,
                "judge_json": response_json,
                "judge_finish_reason": judge_result["finish_reason"],
                "judge_attempts": judge_result["attempts"],
            }
        )
        if not correct:
            error_types.append(map_question_type_to_error(question["question_type"]))

    question_accuracy = compute_question_accuracy(question_results)
    return {
        "benchmark": "tifa",
        "sample_id": sample["sample_id"],
        "prompt": sample["prompt"],
        "category": sample.get("category", ""),
        "score": question_accuracy,
        "question_accuracy": question_accuracy,
        "question_results": question_results,
        "error_types": sorted(set(error_types)),
        "judge_metadata": {
            "judge_model": judge_model,
            "question_count": len(question_results),
            "max_tokens": 64,
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
    image_path = Path(images_root) / "tifa" / variant / f"{sample['sample_id']}.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Missing generated image: {image_path}")
    result = score_tifa_sample(
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
        f"[tifa] variant={variant} total={len(manifest_rows)} pending={len(pending_samples)} "
        f"skipped={len(existing_ids)} workers={max_workers}"
    )
    results: List[Dict[str, Any]] = []
    start_time = time.perf_counter()
    if not pending_samples:
        print(f"[tifa] nothing to do for variant={variant}")
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
                print(f"[tifa] failed sample_id={sample['sample_id']} error={exc}")
                raise
            append_jsonl(output_path, result)
            results.append(result)
            if completed_idx == 1 or completed_idx % max(1, log_every) == 0 or completed_idx == len(pending_samples):
                elapsed = time.perf_counter() - start_time
                rate = completed_idx / elapsed if elapsed > 0 else 0.0
                print(
                    f"[tifa] completed={completed_idx}/{len(pending_samples)} "
                    f"last_sample={sample['sample_id']} rate={rate:.2f} samples/s elapsed={elapsed/60:.1f} min"
                )
    total_elapsed = time.perf_counter() - start_time
    print(
        f"[tifa] finished variant={variant} scored={len(results)} skipped={len(existing_ids)} "
        f"elapsed={total_elapsed/60:.1f} min"
    )
    return results
