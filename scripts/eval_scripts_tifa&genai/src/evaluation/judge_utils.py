import json
import re
import time
from typing import Any, Dict, List


def is_qwen_model(model: str) -> bool:
    return "qwen" in (model or "").lower()


def build_chat_completion_kwargs(model: str, messages: List[Dict[str, Any]], max_tokens: int) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if is_qwen_model(model):
        kwargs["extra_body"] = {"enable_thinking": False}
    return kwargs


def _extract_text_from_part(part: Any) -> str:
    if isinstance(part, str):
        return part
    text_attr = getattr(part, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    content_attr = getattr(part, "content", None)
    if isinstance(content_attr, str):
        return content_attr
    if not isinstance(part, dict):
        return ""
    if isinstance(part.get("text"), str):
        return part["text"]
    if isinstance(part.get("content"), str):
        return part["content"]
    return ""


def extract_text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [_extract_text_from_part(part).strip() for part in content]
        return "\n".join(text for text in texts if text).strip()
    if isinstance(content, dict):
        return _extract_text_from_part(content).strip()
    return str(content).strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}

    candidates: List[str] = []
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)

    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidates.append(match.group(0))

    seen = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def serialize_choice(choice: Any) -> Dict[str, Any]:
    if hasattr(choice, "model_dump"):
        payload = choice.model_dump(mode="json")
        return payload if isinstance(payload, dict) else {"value": payload}
    if isinstance(choice, dict):
        return choice
    return {"value": str(choice)}


def request_json_chat_completion(
    client: Any,
    model: str,
    messages: List[Dict[str, Any]],
    *,
    max_tokens: int,
    max_attempts: int = 3,
    retry_delay_seconds: float = 0.0,
) -> Dict[str, Any]:
    last_result = {
        "raw_text": "",
        "parsed_json": {},
        "finish_reason": None,
        "choice_payload": {},
        "attempts": 0,
    }

    for attempt in range(1, max(1, max_attempts) + 1):
        response = client.chat.completions.create(
            **build_chat_completion_kwargs(model=model, messages=messages, max_tokens=max_tokens)
        )
        choice = response.choices[0]
        choice_payload = serialize_choice(choice)
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason is None and isinstance(choice_payload, dict):
            finish_reason = choice_payload.get("finish_reason")

        message = getattr(choice, "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if content is None and isinstance(choice_payload, dict):
            content = choice_payload.get("message", {}).get("content")

        raw_text = extract_text_content(content)
        parsed_json = extract_json_object(raw_text)
        last_result = {
            "raw_text": raw_text,
            "parsed_json": parsed_json,
            "finish_reason": finish_reason,
            "choice_payload": choice_payload,
            "attempts": attempt,
        }
        if raw_text.strip() and parsed_json:
            return last_result
        if attempt < max_attempts and retry_delay_seconds > 0:
            time.sleep(retry_delay_seconds)

    return last_result
