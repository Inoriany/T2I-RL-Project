"""
GenEval-2 Evaluation (official-aligned, via SiliconFlow API)
=============================================================

Implements the official GenEval-2 / Soft-TIFA evaluation pipeline from
``evaluation.py`` + ``soft_tifa_analysis.py``:

  • For each prompt, run all (question, answer) pairs in ``vqa_list`` through
    a vision-language judge (Qwen3-VL on SiliconFlow).
  • Soft-TIFA AM: per-atom score = P(answer token | image, question).
                  per-prompt  = arithmetic mean of atom scores.
                  overall     = arithmetic mean of per-prompt scores.
  • Soft-TIFA GM: per-prompt  = geometric mean of atom scores, grouped by
                  ``atom_count`` (3..10).
  • Per-skill analysis over the five official skills:
        object / attribute / count / position / verb

Instead of deploying Qwen3-VL locally, this module calls SiliconFlow's
OpenAI-compatible chat completions endpoint. Soft probabilities are obtained
via the ``logprobs`` / ``top_logprobs`` response field; if the provider
doesn't return logprobs, it falls back to hard 0/1 matching (TIFA).

Environment:
  SILICONFLOW_API_KEY (or SILICON_API_KEY): required
  SILICONFLOW_BASE_URL (optional, default https://api.siliconflow.com/v1)
  GENEVAL2_API_MODEL (optional, default Qwen/Qwen3-VL-8B-Instruct)
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Answer-list helpers (mirrors official evaluation.py)
# ---------------------------------------------------------------------------

_NUMBER_WORD_TO_DIGIT = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


def _return_numeric_string(number: str) -> str:
    return _NUMBER_WORD_TO_DIGIT.get(number.lower(), "other")


def _build_answer_list(question: str, answer: str) -> List[str]:
    """Same token variants the official evaluation.py uses."""
    if question.startswith("How many"):
        digit = _return_numeric_string(answer)
        variants = [
            answer, answer.capitalize(), " " + answer, " " + answer.capitalize(),
            digit, " " + digit,
        ]
    else:
        variants = ["Yes", "yes", " yes", " Yes"]
    # de-dup while preserving order
    seen, out = set(), []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# ---------------------------------------------------------------------------
# SiliconFlow Qwen3-VL judge
# ---------------------------------------------------------------------------

class SiliconFlowVLMJudge:
    """
    OpenAI-compatible chat completions client against SiliconFlow.

    Uses ``logprobs=True, top_logprobs=K`` on a single-token generation to
    recover the softmax probability mass over acceptable answer tokens.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 4,
        timeout: int = 60,
        top_logprobs: int = 20,
        request_delay: float = 0.0,
    ):
        self.model_name = (
            model_name
            or os.environ.get("GENEVAL2_API_MODEL")
            or "Qwen/Qwen3-VL-8B-Instruct"
        )
        # Accept generic GENEVAL2_* first (e.g. pointing at a local vLLM),
        # then fall back to the SiliconFlow-specific names for compatibility.
        self.api_key = (
            api_key
            or os.environ.get("GENEVAL2_API_KEY")
            or os.environ.get("SILICONFLOW_API_KEY")
            or os.environ.get("SILICON_API_KEY")
            or "EMPTY"  # vLLM accepts any string; SiliconFlow will reject if missing
        )
        self.base_url = (
            base_url
            or os.environ.get("GENEVAL2_BASE_URL")
            or os.environ.get("SILICONFLOW_BASE_URL")
            or os.environ.get("SILICON_BASE_URL")
            or "https://api.siliconflow.com/v1"
        )
        self.max_retries = max_retries
        self.timeout = timeout
        self.top_logprobs = top_logprobs
        self.request_delay = request_delay
        self._client = None
        # Warn-once flag when provider doesn't return logprobs
        self._warned_no_logprobs = False
        # Sticky flag: once the server rejects logprobs / top_logprobs, stop
        # sending those fields for the rest of the run.
        self._logprobs_supported = True

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    @staticmethod
    def _image_to_data_url(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def query(
        self,
        image_data_url: str,
        question: str,
        answer_list: List[str],
    ) -> Tuple[str, Optional[float]]:
        """
        Return (predicted_token, P(token in answer_list) or None).

        If the server doesn't support ``logprobs`` for this model, prob is None
        and the caller should treat this as hard TIFA.
        """
        client = self._get_client()
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": question},
            ],
        }]

        def _call(with_logprobs: bool):
            kwargs = dict(
                model=self.model_name,
                messages=messages,
                max_tokens=1,
                temperature=0.0,
            )
            if with_logprobs:
                kwargs["logprobs"] = True
                kwargs["top_logprobs"] = self.top_logprobs
            return client.chat.completions.create(**kwargs)

        def _is_logprobs_rejection(err: Exception) -> bool:
            """Detect SiliconFlow 400 responses that reject (top_)logprobs."""
            msg = str(err).lower()
            if "logprobs" not in msg and "top_logprobs" not in msg:
                return False
            # OpenAI SDK BadRequestError has status_code 400; match on text too
            status = getattr(err, "status_code", None)
            return status == 400 or "400" in msg or "not allowed" in msg or "20015" in msg

        resp = None
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = _call(with_logprobs=self._logprobs_supported)
                break
            except TypeError:
                # Some old SDKs don't accept the logprobs kwargs
                self._logprobs_supported = False
                continue
            except Exception as e:
                if self._logprobs_supported and _is_logprobs_rejection(e):
                    # Permanent server-side rejection — disable and retry once.
                    self._logprobs_supported = False
                    if not self._warned_no_logprobs:
                        print(
                            "[GenEval-2] SiliconFlow rejected logprobs for "
                            f"'{self.model_name}' (error: {e}). Falling back "
                            "to hard TIFA (0/1 match) for the rest of this run."
                        )
                        self._warned_no_logprobs = True
                    continue  # retry immediately without logprobs
                last_err = e
                time.sleep(min(2 ** attempt, 8))
        if resp is None:
            raise RuntimeError(f"SiliconFlow call failed after retries: {last_err}")

        if self.request_delay > 0:
            time.sleep(self.request_delay)

        choice = resp.choices[0]
        pred_text = (choice.message.content or "").strip()

        # Try to pull logprobs
        prob: Optional[float] = None
        lp_container = getattr(choice, "logprobs", None)
        if lp_container is not None:
            content = getattr(lp_container, "content", None)
            if content:
                first = content[0]
                top = getattr(first, "top_logprobs", None) or []
                if top:
                    prob = 0.0
                    answer_set = set(answer_list)
                    for entry in top:
                        tok = getattr(entry, "token", None)
                        lp = getattr(entry, "logprob", None)
                        if tok is None or lp is None:
                            continue
                        if tok in answer_set:
                            prob += math.exp(lp)

        if prob is None and not self._warned_no_logprobs:
            print(
                "[GenEval-2] Warning: SiliconFlow did not return logprobs for "
                f"model '{self.model_name}'. Falling back to hard TIFA "
                "(0/1 match). Set GENEVAL2_API_MODEL to a model that supports "
                "logprobs, or accept hard-TIFA scoring."
            )
            self._warned_no_logprobs = True

        return pred_text, prob


# ---------------------------------------------------------------------------
# Per-prompt soft-TIFA scorer
# ---------------------------------------------------------------------------

def _score_prompt(
    judge: SiliconFlowVLMJudge,
    image: Image.Image,
    vqa_list: List[List[str]],
) -> List[float]:
    """
    Return per-atom score list (len == len(vqa_list)).

    Uses logprob-based soft score when available; otherwise a hard 1/0 match
    on the predicted token (TIFA fallback).
    """
    data_url = SiliconFlowVLMJudge._image_to_data_url(image)
    scores: List[float] = []
    for question, answer in vqa_list:
        answer_list = _build_answer_list(question, answer)
        q_prompt = f"{question} Answer in one word."
        pred, prob = judge.query(data_url, q_prompt, answer_list)
        if prob is not None:
            scores.append(float(prob))
        else:
            # Hard TIFA fallback
            scores.append(1.0 if pred in answer_list else 0.0)
    return scores


# ---------------------------------------------------------------------------
# Aggregation (mirrors soft_tifa_analysis.py)
# ---------------------------------------------------------------------------

_OFFICIAL_SKILLS = ("object", "attribute", "count", "position", "verb")


def _per_skill_am(
    all_score_lists: List[List[float]],
    all_skill_lists: List[List[str]],
) -> Dict[str, float]:
    totals = {s: 0.0 for s in _OFFICIAL_SKILLS}
    counts = {s: 0 for s in _OFFICIAL_SKILLS}
    for scores, skills in zip(all_score_lists, all_skill_lists):
        for sc, sk in zip(scores, skills):
            if sk in totals:
                totals[sk] += sc
                counts[sk] += 1
            # unknown skill labels are silently ignored; official would raise
    out = {}
    for s in _OFFICIAL_SKILLS:
        out[s] = (100.0 * totals[s] / counts[s]) if counts[s] else 0.0
    return out


def _gmean(xs: List[float]) -> float:
    if not xs:
        return 0.0
    # Guard zeros — use log-space with tiny epsilon to avoid -inf
    eps = 1e-12
    return math.exp(sum(math.log(max(x, eps)) for x in xs) / len(xs))


def _per_atomicity_gm(
    all_score_lists: List[List[float]],
    atom_counts: List[int],
) -> Dict[int, Dict[str, float]]:
    buckets: Dict[int, Dict[str, float]] = {k: {"correct": 0.0, "total": 0} for k in range(3, 11)}
    for scores, a in zip(all_score_lists, atom_counts):
        if a not in buckets:
            continue
        buckets[a]["correct"] += _gmean(scores)
        buckets[a]["total"] += 1
    for a in buckets:
        t = buckets[a]["total"]
        buckets[a]["accuracy"] = (100.0 * buckets[a]["correct"] / t) if t else 0.0
    return buckets


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class GenEval2Evaluator:
    """
    GenEval-2 evaluator using Qwen3-VL via SiliconFlow (official-aligned).

    Notes:
      • The positional kwargs ``owl_model``, ``clip_model``, ``clip_pretrained``,
        ``owl_threshold`` are accepted for backward compatibility with
        ``scripts/evaluate_benchmarks.py`` but are ignored.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        device: str = "cuda",  # unused
        owl_model: Optional[str] = None,        # ignored
        clip_model: Optional[str] = None,       # ignored
        clip_pretrained: Optional[str] = None,  # ignored
        owl_threshold: Optional[float] = None,  # ignored
        api_model: Optional[str] = None,
        num_workers: int = 4,
    ):
        self.data_dir = Path(data_dir) if data_dir else Path("data/geneval2")
        self.api_model = api_model
        self.num_workers = max(1, int(os.environ.get("GENEVAL2_NUM_WORKERS", num_workers)))
        self.judge: Optional[SiliconFlowVLMJudge] = None
        self._loaded = False

    def load_models(self) -> None:
        self.judge = SiliconFlowVLMJudge(model_name=self.api_model)
        print(
            f"[GenEval-2] VLM judge: {self.judge.model_name} "
            f"@ {self.judge.base_url}"
        )
        self._loaded = True

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self) -> List[Dict[str, Any]]:
        jsonl = self.data_dir / "geneval2_data.jsonl"
        jsn = self.data_dir / "geneval2_data.json"
        if jsonl.exists():
            items = []
            with open(jsonl) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
            return items
        if jsn.exists():
            with open(jsn) as f:
                return json.load(f)
        raise FileNotFoundError(
            f"GenEval-2 data not found at {jsonl} or {jsn}. "
            "Download it from https://github.com/facebookresearch/GenEval2"
        )

    # ------------------------------------------------------------------
    # Main evaluate loop
    # ------------------------------------------------------------------

    def evaluate(
        self,
        generator: Any,
        max_samples: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        assert self._loaded, "Call load_models() first"

        items = self.load_data()
        if max_samples:
            items = items[:max_samples]

        out_dir = Path(output_dir) if output_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        # ── Phase 1: batch generate all images (GPU-bound) ──────────
        GEN_BATCH = 32
        all_images: List[Optional[Image.Image]] = [None] * len(items)
        n_batches = math.ceil(len(items) / GEN_BATCH)
        for start in tqdm(range(0, len(items), GEN_BATCH),
                          desc="GenEval-2 [gen]", total=n_batches):
            batch_prompts = [it["prompt"] for it in items[start : start + GEN_BATCH]]
            try:
                imgs = generator.generate(batch_prompts)
                for j, img in enumerate(imgs):
                    all_images[start + j] = img
            except Exception as e:
                print(f"  [!] Generation failed at {start}: {e}")

        if out_dir:
            (out_dir / "images").mkdir(parents=True, exist_ok=True)
            for idx, img in enumerate(all_images):
                if img is not None:
                    img.save(out_dir / "images" / f"{idx:04d}.png")

        # ── Phase 2: score via SiliconFlow Qwen3-VL (I/O-bound) ─────
        all_score_lists: List[List[float]] = [[] for _ in items]

        def _worker(idx: int) -> Tuple[int, List[float]]:
            it = items[idx]
            img = all_images[idx]
            if img is None:
                return idx, [0.0] * len(it["vqa_list"])
            return idx, _score_prompt(self.judge, img, it["vqa_list"])

        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            futures = [ex.submit(_worker, i) for i in range(len(items))]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="GenEval-2 [score]"):
                try:
                    idx, scores = fut.result()
                    all_score_lists[idx] = scores
                except Exception as e:
                    print(f"  [!] Scoring failed: {e}")

        # ── Aggregation (official) ──────────────────────────────────
        all_skill_lists = [it["skills"] for it in items]
        atom_counts = [int(it.get("atom_count", 0)) for it in items]

        # Soft-TIFA AM overall
        per_prompt_am = [
            (sum(s) / len(s)) if s else 0.0 for s in all_score_lists
        ]
        overall_am = 100.0 * sum(per_prompt_am) / len(per_prompt_am) if per_prompt_am else 0.0

        # Per-skill AM (5 official skills)
        per_skill = _per_skill_am(all_score_lists, all_skill_lists)

        # Soft-TIFA GM per atomicity bucket
        atomicity = _per_atomicity_gm(all_score_lists, atom_counts)

        # ── Results ────────────────────────────────────────────────
        per_item = [
            {
                "prompt": items[i]["prompt"],
                "atom_count": items[i].get("atom_count"),
                "skills": items[i]["skills"],
                "vqa_list": items[i]["vqa_list"],
                "score_list": all_score_lists[i],
                "score_am": per_prompt_am[i],
                "score_gm": _gmean(all_score_lists[i]),
            }
            for i in range(len(items))
        ]

        summary = {
            "benchmark": "GenEval-2",
            "judge": {
                "provider": "openai-compatible",
                "model": self.judge.model_name,
                "base_url": self.judge.base_url,
            },
            "num_samples": len(items),
            # Official Soft-TIFA AM total (expressed on 0-100 scale).
            # Kept as 'overall_score' for backward-compat with evaluate_benchmarks.py.
            "overall_score": overall_am,
            # Official per-skill AM (5 skills, 0-100 scale).
            "per_skill_scores": per_skill,
            # Atomicity buckets for Soft-TIFA GM analysis.
            "atomicity_scores": atomicity,
            "results": per_item,
        }

        if out_dir:
            with open(out_dir / "geneval2_results.json", "w") as f:
                json.dump(summary, f, indent=2, default=_json_default)
            _print_report(summary, out_dir / "geneval2_report.txt")

        return summary


# ---------------------------------------------------------------------------
# Report / IO helpers
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def _print_report(summary: Dict, path: Path) -> None:
    lines = [
        "=" * 60,
        "GenEval-2 Evaluation Report (official-aligned)",
        "=" * 60,
        "",
        f"  Judge:         {summary['judge']['provider']} / {summary['judge']['model']}",
        f"  Num samples:   {summary['num_samples']}",
        f"  Overall (AM):  {summary['overall_score']:.2f}",
        "",
        "  Per-skill (Soft-TIFA AM):",
    ]
    for skill in _OFFICIAL_SKILLS:
        v = summary["per_skill_scores"].get(skill, 0.0)
        lines.append(f"    {skill:<12}: {v:.2f}")
    lines += ["", "  Per-atomicity (Soft-TIFA GM):"]
    for k in sorted(summary["atomicity_scores"].keys()):
        v = summary["atomicity_scores"][k]
        lines.append(
            f"    atom={k}: acc={v['accuracy']:.2f}  (n={v['total']})"
        )
    report = "\n".join(lines)
    print(report)
    with open(path, "w") as f:
        f.write(report)
