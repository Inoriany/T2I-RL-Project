"""
T2I-CompBench Non-MLLM Evaluation
====================================

Reference: https://github.com/Karine-Huang/T2I-CompBench
Paper: NeurIPS 2023 "T2I-CompBench: A Comprehensive Benchmark for
       Open-world Compositional Text-to-image Generation"

Non-MLLM evaluation methods implemented:
  1. Disentangled BLIP-VQA  — color / shape / texture / non-spatial
  2. OwlViT spatial metric   — spatial relationships (left/right/above/below)
  3. CLIPScore               — complex compositions & general alignment

Usage::

    from src.evaluation.t2i_compbench_eval import T2ICompBenchEvaluator
    evaluator = T2ICompBenchEvaluator(data_dir="data/t2i_compbench")
    evaluator.load_models()
    results = evaluator.evaluate(generator)
"""

from __future__ import annotations

import json
import re
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helper: colour/shape/texture vocabulary for BLIP question generation
# ---------------------------------------------------------------------------

COLORS = [
    "red", "orange", "yellow", "green", "blue", "purple", "pink",
    "brown", "black", "white", "gray", "grey", "cyan", "magenta",
    "gold", "silver", "beige", "navy", "teal", "maroon", "violet",
]

SHAPES = [
    "round", "circular", "square", "rectangular", "triangular",
    "oval", "elliptical", "hexagonal", "star-shaped", "heart-shaped",
    "cylindrical", "spherical", "cubic", "pyramidal",
]

TEXTURES = [
    "fluffy", "smooth", "rough", "bumpy", "shiny", "matte", "glossy",
    "furry", "hairy", "silky", "wooden", "metallic", "plastic",
    "woven", "knitted", "striped", "spotted", "checkered", "fuzzy",
    "soft", "hard", "rocky", "sandy",
]

SPATIAL_RELATIONS = {
    "left":   ("left", lambda xA, yA, xB, yB: xA < xB and abs(xA - xB) > abs(yA - yB)),
    "right":  ("right", lambda xA, yA, xB, yB: xA > xB and abs(xA - xB) > abs(yA - yB)),
    "above":  ("above", lambda xA, yA, xB, yB: yA < yB and abs(yA - yB) > abs(xA - xB)),
    "below":  ("below", lambda xA, yA, xB, yB: yA > yB and abs(yA - yB) > abs(xA - xB)),
    "top":    ("above", lambda xA, yA, xB, yB: yA < yB and abs(yA - yB) > abs(xA - xB)),
    "bottom": ("below", lambda xA, yA, xB, yB: yA > yB and abs(yA - yB) > abs(xA - xB)),
    "on":     ("on", lambda xA, yA, xB, yB: yA < yB),
    "under":  ("under", lambda xA, yA, xB, yB: yA > yB),
    "behind": ("behind", lambda xA, yA, xB, yB: True),  # depth—approx by size
    "front":  ("front",  lambda xA, yA, xB, yB: True),
    "next to":   ("near", lambda xA, yA, xB, yB: True),
    "beside":    ("near", lambda xA, yA, xB, yB: True),
    "adjacent":  ("near", lambda xA, yA, xB, yB: True),
}


# ---------------------------------------------------------------------------
# BLIP-VQA evaluator
# ---------------------------------------------------------------------------

class BLIPVQAEvaluator:
    """
    Disentangled BLIP-VQA for attribute binding evaluation.

    For a prompt like "a red apple and a green banana", we ask:
      • "Is there a red apple in the image?" → expected "yes"
      • "Is there a green banana in the image?" → expected "yes"

    The score for a sample is the average of the yes-probabilities.
    """

    def __init__(self, model_name: str = "Salesforce/blip-vqa-base", device: str = "cuda"):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        from transformers import BlipProcessor, BlipForQuestionAnswering
        print(f"[BLIP-VQA] Loading {self.model_name}...")
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(
            self.model_name, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()
        print("[BLIP-VQA] Model loaded.")

    @torch.no_grad()
    def answer_yes_no(self, image: Image.Image, question: str) -> float:
        """Return probability of 'yes' answer in [0, 1]."""
        if self.model is None:
            raise RuntimeError("Call load_model() first")
        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
        # Inference must use generate(), not forward(): BlipForQuestionAnswering.forward
        # requires decoder_input_ids or labels (training), which we do not pass here.
        with torch.autocast(
            device_type=self.device if self.device != "cpu" else "cpu",
            enabled=(self.device == "cuda"),
        ):
            generated = self.model.generate(
                **inputs, max_new_tokens=16, return_dict_in_generate=True
            )
        seq = generated.sequences[0]
        answer = self.processor.tokenizer.decode(seq, skip_special_tokens=True).strip().lower()
        if answer.startswith("yes"):
            return 1.0
        elif answer.startswith("no"):
            return 0.0
        # Soft score: check if "yes" appears in answer
        return 0.5 if "yes" in answer else 0.0

    @torch.no_grad()
    def answer_yes_no_batch(
        self, images: List[Image.Image], questions: List[str], batch_size: int = 32,
    ) -> List[float]:
        """Batched version of ``answer_yes_no`` — processes many (image, question) pairs."""
        if self.model is None:
            raise RuntimeError("Call load_model() first")
        all_results: List[float] = []
        for start in range(0, len(images), batch_size):
            batch_imgs = images[start : start + batch_size]
            batch_qs = questions[start : start + batch_size]
            inputs = self.processor(
                images=batch_imgs, text=batch_qs,
                return_tensors="pt", padding=True,
            ).to(self.device)
            with torch.autocast(
                device_type=self.device if self.device != "cpu" else "cpu",
                enabled=(self.device == "cuda"),
            ):
                generated = self.model.generate(
                    **inputs, max_new_tokens=16, return_dict_in_generate=True,
                )
            for seq in generated.sequences:
                answer = self.processor.tokenizer.decode(
                    seq, skip_special_tokens=True
                ).strip().lower()
                if answer.startswith("yes"):
                    all_results.append(1.0)
                elif answer.startswith("no"):
                    all_results.append(0.0)
                else:
                    all_results.append(0.5 if "yes" in answer else 0.0)
        return all_results

    @torch.no_grad()
    def score_attribute(
        self, image: Image.Image, attribute: str, obj: str
    ) -> float:
        """Score whether image contains an object with a given attribute."""
        question = f"Is there a {attribute} {obj} in the image?"
        return self.answer_yes_no(image, question)


# ---------------------------------------------------------------------------
# OwlViT object detector
# ---------------------------------------------------------------------------

class OwlViTDetector:
    """
    Open-vocabulary object detector using OWL-ViT / OWL-v2.

    Used for:
    1. Verifying object presence
    2. Spatial relationship evaluation (via bounding-box positions)
    """

    def __init__(
        self,
        model_name: str = "google/owlv2-base-patch16-ensemble",
        device: str = "cuda",
        score_threshold: float = 0.10,
    ):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.score_threshold = score_threshold
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        print(f"[OwlViT] Loading {self.model_name}...")
        self.processor = Owlv2Processor.from_pretrained(self.model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(self.model_name).to(self.device).eval()
        print("[OwlViT] Model loaded.")

    @torch.no_grad()
    def detect(
        self, image: Image.Image, text_queries: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in image matching text_queries.

        Returns:
            List of dicts with keys: label, score, box (xyxy normalised)
        """
        if self.model is None:
            raise RuntimeError("Call load_model() first")

        inputs = self.processor(
            text=[text_queries], images=image, return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)
        target_sizes = torch.tensor([[image.height, image.width]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.score_threshold,
        )[0]

        detections = []
        W, H = image.width, image.height
        for score, label_idx, box in zip(
            results["scores"].tolist(),
            results["labels"].tolist(),
            results["boxes"].tolist(),
        ):
            x0, y0, x1, y1 = box
            cx = (x0 + x1) / 2 / W
            cy = (y0 + y1) / 2 / H
            detections.append({
                "label": text_queries[label_idx],
                "score": score,
                "box": [x0 / W, y0 / H, x1 / W, y1 / H],
                "cx": cx,
                "cy": cy,
            })
        return detections

    def best_detection(
        self, image: Image.Image, obj: str
    ) -> Optional[Dict[str, Any]]:
        """Return the highest-confidence detection for ``obj``, or None."""
        dets = self.detect(image, [obj])
        if not dets:
            return None
        return max(dets, key=lambda d: d["score"])

    def check_spatial(
        self, image: Image.Image, objA: str, objB: str, relation: str
    ) -> float:
        """
        Score spatial relationship between objA and objB.
        Returns 1.0 / 0.5 / 0.0.
        """
        detA = self.best_detection(image, objA)
        detB = self.best_detection(image, objB)

        if detA is None or detB is None:
            return 0.0  # Can't evaluate if objects not detected

        xA, yA = detA["cx"], detA["cy"]
        xB, yB = detB["cx"], detB["cy"]

        rel_lower = relation.lower().strip()
        for key, (_, fn) in SPATIAL_RELATIONS.items():
            if key in rel_lower:
                return 1.0 if fn(xA, yA, xB, yB) else 0.0

        # Unknown relation: partial credit if both objects present
        return 0.5


# ---------------------------------------------------------------------------
# CLIP evaluator
# ---------------------------------------------------------------------------

class CLIPEvaluator:
    """CLIP-based text-image alignment score."""

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.tokenizer = None

    def load_model(self) -> None:
        import open_clip
        print(f"[CLIP] Loading {self.model_name}/{self.pretrained}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.eval()
        print("[CLIP] Model loaded.")

    @torch.no_grad()
    def score(self, image: Image.Image, text: str) -> float:
        img_t = self.preprocess(image).unsqueeze(0).to(self.device)
        txt_t = self.tokenizer([text]).to(self.device)
        img_f = self.model.encode_image(img_t)
        txt_f = self.model.encode_text(txt_t)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
        return (img_f * txt_f).sum().item()

    @torch.no_grad()
    def score_batch(
        self, images: List[Image.Image], texts: List[str], batch_size: int = 64,
    ) -> List[float]:
        """Batched CLIP scoring — processes many (image, text) pairs at once."""
        all_scores: List[float] = []
        for start in range(0, len(images), batch_size):
            batch_imgs = images[start : start + batch_size]
            batch_txts = texts[start : start + batch_size]
            img_t = torch.stack(
                [self.preprocess(img) for img in batch_imgs]
            ).to(self.device)
            txt_t = self.tokenizer(batch_txts).to(self.device)
            img_f = self.model.encode_image(img_t)
            txt_f = self.model.encode_text(txt_t)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
            sims = (img_f * txt_f).sum(dim=-1)
            all_scores.extend(sims.tolist())
        return all_scores


# ---------------------------------------------------------------------------
# Prompt parsing helpers
# ---------------------------------------------------------------------------

def extract_color_pairs(prompt: str) -> List[Tuple[str, str]]:
    """
    Extract (color, object) pairs from a prompt.
    e.g. "a red apple and a green banana" → [("red","apple"), ("green","banana")]
    """
    pairs = []
    for color in COLORS:
        pattern = rf'\b{re.escape(color)}\s+(\w+)'
        for m in re.finditer(pattern, prompt, re.IGNORECASE):
            obj = m.group(1).lower()
            if obj not in ("and", "or", "a", "an", "the"):
                pairs.append((color, obj))
    return pairs


def extract_shape_pairs(prompt: str) -> List[Tuple[str, str]]:
    pairs = []
    for shape in SHAPES:
        pattern = rf'\b{re.escape(shape)}\s+(\w+)'
        for m in re.finditer(pattern, prompt, re.IGNORECASE):
            obj = m.group(1).lower()
            if obj not in ("and", "or", "a", "an", "the"):
                pairs.append((shape, obj))
    return pairs


def extract_texture_pairs(prompt: str) -> List[Tuple[str, str]]:
    pairs = []
    for tex in TEXTURES:
        pattern = rf'\b{re.escape(tex)}\s+(\w+)'
        for m in re.finditer(pattern, prompt, re.IGNORECASE):
            obj = m.group(1).lower()
            if obj not in ("and", "or", "a", "an", "the"):
                pairs.append((tex, obj))
    return pairs


def extract_spatial_relation(prompt: str) -> Optional[Tuple[str, str, str]]:
    """
    Try to parse (objectA, relation, objectB) from a spatial prompt.
    Returns None if not parseable.
    """
    for rel in sorted(SPATIAL_RELATIONS.keys(), key=len, reverse=True):
        # Match: <objA> [... ] <rel> [... ] <objB>
        pattern = rf'(\w[\w\s]*?)\s+(?:is\s+)?(?:to the\s+)?{re.escape(rel)}\s+(?:of\s+)?(?:the\s+)?(\w[\w\s]*?)(?:\s*$|[,.])'
        m = re.search(pattern, prompt, re.IGNORECASE)
        if m:
            return m.group(1).strip(), rel, m.group(2).strip()
    return None


# ---------------------------------------------------------------------------
# Main T2I-CompBench Evaluator
# ---------------------------------------------------------------------------

class T2ICompBenchEvaluator:
    """
    Full T2I-CompBench evaluation (non-MLLM methods only).

    Methods per category:
    ┌──────────────┬────────────────────────────────────────────────┐
    │ Category     │ Evaluation Method                              │
    ├──────────────┼────────────────────────────────────────────────┤
    │ color        │ Disentangled BLIP-VQA                          │
    │ shape        │ Disentangled BLIP-VQA                          │
    │ texture      │ Disentangled BLIP-VQA                          │
    │ non_spatial  │ BLIP-VQA (relation question)                   │
    │ spatial      │ OwlViT bounding-box position analysis          │
    │ complex      │ CLIPScore                                      │
    └──────────────┴────────────────────────────────────────────────┘
    """

    CATEGORIES = ["color", "shape", "texture", "spatial", "non_spatial", "complex"]

    def __init__(
        self,
        data_dir: Optional[str] = None,
        device: str = "cuda",
        blip_model: str = "Salesforce/blip-vqa-base",
        owl_model: str = "google/owlv2-base-patch16-ensemble",
        clip_model: str = "ViT-L-14",
        clip_pretrained: str = "openai",
    ):
        self.data_dir = Path(data_dir) if data_dir else Path("data/t2i_compbench")
        self.device = device if torch.cuda.is_available() else "cpu"

        self.blip = BLIPVQAEvaluator(model_name=blip_model, device=self.device)
        self.owl = OwlViTDetector(model_name=owl_model, device=self.device)
        self.clip = CLIPEvaluator(model_name=clip_model, pretrained=clip_pretrained, device=self.device)

        self._models_loaded = False

    def load_models(self) -> None:
        self.blip.load_model()
        self.owl.load_model()
        self.clip.load_model()
        self._models_loaded = True

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_prompts(self, category: str) -> List[Dict[str, Any]]:
        """
        Load prompts for a category.

        Tries (in order):
          1. data_dir/<category>_val.json  (our preprocessed format)
          2. data_dir/<category>_val.txt   (official plain-text format)
          3. Built-in fallback examples
        """
        json_path = self.data_dir / f"{category}_val.json"
        txt_path  = self.data_dir / f"{category}_val.txt"

        if json_path.exists():
            with open(json_path) as f:
                return json.load(f)

        if txt_path.exists():
            prompts = []
            with open(txt_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        prompts.append({"prompt": line})
            return prompts

        # Fallback: built-in examples
        return [{"prompt": p} for p in _FALLBACK_PROMPTS.get(category, [])]

    # ------------------------------------------------------------------
    # Per-category scoring
    # ------------------------------------------------------------------

    def score_image_color(self, image: Image.Image, item: Dict) -> float:
        prompt = item["prompt"]
        # Try metadata fields first
        if "attribute_words" in item and "object_words" in item:
            attrs = item["attribute_words"] if isinstance(item["attribute_words"], list) else [item["attribute_words"]]
            objs  = item["object_words"]  if isinstance(item["object_words"],  list) else [item["object_words"]]
            pairs = list(zip(attrs, objs))
        else:
            pairs = extract_color_pairs(prompt)

        if not pairs:
            return self.clip.score(image, prompt)

        scores = [self.blip.score_attribute(image, attr, obj) for attr, obj in pairs]
        return sum(scores) / len(scores)

    def score_image_shape(self, image: Image.Image, item: Dict) -> float:
        prompt = item["prompt"]
        if "attribute_words" in item and "object_words" in item:
            attrs = item["attribute_words"] if isinstance(item["attribute_words"], list) else [item["attribute_words"]]
            objs  = item["object_words"]  if isinstance(item["object_words"],  list) else [item["object_words"]]
            pairs = list(zip(attrs, objs))
        else:
            pairs = extract_shape_pairs(prompt)

        if not pairs:
            return self.clip.score(image, prompt)
        scores = [self.blip.score_attribute(image, attr, obj) for attr, obj in pairs]
        return sum(scores) / len(scores)

    def score_image_texture(self, image: Image.Image, item: Dict) -> float:
        prompt = item["prompt"]
        if "attribute_words" in item and "object_words" in item:
            attrs = item["attribute_words"] if isinstance(item["attribute_words"], list) else [item["attribute_words"]]
            objs  = item["object_words"]  if isinstance(item["object_words"],  list) else [item["object_words"]]
            pairs = list(zip(attrs, objs))
        else:
            pairs = extract_texture_pairs(prompt)

        if not pairs:
            return self.clip.score(image, prompt)
        scores = [self.blip.score_attribute(image, attr, obj) for attr, obj in pairs]
        return sum(scores) / len(scores)

    def score_image_spatial(self, image: Image.Image, item: Dict) -> float:
        prompt = item["prompt"]
        # Use metadata if available (official format)
        if "object_words" in item and "relation_words" in item:
            objs = item["object_words"]
            if isinstance(objs, list) and len(objs) >= 2:
                objA, objB = objs[0], objs[1]
                relation = item["relation_words"] if isinstance(item["relation_words"], str) else "to the left of"
                return self.owl.check_spatial(image, objA, objB, relation)

        # Fallback: parse from prompt text
        parsed = extract_spatial_relation(prompt)
        if parsed:
            objA, relation, objB = parsed
            return self.owl.check_spatial(image, objA, objB, relation)

        # Last resort: CLIP score
        return self.clip.score(image, prompt)

    def score_image_non_spatial(self, image: Image.Image, item: Dict) -> float:
        prompt = item["prompt"]
        # Ask a binary question from the full prompt description
        q = f"Does the image show: {prompt}? Answer yes or no."
        return self.blip.answer_yes_no(image, q)

    def score_image_complex(self, image: Image.Image, item: Dict) -> float:
        # 3-in-1: CLIPScore + BLIP-VQA + OwlViT object presence
        prompt = item["prompt"]
        clip_s = self.clip.score(image, prompt)

        # BLIP-VQA binary
        q = f"Does this image match the description: {prompt}?"
        blip_s = self.blip.answer_yes_no(image, q)

        # OwlViT: detect major nouns from prompt
        nouns = _extract_nouns(prompt)
        if nouns:
            det_scores = []
            for noun in nouns[:4]:  # limit to avoid slow queries
                det = self.owl.best_detection(image, noun)
                det_scores.append(1.0 if det else 0.0)
            owl_s = sum(det_scores) / len(det_scores)
        else:
            owl_s = clip_s

        return (clip_s + blip_s + owl_s) / 3.0

    # ------------------------------------------------------------------
    # Main evaluation loop
    # ------------------------------------------------------------------

    def evaluate_category(
        self,
        generator: Any,
        category: str,
        max_samples: Optional[int] = None,
        save_images_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Evaluate a single category and return per-sample results + aggregate.

        Two-phase pipeline:
          Phase 1 — batch-generate all images (high GPU utilization).
          Phase 2 — batch-score with BLIP / OwlViT / CLIP (keeps GPU busy).
        """
        assert self._models_loaded, "Call load_models() first"

        items = self.load_prompts(category)
        if max_samples:
            items = items[:max_samples]

        # ── Phase 1: batch generate ──────────────────────────────────
        GEN_BATCH = 32
        all_images: List[Optional[Image.Image]] = [None] * len(items)
        num_gen = math.ceil(len(items) / GEN_BATCH)
        for start in tqdm(
            range(0, len(items), GEN_BATCH),
            desc=f"Gen [{category}]", total=num_gen,
        ):
            batch_items = items[start : start + GEN_BATCH]
            batch_prompts = [it["prompt"] for it in batch_items]
            try:
                imgs = generator.generate(batch_prompts)
                for j, img in enumerate(imgs):
                    all_images[start + j] = img
            except Exception as e:
                print(f"  [!] Generation failed at {start}: {e}")

        if save_images_dir:
            save_images_dir.mkdir(parents=True, exist_ok=True)
            for idx, img in enumerate(all_images):
                if img is not None:
                    img.save(save_images_dir / f"{category}_{idx:04d}.png")

        # ── Phase 2: batch score ─────────────────────────────────────
        scores = self._batch_score(category, all_images, items)

        results = []
        for item, s in zip(items, scores):
            results.append({"prompt": item["prompt"], "score": s})

        valid = [s for s in scores if s is not None]
        mean_score = sum(valid) / len(valid) if valid else 0.0

        return {
            "category": category,
            "num_samples": len(results),
            "mean_score": mean_score,
            "results": results,
        }

    # ------------------------------------------------------------------
    # Batch scoring helpers
    # ------------------------------------------------------------------

    def _batch_score(
        self, category: str,
        images: List[Optional[Image.Image]], items: List[Dict],
    ) -> List[float]:
        if category in ("color", "shape", "texture"):
            return self._batch_score_attribute(category, images, items)
        if category == "non_spatial":
            return self._batch_score_non_spatial(images, items)
        if category == "spatial":
            return self._batch_score_spatial(images, items)
        if category == "complex":
            return self._batch_score_complex(images, items)
        return [0.0] * len(items)

    def _batch_score_attribute(
        self, category: str,
        images: List[Optional[Image.Image]], items: List[Dict],
    ) -> List[float]:
        EXTRACT = {
            "color": extract_color_pairs,
            "shape": extract_shape_pairs,
            "texture": extract_texture_pairs,
        }
        extract_fn = EXTRACT[category]

        task_imgs: List[Image.Image] = []
        task_qs: List[str] = []
        task_idx: List[int] = []
        clip_fallback: List[int] = []

        for idx, (img, item) in enumerate(zip(images, items)):
            if img is None:
                continue
            if "attribute_words" in item and "object_words" in item:
                attrs = item["attribute_words"] if isinstance(item["attribute_words"], list) else [item["attribute_words"]]
                objs = item["object_words"] if isinstance(item["object_words"], list) else [item["object_words"]]
                pairs = list(zip(attrs, objs))
            else:
                pairs = extract_fn(item["prompt"])
            if not pairs:
                clip_fallback.append(idx)
                continue
            for attr, obj in pairs:
                task_imgs.append(img)
                task_qs.append(f"Is there a {attr} {obj} in the image?")
                task_idx.append(idx)

        blip_results = self.blip.answer_yes_no_batch(task_imgs, task_qs) if task_imgs else []

        scores = [0.0] * len(items)
        counts = [0] * len(items)
        for s, i in zip(blip_results, task_idx):
            scores[i] += s
            counts[i] += 1
        for i in range(len(items)):
            if counts[i] > 0:
                scores[i] /= counts[i]

        for i in clip_fallback:
            scores[i] = self.clip.score(images[i], items[i]["prompt"])
        return scores

    def _batch_score_non_spatial(
        self,
        images: List[Optional[Image.Image]], items: List[Dict],
    ) -> List[float]:
        task_imgs: List[Image.Image] = []
        task_qs: List[str] = []
        task_idx: List[int] = []
        for idx, (img, item) in enumerate(zip(images, items)):
            if img is None:
                continue
            task_imgs.append(img)
            task_qs.append(f"Does the image show: {item['prompt']}? Answer yes or no.")
            task_idx.append(idx)

        blip_results = self.blip.answer_yes_no_batch(task_imgs, task_qs) if task_imgs else []

        scores = [0.0] * len(items)
        for s, i in zip(blip_results, task_idx):
            scores[i] = s
        return scores

    def _batch_score_spatial(
        self,
        images: List[Optional[Image.Image]], items: List[Dict],
    ) -> List[float]:
        scores = [0.0] * len(items)
        for idx, (img, item) in enumerate(zip(images, items)):
            if img is None:
                continue
            scores[idx] = self.score_image_spatial(img, item)
        return scores

    def _batch_score_complex(
        self,
        images: List[Optional[Image.Image]], items: List[Dict],
    ) -> List[float]:
        valid = [(i, img, item) for i, (img, item) in enumerate(zip(images, items)) if img is not None]
        if not valid:
            return [0.0] * len(items)

        v_imgs = [img for _, img, _ in valid]
        v_prompts = [item["prompt"] for _, _, item in valid]

        clip_scores = self.clip.score_batch(v_imgs, v_prompts)

        blip_qs = [f"Does this image match the description: {p}?" for p in v_prompts]
        blip_scores = self.blip.answer_yes_no_batch(v_imgs, blip_qs)

        owl_scores: List[float] = []
        for j, (_, img, item) in enumerate(valid):
            nouns = _extract_nouns(item["prompt"])
            if nouns:
                det_s = []
                for noun in nouns[:4]:
                    det = self.owl.best_detection(img, noun)
                    det_s.append(1.0 if det else 0.0)
                owl_scores.append(sum(det_s) / len(det_s))
            else:
                owl_scores.append(clip_scores[j])

        scores = [0.0] * len(items)
        for j, (i, _, _) in enumerate(valid):
            scores[i] = (clip_scores[j] + blip_scores[j] + owl_scores[j]) / 3.0
        return scores

    def evaluate(
        self,
        generator: Any,
        categories: Optional[List[str]] = None,
        max_samples_per_category: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run full T2I-CompBench evaluation.

        Returns aggregated results across all categories.
        """
        assert self._models_loaded, "Call load_models() first"
        categories = categories or self.CATEGORIES
        out_dir = Path(output_dir) if output_dir else None

        all_results: Dict[str, Any] = {}

        for cat in categories:
            print(f"\n{'='*50}")
            print(f"  T2I-CompBench — {cat}")
            print(f"{'='*50}")
            img_dir = (out_dir / "images" / cat) if out_dir else None
            cat_result = self.evaluate_category(
                generator, cat, max_samples_per_category, img_dir
            )
            all_results[cat] = cat_result
            print(f"  → {cat} mean score: {cat_result['mean_score']:.4f}")

        # Aggregate
        category_means = {cat: r["mean_score"] for cat, r in all_results.items()}
        overall = sum(category_means.values()) / len(category_means) if category_means else 0.0

        summary = {
            "benchmark": "T2I-CompBench",
            "categories": all_results,
            "category_means": category_means,
            "overall_mean": overall,
        }

        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "t2i_compbench_results.json", "w") as f:
                json.dump(summary, f, indent=2, default=_json_default)
            _print_report(summary, out_dir / "t2i_compbench_report.txt")

        return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_nouns(text: str) -> List[str]:
    """Simple noun extractor: strips common stop words."""
    stop = {
        "a", "an", "the", "is", "are", "in", "on", "at", "to", "of",
        "and", "or", "with", "next", "front", "back", "left", "right",
        "above", "below", "behind", "under", "near", "beside", "large",
        "small", "big", "tiny", "long", "short", "tall",
    }
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return [w for w in words if w not in stop and len(w) > 2]


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def _print_report(summary: Dict, path: Path) -> None:
    lines = ["=" * 60, "T2I-CompBench Evaluation Report", "=" * 60, ""]
    for cat, mean in summary["category_means"].items():
        lines.append(f"  {cat:<16}: {mean:.4f}")
    lines += ["", f"  {'OVERALL':<16}: {summary['overall_mean']:.4f}", ""]
    report = "\n".join(lines)
    print(report)
    with open(path, "w") as f:
        f.write(report)


# ---------------------------------------------------------------------------
# Fallback prompts (used when data files are not found)
# ---------------------------------------------------------------------------

_FALLBACK_PROMPTS: Dict[str, List[str]] = {
    "color": [
        "a red apple and a green banana",
        "a blue car next to a yellow house",
        "a purple flower in a white vase",
        "a black cat with orange eyes",
        "a pink dress on a brown chair",
        "a gray elephant next to a red fire hydrant",
        "a golden retriever on a green lawn",
        "a white rabbit next to a black cat",
        "a silver fork next to a golden spoon",
        "a teal mug on an orange tray",
    ],
    "shape": [
        "a triangular pizza on a round plate",
        "a square window in a rectangular building",
        "a circular mirror above an oval table",
        "a cylindrical vase next to a cubic box",
        "a heart-shaped balloon over a star-shaped cake",
        "a spherical ball on a flat surface",
        "a hexagonal tile floor with a round rug",
        "a rectangular phone on a round table",
        "a triangular mountain behind a circular lake",
        "an oval face with round glasses",
    ],
    "texture": [
        "a fluffy cat on a smooth marble floor",
        "a rough wooden table with a glossy apple",
        "a shiny metal robot on a fuzzy carpet",
        "a matte ceramic bowl with bumpy oranges",
        "a silky dress draped over a woven basket",
        "a hairy spider on a smooth glass surface",
        "a knitted sweater on a wooden chair",
        "a sandy beach next to rocky cliffs",
        "a striped umbrella on a checkered blanket",
        "a soft pillow on a hard wooden bench",
    ],
    "spatial": [
        "a cat to the left of a dog",
        "a book on top of a table",
        "a bird flying above the clouds",
        "a car parked behind the house",
        "a lamp to the left of the computer",
        "a cup above a plate",
        "a child in front of a school",
        "a bicycle next to a tree",
        "a mountain behind a lake",
        "a dog below a tree branch",
    ],
    "non_spatial": [
        "a dog wearing a hat",
        "a robot holding a flower",
        "a child riding a bicycle",
        "an artist painting a landscape",
        "a chef cooking in a kitchen",
        "a cat sitting in a basket",
        "a bird carrying a worm",
        "a person reading a book",
        "a dog chasing a ball",
        "a horse pulling a cart",
    ],
    "complex": [
        "a red apple and a green banana on a blue plate next to a yellow cup",
        "a small white cat sitting on a large brown dog in front of a red house",
        "two birds flying above three trees beside a lake",
        "a chef in a white hat cooking pasta while a waiter serves wine",
        "a vintage red car parked under a green tree next to a blue bench",
        "a black cat on a white table next to a red flower vase",
        "three yellow ducks swimming in a green pond under a blue sky",
        "a wooden boat with white sails on a blue ocean next to a green island",
        "a small brown rabbit eating orange carrots next to purple flowers",
        "a silver robot holding a red rose standing on a gray pavement",
    ],
}


# ---------------------------------------------------------------------------
# Data downloader
# ---------------------------------------------------------------------------

def download_t2i_compbench_data(save_dir: str = "data/t2i_compbench") -> None:
    """
    Download T2I-CompBench evaluation prompts.

    Strategy:
    1. Try Hugging Face datasets (limingcv/T2I-CompBench)
    2. Fallback: write built-in prompt lists
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"[T2I-CompBench] Downloading data to {save_path}...")

    categories = ["color", "shape", "texture", "spatial", "non_spatial", "complex"]

    try:
        from datasets import load_dataset
        print("  Trying Hugging Face dataset (limingcv/T2I-CompBench)...")
        ds = load_dataset("limingcv/T2I-CompBench", trust_remote_code=True)

        for cat in categories:
            split_key = f"{cat}_val"
            if split_key not in ds:
                print(f"  [!] Split '{split_key}' not found, skipping.")
                continue
            split = ds[split_key]
            items = []
            for row in split:
                item = {"prompt": row["prompt"]}
                # Preserve metadata columns if present
                for col in ["attribute_words", "object_words", "relation_words"]:
                    if col in row:
                        item[col] = row[col]
                items.append(item)

            out_file = save_path / f"{cat}_val.json"
            with open(out_file, "w") as f:
                json.dump(items, f, indent=2)
            print(f"  Saved {len(items)} prompts → {out_file}")

        print("[T2I-CompBench] Download complete.")
        return

    except Exception as e:
        print(f"  HuggingFace download failed: {e}")

    # Fallback: write built-in examples
    print("  Writing built-in fallback prompts...")
    for cat, prompts in _FALLBACK_PROMPTS.items():
        out_file = save_path / f"{cat}_val.json"
        items = [{"prompt": p} for p in prompts]
        with open(out_file, "w") as f:
            json.dump(items, f, indent=2)
        print(f"  Wrote {len(items)} fallback prompts → {out_file}")

    print("[T2I-CompBench] Fallback data ready.")
