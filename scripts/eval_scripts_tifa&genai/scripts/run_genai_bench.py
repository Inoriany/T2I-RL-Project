#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.genai_bench_runner import build_openai_client, evaluate_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GenAI-Bench evaluation on generated images")
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--images_root", type=str, required=True)
    parser.add_argument("--variant", choices=["before", "after"], required=True)
    parser.add_argument("--judge_model", type=str, default=None)
    parser.add_argument("--api_provider", type=str, default="openai")
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENAI_COMPAT_API_KEY")
        or os.environ.get("SILICONFLOW_API_KEY")
    )
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    judge_model = args.judge_model or os.environ.get("JUDGE_MODEL") or "gpt-4.1-mini"
    client = build_openai_client(api_key=api_key, base_url=base_url)
    print(
        f"[genai] manifest={args.manifest_path} variant={args.variant} output={args.output_path} "
        f"judge_model={judge_model} base_url={base_url or 'default'} workers={args.max_workers}"
    )
    evaluate_manifest(
        manifest_path=Path(args.manifest_path),
        images_root=Path(args.images_root),
        variant=args.variant,
        judge_client=client,
        judge_model=judge_model,
        output_path=Path(args.output_path),
        resume=args.resume,
        max_workers=args.max_workers,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
