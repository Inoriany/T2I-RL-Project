import csv
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from src.evaluation.io import write_json


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compute_variant_delta(before_rows: List[Dict[str, Any]], after_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    before_scores = [float(row["score"]) for row in before_rows if "score" in row]
    after_scores = [float(row["score"]) for row in after_rows if "score" in row]
    before_mean = _mean(before_scores)
    after_mean = _mean(after_scores)
    return {
        "before_mean": round(before_mean, 6),
        "after_mean": round(after_mean, 6),
        "delta": round(after_mean - before_mean, 6),
    }


def count_error_types(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(row.get("error_types", []) or [])
    return dict(counter)


def summarize_benchmark_rows(before_rows: List[Dict[str, Any]], after_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = compute_variant_delta(before_rows, after_rows)
    summary["before_count"] = len(before_rows)
    summary["after_count"] = len(after_rows)
    summary["before_errors"] = count_error_types(before_rows)
    summary["after_errors"] = count_error_types(after_rows)
    return summary


def build_summary_report(
    tifa_before: List[Dict[str, Any]],
    tifa_after: List[Dict[str, Any]],
    genai_before: List[Dict[str, Any]],
    genai_after: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "tifa": summarize_benchmark_rows(tifa_before, tifa_after),
        "genai_bench": summarize_benchmark_rows(genai_before, genai_after),
    }


def render_markdown_summary(summary: Dict[str, Any]) -> str:
    lines = ["# Evaluation Summary", ""]
    for benchmark, payload in summary.items():
        lines.append(f"## {benchmark}")
        lines.append(f"- before_mean: {payload['before_mean']:.4f}")
        lines.append(f"- after_mean: {payload['after_mean']:.4f}")
        lines.append(f"- delta: {payload['delta']:.4f}")
        lines.append(f"- before_count: {payload['before_count']}")
        lines.append(f"- after_count: {payload['after_count']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_summary_outputs(output_dir: Path, summary: Dict[str, Any]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    rows = []
    for benchmark, payload in summary.items():
        rows.append(
            {
                "benchmark": benchmark,
                "before_mean": payload["before_mean"],
                "after_mean": payload["after_mean"],
                "delta": payload["delta"],
                "before_count": payload["before_count"],
                "after_count": payload["after_count"],
            }
        )
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "summary.md").write_text(render_markdown_summary(summary), encoding="utf-8")
