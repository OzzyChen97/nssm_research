"""Aggregate official full MMLongBench outputs produced through NSSM."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .mmlongbench_manifest import (
    DEFAULT_FULL_TASKS,
    DEFAULT_LENGTHS,
    OFFICIAL_AGGREGATES,
    RuntimeSettings,
    iter_specs,
)
from .mmlongbench_validation import ValidationResult, inspect_output


def _parse_csv(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _load_manifest(result_dir: Path) -> Dict[str, Any]:
    manifest_path = result_dir / ".nssm_full" / "manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _mean(values: Iterable[Optional[float]], require_all: bool = False) -> Optional[float]:
    items = list(values)
    valid = [float(value) for value in items if value is not None]
    if not valid:
        return None
    if require_all and len(valid) != len(items):
        return None
    return sum(valid) / len(valid)


def _format_optional(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _allowed_statuses(strict_official: bool) -> Sequence[str]:
    if strict_official:
        return ("success_clean",)
    return ("success_clean", "success_degraded", "judge_pending")


def _merge_runtime(
    runtime: Optional[RuntimeSettings],
    manifest_runtime: Optional[Dict[str, Any]],
) -> RuntimeSettings:
    default_runtime = RuntimeSettings()
    if not manifest_runtime:
        return runtime or default_runtime
    if runtime is None:
        return RuntimeSettings(**manifest_runtime)

    merged = asdict(runtime)
    default_values = asdict(default_runtime)
    for field_name, manifest_value in manifest_runtime.items():
        if merged.get(field_name) == default_values.get(field_name):
            merged[field_name] = manifest_value
    return RuntimeSettings(**merged)


def _score_lookup(
    validations: Sequence[ValidationResult],
    strict_official: bool,
) -> Dict[int, Dict[str, float]]:
    allowed = set(_allowed_statuses(strict_official))
    lookup: Dict[int, Dict[str, float]] = defaultdict(dict)
    for validation in validations:
        if validation.status not in allowed:
            continue
        if validation.score_pct is None:
            continue
        job_id_parts = validation.job_id.split("__")
        if len(job_id_parts) < 3:
            continue
        dataset = job_id_parts[1]
        length_part = job_id_parts[2]
        if not length_part.startswith("K"):
            continue
        length_k = int(length_part[1:])
        lookup[length_k][dataset] = float(validation.score_pct)
    return lookup


def _official_length_rows(
    lookup: Dict[int, Dict[str, float]],
    length_list: Sequence[int],
    require_all: bool,
) -> Dict[int, Dict[str, Optional[float]]]:
    rows: Dict[int, Dict[str, Optional[float]]] = {}
    for length_k in sorted(length_list):
        scores = dict(lookup.get(length_k, {}))
        aggregates: Dict[str, Optional[float]] = {}
        for aggregate_name, members in OFFICIAL_AGGREGATES.items():
            values = []
            for member in members:
                if member in aggregates:
                    values.append(aggregates[member])
                else:
                    values.append(scores.get(member))
            aggregates[aggregate_name] = _mean(values, require_all=require_all)
        rows[length_k] = aggregates
    return rows


def _completion_rows(validations: Sequence[ValidationResult], specs: Sequence[Any]) -> List[Dict[str, Any]]:
    expected = defaultdict(int)
    statuses = defaultdict(Counter)
    for spec in specs:
        expected[(spec.task, spec.length_k)] += 1
    for validation in validations:
        parts = validation.job_id.split("__")
        task = parts[0]
        length_k = int(parts[-1][1:])
        statuses[(task, length_k)][validation.status] += 1
    rows: List[Dict[str, Any]] = []
    for key, expected_count in sorted(expected.items(), key=lambda item: (-item[0][1], item[0][0])):
        counter = statuses.get(key, Counter())
        rows.append(
            {
                "task": key[0],
                "length_k": key[1],
                "expected": expected_count,
                "success_clean": counter.get("success_clean", 0),
                "success_degraded": counter.get("success_degraded", 0),
                "judge_pending": counter.get("judge_pending", 0),
                "parse_invalid": counter.get("parse_invalid", 0),
                "missing_output": counter.get("missing_output", 0),
                "other": sum(
                    count
                    for status, count in counter.items()
                    if status
                    not in {"success_clean", "success_degraded", "judge_pending", "parse_invalid", "missing_output"}
                ),
            }
        )
    return rows


def _dataset_rows(validations: Sequence[ValidationResult]) -> List[List[str]]:
    rows: List[List[str]] = []
    for validation in sorted(validations, key=lambda item: item.job_id):
        parts = validation.job_id.split("__")
        task = parts[0]
        dataset = parts[1]
        length_k = parts[2][1:]
        rows.append(
            [
                task,
                dataset,
                str(length_k),
                validation.status,
                _format_optional(validation.score_pct),
                validation.metric_name or "-",
                "yes" if validation.fallback_detected else "no",
                "yes" if validation.oom_detected else "no",
                ",".join(validation.issues) if validation.issues else "-",
            ]
        )
    return rows


def _official_rows_table(rows: Dict[int, Dict[str, Optional[float]]]) -> List[List[str]]:
    output: List[List[str]] = []
    ordered_columns = ["VRAG", "NIAH", "ICL", "Summ", "DocVQA", "Avg"]
    for length_k in sorted(rows):
        output.append([f"K{length_k}", *[_format_optional(rows[length_k].get(column)) for column in ordered_columns]])
    return output


def _length_rows(
    validations: Sequence[ValidationResult],
    specs: Sequence[Any],
    official_rows: Dict[int, Dict[str, Optional[float]]],
) -> List[Dict[str, Any]]:
    expected = defaultdict(int)
    statuses = defaultdict(Counter)
    for spec in specs:
        expected[spec.length_k] += 1
    for validation in validations:
        length_k = int(validation.job_id.split("__")[-1][1:])
        statuses[length_k][validation.status] += 1

    rows: List[Dict[str, Any]] = []
    for length_k in sorted(expected):
        counter = statuses.get(length_k, Counter())
        rows.append(
            {
                "length_k": length_k,
                "expected_jobs": expected[length_k],
                "success_clean": counter.get("success_clean", 0),
                "success_degraded": counter.get("success_degraded", 0),
                "judge_pending": counter.get("judge_pending", 0),
                "parse_invalid": counter.get("parse_invalid", 0),
                "missing_output": counter.get("missing_output", 0),
                "other": sum(
                    count
                    for status, count in counter.items()
                    if status
                    not in {"success_clean", "success_degraded", "judge_pending", "parse_invalid", "missing_output"}
                ),
                "selected_jobs_complete": counter.get("success_clean", 0) == expected[length_k],
                "official_score_complete": official_rows.get(length_k, {}).get("Avg") is not None,
            }
        )
    return rows


def _task_summary(validations: Sequence[ValidationResult]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[ValidationResult]] = defaultdict(list)
    for validation in validations:
        task = validation.job_id.split("__")[0]
        grouped[task].append(validation)
    summary: Dict[str, Dict[str, Any]] = {}
    for task, items in grouped.items():
        summary[task] = {
            "count": len(items),
            "clean": sum(1 for item in items if item.status == "success_clean"),
            "degraded": sum(1 for item in items if item.status == "success_degraded"),
            "judge_pending": sum(1 for item in items if item.status == "judge_pending"),
            "missing": sum(1 for item in items if item.status == "missing_output"),
            "invalid": sum(1 for item in items if item.status == "parse_invalid"),
        }
    return summary


def _failure_examples(validations: Sequence[ValidationResult], limit: int = 20) -> List[str]:
    lines: List[str] = []
    for validation in validations:
        if validation.status in {"success_clean", "success_degraded", "judge_pending"}:
            continue
        lines.append(
            f"- `{validation.job_id}` status={validation.status}"
            + (f" issues={','.join(validation.issues)}" if validation.issues else "")
        )
        if len(lines) >= limit:
            break
    if not lines:
        lines.append("- None")
    return lines


def build_report(
    result_dir: Path,
    output_dir: Path,
    bench_root: Optional[Path] = None,
    task_list: Optional[Sequence[str]] = None,
    length_list: Optional[Sequence[int]] = None,
    runtime: Optional[RuntimeSettings] = None,
) -> Dict[str, Any]:
    manifest = _load_manifest(result_dir=result_dir)
    task_list = list(task_list) if task_list is not None else None
    length_list = [int(item) for item in length_list] if length_list is not None else None
    if task_list is None:
        task_list = [str(item) for item in manifest.get("task_list", DEFAULT_FULL_TASKS)]
    if length_list is None:
        length_list = [int(item) for item in manifest.get("length_list", DEFAULT_LENGTHS)]
    if bench_root is None and manifest.get("bench_dir"):
        bench_root = Path(manifest["bench_dir"])
    runtime = _merge_runtime(runtime=runtime, manifest_runtime=manifest.get("runtime"))

    specs = iter_specs(
        bench_root=bench_root,
        task_list=task_list,
        length_list=length_list,
    )
    meta_dir = result_dir / ".nssm_full"
    validations = [
        inspect_output(result_dir=result_dir, spec=spec, runtime=runtime, meta_dir=meta_dir)
        for spec in specs
    ]

    status_counter = Counter(validation.status for validation in validations)
    official_lookup = _score_lookup(validations, strict_official=True)
    diagnostic_lookup = _score_lookup(validations, strict_official=False)
    length_order = list(length_list)
    official_rows = _official_length_rows(official_lookup, length_order, require_all=True)
    diagnostic_rows = _official_length_rows(diagnostic_lookup, length_order, require_all=False)
    completion = _completion_rows(validations, specs)
    length_summary = _length_rows(validations, specs, official_rows)

    summary = {
        "overview": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "result_dir": str(result_dir),
            "bench_root": str(bench_root) if bench_root is not None else None,
            "total_expected": len(specs),
            "status_counts": dict(status_counter),
        },
        "manifest": manifest,
        "runtime": asdict(runtime),
        "length_summary": length_summary,
        "completion": completion,
        "task_summary": _task_summary(validations),
        "dataset_results": [validation.to_dict() for validation in validations],
        "official_by_length": official_rows,
        "diagnostic_by_length": diagnostic_rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dataset_table = _dataset_rows(validations)
    completion_table = _markdown_table(
        ["Task", "K", "Expected", "Clean", "Degraded", "Judge Pending", "Invalid", "Missing", "Other"],
        [
            [
                row["task"],
                f"K{row['length_k']}",
                str(row["expected"]),
                str(row["success_clean"]),
                str(row["success_degraded"]),
                str(row["judge_pending"]),
                str(row["parse_invalid"]),
                str(row["missing_output"]),
                str(row["other"]),
            ]
            for row in completion
        ],
    )
    length_table = _markdown_table(
        [
            "Length",
            "Expected Jobs",
            "Clean",
            "Degraded",
            "Judge Pending",
            "Invalid",
            "Missing",
            "Selected Jobs Complete",
            "Official Score Complete",
        ],
        [
            [
                f"K{row['length_k']}",
                str(row["expected_jobs"]),
                str(row["success_clean"]),
                str(row["success_degraded"]),
                str(row["judge_pending"]),
                str(row["parse_invalid"]),
                str(row["missing_output"]),
                "yes" if row["selected_jobs_complete"] else "no",
                "yes" if row["official_score_complete"] else "no",
            ]
            for row in length_summary
        ],
    )
    official_table = _markdown_table(
        ["Length", "VRAG", "NIAH", "ICL", "Summ", "DocVQA", "Avg"],
        _official_rows_table(official_rows),
    )
    diagnostic_table = _markdown_table(
        ["Length", "VRAG", "NIAH", "ICL", "Summ", "DocVQA", "Avg"],
        _official_rows_table(diagnostic_rows),
    )
    status_lines = [f"- `{status}`: {count}" for status, count in sorted(status_counter.items())]
    markdown = "\n".join(
        [
            "# NSSM Full MMLongBench Report",
            "",
            "## Overview",
            f"- Generated at: {summary['overview']['generated_at_utc']}",
            f"- Result dir: `{result_dir}`",
            f"- Total expected dataset jobs: {len(specs)}",
            *status_lines,
            "",
            "## Completion",
            completion_table,
            "",
            "## Length Completeness",
            length_table,
            "",
            "## Official Clean Totals",
            official_table,
            "",
            "If any cell is `-`, the full official score for that length is incomplete.",
            "",
            "## Partial Diagnostic Totals",
            diagnostic_table,
            "",
            "Diagnostic totals may include degraded runs or unjudged summarization outputs. They are not official final numbers.",
            "",
            "## Dataset Results",
            _markdown_table(
                ["Task", "Dataset", "K", "Status", "Score", "Metric", "Fallback", "OOM", "Issues"],
                dataset_table,
            ),
            "",
            "## Failure Examples",
            *_failure_examples(validations),
            "",
        ]
    )
    (output_dir / "summary.md").write_text(markdown + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate NSSM full MMLongBench result files.")
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--bench_root", type=str, default=None)
    parser.add_argument("--task_list", type=str, default=None)
    parser.add_argument("--length_list", type=str, default=None)
    parser.add_argument("--generation_min_length", type=int, default=None)
    parser.add_argument("--do_sample", type=str, choices=["True", "False"], default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = None
    if any(
        value is not None
        for value in (
            args.generation_min_length,
            args.do_sample,
            args.temperature,
            args.top_p,
            args.seed,
            args.max_test_samples,
        )
    ):
        runtime = RuntimeSettings(
            generation_min_length=int(args.generation_min_length or 0),
            do_sample=args.do_sample == "True" if args.do_sample is not None else False,
            temperature=float(args.temperature if args.temperature is not None else 1.0),
            top_p=float(args.top_p if args.top_p is not None else 1.0),
            seed=int(args.seed if args.seed is not None else 42),
            max_test_samples_override=args.max_test_samples,
        )
    build_report(
        result_dir=Path(args.result_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        bench_root=Path(args.bench_root).resolve() if args.bench_root else None,
        task_list=_parse_csv(args.task_list) if args.task_list else None,
        length_list=[int(item) for item in _parse_csv(args.length_list)] if args.length_list else None,
        runtime=runtime,
    )


if __name__ == "__main__":
    main()
