"""Validation helpers for full MMLongBench outputs."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .mmlongbench_manifest import DatasetRunSpec, RuntimeSettings


YES_NO_RE = re.compile(r"\b(?:yes|no)\b", flags=re.IGNORECASE)
CHOICE_RE = re.compile(r"\b([A-Z])\b")
JSON_LIST_RE = re.compile(r"\[[^\[\]]*\]")


@dataclass
class ValidationResult:
    job_id: str
    status: str
    output_path: Optional[str] = None
    raw_output_path: Optional[str] = None
    judged_output_path: Optional[str] = None
    sample_count: int = 0
    metric_name: Optional[str] = None
    score_pct: Optional[float] = None
    issues: List[str] = field(default_factory=list)
    degraded: bool = False
    judge_pending: bool = False
    used_judged_output: bool = False
    runtime_status: Optional[str] = None
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    fallback_detected: bool = False
    oom_detected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_metric_value(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def find_raw_output(result_dir: Path, spec: DatasetRunSpec, runtime: RuntimeSettings) -> Optional[Path]:
    exact = result_dir / spec.expected_output_name(runtime)
    if exact.exists():
        return exact
    candidates = sorted(
        path
        for path in result_dir.glob(f"{spec.output_prefix()}*.json")
        if path.name != "summary.json" and not path.name.endswith("-gpt4eval_o.json")
    )
    return _first_existing(candidates)


def find_score_output(raw_output: Path) -> Optional[Path]:
    score_path = Path(f"{raw_output}.score")
    if score_path.exists():
        return score_path
    return None


def find_judged_output(raw_output: Path) -> Optional[Path]:
    judged_path = raw_output.with_name(raw_output.name.replace(".json", "-gpt4eval_o.json"))
    if judged_path.exists():
        return judged_path
    return None


def _select_metric_name(spec: DatasetRunSpec, averaged_metrics: Dict[str, Any]) -> Optional[str]:
    for metric_name in spec.expected_metric_names():
        if _normalize_metric_value(averaged_metrics.get(metric_name)) is not None:
            return metric_name
    for fallback_name in ("doc_qa_llm", "doc_qa", "sub_em", "acc", "soft_acc", "mc_acc", "cls_acc"):
        if _normalize_metric_value(averaged_metrics.get(fallback_name)) is not None:
            return fallback_name
    return None


def _looks_like_bad_vh(parsed_output: str) -> bool:
    lowered = parsed_output.lower()
    return "<image>" in lowered or lowered.count("answer:") > 1 or len(parsed_output.strip()) > 64


def _looks_like_bad_icl(parsed_output: str) -> bool:
    return "\n" in parsed_output or len(parsed_output.strip()) > 32 or "<image>" in parsed_output.lower()


def _looks_like_bad_choice(parsed_output: str) -> bool:
    matches = CHOICE_RE.findall(parsed_output)
    return not matches or len(parsed_output.strip()) > 24


def _looks_like_bad_counting(parsed_output: str) -> bool:
    return JSON_LIST_RE.search(parsed_output) is None and len(re.findall(r"\d+", parsed_output)) == 0


def _detect_degraded_rows(spec: DatasetRunSpec, rows: Sequence[Dict[str, Any]]) -> List[str]:
    if not rows:
        return ["no_rows"]
    bad = 0
    for row in rows:
        parsed_output = str(row.get("parsed_output", row.get("output", "")))
        if spec.task == "vh" and _looks_like_bad_vh(parsed_output):
            bad += 1
        elif spec.task == "icl" and _looks_like_bad_icl(parsed_output):
            bad += 1
        elif spec.dataset.endswith("-image") and ("retrieval" in spec.dataset or "reasoning" in spec.dataset):
            if _looks_like_bad_choice(parsed_output):
                bad += 1
        elif "counting" in spec.dataset and _looks_like_bad_counting(parsed_output):
            bad += 1
    if bad == 0:
        return []
    ratio = bad / max(len(rows), 1)
    if ratio >= 0.2:
        return [f"suspicious_outputs:{bad}/{len(rows)}"]
    return []


def _load_status_metadata(meta_dir: Path, job_id: str) -> Dict[str, Any]:
    status_path = meta_dir / "job_status" / f"{job_id}.json"
    if not status_path.exists():
        return {}
    with status_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def inspect_output(
    result_dir: Path,
    spec: DatasetRunSpec,
    runtime: RuntimeSettings,
    meta_dir: Optional[Path] = None,
) -> ValidationResult:
    raw_output = find_raw_output(result_dir=result_dir, spec=spec, runtime=runtime)
    if raw_output is None:
        result = ValidationResult(job_id=spec.job_id, status="missing_output")
    else:
        judged_output = find_judged_output(raw_output)
        chosen_output = judged_output if judged_output is not None else raw_output
        with chosen_output.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        averaged_metrics = payload.get("averaged_metrics", {}) if isinstance(payload, dict) else {}
        metric_name = _select_metric_name(spec, averaged_metrics)
        issues: List[str] = []
        judge_pending = False
        degraded = False
        status = "success_clean"
        if not isinstance(payload, dict):
            issues.append("payload_not_dict")
            status = "parse_invalid"
        if metric_name is None:
            issues.append("metric_missing")
            status = "parse_invalid"
        score_pct = None
        if metric_name is not None:
            metric_value = _normalize_metric_value(averaged_metrics.get(metric_name))
            if metric_value is None:
                issues.append(f"metric_not_numeric:{metric_name}")
                status = "parse_invalid"
            else:
                score_pct = metric_value
        if len(rows) == 0:
            issues.append("no_rows")
            status = "parse_invalid"
        degraded_issues = _detect_degraded_rows(spec, rows)
        if degraded_issues and status == "success_clean":
            degraded = True
            status = "success_degraded"
            issues.extend(degraded_issues)
        if spec.is_summ and judged_output is None:
            judge_pending = True
            if status == "success_clean":
                status = "judge_pending"
            elif status == "success_degraded":
                status = "judge_pending"
        result = ValidationResult(
            job_id=spec.job_id,
            status=status,
            output_path=str(chosen_output),
            raw_output_path=str(raw_output),
            judged_output_path=str(judged_output) if judged_output is not None else None,
            sample_count=len(rows),
            metric_name=metric_name,
            score_pct=score_pct,
            issues=issues,
            degraded=degraded,
            judge_pending=judge_pending,
            used_judged_output=judged_output is not None,
        )

    if meta_dir is not None:
        metadata = _load_status_metadata(meta_dir=meta_dir, job_id=spec.job_id)
        if metadata:
            result.runtime_status = metadata.get("final_status")
            result.attempts = metadata.get("attempts", [])
            result.fallback_detected = bool(metadata.get("fallback_detected", False))
            result.oom_detected = bool(metadata.get("oom_detected", False))
            if result.status.startswith("success") and metadata.get("final_status") == "success_degraded":
                result.status = "success_degraded"
                result.degraded = True
            if metadata.get("final_status") == "parse_invalid":
                result.status = "parse_invalid"
            if metadata.get("final_status") == "missing_output":
                result.status = "missing_output"
    return result
