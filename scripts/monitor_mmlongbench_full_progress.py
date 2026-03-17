#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import heapq
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3}")
START_RE = re.compile(r"gpu(\d+) start ([^\s]+) profile=([^\s]+) attempt=(\d+)")
FINISH_RE = re.compile(r"gpu(\d+) finish ([^\s]+) final_status=([^\s]+)")

PCT_LINE_RE = re.compile(
    r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[[^\]]*<([^,\]]+),\s*([0-9]*\.?[0-9]+)s/it\]"
)
PCT_LINE_ITPS_RE = re.compile(
    r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[[^\]]*<([^,\]]+),\s*([0-9]*\.?[0-9]+)it/s\]"
)


RISK_SAFE_TASKS = {"vrag", "vh", "mm_niah_text", "mm_niah_image", "summ", "docqa"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor full MMLongBench run with global ETA.")
    parser.add_argument("--model-output-dir", type=str, required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--output-log", type=str, default=None)
    parser.add_argument("--tail-bytes", type=int, default=200_000)
    return parser.parse_args()


def parse_ts(line: str) -> Optional[dt.datetime]:
    match = TS_RE.search(line)
    if not match:
        return None
    return dt.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")


def _risk_level(task: str, length_k: int) -> str:
    if length_k >= 128 and task in RISK_SAFE_TASKS:
        return "safe"
    if length_k >= 64 and task in RISK_SAFE_TASKS:
        return "balanced"
    return "fast"


def _job_id_from_spec(spec: Dict[str, object]) -> Optional[str]:
    task = str(spec.get("task", "")).strip()
    dataset = str(spec.get("dataset", "")).strip()
    length_k_raw = spec.get("length_k", 0)
    try:
        length_k = int(length_k_raw or 0)
    except (TypeError, ValueError):
        length_k = 0
    if not task or not dataset or length_k <= 0:
        return None
    return f"{task}__{dataset}__K{length_k}"


def read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def read_tail_lines(path: Path, max_bytes: int) -> List[str]:
    if not path.exists():
        return []
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes), os.SEEK_SET)
        data = handle.read().decode("utf-8", errors="ignore")
    return data.splitlines()


def parse_progress_from_error_log(path: Path, max_bytes: int) -> Optional[Dict[str, float]]:
    lines = read_tail_lines(path, max_bytes)
    for line in reversed(lines):
        match = PCT_LINE_RE.search(line)
        if match:
            done = int(match.group(2))
            total = int(match.group(3))
            sec_per_it = float(match.group(5))
            remain = max(total - done, 0)
            return {
                "done": done,
                "total": total,
                "sec_per_it": sec_per_it,
                "remain_sec": remain * sec_per_it,
            }
        match_itps = PCT_LINE_ITPS_RE.search(line)
        if match_itps:
            done = int(match_itps.group(2))
            total = int(match_itps.group(3))
            it_per_sec = float(match_itps.group(5))
            if it_per_sec <= 0:
                continue
            sec_per_it = 1.0 / it_per_sec
            remain = max(total - done, 0)
            return {
                "done": done,
                "total": total,
                "sec_per_it": sec_per_it,
                "remain_sec": remain * sec_per_it,
            }
    return None


def parse_eval_log(eval_log: Path) -> Dict[str, object]:
    starts: Dict[str, dt.datetime] = {}
    running_gpu: Dict[str, str] = {}
    running_attempt: Dict[str, int] = {}
    finishes: Dict[str, str] = {}

    for raw_line in read_text(eval_log).splitlines():
        line = raw_line.strip()
        if not line:
            continue

        start_match = START_RE.search(line)
        if start_match:
            ts = parse_ts(line)
            if ts is None:
                continue
            gpu_id = start_match.group(1)
            job_id = start_match.group(2)
            attempt = int(start_match.group(4))
            starts[job_id] = ts
            running_gpu[job_id] = gpu_id
            running_attempt[job_id] = attempt
            continue

        finish_match = FINISH_RE.search(line)
        if finish_match:
            job_id = finish_match.group(2)
            final_status = finish_match.group(3)
            finishes[job_id] = final_status

    running = sorted(set(starts.keys()) - set(finishes.keys()))
    return {
        "starts": starts,
        "running_gpu": running_gpu,
        "running_attempt": running_attempt,
        "finishes": finishes,
        "running": running,
    }


def collect_statuses(meta_dir: Path) -> Dict[str, Dict[str, object]]:
    status_dir = meta_dir / "job_status"
    statuses: Dict[str, Dict[str, object]] = {}
    if not status_dir.exists():
        return statuses
    for path in status_dir.glob("*.json"):
        payload = read_json(path)
        if not payload:
            continue
        job_id = str(payload.get("job_id", path.stem))
        statuses[job_id] = payload
    return statuses


def summarize_duration_stats(
    statuses: Dict[str, Dict[str, object]],
    job_to_spec: Dict[str, Dict[str, object]],
) -> Dict[str, Dict[object, float]]:
    by_task_len: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    by_len: Dict[int, List[float]] = defaultdict(list)
    by_risk: Dict[str, List[float]] = defaultdict(list)
    all_vals: List[float] = []

    for job_id, payload in statuses.items():
        attempts = payload.get("attempts", [])
        if not isinstance(attempts, list) or not attempts:
            continue

        elapsed_sum = 0.0
        for item in attempts:
            if isinstance(item, dict):
                elapsed_sum += float(item.get("elapsed_sec", 0.0) or 0.0)

        if elapsed_sum <= 0:
            continue

        spec = job_to_spec.get(job_id)
        if not spec:
            continue
        task = str(spec.get("task", ""))
        length_k = int(spec.get("length_k", 0) or 0)
        if not task or length_k <= 0:
            continue

        by_task_len[(task, length_k)].append(elapsed_sum)
        by_len[length_k].append(elapsed_sum)
        by_risk[_risk_level(task, length_k)].append(elapsed_sum)
        all_vals.append(elapsed_sum)

    task_len_median = {key: sorted(vals)[len(vals) // 2] for key, vals in by_task_len.items()}
    len_median = {key: sorted(vals)[len(vals) // 2] for key, vals in by_len.items()}
    risk_median = {key: sorted(vals)[len(vals) // 2] for key, vals in by_risk.items()}
    global_median = sorted(all_vals)[len(all_vals) // 2] if all_vals else 3600.0

    return {
        "task_len_median": task_len_median,
        "len_median": len_median,
        "risk_median": risk_median,
        "global_median": global_median,
    }


def estimate_default_duration(
    task: str,
    length_k: int,
    duration_stats: Dict[str, Dict[object, float]],
) -> float:
    task_len_median = duration_stats["task_len_median"]
    len_median = duration_stats["len_median"]
    risk_median = duration_stats["risk_median"]
    global_median = float(duration_stats["global_median"])

    key = (task, length_k)
    if key in task_len_median:
        return float(task_len_median[key])
    if length_k in len_median:
        return float(len_median[length_k])

    risk = _risk_level(task, length_k)
    if risk in risk_median:
        return float(risk_median[risk])

    known_lengths = sorted(len_median.keys())
    if known_lengths:
        nearest = min(known_lengths, key=lambda known: abs(known - length_k))
        base = float(len_median[nearest])
        if nearest > 0:
            scale = max(length_k, 1) / nearest
            base = base * (scale ** 0.9)
        return base

    return global_median


def simulate_global_eta(
    running_remaining_secs: List[float],
    pending_secs: List[float],
    slots: int,
) -> float:
    if slots <= 0:
        slots = 1
    heap = running_remaining_secs[:slots]
    if len(heap) < slots:
        heap.extend([0.0] * (slots - len(heap)))
    heapq.heapify(heap)

    for duration in pending_secs:
        earliest = heapq.heappop(heap)
        heapq.heappush(heap, earliest + max(duration, 0.0))

    return max(heap) if heap else 0.0


def format_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    sec = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def gpu_status() -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception as exc:
        return f"gpu_error={exc}"

    parts: List[str] = []
    for row in out.strip().splitlines():
        fields = [item.strip() for item in row.split(",")]
        if len(fields) < 4:
            continue
        parts.append(f"gpu{fields[0]}:{fields[1]}%/{fields[2]}MiB")
    return " ".join(parts)


def _iter_specs(manifest: Dict[str, object]) -> Iterable[Dict[str, object]]:
    specs = manifest.get("specs", [])
    if not isinstance(specs, list):
        return []
    return [item for item in specs if isinstance(item, dict)]


def build_line(model_output_dir: Path, tail_bytes: int) -> str:
    now = dt.datetime.now()
    meta_dir = model_output_dir / ".nssm_full"
    manifest = read_json(meta_dir / "manifest.json")
    eval_log = model_output_dir / "eval_log_full.log"

    specs = list(_iter_specs(manifest))
    job_to_spec: Dict[str, Dict[str, object]] = {}
    for item in specs:
        job_id = str(item.get("job_id", "")).strip() or (_job_id_from_spec(item) or "")
        if job_id:
            job_to_spec[job_id] = item

    total_jobs = len(job_to_spec)
    tasks = sorted({str(item.get("task", "")) for item in specs if item.get("task")})
    lengths = sorted({int(item.get("length_k", 0)) for item in specs if int(item.get("length_k", 0) or 0) > 0})
    gpu_list = manifest.get("gpu_list", [])
    slots = len(gpu_list) if isinstance(gpu_list, list) and gpu_list else 4

    statuses = collect_statuses(meta_dir)
    status_counts: Dict[str, int] = defaultdict(int)
    for payload in statuses.values():
        status_counts[str(payload.get("final_status", "unknown"))] += 1

    log_meta = parse_eval_log(eval_log)
    starts: Dict[str, dt.datetime] = log_meta["starts"]  # type: ignore[assignment]
    running: List[str] = log_meta["running"]  # type: ignore[assignment]
    running_gpu: Dict[str, str] = log_meta["running_gpu"]  # type: ignore[assignment]
    running_attempt: Dict[str, int] = log_meta["running_attempt"]  # type: ignore[assignment]

    completed_jobs = set(statuses.keys())
    running_jobs = [job_id for job_id in running if job_id not in completed_jobs]
    pending_jobs = [job_id for job_id in job_to_spec.keys() if job_id not in completed_jobs and job_id not in running_jobs]

    duration_stats = summarize_duration_stats(statuses=statuses, job_to_spec=job_to_spec)

    running_parts: List[str] = []
    running_remaining_secs: List[float] = []
    for job_id in running_jobs:
        spec = job_to_spec.get(job_id, {})
        task = str(spec.get("task", "unknown"))
        length_k = int(spec.get("length_k", 0) or 0)

        attempt_idx = int(running_attempt.get(job_id, 1))
        error_log = meta_dir / "jobs" / job_id / f"attempt{attempt_idx}.error.log"
        parsed = parse_progress_from_error_log(error_log, max_bytes=tail_bytes)

        if parsed is not None:
            remain_sec = float(parsed["remain_sec"])
            running_parts.append(
                f"{job_id}@gpu{running_gpu.get(job_id, '?')}:{int(parsed['done'])}/{int(parsed['total'])},eta={format_hms(remain_sec)}"
            )
        else:
            default_total = estimate_default_duration(task, length_k, duration_stats)
            elapsed = 0.0
            if job_id in starts:
                elapsed = max((now - starts[job_id]).total_seconds(), 0.0)
            remain_sec = max(default_total - elapsed, default_total * 0.12)
            running_parts.append(
                f"{job_id}@gpu{running_gpu.get(job_id, '?')}:eta~{format_hms(remain_sec)}"
            )
        running_remaining_secs.append(remain_sec)

    pending_secs: List[float] = []
    for job_id in pending_jobs:
        spec = job_to_spec.get(job_id, {})
        task = str(spec.get("task", "unknown"))
        length_k = int(spec.get("length_k", 0) or 0)
        pending_secs.append(estimate_default_duration(task, length_k, duration_stats))

    global_eta_sec = simulate_global_eta(running_remaining_secs, pending_secs, slots=max(slots, len(running_jobs), 1))
    finish_at = now + dt.timedelta(seconds=global_eta_sec)

    running_block = " | ".join(running_parts) if running_parts else "none"

    done_count = len(completed_jobs)
    total = total_jobs if total_jobs > 0 else (done_count + len(running_jobs) + len(pending_jobs))
    status_summary = ",".join(f"{k}:{v}" for k, v in sorted(status_counts.items())) or "none"
    lengths_txt = ",".join(str(item) for item in lengths) if lengths else "unknown"
    tasks_txt = ",".join(tasks) if tasks else "unknown"

    return (
        f"[{now.strftime('%F %T')}] done={done_count}/{total} running={len(running_jobs)} "
        f"pending={len(pending_jobs)} slots={slots} statuses={status_summary} "
        f"ETA={format_hms(global_eta_sec)} finish_at={finish_at.strftime('%F %T')} "
        f"lengths={lengths_txt} tasks={tasks_txt} | running: {running_block} | gpu: {gpu_status()}"
    )


def main() -> None:
    args = parse_args()
    model_output_dir = Path(args.model_output_dir).resolve()
    output_log = (
        Path(args.output_log).resolve()
        if args.output_log
        else model_output_dir.parent / "monitor_full_live.log"
    )
    output_log.parent.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            line = build_line(model_output_dir=model_output_dir, tail_bytes=args.tail_bytes)
        except Exception as exc:
            line = f"[{dt.datetime.now().strftime('%F %T')}] monitor_error={exc}"

        print(line, flush=True)
        with output_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        time.sleep(max(args.interval, 5))


if __name__ == "__main__":
    main()
