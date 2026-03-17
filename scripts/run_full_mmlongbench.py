#!/usr/bin/env python3
"""Run official full MMLongBench through NSSM with dataset-level recovery."""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import yaml

from src.eval.mmlongbench_manifest import (
    DEFAULT_FULL_TASKS,
    DEFAULT_LENGTHS,
    DatasetRunSpec,
    RuntimeSettings,
    iter_specs,
)
from src.eval.mmlongbench_validation import inspect_output


LOGGER = logging.getLogger("nssm_mmlongbench_full_runner")


@dataclass(frozen=True)
class AttemptProfile:
    name: str
    nssm_overrides: Dict[str, Any]
    model_overrides: Dict[str, Any]
    num_workers: int
    preprocessing_num_workers: int
    allow_degraded: bool = False


def _collect_cuda_diagnostics(gpu_list: Sequence[str]) -> List[str]:
    diagnostics: List[str] = []
    diagnostics.append(f"torch={torch.__version__}")
    diagnostics.append(f"torch.version.cuda={torch.version.cuda}")
    diagnostics.append(f"cuda_available={torch.cuda.is_available()}")
    diagnostics.append(f"cuda_device_count={torch.cuda.device_count()}")
    return diagnostics


def _model_tag(model_name: str) -> str:
    return os.path.basename(os.path.normpath(model_name))


def _parse_csv(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official full MMLongBench via NSSM.")
    parser.add_argument("--bench_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task_list", type=str, default=",".join(DEFAULT_FULL_TASKS))
    parser.add_argument("--length_list", type=str, default=",".join(str(item) for item in DEFAULT_LENGTHS))
    parser.add_argument("--gpu_list", type=str, default="0,1,2,3")
    parser.add_argument("--result_base_path", type=str, required=True)
    parser.add_argument("--test_file_root", type=str, required=True)
    parser.add_argument("--image_file_root", type=str, required=True)
    parser.add_argument("--python_exec", type=str, default=sys.executable)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--preprocessing_num_workers", type=int, default=12)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume_missing", action="store_true")
    parser.add_argument("--strict_completeness", action="store_true")
    parser.add_argument("--allow_degraded", action="store_true")
    parser.add_argument("--docqa_llm_judge", type=str, choices=["True", "False"], default="False")
    parser.add_argument("--nssm_config", type=str, required=True)
    parser.add_argument("--memory_profile", type=str, default="a6000_4gpu")
    parser.add_argument("--generation_min_length", type=int, default=0)
    parser.add_argument("--do_sample", type=str, choices=["True", "False"], default="False")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_attempts", type=int, default=4)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def build_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER.name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(output_dir / "eval_log_full.log")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def preflight_cuda(gpu_list: Sequence[str]) -> None:
    if not torch.cuda.is_available():
        diagnostics = "; ".join(_collect_cuda_diagnostics(gpu_list))
        raise RuntimeError(f"CUDA is not available. Diagnostics: {diagnostics}")
    device_count = torch.cuda.device_count()
    if device_count <= 0:
        raise RuntimeError("PyTorch reports zero CUDA devices.")
    invalid_ids = [gpu_id for gpu_id in gpu_list if gpu_id.isdigit() and int(gpu_id) >= device_count]
    if invalid_ids:
        raise RuntimeError(f"Requested GPU ids {invalid_ids} exceed visible CUDA device count {device_count}.")


def _risk_level(spec: DatasetRunSpec) -> str:
    if spec.length_k >= 128 and spec.task in {"vrag", "vh", "mm_niah_text", "mm_niah_image", "summ", "docqa"}:
        return "safe"
    if spec.length_k >= 64 and spec.task in {"vrag", "vh", "mm_niah_text", "mm_niah_image", "summ", "docqa"}:
        return "balanced"
    return "fast"


def _schedule_specs(specs: Sequence[DatasetRunSpec]) -> List[DatasetRunSpec]:
    buckets: Dict[str, List[DatasetRunSpec]] = {"safe": [], "balanced": [], "fast": []}
    for spec in specs:
        buckets[_risk_level(spec)].append(spec)
    for items in buckets.values():
        items.sort(key=lambda item: (-item.length_k, item.task, item.dataset))

    schedule: List[DatasetRunSpec] = []
    round_robin = ["safe", "balanced", "fast", "safe", "balanced", "fast"]
    while any(buckets.values()):
        progressed = False
        for risk in round_robin:
            if buckets[risk]:
                schedule.append(buckets[risk].pop(0))
                progressed = True
        if not progressed:
            break
    return schedule


def _profile(
    *,
    name: str,
    args: argparse.Namespace,
    nssm_overrides: Optional[Dict[str, Any]] = None,
    model_overrides: Optional[Dict[str, Any]] = None,
    num_workers: Optional[int] = None,
    preprocessing_num_workers: Optional[int] = None,
    allow_degraded: bool = False,
) -> AttemptProfile:
    return AttemptProfile(
        name=name,
        nssm_overrides=nssm_overrides or {},
        model_overrides=model_overrides or {},
        num_workers=int(num_workers if num_workers is not None else args.num_workers),
        preprocessing_num_workers=int(
            preprocessing_num_workers
            if preprocessing_num_workers is not None
            else args.preprocessing_num_workers
        ),
        allow_degraded=allow_degraded,
    )


def build_attempt_profiles(spec: DatasetRunSpec, args: argparse.Namespace) -> List[AttemptProfile]:
    if args.memory_profile != "a6000_4gpu":
        return [_profile(name="default", args=args)]

    safe_text_rich = spec.task in {"summ", "docqa"} and spec.length_k >= 64
    safe_visual = spec.task in {"vh", "mm_niah_image"} and spec.length_k >= 64
    safe_vrag = spec.task == "vrag" and spec.length_k >= 64
    safe_text = spec.task == "mm_niah_text" and spec.length_k >= 64

    if safe_text_rich:
        return [
            _profile(
                name="safe_text_rich_base",
                args=args,
                nssm_overrides={"num_dynamic_slots": 192, "router_top_k": 24, "max_visual_tokens_raw": 80000},
                model_overrides={"torch_compile": False, "image_resize": 0.75},
                num_workers=min(args.num_workers, 20),
                preprocessing_num_workers=min(args.preprocessing_num_workers, 10),
            ),
            _profile(
                name="safe_text_rich_reduced",
                args=args,
                nssm_overrides={"num_dynamic_slots": 128, "router_top_k": 16, "max_visual_tokens_raw": 60000},
                model_overrides={"torch_compile": False, "image_resize": 0.67},
                num_workers=min(args.num_workers, 16),
                preprocessing_num_workers=min(args.preprocessing_num_workers, 8),
            ),
            _profile(
                name="safe_text_rich_textual_fallback",
                args=args,
                nssm_overrides={"num_dynamic_slots": 128, "router_top_k": 16, "max_visual_tokens_raw": 50000},
                model_overrides={
                    "torch_compile": False,
                    "image_resize": 0.67,
                    "force_textual_fallback": True,
                },
                num_workers=min(args.num_workers, 16),
                preprocessing_num_workers=min(args.preprocessing_num_workers, 8),
                allow_degraded=True,
            ),
        ]
    if safe_visual:
        return [
            _profile(
                name="safe_visual_base",
                args=args,
                nssm_overrides={"num_dynamic_slots": 192, "router_top_k": 24, "max_visual_tokens_raw": 70000},
                model_overrides={"torch_compile": False},
                num_workers=min(args.num_workers, 20),
                preprocessing_num_workers=min(args.preprocessing_num_workers, 10),
            ),
            _profile(
                name="safe_visual_reduced",
                args=args,
                nssm_overrides={"num_dynamic_slots": 128, "router_top_k": 16, "max_visual_tokens_raw": 55000},
                model_overrides={"torch_compile": False, "image_resize": 0.8},
                num_workers=min(args.num_workers, 16),
                preprocessing_num_workers=min(args.preprocessing_num_workers, 8),
            ),
            _profile(
                name="safe_visual_textual_fallback",
                args=args,
                nssm_overrides={"num_dynamic_slots": 128, "router_top_k": 16, "max_visual_tokens_raw": 50000},
                model_overrides={
                    "torch_compile": False,
                    "image_resize": 0.75,
                    "force_textual_fallback": True,
                },
                num_workers=min(args.num_workers, 16),
                preprocessing_num_workers=min(args.preprocessing_num_workers, 8),
                allow_degraded=True,
            ),
        ]
    if safe_vrag:
        return [
            _profile(
                name="safe_vrag_base",
                args=args,
                nssm_overrides={"num_dynamic_slots": 192, "router_top_k": 24, "max_visual_tokens_raw": 90000},
                model_overrides={"torch_compile": False},
                num_workers=min(args.num_workers, 24),
                preprocessing_num_workers=min(args.preprocessing_num_workers, 10),
            ),
            _profile(
                name="safe_vrag_reduced",
                args=args,
                nssm_overrides={"num_dynamic_slots": 128, "router_top_k": 16, "max_visual_tokens_raw": 70000},
                model_overrides={"torch_compile": False},
                num_workers=min(args.num_workers, 20),
                preprocessing_num_workers=min(args.preprocessing_num_workers, 8),
            ),
            _profile(
                name="safe_vrag_textual_fallback",
                args=args,
                nssm_overrides={"num_dynamic_slots": 128, "router_top_k": 16, "max_visual_tokens_raw": 70000},
                model_overrides={"torch_compile": False, "force_textual_fallback": True},
                num_workers=min(args.num_workers, 20),
                preprocessing_num_workers=min(args.preprocessing_num_workers, 8),
                allow_degraded=True,
            ),
        ]
    if safe_text:
        return [
            _profile(
                name="safe_text_base",
                args=args,
                model_overrides={"torch_compile": False},
                num_workers=min(args.num_workers, 24),
                preprocessing_num_workers=min(args.preprocessing_num_workers, 10),
            ),
            _profile(
                name="safe_text_retry",
                args=args,
                model_overrides={"torch_compile": False},
                num_workers=min(args.num_workers, 20),
                preprocessing_num_workers=min(args.preprocessing_num_workers, 8),
            ),
        ]
    return [
        _profile(name="default_fast", args=args),
        _profile(
            name="default_fast_retry",
            args=args,
            model_overrides={"torch_compile": False},
            num_workers=min(args.num_workers, 24),
            preprocessing_num_workers=min(args.preprocessing_num_workers, 12),
        ),
    ]


def apply_config_overrides(
    base_config_path: Path,
    output_path: Path,
    model_name: str,
    profile: AttemptProfile,
) -> None:
    with base_config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config.setdefault("model", {})
    config.setdefault("nssm", {})
    config["model"].update(profile.model_overrides)
    config["nssm"].update(profile.nssm_overrides)
    if os.path.exists(model_name):
        config["model"]["model_local_path"] = model_name
    else:
        config["model"]["model_name"] = model_name
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def build_command(
    args: argparse.Namespace,
    spec: DatasetRunSpec,
    output_dir: Path,
    nssm_config_path: Path,
    profile: AttemptProfile,
    runtime: RuntimeSettings,
) -> List[str]:
    command = [
        args.python_exec,
        "eval.py",
        "--config",
        f"configs/{spec.config_name}",
        "--model_name_or_path",
        args.model_name,
        "--output_dir",
        str(output_dir),
        "--test_file_root",
        args.test_file_root,
        "--image_file_root",
        args.image_file_root,
        "--num_workers",
        str(profile.num_workers),
        "--preprocessing_num_workers",
        str(profile.preprocessing_num_workers),
        "--test_length",
        str(spec.length_k),
        "--docqa_llm_judge",
        args.docqa_llm_judge,
        "--datasets",
        spec.dataset,
        "--test_files",
        spec.test_file,
        "--input_max_length",
        str(spec.input_max_length),
        "--generation_max_length",
        str(spec.generation_max_length),
        "--use_chat_template",
        str(spec.use_chat_template),
        "--generation_min_length",
        str(runtime.generation_min_length),
        "--do_sample",
        str(runtime.do_sample),
        "--temperature",
        str(runtime.temperature),
        "--top_p",
        str(runtime.top_p),
        "--seed",
        str(runtime.seed),
        "--use_nssm",
        "--nssm_config",
        str(nssm_config_path),
    ]
    max_test_samples = spec.effective_max_test_samples(runtime)
    if max_test_samples is not None:
        command.extend(["--max_test_samples", str(max_test_samples)])
    command.append("--overwrite")
    return command


def detect_runtime_signals(*log_paths: Path) -> Dict[str, bool]:
    oom_detected = False
    fallback_detected = False
    for path in log_paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "CUDA out of memory" in text or "OutOfMemoryError" in text:
            oom_detected = True
        if "Falling back to textual memory prompt" in text or "forcing textual fallback" in text:
            fallback_detected = True
    return {
        "oom_detected": oom_detected,
        "fallback_detected": fallback_detected,
    }


def write_status(meta_dir: Path, spec: DatasetRunSpec, payload: Dict[str, Any]) -> None:
    status_dir = meta_dir / "job_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    with (status_dir / f"{spec.job_id}.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def should_skip_existing(
    validation_status: str,
    strict_completeness: bool,
    allow_degraded: bool,
) -> bool:
    if validation_status in {"success_clean", "judge_pending"}:
        return True
    if validation_status == "success_degraded":
        return not strict_completeness or allow_degraded
    return False


def determine_effective_status(
    validation_status: str,
    signals: Dict[str, bool],
    profile: AttemptProfile,
) -> str:
    effective = "success_clean" if validation_status == "judge_pending" else validation_status
    if effective == "success_clean" and signals["fallback_detected"]:
        return "success_degraded"
    if effective == "success_clean" and profile.allow_degraded:
        return "success_degraded"
    return effective


def run_spec(
    spec: DatasetRunSpec,
    gpu_id: str,
    args: argparse.Namespace,
    runtime: RuntimeSettings,
    bench_dir: Path,
    output_dir: Path,
    meta_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    job_dir = meta_dir / "jobs" / spec.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    attempt_profiles = build_attempt_profiles(spec=spec, args=args)[: args.max_attempts]
    attempts: List[Dict[str, Any]] = []
    final_status = "missing_output"
    final_validation = None
    any_oom = False
    any_fallback = False

    for attempt_idx, profile in enumerate(attempt_profiles, start=1):
        nssm_config_path = job_dir / f"attempt{attempt_idx}.nssm.yaml"
        apply_config_overrides(
            base_config_path=Path(args.nssm_config),
            output_path=nssm_config_path,
            model_name=args.model_name,
            profile=profile,
        )
        stdout_log = job_dir / f"attempt{attempt_idx}.stdout.log"
        stderr_log = job_dir / f"attempt{attempt_idx}.error.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

        command = build_command(
            args=args,
            spec=spec,
            output_dir=output_dir,
            nssm_config_path=nssm_config_path,
            profile=profile,
            runtime=runtime,
        )
        logger.info(
            "gpu%s start %s profile=%s attempt=%d",
            gpu_id,
            spec.job_id,
            profile.name,
            attempt_idx,
        )
        logger.info("gpu%s command=%s", gpu_id, " ".join(command))
        start_time = time.time()
        with stdout_log.open("w", encoding="utf-8") as stdout_handle, stderr_log.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            completed = subprocess.run(
                command,
                cwd=bench_dir,
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                check=False,
            )
        elapsed = time.time() - start_time
        signals = detect_runtime_signals(stdout_log, stderr_log)
        any_oom = any_oom or signals["oom_detected"]
        any_fallback = any_fallback or signals["fallback_detected"]
        validation = inspect_output(
            result_dir=output_dir,
            spec=spec,
            runtime=runtime,
            meta_dir=None,
        )
        effective_status = determine_effective_status(
            validation_status=validation.status,
            signals=signals,
            profile=profile,
        )
        attempt_payload = {
            "attempt": attempt_idx,
            "profile": profile.name,
            "returncode": completed.returncode,
            "elapsed_sec": elapsed,
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "nssm_config_path": str(nssm_config_path),
            "validation_status": validation.status,
            "effective_status": effective_status,
            "issues": list(validation.issues),
            "oom_detected": signals["oom_detected"],
            "fallback_detected": signals["fallback_detected"],
            "allow_degraded": profile.allow_degraded,
        }
        attempts.append(attempt_payload)
        final_validation = validation

        if effective_status == "success_clean":
            final_status = "success_clean"
            break
        if effective_status == "success_degraded":
            final_status = "success_degraded"
            if args.allow_degraded or not args.strict_completeness:
                break
            continue
        if signals["oom_detected"]:
            final_status = "oom"
            continue
        final_status = effective_status

    result = {
        "job_id": spec.job_id,
        "task": spec.task,
        "dataset": spec.dataset,
        "length_k": spec.length_k,
        "gpu_id": gpu_id,
        "final_status": final_status,
        "attempts": attempts,
        "fallback_detected": any_fallback,
        "oom_detected": any_oom,
    }
    if final_validation is not None:
        result["validation"] = final_validation.to_dict()
    write_status(meta_dir=meta_dir, spec=spec, payload=result)
    logger.info("gpu%s finish %s final_status=%s", gpu_id, spec.job_id, final_status)
    return result


def write_job_plan(
    meta_dir: Path,
    specs: Sequence[DatasetRunSpec],
    args: argparse.Namespace,
) -> None:
    plan_rows: List[Dict[str, Any]] = []
    for spec in specs:
        profiles = build_attempt_profiles(spec=spec, args=args)[: args.max_attempts]
        plan_rows.append(
            {
                "job_id": spec.job_id,
                "task": spec.task,
                "dataset": spec.dataset,
                "length_k": spec.length_k,
                "risk_level": _risk_level(spec),
                "profiles": [
                    {
                        "name": profile.name,
                        "nssm_overrides": dict(profile.nssm_overrides),
                        "model_overrides": dict(profile.model_overrides),
                        "num_workers": profile.num_workers,
                        "preprocessing_num_workers": profile.preprocessing_num_workers,
                        "allow_degraded": profile.allow_degraded,
                    }
                    for profile in profiles
                ],
            }
        )
    with (meta_dir / "job_plan.json").open("w", encoding="utf-8") as handle:
        json.dump(plan_rows, handle, ensure_ascii=False, indent=2)


def worker(
    gpu_id: str,
    args: argparse.Namespace,
    runtime: RuntimeSettings,
    bench_dir: Path,
    output_dir: Path,
    meta_dir: Path,
    spec_queue: "queue.Queue[DatasetRunSpec]",
    results: List[Dict[str, Any]],
    results_lock: threading.Lock,
    logger: logging.Logger,
) -> None:
    while True:
        try:
            spec = spec_queue.get_nowait()
        except queue.Empty:
            logger.info("gpu%s no more jobs, exiting", gpu_id)
            return
        try:
            result = run_spec(
                spec=spec,
                gpu_id=gpu_id,
                args=args,
                runtime=runtime,
                bench_dir=bench_dir,
                output_dir=output_dir,
                meta_dir=meta_dir,
                logger=logger,
            )
            with results_lock:
                results.append(result)
        finally:
            spec_queue.task_done()


def main() -> None:
    args = parse_args()
    bench_dir = Path(args.bench_dir).resolve()
    model_tag = _model_tag(args.model_name)
    output_dir = Path(args.result_base_path).resolve() / model_tag
    meta_dir = output_dir / ".nssm_full"
    meta_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(output_dir)

    runtime = RuntimeSettings(
        generation_min_length=int(args.generation_min_length),
        do_sample=args.do_sample == "True",
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        seed=int(args.seed),
        max_test_samples_override=args.max_test_samples,
    )

    task_list = _parse_csv(args.task_list)
    length_list = [int(item) for item in _parse_csv(args.length_list)]
    gpu_list = _parse_csv(args.gpu_list)
    if not args.dry_run:
        preflight_cuda(gpu_list)

    specs = iter_specs(bench_root=bench_dir, task_list=task_list, length_list=length_list)
    scheduled_specs = _schedule_specs(specs)
    manifest = {
        "bench_dir": str(bench_dir),
        "output_dir": str(output_dir),
        "task_list": task_list,
        "length_list": length_list,
        "gpu_list": gpu_list,
        "runtime": asdict(runtime),
        "specs": [asdict(spec) for spec in scheduled_specs],
    }
    with (meta_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    write_job_plan(meta_dir=meta_dir, specs=scheduled_specs, args=args)

    if args.dry_run:
        risk_counts = Counter(_risk_level(spec) for spec in scheduled_specs)
        logger.info(
            "Dry run complete: total_jobs=%d safe=%d balanced=%d fast=%d",
            len(scheduled_specs),
            risk_counts.get("safe", 0),
            risk_counts.get("balanced", 0),
            risk_counts.get("fast", 0),
        )
        return

    spec_queue: "queue.Queue[DatasetRunSpec]" = queue.Queue()
    pending_specs: List[DatasetRunSpec] = []
    for spec in scheduled_specs:
        if not args.overwrite:
            validation = inspect_output(
                result_dir=output_dir,
                spec=spec,
                runtime=runtime,
                meta_dir=meta_dir,
            )
            if should_skip_existing(
                validation_status=validation.status,
                strict_completeness=args.strict_completeness,
                allow_degraded=args.allow_degraded,
            ):
                logger.info("skip existing %s status=%s", spec.job_id, validation.status)
                continue
            if not args.resume_missing and validation.status != "missing_output":
                logger.info("rerun %s status=%s", spec.job_id, validation.status)
        pending_specs.append(spec)
        spec_queue.put(spec)

    logger.info("Pending dataset jobs: %d / %d", len(pending_specs), len(scheduled_specs))
    logger.info("Tasks=%s Lengths=%s GPUs=%s", task_list, length_list, gpu_list)

    results: List[Dict[str, Any]] = []
    results_lock = threading.Lock()
    threads: List[threading.Thread] = []
    for gpu_id in gpu_list:
        thread = threading.Thread(
            target=worker,
            args=(gpu_id, args, runtime, bench_dir, output_dir, meta_dir, spec_queue, results, results_lock, logger),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    success_clean = sum(1 for item in results if item["final_status"] == "success_clean")
    success_degraded = sum(1 for item in results if item["final_status"] == "success_degraded")
    logger.info(
        "Run complete: clean=%d degraded=%d total_attempted=%d total_specs=%d",
        success_clean,
        success_degraded,
        len(results),
        len(scheduled_specs),
    )


if __name__ == "__main__":
    main()
