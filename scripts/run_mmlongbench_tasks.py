#!/usr/bin/env python3
"""Run visual MMLongBench tasks through NSSM without multiprocessing.Manager."""

from __future__ import annotations

import argparse
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from itertools import product
from pathlib import Path
from typing import List, Sequence, Tuple

import torch


TASK_CONFIG_MAP = {
    "vrag": "vrag_all.yaml",
    "vh": "vh_all.yaml",
    "mm_niah_image": "mm_niah_image_all.yaml",
    "icl": "icl_all.yaml",
    "docqa": "docqa_all.yaml",
}


def _collect_cuda_diagnostics(gpu_list: Sequence[str]) -> List[str]:
    diagnostics: List[str] = []
    diagnostics.append(f"torch={torch.__version__}")
    diagnostics.append(f"torch.version.cuda={torch.version.cuda}")

    device_nodes = sorted(path.name for path in Path("/dev").glob("nvidia*"))
    diagnostics.append(f"/dev nodes={device_nodes}")

    for special_node in ("nvidiactl", "nvidia-uvm"):
        path = Path("/dev") / special_node
        if not path.exists():
            diagnostics.append(f"{path} missing")
            continue
        try:
            fd = os.open(path, os.O_RDWR)
        except OSError as exc:
            diagnostics.append(f"{path} open failed: {exc.__class__.__name__}: {exc}")
        else:
            os.close(fd)
            diagnostics.append(f"{path} open ok")

    missing_gpu_nodes: List[str] = []
    blocked_gpu_nodes: List[str] = []
    for gpu_id in gpu_list:
        if not gpu_id.isdigit():
            continue
        gpu_path = Path(f"/dev/nvidia{gpu_id}")
        if not gpu_path.exists():
            missing_gpu_nodes.append(str(gpu_path))
            continue
        try:
            fd = os.open(gpu_path, os.O_RDWR)
        except OSError as exc:
            blocked_gpu_nodes.append(f"{gpu_path}: {exc.__class__.__name__}: {exc}")
        else:
            os.close(fd)
    if missing_gpu_nodes:
        diagnostics.append(f"missing requested gpu device nodes={missing_gpu_nodes}")
    if blocked_gpu_nodes:
        diagnostics.append(f"requested gpu device open failures={blocked_gpu_nodes}")

    proc_gpu_info = Path("/proc/driver/nvidia/gpus")
    if proc_gpu_info.exists():
        proc_gpus = sorted(path.name for path in proc_gpu_info.iterdir() if path.is_dir())
        diagnostics.append(f"/proc/driver/nvidia/gpus={proc_gpus}")

    return diagnostics


def _model_tag(model_name: str) -> str:
    return os.path.basename(os.path.normpath(model_name))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local NSSM task manager for MMLongBench.")
    parser.add_argument("--bench_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task_list", type=str, default="vrag,vh,mm_niah_image,icl,docqa")
    parser.add_argument("--length_list", type=str, default="8,16,32,64,128")
    parser.add_argument("--gpu_list", type=str, default="0,1,2,3")
    parser.add_argument("--result_base_path", type=str, required=True)
    parser.add_argument("--test_file_root", type=str, required=True)
    parser.add_argument("--image_file_root", type=str, required=True)
    parser.add_argument("--python_exec", type=str, default=sys.executable)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--preprocessing_num_workers", type=int, default=16)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--docqa_llm_judge", type=str, choices=["True", "False"], default="False")
    parser.add_argument("--nssm_config", type=str, required=True)
    parser.add_argument("--processes_per_gpu", type=int, default=1)
    return parser.parse_args()


def build_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("nssm_mmlongbench_runner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(output_dir / "eval_log.log")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def preflight_cuda(gpu_list: Sequence[str]) -> None:
    if not torch.cuda.is_available():
        diagnostics = "; ".join(_collect_cuda_diagnostics(gpu_list))
        raise RuntimeError(
            "CUDA is not available in the current `mmlongbench` environment. "
            "This runner is intended for GPU execution and will stop before launching tasks. "
            f"Diagnostics: {diagnostics}"
        )
    device_count = torch.cuda.device_count()
    if device_count <= 0:
        raise RuntimeError("PyTorch reports zero CUDA devices.")

    invalid_ids: List[str] = []
    for gpu_id in gpu_list:
        if not gpu_id.isdigit():
            continue
        if int(gpu_id) >= device_count:
            invalid_ids.append(gpu_id)
    if invalid_ids:
        raise RuntimeError(
            f"Requested GPU ids {invalid_ids} exceed visible CUDA device count {device_count}."
        )


def build_command(args: argparse.Namespace, task: str, length_k: int, output_dir: Path) -> List[str]:
    config_name = TASK_CONFIG_MAP[task]
    command = [
        args.python_exec,
        "eval.py",
        "--config",
        f"configs/{config_name}",
        "--model_name_or_path",
        args.model_name,
        "--output_dir",
        str(output_dir),
        "--test_file_root",
        args.test_file_root,
        "--image_file_root",
        args.image_file_root,
        "--num_workers",
        str(args.num_workers),
        "--preprocessing_num_workers",
        str(args.preprocessing_num_workers),
        "--test_length",
        str(length_k),
        "--docqa_llm_judge",
        args.docqa_llm_judge,
        "--use_nssm",
        "--nssm_config",
        args.nssm_config,
    ]
    if args.max_test_samples is not None:
        command.extend(["--max_test_samples", str(args.max_test_samples)])
    if args.overwrite:
        command.append("--overwrite")
    return command


def worker(
    worker_name: str,
    gpu_id: str,
    bench_dir: Path,
    output_dir: Path,
    task_queue: "queue.Queue[Tuple[str, int]]",
    results: List[Tuple[str, int, bool]],
    result_lock: threading.Lock,
    logger: logging.Logger,
    args: argparse.Namespace,
) -> None:
    while True:
        try:
            task, length_k = task_queue.get_nowait()
        except queue.Empty:
            logger.info("%s: no more tasks, exiting", worker_name)
            return

        task_log_dir = output_dir / f"{task}_{length_k}"
        task_log_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = task_log_dir / "stdout.log"
        stderr_log = task_log_dir / "error.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

        command = build_command(args=args, task=task, length_k=length_k, output_dir=output_dir)
        logger.info("%s: start task=%s length=%s gpu=%s", worker_name, task, length_k, gpu_id)
        logger.info("%s: command=%s", worker_name, " ".join(command))
        start_time = time.time()
        success = False
        try:
            with stdout_log.open("w", encoding="utf-8") as stdout_handle, stderr_log.open(
                "w", encoding="utf-8"
            ) as stderr_handle:
                completed = subprocess.run(
                    command,
                    cwd=bench_dir,
                    env=env,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    check=False,
                    text=True,
                )
            success = completed.returncode == 0
            elapsed = time.time() - start_time
            if success:
                logger.info(
                    "%s: finished task=%s length=%s in %.1fs",
                    worker_name,
                    task,
                    length_k,
                    elapsed,
                )
            else:
                logger.error(
                    "%s: failed task=%s length=%s returncode=%s",
                    worker_name,
                    task,
                    length_k,
                    completed.returncode,
                )
        except Exception as exc:  # pragma: no cover - defensive runtime path
            logger.exception("%s: exception while running %s@%s: %s", worker_name, task, length_k, exc)
        finally:
            with result_lock:
                results.append((task, length_k, success))
            task_queue.task_done()


def main() -> None:
    args = parse_args()
    bench_dir = Path(args.bench_dir).resolve()
    model_tag = _model_tag(args.model_name)
    output_dir = Path(args.result_base_path).resolve() / model_tag
    logger = build_logger(output_dir)

    task_list = [item for item in args.task_list.split(",") if item]
    length_list = sorted([int(item) for item in args.length_list.split(",") if item], reverse=True)
    gpu_list = [item for item in args.gpu_list.split(",") if item]
    preflight_cuda(gpu_list)

    task_queue: "queue.Queue[Tuple[str, int]]" = queue.Queue()
    for length_k, task in product(length_list, task_list):
        task_queue.put((task, length_k))
    total_tasks = task_queue.qsize()

    logger.info("Bench dir: %s", bench_dir)
    logger.info("Output dir: %s", output_dir)
    logger.info("Tasks: %s", task_list)
    logger.info("Lengths: %s", length_list)
    logger.info("GPUs: %s", gpu_list)
    logger.info("Processes per GPU: %s", args.processes_per_gpu)
    logger.info("Total task groups: %s", total_tasks)

    results: List[Tuple[str, int, bool]] = []
    result_lock = threading.Lock()
    threads: List[threading.Thread] = []
    for gpu_id in gpu_list:
        for slot_idx in range(args.processes_per_gpu):
            worker_name = f"gpu{gpu_id}-slot{slot_idx}"
            thread = threading.Thread(
                target=worker,
                args=(worker_name, gpu_id, bench_dir, output_dir, task_queue, results, result_lock, logger, args),
                daemon=True,
            )
            thread.start()
            threads.append(thread)

    for thread in threads:
        thread.join()

    success_count = sum(1 for _, _, success in results if success)
    logger.info("Finished %s/%s task groups successfully.", success_count, total_tasks)
    if success_count != total_tasks:
        failed = [(task, length_k) for task, length_k, success in results if not success]
        logger.info("Failed task groups: %s", failed)


if __name__ == "__main__":
    main()
