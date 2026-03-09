"""Evaluation metrics for NSSM research experiments."""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Iterable, List

import numpy as np
import torch


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def compute_accuracy(prediction: str, answer: Any) -> float:
    """Compute exact-match style accuracy with list-answer support."""

    pred = _normalize_text(str(prediction))
    if isinstance(answer, list):
        normalized = [_normalize_text(str(a)) for a in answer]
        return float(pred in normalized)
    return float(pred == _normalize_text(str(answer)))


def reset_peak_vram_stats() -> None:
    """Reset CUDA peak-memory tracking across all visible devices."""

    if not torch.cuda.is_available():
        return
    for idx in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(idx)


def measure_peak_vram_gb() -> float:
    """Return total peak VRAM (GB) across all CUDA devices."""

    if not torch.cuda.is_available():
        return 0.0
    total_bytes = 0
    for idx in range(torch.cuda.device_count()):
        total_bytes += torch.cuda.max_memory_allocated(idx)
    return float(total_bytes / (1024**3))


def measure_latency_ms(start_time: float, end_time: float) -> float:
    """Compute wall-clock latency in milliseconds."""

    return float((end_time - start_time) * 1000.0)


def aggregate_metrics(rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate per-example metrics into corpus-level means."""

    rows = list(rows)
    if not rows:
        return {
            "accuracy": 0.0,
            "latency_ms": 0.0,
            "vram_peak_gb": 0.0,
        }

    acc = np.mean([float(item.get("accuracy", 0.0)) for item in rows])
    latency = np.mean([float(item.get("latency_ms", 0.0)) for item in rows])
    vram = np.mean([float(item.get("vram_peak_gb", 0.0)) for item in rows])
    return {
        "accuracy": float(acc),
        "latency_ms": float(latency),
        "vram_peak_gb": float(vram),
    }


class LatencyTimer:
    """Small helper timer for stage-wise latency accounting."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self._end: float = 0.0

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> float:
        self._end = time.perf_counter()
        return measure_latency_ms(self._start, self._end)

