"""Evaluation helpers for NSSM."""

from .metrics import aggregate_metrics, compute_accuracy, measure_peak_vram_gb
from .mmlongbench_loader import MMLongBenchSample, load_mmlongbench_samples


def build_report(*args, **kwargs):
    from .mmlongbench_report import build_report as _build_report

    return _build_report(*args, **kwargs)

__all__ = [
    "MMLongBenchSample",
    "load_mmlongbench_samples",
    "compute_accuracy",
    "measure_peak_vram_gb",
    "aggregate_metrics",
    "build_report",
]
