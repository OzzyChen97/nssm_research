#!/usr/bin/env python3
"""Scan existing NSSM outputs and print missing / degraded jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.mmlongbench_manifest import RuntimeSettings
from src.eval.mmlongbench_report import build_report


def _parse_csv(raw: str):
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan full MMLongBench outputs.")
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--bench_root", type=str, default=None)
    parser.add_argument("--task_list", type=str, default=None)
    parser.add_argument("--length_list", type=str, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_report(
        result_dir=Path(args.result_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        bench_root=Path(args.bench_root).resolve() if args.bench_root else None,
        task_list=_parse_csv(args.task_list) if args.task_list else None,
        length_list=[int(item) for item in _parse_csv(args.length_list)] if args.length_list else None,
        runtime=(
            RuntimeSettings(max_test_samples_override=args.max_test_samples)
            if args.max_test_samples is not None
            else None
        ),
    )
    print(json.dumps(summary["overview"]["status_counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
