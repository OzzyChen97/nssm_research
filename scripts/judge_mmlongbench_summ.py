#!/usr/bin/env python3
"""Run official summarization judge for NSSM full MMLongBench outputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import List

import openai

from src.eval.mmlongbench_manifest import DEFAULT_LENGTHS, RuntimeSettings, iter_specs
from src.eval.mmlongbench_validation import find_judged_output, find_raw_output


def _load_official_eval_module(bench_dir: Path):
    module_path = bench_dir / "scripts" / "eval_gpt4_summ.py"
    spec = importlib.util.spec_from_file_location("official_eval_gpt4_summ", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load official summarization judge from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_manifest(result_dir: Path):
    manifest_path = result_dir / ".nssm_full" / "manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge NSSM summarization outputs with official script.")
    parser.add_argument("--bench_dir", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--data_base_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gpt-4o-2024-11-20")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_base_url", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bench_dir = Path(args.bench_dir).resolve()
    result_dir = Path(args.result_dir).resolve()
    manifest = _load_manifest(result_dir=result_dir)
    manifest_runtime = manifest.get("runtime", {})
    runtime = RuntimeSettings(
        max_test_samples_override=(
            args.max_test_samples
            if args.max_test_samples is not None
            else manifest_runtime.get("max_test_samples_override")
        )
    )
    official_eval = _load_official_eval_module(bench_dir=bench_dir)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for summarization judge.")
    client_kwargs = {"api_key": api_key}
    if args.api_base_url:
        client_kwargs["base_url"] = args.api_base_url
    client = openai.OpenAI(**client_kwargs)

    summ_specs = [
        spec
        for spec in iter_specs(
            bench_root=bench_dir,
            task_list=["summ"],
            length_list=manifest.get("length_list", DEFAULT_LENGTHS),
        )
    ]
    pending: List[Path] = []
    for spec in summ_specs:
        raw_output = find_raw_output(result_dir=result_dir, spec=spec, runtime=runtime)
        if raw_output is None:
            continue
        judged_output = find_judged_output(raw_output)
        if judged_output is not None and not args.overwrite:
            continue
        pending.append(raw_output)

    pending = pending[args.shard_idx :: args.num_shards]
    print(f"Pending summarization judge files: {len(pending)}")
    for raw_output in pending:
        judged_output = raw_output.with_name(raw_output.name.replace(".json", "-gpt4eval_o.json"))
        print(f"Judging {raw_output.name}")
        official_eval.check_metrics(
            model=client,
            results_file=str(raw_output),
            output_file=str(judged_output),
            model_name=args.model_name,
            data_base_path=args.data_base_path,
        )


if __name__ == "__main__":
    main()
