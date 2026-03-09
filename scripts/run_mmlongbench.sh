#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/configs/exp_mmlongbench_128k.yaml}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

NUM_PROCESSES="${NUM_PROCESSES:-2}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision "${MIXED_PRECISION}" \
  -m src.pipeline.inference_engine \
  --config "${CONFIG_PATH}" \
  "${@:2}"

