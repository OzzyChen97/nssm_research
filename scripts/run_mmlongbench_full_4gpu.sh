#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
BENCH_DIR="${REPO_DIR}/MMLongBench"

CONDA_ENV="${CONDA_ENV:-mmlongbench}"
CONDA_BIN="${CONDA_BIN:-/workspace/home/miniconda3/bin/conda}"
MODEL_NAME="${MODEL_NAME:-/workspace/helandi/model/Qwen-7b/Qwen2.5-VL-7B-Instruct/}"
NSSM_CONFIG="${NSSM_CONFIG:-${ROOT_DIR}/configs/exp_mmlongbench_full_4gpu_a6000.yaml}"
TASK_LIST="${TASK_LIST:-vrag,vh,mm_niah_text,mm_niah_image,icl,summ,docqa}"
LENGTH_LIST="${LENGTH_LIST:-8,16,32,64,128}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
NUM_WORKERS="${NUM_WORKERS:-32}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-16}"
TEST_FILE_ROOT="${TEST_FILE_ROOT:-${REPO_DIR}/data/mmlb_data}"
IMAGE_FILE_ROOT="${IMAGE_FILE_ROOT:-${REPO_DIR}/data/mmlb_image}"
RESULT_BASE_PATH="${RESULT_BASE_PATH:-${ROOT_DIR}/outputs/mmlongbench_full_4gpu_a6000}"
DOCQA_LLM_JUDGE="${DOCQA_LLM_JUDGE:-False}"
PYTHON_EXEC="${PYTHON_EXEC:-python}"
AUTO_JUDGE="${AUTO_JUDGE:-0}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-gpt-4o-2024-11-20}"
JUDGE_API_BASE_URL="${JUDGE_API_BASE_URL:-}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS
export TOKENIZERS_PARALLELISM
export PYTORCH_CUDA_ALLOC_CONF

MODEL_TAG="$(basename "${MODEL_NAME%/}")"
MODEL_OUTPUT_DIR="${RESULT_BASE_PATH}/${MODEL_TAG}"
REPORT_DIR="${MODEL_OUTPUT_DIR}/report"
DRY_RUN=0

for arg in "$@"; do
  if [[ "${arg}" == "--dry_run" ]]; then
    DRY_RUN=1
    break
  fi
done

mkdir -p "${RESULT_BASE_PATH}"

"${CONDA_BIN}" run -n "${CONDA_ENV}" python "${ROOT_DIR}/scripts/run_full_mmlongbench.py" \
  --bench_dir "${BENCH_DIR}" \
  --model_name "${MODEL_NAME}" \
  --task_list "${TASK_LIST}" \
  --length_list "${LENGTH_LIST}" \
  --gpu_list "${GPU_LIST}" \
  --result_base_path "${RESULT_BASE_PATH}" \
  --test_file_root "${TEST_FILE_ROOT}" \
  --image_file_root "${IMAGE_FILE_ROOT}" \
  --python_exec "${PYTHON_EXEC}" \
  --num_workers "${NUM_WORKERS}" \
  --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS}" \
  --docqa_llm_judge "${DOCQA_LLM_JUDGE}" \
  --nssm_config "${NSSM_CONFIG}" \
  --memory_profile a6000_4gpu \
  --strict_completeness \
  --resume_missing \
  "$@"

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

if [[ "${AUTO_JUDGE}" == "1" ]]; then
  JUDGE_CMD=(
    "${CONDA_BIN}" run -n "${CONDA_ENV}" python "${ROOT_DIR}/scripts/judge_mmlongbench_summ.py"
    --bench_dir "${BENCH_DIR}"
    --result_dir "${MODEL_OUTPUT_DIR}"
    --data_base_path "${TEST_FILE_ROOT}"
    --model_name "${JUDGE_MODEL_NAME}"
  )
  if [[ -n "${JUDGE_API_BASE_URL}" ]]; then
    JUDGE_CMD+=(--api_base_url "${JUDGE_API_BASE_URL}")
  fi
  "${JUDGE_CMD[@]}"
fi

"${CONDA_BIN}" run -n "${CONDA_ENV}" python -m src.eval.mmlongbench_report \
  --result_dir "${MODEL_OUTPUT_DIR}" \
  --output_dir "${REPORT_DIR}" \
  --bench_root "${BENCH_DIR}"
