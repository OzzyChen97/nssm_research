#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
BENCH_DIR="${REPO_DIR}/MMLongBench"

CONDA_ENV="${CONDA_ENV:-mmlongbench}"
CONDA_BIN="${CONDA_BIN:-/workspace/home/miniconda3/bin/conda}"
MODEL_NAME="${MODEL_NAME:-/workspace/helandi/model/Qwen-7b/Qwen2.5-VL-7B-Instruct/}"
NSSM_CONFIG="${NSSM_CONFIG:-${ROOT_DIR}/configs/exp_mmlongbench_vision_4gpu.yaml}"
TASK_LIST="${TASK_LIST:-vrag,vh,mm_niah_image,icl,docqa}"
LENGTH_LIST="${LENGTH_LIST:-8,16,32,64,128}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
GPU_GROUP_SIZE="${GPU_GROUP_SIZE:-1}"
PROCESSES_PER_GPU="${PROCESSES_PER_GPU:-1}"
NUM_WORKERS="${NUM_WORKERS:-24}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-16}"
TEST_FILE_ROOT="${TEST_FILE_ROOT:-${REPO_DIR}/data/mmlb_data}"
IMAGE_FILE_ROOT="${IMAGE_FILE_ROOT:-${REPO_DIR}/data/mmlb_image}"
RESULT_BASE_PATH="${RESULT_BASE_PATH:-${ROOT_DIR}/outputs/mmlongbench_vision_4gpu}"
DOCQA_LLM_JUDGE="${DOCQA_LLM_JUDGE:-False}"
PYTHON_EXEC="${PYTHON_EXEC:-python}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${ROOT_DIR}/outputs/torchinductor_cache}"

mkdir -p "${RESULT_BASE_PATH}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

export OMP_NUM_THREADS
export TOKENIZERS_PARALLELISM
export PYTORCH_CUDA_ALLOC_CONF
export TORCHINDUCTOR_CACHE_DIR

"${CONDA_BIN}" run -n "${CONDA_ENV}" python "${ROOT_DIR}/scripts/run_mmlongbench_tasks.py" \
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
  --processes_per_gpu "${PROCESSES_PER_GPU}" \
  "$@"
