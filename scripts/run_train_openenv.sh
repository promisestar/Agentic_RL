#!/usr/bin/env bash
# Run OpenEnv GRPO training (train_openenv.py) with the recommended defaults.
#
# Usage (from anywhere):
#   bash scripts/run_train_openenv.sh
#   bash scripts/run_train_openenv.sh --max_steps 50          # override / append args
#   CUDA_VISIBLE_DEVICES=3 bash scripts/run_train_openenv.sh # pick another GPU
#   LOG_TO_FILE=0 bash scripts/run_train_openenv.sh          # disable log file
#   LOG_PATH=/tmp/run.log bash scripts/run_train_openenv.sh  # custom log path
#
# Prerequisites:
#   uv sync --extra train

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
# Single-GPU by default. Without this, PyTorch/Unsloth sees every GPU on the
# node and may create CUDA contexts on all of them (looks like "6-card" usage).
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ECOM_RLVE_LLM_BACKEND=openai
export ECOM_RLVE_LLM_BASE_URL=http://localhost:8000/v1
export ECOM_RLVE_LLM_MODEL=Qwen3.5-4B

echo "[run_train_openenv] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Compose the optional --log_path / --log_to_file flags.
# - Default: tee training logs to <output_dir>/train.log via --log_to_file.
# - LOG_TO_FILE=0 disables log file output (only console).
# - LOG_PATH=/some/path.log overrides the default location.
LOG_ARGS=()
if [[ "${LOG_TO_FILE:-1}" == "1" || "${LOG_TO_FILE:-1}" == "true" ]]; then
  if [[ -n "${LOG_PATH:-}" ]]; then
    LOG_ARGS+=(--log_path "${LOG_PATH}")
  else
    LOG_ARGS+=(--log_to_file)
  fi
fi

uv run python scripts/train_openenv.py \
  --model /data_160TB/2024/hanshuaiteng/LLM/Qwen3.5-4B/ \
  --collection C1 \
  --max_steps 300 \
  --lora_rank 16 \
  --num_generations 4 \
  --n_prompts 1000 \
  --load_in_4bit \
  --output_dir outputs/ecomrlve_grpo \
  "${LOG_ARGS[@]}" \
  "$@"
