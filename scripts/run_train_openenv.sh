#!/usr/bin/env bash
# Run OpenEnv GRPO training (train_openenv.py) with the recommended defaults.
#
# Usage (from anywhere):
#   bash scripts/run_train_openenv.sh
#   bash scripts/run_train_openenv.sh --max_steps 50          # override / append args
#   CUDA_VISIBLE_DEVICES=3 bash scripts/run_train_openenv.sh # pick another GPU
#   LOG_TO_FILE=0 bash scripts/run_train_openenv.sh          # disable log file
#   LOG_PATH=/tmp/run.log bash scripts/run_train_openenv.sh  # custom log path
#   DUMP_PROMPTS=0 bash scripts/run_train_openenv.sh         # disable prompt dump
#   DUMP_PATH=/tmp/p.jsonl bash scripts/run_train_openenv.sh # custom dump file
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
# - Default: tee training logs to <run_dir>/train.log via --log_to_file.
# - Each run creates a timestamped subdirectory under --output_dir, e.g.
#   outputs/ecomrlve_grpo/20260818_093012/{train.log,prompts.jsonl,final,...}.
# - Pass --no_timestamp (via "$@") to write directly into --output_dir.
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

# Compose the optional --dump_prompts / --dump_path flags.
# - Default: append every sampled prompt + matching reward result to
#   <run_dir>/prompts.jsonl. Useful for verifying the environment
#   actually generates sensible user messages and hidden goals.
# - DUMP_PROMPTS=0 disables the dump.
# - DUMP_PATH=/some/path.jsonl overrides the default location.
DUMP_ARGS=()
if [[ "${DUMP_PROMPTS:-1}" == "1" || "${DUMP_PROMPTS:-1}" == "true" ]]; then
  if [[ -n "${DUMP_PATH:-}" ]]; then
    DUMP_ARGS+=(--dump_path "${DUMP_PATH}")
  else
    DUMP_ARGS+=(--dump_prompts)
  fi
fi

uv run python scripts/train_openenv.py \
  --model /data_160TB/2024/hanshuaiteng/LLM/Qwen3.5-4B/ \
  --max_seq_length 8192 \
  --collection C1 \
  --max_steps 300 \
  --lora_rank 16 \
  --num_generations 4 \
  --n_prompts 1000 \
  --load_in_4bit \
  --output_dir outputs/ecomrlve_grpo \
  "${LOG_ARGS[@]}" \
  "${DUMP_ARGS[@]}" \
  "$@"
