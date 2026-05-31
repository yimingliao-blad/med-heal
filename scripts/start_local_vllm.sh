#!/usr/bin/env bash
set -euo pipefail

SOURCE_REPO="${MED_HEAL_SOURCE_REPO:-/home/ra/Projects/llm-ehr-hallucination}"
MODEL_PATH="${1:?usage: start_local_vllm.sh MODEL_PATH [SERVED_MODEL_NAME]}"
SERVED_MODEL_NAME="${2:-$(basename "$MODEL_PATH")}"
PORT="${PORT:-8003}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DTYPE="${DTYPE:-float16}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$SOURCE_REPO"
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --dtype "$DTYPE"
