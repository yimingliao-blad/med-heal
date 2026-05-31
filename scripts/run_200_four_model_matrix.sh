#!/usr/bin/env bash
set -euo pipefail

ROOT="${MED_HEAL_SOURCE_REPO:-/home/ra/Projects/llm-ehr-hallucination}"
PORT="${PORT:-8003}"
CONCURRENCY="${CONCURRENCY:-6}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DTYPE="${DTYPE:-float16}"
SEED="${SEED:-42}"
N_WRONG="${N_WRONG:-100}"
N_CORRECT="${N_CORRECT:-100}"
JUDGE_FLAG="${JUDGE_FLAG:---judge}"
WAIT_SECS="${WAIT_SECS:-5}"
RUN_FROM="${RUN_FROM:-}"
STARTED=""

# Format: run_tag|model_path|served_name|step8_input_model|served_contains
# Edit this list if you want a different four-model set.
MODELS=(
  "biomistral-7b|$ROOT/models/biomistral-7b|biomistral-7b|biomistral-7b|biomistral"
  "qwen3-8b|$ROOT/models/qwen3-8b|qwen3-8b|qwen3-8b|qwen3"
  "deepseek-r1-distill-llama-8b|$ROOT/models/DeepSeek-R1-Distill-Llama-8B|deepseek-r1-distill-llama-8b|deepseek-r1-distill-llama-8b|deepseek"
  "llama-3.1-8b-instruct|$ROOT/models/Llama-3.1-8B-Instruct|llama-3.1-8b-instruct|llama-3.1-8b-instruct|llama"
)

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -INT "-$SERVER_PID" 2>/dev/null || kill -INT "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait_for_server() {
  local served_contains="$1"
  for _ in $(seq 1 180); do
    if /home/ra/Projects/llm-ehr-hallucination/.venv/bin/python scripts/wait_for_vllm.py \
      --port "$PORT" \
      --served-model-contains "$served_contains" \
      --once; then
      return 0
    fi
    sleep "$WAIT_SECS"
  done
  return 1
}

for spec in "${MODELS[@]}"; do
  IFS='|' read -r tag model_path served_name input_model served_contains <<< "$spec"
  if [[ -n "$RUN_FROM" && -z "$STARTED" && "$tag" != "$RUN_FROM" ]]; then
    echo "=== Skipping $tag before RUN_FROM=$RUN_FROM ==="
    continue
  fi
  STARTED=1
  echo "=== Starting $tag from $model_path ==="
  PORT="$PORT" MAX_MODEL_LEN="$MAX_MODEL_LEN" GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" DTYPE="$DTYPE" \
    setsid bash scripts/start_local_vllm.sh "$model_path" "$served_name" &
  SERVER_PID=$!
  wait_for_server "$served_contains"

  echo "=== Running 200-case context matrix for $tag ==="
  PORT="$PORT" CONCURRENCY="$CONCURRENCY" N_WRONG="$N_WRONG" N_CORRECT="$N_CORRECT" SEED="$SEED" \
    INPUT_MODEL="$input_model" RUN_MODEL_TAG="$tag" SERVED_MODEL_CONTAINS="$served_contains" JUDGE_FLAG="$JUDGE_FLAG" \
    bash scripts/run_200_context_matrix.sh

  cleanup
  SERVER_PID=""
  sleep 10
done
