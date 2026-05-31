#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8003}"
CONCURRENCY="${CONCURRENCY:-6}"
N_WRONG="${N_WRONG:-100}"
N_CORRECT="${N_CORRECT:-100}"
SEED="${SEED:-42}"
INPUT_MODEL="${INPUT_MODEL:-qwen2.5-7b-instruct}"
RUN_MODEL_TAG="${RUN_MODEL_TAG:-qwen2.5-7b-instruct}"
SERVED_MODEL_CONTAINS="${SERVED_MODEL_CONTAINS:-}"
PYTHON_BIN="${PYTHON_BIN:-/home/ra/Projects/llm-ehr-hallucination/.venv/bin/python}"
JUDGE_FLAG="${JUDGE_FLAG:---judge}"

COMMON=(
  scripts/run_selfdetect_raicl_verdict.py
  --port "$PORT"
  --concurrency "$CONCURRENCY"
  --n-wrong "$N_WRONG"
  --n-correct "$N_CORRECT"
  --seed "$SEED"
  --input-model "$INPUT_MODEL"
  --run-model-tag "$RUN_MODEL_TAG"
  --det-prompt meta_plan_confirm_natural
  --det-parse-backend gpt4o-mini-helper-v2
  --correction-prompt operation_guided
  --verdict-prompt multi_dimension
  --verdict-k 1
  --det-temperature 0.0
  --correction-temperature 0.0
  --verdict-temperature 0.0
)

if [[ -n "$SERVED_MODEL_CONTAINS" ]]; then
  COMMON+=(--served-model-contains "$SERVED_MODEL_CONTAINS")
fi
if [[ -n "$JUDGE_FLAG" ]]; then
  COMMON+=($JUDGE_FLAG)
fi

"$PYTHON_BIN" "${COMMON[@]}" --note-context first18k
"$PYTHON_BIN" "${COMMON[@]}" --note-context dynamic_spans
