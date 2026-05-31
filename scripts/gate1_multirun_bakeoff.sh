#!/usr/bin/env bash
# Gate 1: multirun K-stage bakeoff on Qwen2.5-7B-Instruct.
#
# Four arms × 50 cases (25 wrong / 25 correct, seed 42). Same global prompt
# (meta_plan_confirm_natural + gpt4o-mini-helper-v2 + operation_guided + multi_dimension)
# Same note-context. K varies per stage. Temperatures: 0.0 for K=1 stages, 0.3 for K>1.
#
# Pre-flight: Qwen2.5-7B-Instruct must be loaded on vLLM port 8003.
#   bash scripts/start_qwen25_vllm.sh   # or the qwen2.5 path in scripts/start_local_vllm.sh
#
# Output:
#   runs/selfdetect_raicl_verdict/qwen2.5-7b-instruct_input-qwen2.5-7b-instruct_nw25_nc25_seed42_<...>/
#
# Each arm writes its own folder; arm tags are encoded in the run-id slug via dk/gk/vk.

set -euo pipefail

PORT="${PORT:-8003}"
CONCURRENCY="${CONCURRENCY:-6}"
N_WRONG="${N_WRONG:-25}"
N_CORRECT="${N_CORRECT:-25}"
SEED="${SEED:-42}"
NOTE_CTX="${NOTE_CTX:-dynamic_spans}"
PYTHON_BIN="${PYTHON_BIN:-/home/ra/Projects/llm-ehr-hallucination/.venv/bin/python}"

COMMON=(
  scripts/run_selfdetect_raicl_verdict.py
  --port "$PORT"
  --concurrency "$CONCURRENCY"
  --n-wrong "$N_WRONG"
  --n-correct "$N_CORRECT"
  --seed "$SEED"
  --input-model qwen2.5-7b-instruct
  --run-model-tag qwen2.5-7b-instruct
  --served-model-contains qwen2
  --det-prompt meta_plan_confirm_natural
  --det-parse-backend gpt4o-mini-helper-v2
  --correction-prompt operation_guided
  --verdict-prompt multi_dimension
  --note-context "$NOTE_CTX"
  --judge
)

echo "=== Arm 1: K=1 baseline ==="
"$PYTHON_BIN" "${COMMON[@]}" \
  --detect-k 1 --gen-k 1 --verdict-k 1 \
  --det-temperature 0.0 --correction-temperature 0.0 --verdict-temperature 0.0

echo "=== Arm 2: K=3 detection only ==="
"$PYTHON_BIN" "${COMMON[@]}" \
  --detect-k 3 --gen-k 1 --verdict-k 1 \
  --detect-temperature-multirun 0.3 \
  --det-temperature 0.0 --correction-temperature 0.0 --verdict-temperature 0.0

echo "=== Arm 3: K=3 detection + K=3 verdict ==="
"$PYTHON_BIN" "${COMMON[@]}" \
  --detect-k 3 --gen-k 1 --verdict-k 3 \
  --detect-temperature-multirun 0.3 \
  --det-temperature 0.0 --correction-temperature 0.0 --verdict-temperature 0.3

echo "=== Arm 4: K=3 detection + K=3 generation ==="
"$PYTHON_BIN" "${COMMON[@]}" \
  --detect-k 3 --gen-k 3 --verdict-k 1 \
  --detect-temperature-multirun 0.3 \
  --correction-temperature-multirun 0.3 \
  --det-temperature 0.0 --correction-temperature 0.0 --verdict-temperature 0.0

echo "=== Done. Compare summaries with scripts/gate1_summarize.py ==="
