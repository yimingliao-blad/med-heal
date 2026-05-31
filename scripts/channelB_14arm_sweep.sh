#!/usr/bin/env bash
# 3B-1: Channel-B 14-arm sweep at small size for Qwen2.5.
#
# Single-stage pipeline (zero-shot wrong answer -> correction with same-patient
# spans). Same retrieval (gtr_q_answer, k=5, agreement scoring) across all arms.
# Only the correction prompt varies.
#
# After the user reviews results, narrow to the top 3-4 arms and rerun at 100w/100c
# for confirmation; the existing 218-case run is the upper-size baseline.
#
# Pre-flight: Qwen2.5-7B-Instruct must be loaded on vLLM port 8003.
# Estimated runtime: ~15-20 min on Qwen2.5 at concurrency 8.
# Estimated oracle cost: ~$5.60 (14 arms x 50 cases x 1 judge call at gpt-4o ~$0.008).
#
# Output:
#   runs/retrieval_correction/qwen25_gtr_q_answer_nw25_nc25_seed42_<arm-tag>/

set -euo pipefail

PORT="${PORT:-8003}"
CONCURRENCY="${CONCURRENCY:-8}"
RETRIEVAL_WORKERS="${RETRIEVAL_WORKERS:-4}"
N_WRONG="${N_WRONG:-25}"
N_CORRECT="${N_CORRECT:-25}"
SEED="${SEED:-42}"
K="${K:-5}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-gtr_q_answer}"
PYTHON_BIN="${PYTHON_BIN:-/home/ra/Projects/llm-ehr-hallucination/.venv/bin/python}"

ARMS=(
  evidence_only
  taxonomy_evidence
  conservative_keep_gate
  quote_then_revise
  minimal_patch
  answer_from_evidence_then_compare
  contradiction_first
  omission_first
  focus_first
  claim_table_private
  error_type_router
  no_new_entities
  abstain_if_uncertain
  oracle_error_description
)

"$PYTHON_BIN" scripts/qwen25_retrieval_correction_quicktest.py \
  --port "$PORT" \
  --concurrency "$CONCURRENCY" \
  --retrieval-workers "$RETRIEVAL_WORKERS" \
  --n-wrong "$N_WRONG" \
  --n-correct "$N_CORRECT" \
  --seed "$SEED" \
  --k "$K" \
  --retrieval-mode "$RETRIEVAL_MODE" \
  --arms "${ARMS[@]}" \
  --judge

echo "=== 3B-1 done. Compare arms in summary.json under runs/retrieval_correction/. ==="
