#!/usr/bin/env bash
# Self-advancing pipeline driver.
#
# Runs a sequence of stages. Each stage = a command that writes a summary.json.
# Between stages, a GATE validates the just-finished summary (file exists, valid
# JSON, n_cases>0, error rate acceptable). The next stage starts ONLY if the gate
# passes. On any gate failure the pipeline stops and prints why, so a bad stage
# never silently feeds the next.
#
# Add stages by appending to the STAGES array: "label|summary_path|command..."
# Run detached:  nohup bash scripts/pipeline_driver.sh > /tmp/pipeline.log 2>&1 &
# A STATUS file is updated after every stage so progress is auditable any time.

set -uo pipefail

PY="${PY:-/home/ra/Projects/llm-ehr-hallucination/.venv/bin/python}"
ROOT="/home/ra/Projects/med-heal"
RUNS="$ROOT/runs/phase2b_extract_compare"
STATUS="$ROOT/runs/pipeline_status.txt"
MAX_ERR_FRAC="${MAX_ERR_FRAC:-0.05}"   # fail the gate if >5% of cases errored

cd "$ROOT"
: > "$STATUS"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
log()   { echo "[$(stamp)] $*"; echo "[$(stamp)] $*" >> "$STATUS"; }

gate() {
  # gate <summary_path> : returns 0 if the run finished cleanly
  local s="$1"
  if [[ ! -f "$s" ]]; then log "GATE FAIL: missing $s"; return 1; fi
  "$PY" - "$s" "$MAX_ERR_FRAC" <<'PYEOF'
import json, sys
s, maxerr = sys.argv[1], float(sys.argv[2])
try:
    d = json.load(open(s))
except Exception as e:
    print("GATE FAIL: bad json", e); sys.exit(1)
n = d.get("n_cases", 0)
if n <= 0:
    print("GATE FAIL: n_cases<=0"); sys.exit(1)
# error count may live at top-level 'errors' or be absent; treat absent as 0
errs = d.get("errors", 0)
if isinstance(errs, dict):
    errs = sum(errs.values())
frac = (errs or 0) / max(1, n)
if frac > maxerr:
    print(f"GATE FAIL: error fraction {frac:.2f} > {maxerr}"); sys.exit(1)
print(f"GATE PASS: n={n} errors={errs}")
sys.exit(0)
PYEOF
}

run_stage() {
  local label="$1" summary="$2"; shift 2
  log "START  $label"
  "$@"
  local rc=$?
  if [[ $rc -ne 0 ]]; then log "STAGE FAIL: $label exited rc=$rc — stopping pipeline"; exit 1; fi
  if ! gate "$summary"; then log "GATE FAIL after $label — stopping pipeline"; exit 1; fi
  log "DONE   $label  (gate passed)"
}

# ------------------------------------------------------------------
# STAGES — edit this list to define the pipeline. Order matters.
# Each: run_stage "label" "summary.json path" <command...>
# ------------------------------------------------------------------

PORT="${PORT:-8003}"
C="${C:-8}"

# Stage: full k3union detection run (natural x3 @ T=0.7, union flag, planner designs fix).
run_stage "k3union_full" \
  "$RUNS/qwen25_nw-1_nc50_seed42_helper-v2_k3union/summary.json" \
  "$PY" scripts/phase2b_extract_compare_detection.py \
    --port "$PORT" --concurrency "$C" --n-wrong -1 --n-correct 50 --seed 42 \
    --parser helper-v2 --detect-mode k3union

# Add further stages below as they are built. They auto-advance only if the prior gate passed.
# run_stage "next_stage" "<summary path>" "$PY" scripts/<next>.py ...

log "PIPELINE COMPLETE"
