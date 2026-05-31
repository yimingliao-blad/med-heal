"""Build the 112-item Sara∩Jose unanimous human-gold judge dataset.

Source files (all already in repo):
  - output/step9_v2/sample200_with_gold.csv
       cols: patient_id, Caitlin Schwanke, Jose E. Lizarraga Mazab, Kushali darak,
             Sara Saif, gpt, gold, n_raters
  - output/EHRNoteQA_processed.jsonl  (question + ground_truth + answer letter)
  - output/ours_biomistral-7b_EHRNoteQA_processed.csv  (BioMistral-7B model_answer)

Filter rule for the canonical 112:
  - Sara Saif rated AND Jose E. Lizarraga Mazab rated (both non-null).
  - They UNANIMOUSLY agreed (Sara == Jose).
  - The shared rating is the gold label.

This matches the validated GPT-4o reference set (92% agreement, κ=0.75 vs human).

Output schema (JSONL, one row per item):
  {patient_id, question, ground_truth, model_answer, sara, jose, gold,
   gpt_4o_label, fold_id, target}

`target` is set to "biomistral-7b" since these are BioMistral open-ended outputs.
`fold_id` is set to "human_gold" (sentinel; not from the EHRNoteQA folds).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
SAMPLE200 = ROOT / "output" / "step9_v2" / "sample200_with_gold.csv"
EHRNOTEQA = ROOT / "output" / "EHRNoteQA_processed.jsonl"
BIOMISTRAL = ROOT / "output" / "ours_biomistral-7b_EHRNoteQA_processed.csv"
OUT = ROOT / "output" / "ichl" / "mlx_judge" / "splits" / "gold_112.jsonl"


def main() -> None:
    print("Loading sources…")
    s200 = pd.read_csv(SAMPLE200)
    ehr = pd.read_json(EHRNOTEQA, lines=True)
    bm = pd.read_csv(BIOMISTRAL)

    print(f"  sample200: {len(s200)} rows")
    print(f"  EHRNoteQA: {len(ehr)} rows  cols={list(ehr.columns)[:8]}…")
    print(f"  BioMistral CSV: {len(bm)} rows  cols={list(bm.columns)[:8]}…")

    # 1) Filter to Sara∩Jose unanimous (the canonical 112-item set)
    s = "Sara Saif"
    j = "Jose E. Lizarraga Mazab"
    both_rated = s200[s].notna() & s200[j].notna()
    unanimous = s200[s] == s200[j]
    sj = s200[both_rated & unanimous].copy()
    print(f"\n  Sara rated: {s200[s].notna().sum()}")
    print(f"  Jose rated: {s200[j].notna().sum()}")
    print(f"  Both rated: {both_rated.sum()}")
    print(f"  Both rated AND unanimous: {len(sj)}")

    # Sanity: gold should equal Sara==Jose where they agree
    if "gold" in sj.columns:
        mismatches = (sj["gold"] != sj[s]).sum()
        print(f"  Gold-vs-Sara mismatches in unanimous subset: {mismatches} (should be 0)")

    # 2) Locate question + ground_truth
    # EHRNoteQA has columns including patient_id, question, choice_A..E, answer (letter)
    # Reconstruct ground_truth as "<letter>: <choice text>"
    def gt_from_row(r) -> str:
        letter = str(r.get("answer", "")).strip().upper()
        col = f"choice_{letter}"
        choice_text = str(r.get(col, "")).strip() if col in r.index else ""
        if letter and choice_text:
            return f"{letter}: {choice_text}"
        # Fallback for any other field name
        return choice_text or str(r.get("ground_truth", "")) or letter

    ehr["ground_truth_built"] = ehr.apply(gt_from_row, axis=1)
    ehr_lookup = {int(r["patient_id"]): r for _, r in ehr.iterrows()}

    # 3) BioMistral model answers — column name varies across runs
    candidate_cols = [c for c in bm.columns if "answer" in c.lower() or "openended" in c.lower()]
    print(f"  BioMistral candidate answer cols: {candidate_cols}")
    # Prefer 'openended_answer' if present, else first 'answer' col
    if "openended_answer" in bm.columns:
        ans_col = "openended_answer"
    elif candidate_cols:
        ans_col = candidate_cols[0]
    else:
        raise SystemExit(f"Could not find a model-answer column in {BIOMISTRAL}")
    print(f"  using BioMistral answer column: {ans_col!r}")
    bm_lookup = {int(r["patient_id"]): str(r.get(ans_col, "")) for _, r in bm.iterrows()}

    # 4) Assemble rows
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_built = 0
    n_missing_ehr = 0
    n_missing_bm = 0
    with OUT.open("w") as f:
        for _, row in sj.iterrows():
            pid = int(row["patient_id"])
            ehr_row = ehr_lookup.get(pid)
            if ehr_row is None:
                n_missing_ehr += 1
                continue
            ans = bm_lookup.get(pid, "")
            if not ans or pd.isna(ans):
                n_missing_bm += 1
                continue
            out_row = {
                "patient_id": pid,
                "fold_id": "human_gold",
                "target": "biomistral-7b",
                "question": str(ehr_row.get("question", "")),
                "ground_truth": str(ehr_row["ground_truth_built"]),
                "model_answer": ans,
                "sara": int(row[s]),
                "jose": int(row[j]),
                "gold": int(row["gold"]),
                "gpt_4o_label": int(row["gpt"]) if not pd.isna(row.get("gpt")) else None,
                "binary_correct": int(row["gold"]),  # alias used by run_vllm_judge_v2 sampler
            }
            f.write(json.dumps(out_row, default=str) + "\n")
            n_built += 1

    print(f"\nBuilt: {n_built} rows  →  {OUT}")
    print(f"  dropped (missing EHRNoteQA):  {n_missing_ehr}")
    print(f"  dropped (missing BioMistral): {n_missing_bm}")

    # Sanity: gold balance
    rows = [json.loads(l) for l in OUT.open() if l.strip()]
    g1 = sum(1 for r in rows if r["gold"] == 1)
    g0 = sum(1 for r in rows if r["gold"] == 0)
    gpt_correct = sum(1 for r in rows if r.get("gpt_4o_label") is not None and r["gpt_4o_label"] == r["gold"])
    n_with_gpt = sum(1 for r in rows if r.get("gpt_4o_label") is not None)
    print(f"\n  Gold balance: g1={g1}, g0={g0}")
    if n_with_gpt:
        print(f"  GPT-4o vs gold (sanity reference): {gpt_correct}/{n_with_gpt} = {100*gpt_correct/n_with_gpt:.1f}%")


if __name__ == "__main__":
    main()
