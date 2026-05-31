#!/usr/bin/env python3
"""Experiment J — two-round BLIND consistency detector + correction.

Idea (user): split detection into two roles to remove self-justification bias.
  Round 1 (paraphrase): Qwen restates its ZS answer as plain factual claims — no note,
                        just rephrase the claims. (expG: it is ~90% accurate at handling
                        its own claims.)
  Round 2 (consistency, BLIND to the ZS answer): a fresh Qwen checks each paraphrased claim
                        against the note + question -> CONSISTENT / INCONSISTENT. It does NOT
                        know these came from the model's own answer, so it judges objectively.
                        INCONSISTENT claims = potential errors, and it states what the note
                        actually says -> the correction target.

Measure: detection recall / over-flag / F1; localization (does the inconsistency match the
real error, via GPT-4o); then feed the inconsistency + note fact as the correction
instruction -> fix-rate / break-rate / net vs zeroshot.

Wrong + correct cases, real notes (24k), c=8, ledger, blocking.
Output: runs/expJ_two_round_consistency/qwen25_nw{NW}_nc{NC}/{judged_outputs.jsonl, summary.json}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import phase2b_extract_compare_detection as P2  # noqa: E402
from llm_audit import set_ledger, log_call  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "runs" / "expJ_two_round_consistency"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---------- Round 1: paraphrase the answer into claims (no note) ----------

R1_SYS = "You restate a clinical answer as a plain list of the factual claims it makes."
R1_PARAPHRASE = """Question:
{question}

Answer:
{answer}

Restate this answer as a numbered list of the distinct factual claims it makes in response to the question. Just rephrase each claim plainly in your own words. Do not judge whether they are correct — only restate them."""


def round1_paraphrase(row, port) -> str:
    return P2.vllm_chat(R1_SYS, R1_PARAPHRASE.format(question=row["question"], answer=row["original_answer"][:1500]), port, 500, 0.0, tag="r1.paraphrase")


# ---------- Round 2: consistency check, BLIND to the ZS answer ----------

R2_SYS = "You check whether a set of claims is consistent with a discharge note. You are NOT shown the original answer; judge the claims on their own."
R2_CONSISTENCY = """Discharge note:
{note}

Question:
{question}

Someone made the following claims in answer to the question:
{claims}

Check EACH claim against the discharge note. For each, decide CONSISTENT (the note clearly supports it) or INCONSISTENT (the note contradicts it, does not support it, or it does not actually answer the question).

End with exactly this section:
INCONSISTENT: for each inconsistent claim, state the claim and what the note actually says instead. Write "none" if every claim is consistent and together they correctly answer the question."""


def round2_consistency(row, claims, port) -> dict[str, Any]:
    raw = P2.vllm_chat(R2_SYS, R2_CONSISTENCY.format(note=row["note"][:24000], question=row["question"], claims=claims[:3000]), port, 700, 0.0, tag="r2.consistency")
    m = re.search(r"INCONSISTENT\s*:?\s*(.+)$", raw or "", re.I | re.S)
    inc = (m.group(1).strip() if m else "").strip()
    flagged = bool(inc) and not re.match(r"^\(?none\)?\.?\s*$", inc.strip(), re.I)
    return {"raw": raw, "inconsistent": inc[:1800], "flagged": flagged}


# ---------- localization judge ----------

def gpt_localize(row, inconsistent) -> dict[str, Any]:
    user = (f"Discharge note:\n{row['note'][:18000]}\n\nQuestion:\n{row['question']}\n\n"
            f"Answer given:\n{row['original_answer'][:1500]}\n\nCorrect answer (gold):\n{row['ground_truth'][:800]}\n\n"
            f"A blind check found these inconsistent claims:\n{inconsistent[:1500]}\n\n"
            "Is the answer actually wrong? And does the inconsistency correctly identify the real error?\n"
            'Return JSON: {"answer_is_wrong":true/false, "matches_real_error":true/false}')
    raw = P2.gpt("gpt-4o", "Judge whether a blind consistency check found the real error. Return JSON only.", user, max_tokens=120, temperature=0.0, json_mode=True, tag="gpt.localize")
    m = re.search(r"\{[\s\S]*\}", raw or "")
    try:
        return json.loads(m.group()) if m else {}
    except Exception:
        return {}


# ---------- correction (uses the inconsistency as the fix target) ----------

CORR_SYS = ("You are a careful clinical QA assistant. Fix the previous answer to resolve the inconsistencies listed, "
            "grounded in what the note says. Do not add facts not in the note. Return only the final answer.")


def run_correction(row, inconsistent, port) -> str:
    user = f"""Discharge note:
{row['note'][:24000]}

Question:
{row['question']}

Previous answer:
{row['original_answer']}

A check against the note found these inconsistencies (and what the note actually says):
{inconsistent[:2500]}

Rewrite the previous answer to fix these inconsistencies, grounded in the note. Return only the final answer."""
    return P2.vllm_chat(CORR_SYS, user, port, 700, 0.0, tag="correction")


# ---------- orchestration ----------

def process_one(row, port) -> dict[str, Any]:
    out = {k: row[k] for k in ["fold", "idx", "patient_id", "question", "ground_truth", "original_answer", "stored_label"]}
    try:
        out["judge_original"] = P2.judge(row, row["original_answer"])
        claims = round1_paraphrase(row, port)
        out["paraphrase"] = claims
        r2 = round2_consistency(row, claims, port)
        out["flagged"] = r2["flagged"]
        out["inconsistent"] = r2["inconsistent"]
        if r2["flagged"]:
            out["localize"] = gpt_localize(row, r2["inconsistent"])
            corrected = run_correction(row, r2["inconsistent"], port)
            out["corrected"] = corrected
            out["judge_corrected"] = P2.judge(row, corrected)
        else:
            out["judge_corrected"] = None
    except Exception as e:
        out["error"] = str(e)
    return out


def summarize(rows):
    wrong = [r for r in rows if (r.get("judge_original") or {}).get("label") == 0]
    correct = [r for r in rows if (r.get("judge_original") or {}).get("label") == 1]
    nw, nc = len(wrong), len(correct)
    fw = sum(1 for r in wrong if r.get("flagged"))
    fc = sum(1 for r in correct if r.get("flagged"))
    rec = fw / max(1, nw)
    prec = fw / max(1, fw + fc)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    flagged_wrong = [r for r in wrong if r.get("flagged")]
    loc = sum(1 for r in flagged_wrong if (r.get("localize") or {}).get("matches_real_error"))
    fixes = sum(1 for r in wrong if (r.get("judge_corrected") or {}).get("label") == 1)
    breaks = sum(1 for r in correct if (r.get("judge_corrected") or {}).get("label") == 0)
    W, C = 109, 853
    proj_fix = round(W * (fixes / max(1, nw))); proj_brk = round(C * (breaks / max(1, nc)))
    return {
        "n_wrong": nw, "n_correct": nc,
        "GATE": {"recall_on_wrong": round(rec, 3), "over_flag_on_correct": round(fc / max(1, nc), 3),
                 "precision": round(prec, 3), "f1": round(f1, 3),
                 "localization_of_flagged_wrong": round(loc / max(1, len(flagged_wrong)), 3), "localization_n": f"{loc}/{len(flagged_wrong)}",
                 "compare": {"union": "rec0.95/prec0.64/of0.79/loc0.15", "doubt_flipside": "rec0.93/prec0.54/of0.80/loc0.30", "confirm_gate": "rec0.68/prec0.51/of0.60"}},
        "DOWNSTREAM": {"fix": fixes, "break": breaks, "fix_rate_on_wrong": round(fixes / max(1, nw), 3),
                       "break_rate_on_correct": round(breaks / max(1, nc), 3), "net_balanced": fixes - breaks},
        "full_scale_projection_pre_verdict": {"fixes": proj_fix, "breaks": proj_brk, "net": proj_fix - proj_brk},
        "errors": sum(1 for r in rows if r.get("error")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--n-wrong", type=int, default=40)
    ap.add_argument("--n-correct", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    served = P2.served_model_id(args.port)
    if "qwen2" not in served.lower():
        raise RuntimeError(f"expected Qwen2.5, found {served}")
    sample = P2.load_rows(args.n_wrong, args.n_correct, args.seed)
    out_dir = OUT_ROOT / f"qwen25_nw{args.n_wrong}_nc{args.n_correct}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expJ_two_round_consistency", served=served, args=vars(args))
    print(f"sample={len(sample)} c={args.concurrency} out={out_dir}", flush=True)
    if sample:
        P2.topk_spans(sample[0]["note"], [sample[0]["question"]], k=1, scoring="agreement")
    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, r, args.port) for r in sample]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 5 == 0 or i == len(futs):
                print(f"processed {i}/{len(futs)}", flush=True)
    with (out_dir / "judged_outputs.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    summary = summarize(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
