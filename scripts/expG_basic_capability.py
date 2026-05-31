#!/usr/bin/env python3
"""Experiment G — can Qwen2.5 do the BASIC detection work: correct vs suspicious claims?

Before optimizing detection, confirm the fundamental capability. Bidirectional test:

  GENERATION (can Qwen produce a correct categorization?)
    1. Qwen lists, for its ZS answer, the claims it thinks are CORRECT and the claims it
       thinks are SUSPICIOUS (vs question + note).
    2. GPT-4o judges Qwen's categorization accuracy (using ground truth).

  RECOGNITION (can Qwen confirm a correct categorization?)
    3. GPT-4o independently lists the correct + suspicious claims (gold).
    4. Qwen is shown GPT's gold list and confirms each (AGREE/DISAGREE).
       Agreement with gold = recognition accuracy.

If Qwen's GENERATION accuracy is low but RECOGNITION is high -> it can tell right from wrong
when shown, but can't produce it (a precision/generation gap). If RECOGNITION is also low ->
a deeper comprehension failure. Split by wrong vs correct ZS answers.

Real notes (24k). GPT-4o = oracle judge/lister. c=8, ledger, blocking.
Output: runs/expG_basic_capability/qwen25_nw{NW}_nc{NC}/{judged_outputs.jsonl, summary.json}
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

OUT_ROOT = PROJECT_ROOT / "runs" / "expG_basic_capability"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


# ---------- 1. Qwen lists correct vs suspicious ----------

QWEN_LIST_SYS = "You review a clinical answer against the discharge note and separate what is solid from what is shaky."
QWEN_LIST = """Discharge note:
{note}

Question:
{question}

Answer that was given:
{answer}

Go through this answer. List:
CORRECT: the specific claims in the answer that are clearly supported by the note and answer the question.
SUSPICIOUS: the specific claims that may be wrong, unsupported by the note, or not what the question asks.
Be specific and quote the claim. If there is nothing suspicious, say so."""


def qwen_list(row, port) -> str:
    return P2.vllm_chat(QWEN_LIST_SYS, QWEN_LIST.format(note=row["note"][:24000], question=row["question"], answer=row["original_answer"][:1500]), port, 700, 0.0, tag="qwen.list")


# ---------- 2. GPT judges Qwen's categorization ----------

def gpt_judge_qwen(row, qlist) -> dict[str, Any]:
    user = (f"Discharge note:\n{row['note'][:18000]}\n\nQuestion:\n{row['question']}\n\n"
            f"Answer given:\n{row['original_answer'][:1500]}\n\nCorrect answer (gold):\n{row['ground_truth'][:800]}\n\n"
            f"A model categorized the answer's claims as CORRECT or SUSPICIOUS:\n{qlist[:3000]}\n\n"
            "Judge the model's categorization against the gold. What fraction of the claims it called CORRECT are actually correct, "
            "and what fraction it called SUSPICIOUS are actually problematic? Also did it correctly identify the main error (if the answer is wrong)?\n"
            'Return JSON: {"correct_label_accuracy":0-1, "suspicious_label_accuracy":0-1, "found_main_error":true/false, "overall":0-1}')
    raw = P2.gpt("gpt-4o", "You grade a model's claim categorization against the gold answer. Return JSON only.", user, max_tokens=200, temperature=0.0, json_mode=True, tag="gpt.judge_qwen")
    return _parse_json(raw)


# ---------- 3. GPT lists gold correct/suspicious ----------

def gpt_list(row) -> dict[str, Any]:
    user = (f"Discharge note:\n{row['note'][:18000]}\n\nQuestion:\n{row['question']}\n\n"
            f"Answer given:\n{row['original_answer'][:1500]}\n\nCorrect answer (gold):\n{row['ground_truth'][:800]}\n\n"
            "List the answer's claims that are CORRECT (supported by note, answer the question) and the claims that are SUSPICIOUS (wrong/unsupported/wrong-focus). Be specific.\n"
            'Return JSON: {"correct":["..."], "suspicious":["..."]}')
    raw = P2.gpt("gpt-4o", "You list which claims in an answer are correct vs suspicious, using the gold answer. Return JSON only.", user, max_tokens=400, temperature=0.0, json_mode=True, tag="gpt.list")
    return _parse_json(raw)


# ---------- 4. Qwen confirms GPT's gold list ----------

QWEN_CONFIRM_SYS = "You review an expert's labels of an answer's claims and say whether you agree with each."
QWEN_CONFIRM = """Discharge note:
{note}

Question:
{question}

Answer that was given:
{answer}

An expert labeled the answer's claims. For EACH item, reply AGREE or DISAGREE on its own line, in order.

CLAIMS LABELED CORRECT:
{correct_items}

CLAIMS LABELED SUSPICIOUS:
{suspicious_items}

Reply with one AGREE or DISAGREE per item, correct items first then suspicious items, one per line."""


def qwen_confirm(row, gold, port) -> dict[str, Any]:
    corr = gold.get("correct") or []
    susp = gold.get("suspicious") or []
    corr_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(corr)) or "(none)"
    susp_block = "\n".join(f"{i+1}. {s}" for i, s in enumerate(susp)) or "(none)"
    raw = P2.vllm_chat(QWEN_CONFIRM_SYS, QWEN_CONFIRM.format(note=row["note"][:24000], question=row["question"], answer=row["original_answer"][:1500], correct_items=corr_block, suspicious_items=susp_block), port, 400, 0.0, tag="qwen.confirm")
    decisions = [m.group(1) for m in re.finditer(r"\b(AGREE|DISAGREE)\b", (raw or "").upper())]
    n_items = len(corr) + len(susp)
    agree = sum(1 for d in decisions[:n_items] if d == "AGREE")
    return {"n_items": n_items, "n_decisions": len(decisions), "agree": agree,
            "agreement_rate": round(agree / max(1, min(n_items, len(decisions))), 3) if n_items else None}


# ---------- gpt helpers ----------

def _parse_json(raw: str) -> dict[str, Any]:
    m = re.search(r"\{[\s\S]*\}", raw or "")
    if not m:
        return {}
    try:
        return json.loads(m.group())
    except Exception:
        return {}


# ---------- orchestration ----------

def process_one(row, port) -> dict[str, Any]:
    out = {k: row[k] for k in ["fold", "idx", "patient_id", "question", "ground_truth", "original_answer", "stored_label"]}
    try:
        out["judge_original"] = P2.judge(row, row["original_answer"])
        ql = qwen_list(row, port)
        out["qwen_list"] = ql
        out["gpt_judge_qwen"] = gpt_judge_qwen(row, ql)
        gold = gpt_list(row)
        out["gpt_gold"] = gold
        out["qwen_confirm"] = qwen_confirm(row, gold, port)
    except Exception as e:
        out["error"] = str(e)
    return out


def _avg(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / max(1, len(xs)), 3) if xs else None


def summarize(rows):
    def block(subset, label):
        gj = [r.get("gpt_judge_qwen") or {} for r in subset]
        return {
            "n": len(subset),
            "GEN_correct_label_acc": _avg([g.get("correct_label_accuracy") for g in gj]),
            "GEN_suspicious_label_acc": _avg([g.get("suspicious_label_accuracy") for g in gj]),
            "GEN_found_main_error_rate": _avg([1.0 if g.get("found_main_error") else 0.0 for g in gj]),
            "GEN_overall": _avg([g.get("overall") for g in gj]),
            "RECOG_confirm_agreement": _avg([(r.get("qwen_confirm") or {}).get("agreement_rate") for r in subset]),
        }
    wrong = [r for r in rows if (r.get("judge_original") or {}).get("label") == 0]
    correct = [r for r in rows if (r.get("judge_original") or {}).get("label") == 1]
    return {"n_cases": len(rows), "errors": sum(1 for r in rows if r.get("error")),
            "ALL": block(rows, "all"), "WRONG_zs": block(wrong, "wrong"), "CORRECT_zs": block(correct, "correct")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--n-wrong", type=int, default=30)
    ap.add_argument("--n-correct", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    served = P2.served_model_id(args.port)
    if "qwen2" not in served.lower():
        raise RuntimeError(f"expected Qwen2.5, found {served}")
    sample = P2.load_rows(args.n_wrong, args.n_correct, args.seed)
    out_dir = OUT_ROOT / f"qwen25_nw{args.n_wrong}_nc{args.n_correct}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expG_basic_capability", served=served, args=vars(args))
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
