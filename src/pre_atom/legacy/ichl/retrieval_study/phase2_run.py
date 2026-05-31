"""Phase 2 Generate + Judge: 5 variants on fold_0 test (193 items).

Variants:
  B0       zero-shot baseline (fresh)
  B1       self-critique, no ICL
  P_weak   RA self-correct, minimal "use as guidance"
  P_struct RA self-correct, structural-comparison instruction
  P_critic RA self-correct, critic-style instruction

Generation: Qwen2.5-7B-Instruct via vLLM port 8003 (T=0, max_tokens=512).
Judge: GPT-4o Stage-1 binary.
Pool retrieval: phase2/retrievals_qwen25_fold0.jsonl (top-1, locked 3-cos scorer).

Output: phase2/{variant}_generated.jsonl
        phase2/{variant}_judged.jsonl
        phase2/results_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RS = ROOT / "output" / "ichl" / "retrieval_study"
PHASE2 = RS / "phase2"
RETRIEVALS = PHASE2 / "retrievals_qwen25_fold0.jsonl"

FOLD_TEST = ROOT / "output" / "folds" / "fold_0" / "test.jsonl"
FOLD_TRAIN = ROOT / "output" / "folds" / "fold_0" / "train.jsonl"
QWEN25_ZS = ROOT / "output" / "step8" / "qwen2.5-7b-instruct" / "fold_0" / "zeroshot_generated.csv"
ITEMS = RS / "pool_index" / "items.jsonl"

VLLM_URL = "http://localhost:8003/v1"
VLLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# === Prompt templates ===
SYSTEM_DEFAULT = "You are a medical assistant. Read the clinical note carefully and answer the question concisely."
SYSTEM_REVISE = "You are a medical assistant reviewing your previous answer to a clinical question. Revise your answer if needed for accuracy."

PROMPT_B0 = """NOTE:
{note}

QUESTION: {question}

ANSWER:"""

PROMPT_B1 = """NOTE:
{note}

QUESTION: {question}

YOUR PREVIOUS ANSWER: {your_zs}

If the previous answer is correct, restate it. If wrong, provide a corrected answer based only on the note.

REVISED ANSWER:"""

PROMPT_RA_BASE = """REFERENCE CASE (a similar clinical case with a verified-correct answer):
NOTE: {ref_note}
QUESTION: {ref_question}
CORRECT ANSWER: {ref_answer}

YOUR CASE:
NOTE: {note}
QUESTION: {question}
YOUR PREVIOUS ANSWER: {your_zs}

{instruction}

REVISED ANSWER:"""

INSTR_WEAK = "Use the reference case as guidance. If your previous answer is correct, restate it; otherwise provide a corrected answer based on the note."

INSTR_STRUCT = ("Compare the structure of your previous answer to the reference's correct answer. "
                "Identify any mismatches in clinical reasoning or content, then provide your revised answer "
                "based on the note.")

INSTR_CRITIC = ("First, briefly identify what makes the reference's answer correct (the key clinical fact "
                "or reasoning). Then check whether your previous answer follows the same form. Finally, "
                "provide your revised answer based on the note.")


def make_prompt(variant: str, item: dict, note: str, your_zs: str | None,
                ref: dict | None) -> tuple[str, str]:
    """Return (system_prompt, user_prompt)."""
    if variant == "B0":
        return SYSTEM_DEFAULT, PROMPT_B0.format(note=note, question=item["question"])
    if variant == "B1":
        return SYSTEM_REVISE, PROMPT_B1.format(note=note, question=item["question"], your_zs=your_zs)
    # RA variants
    instr = {"P_weak": INSTR_WEAK, "P_struct": INSTR_STRUCT, "P_critic": INSTR_CRITIC}[variant]
    return SYSTEM_REVISE, PROMPT_RA_BASE.format(
        ref_note=ref["note"], ref_question=ref["question"], ref_answer=ref["answer"],
        note=note, question=item["question"], your_zs=your_zs, instruction=instr,
    )


def truncation_detect(text: str, finish_reason: str | None, max_tokens: int) -> dict:
    """Mandatory per MEMORY truncation-detection rule."""
    text = text or ""
    certain = finish_reason == "length"
    suspicious = (
        len(text) > 0 and not text.rstrip().endswith((".", "?", "!", '"', "'", "]", ")", ":"))
        and len(text.split()) > 30
    )
    return {"finish_reason": finish_reason, "certain": certain,
            "suspicious_no_terminal": suspicious, "char_len": len(text)}


def vllm_call(client, system: str, user: str, max_tokens: int = 512) -> dict:
    try:
        r = client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0, max_tokens=max_tokens,
        )
        text = r.choices[0].message.content or ""
        fr = r.choices[0].finish_reason
        usage = r.usage
        return {"text": text, "finish_reason": fr,
                "prompt_tok": usage.prompt_tokens if usage else None,
                "comp_tok": usage.completion_tokens if usage else None,
                "trunc": truncation_detect(text, fr, max_tokens)}
    except Exception as e:
        return {"_err": str(e)[:200]}


# === Stage-1 binary GPT-4o judge ===
JUDGE_SYSTEM = ("You are a medical evaluator. Compare a candidate answer to the ground truth and "
                "determine if the candidate is correct.")

JUDGE_USER_TMPL = """Question: {question}

Ground truth answer: {gt}

Candidate answer: {candidate}

Is the candidate answer correct (semantically matches the ground truth)? Respond with ONLY a single digit: 1 for correct, 0 for incorrect."""


def gpt4o_judge(client, item_q: str, gt: str, cand: str) -> dict:
    try:
        r = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": JUDGE_USER_TMPL.format(
                    question=item_q, gt=gt, candidate=cand)}
            ],
            temperature=0.0, max_tokens=10,
        )
        txt = (r.choices[0].message.content or "").strip()
        m = re.search(r"[01]", txt)
        score = int(m.group(0)) if m else None
        usage = r.usage
        return {"binary_correct": score, "raw": txt[:30],
                "prompt_tok": usage.prompt_tokens if usage else None,
                "comp_tok": usage.completion_tokens if usage else None}
    except Exception as e:
        return {"_err": str(e)[:200]}


def load_targets() -> list[dict]:
    test = [json.loads(l) for l in FOLD_TEST.open() if l.strip()]
    qwen_df = pd.read_csv(QWEN25_ZS)
    qwen_by_pid = {int(r["patient_id"]): str(r["model_answer"] or "") for _, r in qwen_df.iterrows()}
    out = []
    for ti, r in enumerate(test):
        pid = int(r["patient_id"])
        # Concatenate notes the same way step8 did. NEVER truncate per
        # [Workflow] No Silent Truncation. Max EHRNoteQA note ~5709 nomic tokens
        # (~21K chars); fits any 8K+ context model. The earlier [:16000] cap was
        # an unnecessary defensive shortcut and is removed.
        notes = [f"[Note {i}]\n{str(r.get(f'note_{i}', '') or '').strip()}"
                 for i in [1, 2, 3]
                 if r.get(f"note_{i}") and str(r.get(f"note_{i}")).strip()
                 and str(r.get(f"note_{i}")).lower() != "nan"]
        note = "\n\n".join(notes)
        # Ground truth
        letter = str(r.get("answer", "")).strip().upper()
        gt_text = str(r.get(f"choice_{letter}", "")).strip() if letter else ""
        gt = f"{letter}: {gt_text}" if (letter and gt_text) else gt_text
        out.append({
            "test_idx": ti, "patient_id": pid, "question": str(r["question"]),
            "note": note, "ground_truth": gt,
            "qwen_zs": qwen_by_pid.get(pid, ""),
        })
    return out


def load_pool_refs() -> dict[int, dict]:
    """row_id -> {note, question, answer (GT)}."""
    train = [json.loads(l) for l in FOLD_TRAIN.open() if l.strip()]
    items = [json.loads(l) for l in ITEMS.open() if l.strip()]
    pid_to_train = {int(r["patient_id"]): r for r in train}
    out = {}
    for it in items:
        pid = int(it["patient_id"])
        if pid not in pid_to_train: continue
        tr = pid_to_train[pid]
        notes = [str(tr.get(f"note_{i}", "") or "") for i in [1, 2, 3]]
        notes = [n for n in notes if n.strip() and n.lower() != "nan"]
        note = "\n\n".join(notes)[:8000]  # smaller cap for ref side
        letter = str(tr.get("answer", "")).strip().upper()
        gt_text = str(tr.get(f"choice_{letter}", "")).strip() if letter else ""
        gt = f"{letter}: {gt_text}" if (letter and gt_text) else gt_text
        out[int(it["row_id"])] = {
            "patient_id": pid, "question": str(tr["question"]),
            "note": note, "answer": gt,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+",
                    default=["B0", "B1", "P_weak", "P_struct", "P_critic"])
    ap.add_argument("--gen-workers", type=int, default=8)
    ap.add_argument("--judge-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="limit test items (debug)")
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--skip-judge", action="store_true")
    args = ap.parse_args()

    print(f"Loading targets, retrievals, pool refs...")
    targets = load_targets()
    if args.limit > 0: targets = targets[:args.limit]
    retrievals = {int(json.loads(l)["test_idx"]): json.loads(l)
                  for l in RETRIEVALS.open() if l.strip()}
    pool_refs = load_pool_refs()
    print(f"  targets={len(targets)}  retrievals={len(retrievals)}  pool_refs={len(pool_refs)}")

    # === API clients ===
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        env = ROOT / ".env"
        for line in env.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                api_key = line.split("=", 1)[1].strip(); break
    vllm_client = OpenAI(base_url=VLLM_URL, api_key="not-needed", timeout=120)
    judge_client = OpenAI(api_key=api_key, timeout=60) if api_key else None

    # === Generation ===
    if not args.skip_gen:
        for variant in args.variants:
            out_file = PHASE2 / f"{variant}_generated.jsonl"
            if out_file.exists():
                print(f"  [skip-gen] {variant}: {out_file.name} already exists ({sum(1 for _ in out_file.open())} rows)")
                continue
            print(f"\n=== Generation: {variant} ({len(targets)} items, {args.gen_workers} workers) ===")
            t0 = time.monotonic()

            def gen_one(t):
                ref = None
                if variant in ("P_weak", "P_struct", "P_critic"):
                    rt = retrievals.get(t["test_idx"])
                    if rt is None: return {"test_idx": t["test_idx"], "_err": "no retrieval"}
                    ref = pool_refs.get(rt["top1_pool_rid"])
                    if ref is None: return {"test_idx": t["test_idx"], "_err": "no pool_ref"}
                sysm, userm = make_prompt(variant, t, t["note"], t["qwen_zs"], ref)
                r = vllm_call(vllm_client, sysm, userm)
                out = {"test_idx": t["test_idx"], "patient_id": t["patient_id"],
                       "variant": variant}
                if "_err" in r: out["_err"] = r["_err"]
                else:
                    out["model_answer"] = r["text"]
                    out["finish_reason"] = r["finish_reason"]
                    out["truncation"] = r["trunc"]
                    out["prompt_tok"] = r["prompt_tok"]; out["comp_tok"] = r["comp_tok"]
                if ref: out["ref_pool_rid"] = retrievals[t["test_idx"]]["top1_pool_rid"]
                return out

            n_done = 0; n_err = 0; n_trunc_certain = 0; n_trunc_susp = 0
            with out_file.open("w") as f, ThreadPoolExecutor(max_workers=args.gen_workers) as ex:
                for r in ex.map(gen_one, targets):
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                    n_done += 1
                    if "_err" in r: n_err += 1
                    elif r.get("truncation", {}).get("certain"): n_trunc_certain += 1
                    elif r.get("truncation", {}).get("suspicious_no_terminal"): n_trunc_susp += 1
                    if n_done % 50 == 0:
                        dt = time.monotonic() - t0
                        eta = dt * (len(targets) - n_done) / n_done
                        print(f"  {n_done}/{len(targets)}  elapsed={dt:.0f}s  eta={eta:.0f}s "
                              f"err={n_err}  trunc_certain={n_trunc_certain}  trunc_susp={n_trunc_susp}")
            dt = time.monotonic() - t0
            print(f"  DONE {variant}: {n_done} items in {dt:.0f}s  err={n_err}  "
                  f"trunc_certain={n_trunc_certain}  trunc_susp={n_trunc_susp}")
            if n_trunc_certain > len(targets) * 0.05:
                print(f"  WARNING: certain-truncation rate {100*n_trunc_certain/len(targets):.1f}% > 5%")

    # === Judging ===
    if not args.skip_judge:
        if judge_client is None:
            print("\nNo OPENAI_API_KEY — skipping judge phase")
            return
        for variant in args.variants:
            in_file = PHASE2 / f"{variant}_generated.jsonl"
            out_file = PHASE2 / f"{variant}_judged.jsonl"
            if not in_file.exists():
                print(f"  [skip-judge] {variant}: no generated file")
                continue
            if out_file.exists():
                print(f"  [skip-judge] {variant}: judged file exists")
                continue
            print(f"\n=== Judging: {variant} ===")
            gens = [json.loads(l) for l in in_file.open() if l.strip()]
            t_by_idx = {t["test_idx"]: t for t in targets}
            t0 = time.monotonic()
            cost_in, cost_out = 0, 0

            def judge_one(g):
                if "_err" in g:
                    return {**g, "_judge_err": "no_gen"}
                t = t_by_idx.get(g["test_idx"])
                if t is None:
                    return {**g, "_judge_err": "no_target"}
                jr = gpt4o_judge(judge_client, t["question"], t["ground_truth"], g["model_answer"])
                return {**g, **jr}

            with out_file.open("w") as f, ThreadPoolExecutor(max_workers=args.judge_workers) as ex:
                n_done = 0
                for r in ex.map(judge_one, gens):
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                    n_done += 1
                    cost_in += r.get("prompt_tok", 0) or 0
                    cost_out += r.get("comp_tok", 0) or 0
                    if n_done % 50 == 0:
                        dt = time.monotonic() - t0
                        cost_so_far = cost_in * 5e-6 + cost_out * 1.5e-5
                        print(f"  {n_done}/{len(gens)}  elapsed={dt:.0f}s  cost~${cost_so_far:.2f}")
            cost = cost_in * 5e-6 + cost_out * 1.5e-5
            print(f"  DONE judging {variant}: {n_done} in {time.monotonic()-t0:.0f}s  cost~${cost:.2f}")

    # === Aggregate ===
    print("\n=== Final accuracy by variant ===")
    summary = {}
    for variant in args.variants:
        f = PHASE2 / f"{variant}_judged.jsonl"
        if not f.exists():
            print(f"  {variant}: NO JUDGED FILE")
            continue
        rows = [json.loads(l) for l in f.open() if l.strip()]
        valid = [r for r in rows if r.get("binary_correct") in (0, 1)]
        n_correct = sum(1 for r in valid if r["binary_correct"] == 1)
        summary[variant] = {
            "n": len(rows), "n_judged": len(valid), "n_correct": n_correct,
            "accuracy": round(n_correct / max(len(valid), 1), 4),
        }
        print(f"  {variant:10s}  n={len(rows)}  judged={len(valid)}  acc={summary[variant]['accuracy']:.3f} ({n_correct}/{len(valid)})")

    (PHASE2 / "results_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {PHASE2/'results_summary.json'}")


if __name__ == "__main__":
    main()
