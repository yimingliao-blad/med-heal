#!/usr/bin/env python3
"""
Correction + Verdict test on 5-fold detected items.
Uses reclassified detection results.
P1 and P1+pool correction × V1/V2/V3/None verdict = 8 combinations.
Resume-safe.

Usage:
    python run_correction_verdict_5fold.py --port 8003
"""
import json, random, re, os, time, argparse
import numpy as np
from pathlib import Path
from collections import Counter
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = Path(__file__).parent

key = None
for line in open(PROJECT_ROOT / ".env"):
    line = line.strip()
    if line.startswith("OPENAI_API_KEY=") and not line.startswith("#"):
        key = line.split("=", 1)[1]; break
from openai import OpenAI
gpt_client = OpenAI(api_key=key)
spending = {"calls": 0, "cost": 0.0}

PORT = 8003
QWEN32B_URL = "http://192.168.68.107:8090/v1/chat/completions"

# ============================================================
# TYPE-ROUTED CORRECTION PROMPTS
# ============================================================

COR_CONTRADICTION = """Discharge summary:
{note}

Question: {question}

Your previous answer contained a factual error:
YOUR ANSWER SAID: {error_statement}
BUT THE NOTES SAY: {correct_statement}

Re-read the relevant section of the discharge notes. Then re-answer the question based on what the notes actually say.
Answer in 1-3 direct sentences."""

COR_OMISSION = """Discharge summary:
{note}

Question: {question}

Your previous answer was missing critical information:
MISSING: {error_statement}
THE NOTES SAY: {correct_statement}

Re-answer the question, making sure to include this information.
Answer in 1-3 direct sentences."""

COR_QMIS = """Discharge summary:
{note}

Question: {question}

Your previous answer addressed the wrong aspect of the question:
ISSUE: {error_statement}

Re-read the question carefully. Pay attention to which visit, time period, and clinical focus it asks about.
Re-answer the question correctly.
Answer in 1-3 direct sentences."""

COR_POOL_CONTRADICTION = """Discharge summary:
{note}

Question: {question}

Your previous answer contained a factual error:
YOUR ANSWER SAID: {error_statement}
BUT THE NOTES SAY: {correct_statement}

Here is an example of how a similar error was corrected in another case:
  Original claim: "{pool_wrong}"
  Corrected to: "{pool_correct}"
  Approach: check the discharge notes for the exact information and use what the notes state.

Apply the same approach and re-answer the question.
Answer in 1-3 direct sentences."""

COR_POOL_OMISSION = """Discharge summary:
{note}

Question: {question}

Your previous answer was missing critical information:
MISSING: {error_statement}
THE NOTES SAY: {correct_statement}

Here is an example of a similar omission corrected in another case:
  Original: "{pool_wrong}"
  Corrected to: "{pool_correct}"

Re-answer including the missing information.
Answer in 1-3 direct sentences."""

COR_POOL_QMIS = """Discharge summary:
{note}

Question: {question}

Your previous answer addressed the wrong aspect:
ISSUE: {error_statement}

Here is an example of a similar error corrected in another case:
  Original: "{pool_wrong}"
  Corrected to: "{pool_correct}"

Re-read the question and re-answer correctly.
Answer in 1-3 direct sentences."""

# ============================================================
# VERDICT PROMPTS
# ============================================================

V1_PROMPT = """Discharge summary:
{note}

Question: {question}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Count how many factual claims in each answer CONTRADICT the discharge notes.

Answer A contradictions: <number>
Answer B contradictions: <number>
Better answer: A or B"""

V2_PROMPT = """Discharge summary:
{note}

Question: {question}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Compare on three criteria:
1. Which better addresses what the question asks?
2. Which has fewer factual conflicts with the notes?
3. Which better covers critical information?

Better answer: A or B"""

V3_PROMPT = """Discharge summary:
{note}

Question: {question}

An error was detected in the original answer:
ERROR: {error_statement}

ORIGINAL ANSWER:
{original}

CORRECTED ANSWER:
{corrected}

Does the corrected answer fix the error without introducing new errors?
Better answer: ORIGINAL or CORRECTED"""

# ============================================================
# HELPERS
# ============================================================

def build_chatml(s, u):
    return f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n"

def vllm_gen(prompt, max_tokens=512, temperature=1.0):
    model = requests.get(f"http://localhost:{PORT}/v1/models", timeout=5).json()["data"][0]["id"]
    resp = requests.post(f"http://localhost:{PORT}/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": max_tokens,
              "temperature": temperature, "stop": ["<|im_end|>", "<|endoftext|>"]}, timeout=120)
    return resp.json()["choices"][0]["text"].strip()

def q32_verdict(raw):
    try:
        resp = requests.post(QWEN32B_URL, json={
            "model": "Qwen/Qwen3-32B-MLX-bf16",
            "messages": [{"role": "system", "content": "Extract verdict. JSON only."},
                         {"role": "user", "content": f'/nothink\nWhich answer picked?\n\nTEXT:\n{raw[:1500]}\n\n{{"pick": "A" or "B" or "ORIGINAL" or "CORRECTED"}}'}],
            "max_tokens": 100, "temperature": 0.0,
        }, timeout=60)
        text = resp.json()["choices"][0]["message"]["content"].strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if m: return json.loads(m.group()).get("pick", "UNCLEAR").upper()
    except: pass
    return "UNCLEAR"

def gpt4o_eval(note, question, gt, answer):
    time.sleep(1.5)
    try:
        r = gpt_client.chat.completions.create(
            model="gpt-4o", messages=[
                {"role": "system", "content": "You are a medical expert evaluating an AI model's answer."},
                {"role": "user", "content": (
                    f"DISCHARGE SUMMARY:\n{note}\n\nQUESTION:\n{question}\n\n"
                    f"CORRECT ANSWER (Ground Truth):\n{gt}\n\nMODEL'S ANSWER:\n{answer}\n\n"
                    f"Respond with ONLY a single digit: 1 = Correct, 0 = Incorrect"
                )},
            ], max_tokens=10, temperature=0.1,
        )
        text = r.choices[0].message.content.strip()
        cost = r.usage.prompt_tokens * 2.5 / 1e6 + r.usage.completion_tokens * 10.0 / 1e6
        spending["calls"] += 1; spending["cost"] += cost
        return 1 if text.startswith("1") else 0 if text.startswith("0") else -1
    except Exception as e:
        print(f"  GPT-4o error: {e}", flush=True); time.sleep(5); return -1

def load_notes():
    notes_df = pd.read_json(PROJECT_ROOT / "output" / "EHRNoteQA_processed.jsonl", lines=True)
    lookup = {}
    for _, r in notes_df.iterrows():
        pid = str(r.get("patient_id", ""))
        parts = []
        for i in [1,2,3]:
            col = f"note_{i}"
            if col in r and pd.notna(r[col]):
                t = str(r[col]).strip()
                if t and t.lower() != "nan":
                    parts.append(f"[Note {i}]\n{t}")
        lookup[pid] = "\n\n".join(parts)
    return lookup

embedder = None
def get_embedder():
    global embedder
    if embedder is None:
        embedder = SentenceTransformer("sentence-transformers/gtr-t5-base", device="cpu")
    return embedder

def retrieve_pool(error_stmt, error_type, fold_id):
    pool_f = PROJECT_ROOT / "workspace" / "self_critique" / "data" / "bm_atomic_pool" / f"fold_{fold_id}_atoms.json"
    emb_f = PROJECT_ROOT / "workspace" / "self_critique" / "data" / "bm_atomic_pool" / f"fold_{fold_id}_atom_embeddings.npy"
    if not pool_f.exists() or not emb_f.exists(): return None
    pool = json.load(open(pool_f))
    pool_emb = np.load(emb_f)
    type_map = {"CONTRADICTION": "factual_error", "OMISSION": "omission", "QUESTION_MISALIGNMENT": "factual_error"}
    target = type_map.get(error_type, "factual_error")
    indices = [i for i, a in enumerate(pool) if a.get("main_error_type") == target and a.get("gt_atom_raw")]
    if not indices: indices = [i for i, a in enumerate(pool) if a.get("gt_atom_raw")]
    if not indices: return None
    query_emb = get_embedder().encode([error_stmt], normalize_embeddings=True)
    sims = np.dot(pool_emb[indices], query_emb.T).flatten()
    top = np.argsort(-sims)[0]
    atom = pool[indices[top]]
    return {"text_raw": atom["text_raw"], "gt_atom_raw": atom["gt_atom_raw"], "sim": float(sims[top])}

def get_best_error(reclassified_item):
    """Get the most critical error from reclassified detection results."""
    rc = reclassified_item["reclassified"]
    # Priority: use S1 sub-prompts first (qmis > contra > omis), then S3, then D1
    for dk in ["qmis", "contra", "omis"]:
        if rc[dk]["verdict"] == "INCORRECT":
            return rc[dk]["type"], rc[dk]["error_statement"], rc[dk]["correct_statement"], dk
    for dk in ["S3", "D1"]:
        if rc[dk]["verdict"] == "INCORRECT":
            return rc[dk]["type"], rc[dk]["error_statement"], rc[dk]["correct_statement"], dk
    return "NONE", "", "", "none"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()
    global PORT; PORT = args.port

    notes = load_notes()
    dfs = []
    for fold in range(5):
        f = PROJECT_ROOT / "output" / "step8" / "qwen2.5-7b-instruct" / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        if f.exists(): df = pd.read_csv(f); df["fold"] = fold; dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    # Load reclassified detection
    with open(OUTPUT_DIR / "all_detection_5fold_reclassified.json") as f:
        reclassified = json.load(f)

    # Get items where S1_all detected (any of contra/qmis/omis)
    detected_items = []
    for r in reclassified:
        rc = r["reclassified"]
        if rc["contra"]["verdict"]=="INCORRECT" or rc["qmis"]["verdict"]=="INCORRECT" or rc["omis"]["verdict"]=="INCORRECT":
            detected_items.append(r)

    print(f"Correction+Verdict: {len(detected_items)} detected items "
          f"({sum(1 for d in detected_items if d['label']=='wrong')} TP, "
          f"{sum(1 for d in detected_items if d['label']=='correct')} FP)", flush=True)
    print("Methods: P1, P1+pool × V1, V2, V3, None", flush=True)
    print("=" * 70, flush=True)

    save_file = OUTPUT_DIR / "correction_verdict_5fold.json"
    results = []
    done_keys = set()
    if save_file.exists():
        results = json.load(open(save_file))
        done_keys = {(r["fold"], r["idx"]) for r in results}
        print(f"Resuming: {len(done_keys)} done", flush=True)

    for di in detected_items:
        if (di["fold"], di["idx"]) in done_keys: continue

        row = all_df[(all_df["fold"]==di["fold"]) & (all_df["idx"]==di["idx"])]
        if len(row) == 0: continue
        row = row.iloc[0]
        note = notes.get(str(row["patient_id"]), "")
        if not note: continue
        answer = str(row.get("openended_answer", row.get("model_answer", "")))
        gt = row["ground_truth"]

        error_type, error_stmt, correct_stmt, source = get_best_error(di)

        # === CORRECTION: P1 (type-routed) ===
        if error_type == "CONTRADICTION":
            msg = COR_CONTRADICTION.format(note=note, question=row["question"],
                error_statement=error_stmt[:200], correct_statement=correct_stmt[:200])
        elif error_type == "QUESTION_MISALIGNMENT":
            msg = COR_QMIS.format(note=note, question=row["question"],
                error_statement=error_stmt[:200])
        else:
            msg = COR_OMISSION.format(note=note, question=row["question"],
                error_statement=error_stmt[:200], correct_statement=correct_stmt[:200])
        cor_p1 = vllm_gen(build_chatml("You are a medical expert.", msg))

        # === CORRECTION: P1+pool ===
        pool_ex = retrieve_pool(error_stmt, error_type, di["fold"])
        if pool_ex:
            if error_type == "CONTRADICTION":
                msg = COR_POOL_CONTRADICTION.format(note=note, question=row["question"],
                    error_statement=error_stmt[:200], correct_statement=correct_stmt[:200],
                    pool_wrong=pool_ex["text_raw"][:150], pool_correct=pool_ex["gt_atom_raw"][:150])
            elif error_type == "QUESTION_MISALIGNMENT":
                msg = COR_POOL_QMIS.format(note=note, question=row["question"],
                    error_statement=error_stmt[:200],
                    pool_wrong=pool_ex["text_raw"][:150], pool_correct=pool_ex["gt_atom_raw"][:150])
            else:
                msg = COR_POOL_OMISSION.format(note=note, question=row["question"],
                    error_statement=error_stmt[:200], correct_statement=correct_stmt[:200],
                    pool_wrong=pool_ex["text_raw"][:150], pool_correct=pool_ex["gt_atom_raw"][:150])
            cor_pp = vllm_gen(build_chatml("You are a medical expert.", msg))
        else:
            cor_pp = cor_p1  # fallback

        # === GPT-4o EVAL ===
        ev_p1 = gpt4o_eval(note, row["question"], gt, cor_p1)
        ev_pp = gpt4o_eval(note, row["question"], gt, cor_pp)

        # === VERDICTS ===
        verdicts = {}
        for ck, corrected in [("P1", cor_p1), ("PP", cor_pp)]:
            rng = random.Random(42 + hash(str(di["idx"])) + hash(ck))
            orig_is_a = rng.random() > 0.5
            ans_a = answer[:500] if orig_is_a else corrected[:500]
            ans_b = corrected[:500] if orig_is_a else answer[:500]

            # V1
            raw = vllm_gen(build_chatml("You are a medical expert comparing answers.",
                V1_PROMPT.format(note=note, question=row["question"], answer_a=ans_a, answer_b=ans_b)),
                max_tokens=512, temperature=0.0)
            pick = q32_verdict(raw)
            verdicts[f"{ck}_V1"] = (pick == "B") if orig_is_a else (pick == "A")

            # V2
            raw = vllm_gen(build_chatml("You are a medical expert comparing answers.",
                V2_PROMPT.format(note=note, question=row["question"], answer_a=ans_a, answer_b=ans_b)),
                max_tokens=512, temperature=0.0)
            pick = q32_verdict(raw)
            verdicts[f"{ck}_V2"] = (pick == "B") if orig_is_a else (pick == "A")

            # V3
            raw = vllm_gen(build_chatml("You are a medical expert.",
                V3_PROMPT.format(note=note, question=row["question"],
                    error_statement=error_stmt[:200], original=answer[:500], corrected=corrected[:500])),
                max_tokens=512, temperature=0.0)
            pick = q32_verdict(raw)
            verdicts[f"{ck}_V3"] = pick in ("CORRECTED", "B")

        entry = {
            "idx": di["idx"], "fold": di["fold"], "label": di["label"],
            "eval_orig": di["eval_orig"],
            "error_type": error_type, "source": source,
            "ev_p1": ev_p1, "ev_pp": ev_pp,
            "verdicts": verdicts,
            "pool_sim": pool_ex["sim"] if pool_ex else 0,
            "cor_p1": cor_p1[:200], "cor_pp": cor_pp[:200],
        }
        results.append(entry)
        with open(save_file, "w") as f: json.dump(results, f)

        p1s = "FIX" if ev_p1==1 else "FAIL"
        pps = "FIX" if ev_pp==1 else "FAIL"
        p1_acc = [k.split("_")[1] for k in verdicts if k.startswith("P1_") and verdicts[k]]
        pp_acc = [k.split("_")[1] for k in verdicts if k.startswith("PP_") and verdicts[k]]
        print(f"  [{di['label']:>7}] idx={di['idx']} f={di['fold']} {error_type[:5]:>5} "
              f"P1={p1s}[{','.join(p1_acc) or '-'}] PP={pps}[{','.join(pp_acc) or '-'}] "
              f"${spending['cost']:.2f}", flush=True)

    # === SUMMARY ===
    with open(save_file, "w") as f: json.dump(results, f, indent=2)
    tp = [r for r in results if r["label"] == "wrong"]
    fp = [r for r in results if r["label"] == "correct"]

    print(f"\n{'='*70}", flush=True)
    print(f"RESULTS: {len(tp)} TP + {len(fp)} FP", flush=True)
    print(f"{'='*70}", flush=True)

    total=962; total_w=109; total_c=853; base_acc=total_c/total
    n_w_test=50; n_c_test=50
    det_w_rate = len(tp)/n_w_test; det_c_rate = len(fp)/n_c_test

    print(f"\n{'Method':<15} {'TP fix':>7} {'FP brk':>7} {'Net':>5} {'Proj fix':>9} {'Proj brk':>9} {'Proj acc':>10}", flush=True)
    print("-" * 68, flush=True)

    for ck, ev_key in [("P1", "ev_p1"), ("PP", "ev_pp")]:
        # Raw
        tf = sum(1 for r in tp if r[ev_key]==1)
        fb = sum(1 for r in fp if r[ev_key]==0)
        fix_r = tf/max(len(tp),1); brk_r = fb/max(len(fp),1)
        pf = fix_r*det_w_rate*total_w; pb = brk_r*det_c_rate*total_c
        pa = base_acc + (pf-pb)/total
        print(f"  {ck+' raw':<15} {tf:>5}   {fb:>5}  {tf-fb:>+4}  {pf:>7.0f}   {pb:>7.0f}   {100*pa:.2f}% ({100*(pa-base_acc):+.2f}pp)", flush=True)

        for vk in ["V1", "V2", "V3"]:
            combo = f"{ck}_{vk}"
            tf_v = sum(1 for r in tp if r["verdicts"].get(combo) and r[ev_key]==1)
            fb_v = sum(1 for r in fp if r["verdicts"].get(combo) and r[ev_key]==0)
            fix_r = tf_v/max(len(tp),1); brk_r = fb_v/max(len(fp),1)
            pf = fix_r*det_w_rate*total_w; pb = brk_r*det_c_rate*total_c
            pa = base_acc + (pf-pb)/total
            print(f"  {ck+'+'+vk:<15} {tf_v:>5}   {fb_v:>5}  {tf_v-fb_v:>+4}  {pf:>7.0f}   {pb:>7.0f}   {100*pa:.2f}% ({100*(pa-base_acc):+.2f}pp)", flush=True)

    # Per error type
    print(f"\nBy error type:", flush=True)
    for et in ["CONTRADICTION", "OMISSION", "QUESTION_MISALIGNMENT"]:
        et_tp = [r for r in tp if r["error_type"]==et]
        et_fp = [r for r in fp if r["error_type"]==et]
        if et_tp or et_fp:
            tf = sum(1 for r in et_tp if r["ev_p1"]==1)
            fb = sum(1 for r in et_fp if r["ev_p1"]==0)
            print(f"  {et}: TP {tf}/{len(et_tp)} fix | FP {fb}/{len(et_fp)} brk", flush=True)

    print(f"\nGPT-4o: {spending['calls']} calls, ${spending['cost']:.3f}", flush=True)
    print(f"Saved to {save_file}", flush=True)


if __name__ == "__main__":
    main()
