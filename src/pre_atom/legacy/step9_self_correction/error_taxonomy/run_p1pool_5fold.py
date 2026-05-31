#!/usr/bin/env python3
"""
P1+pool pipeline: 10 TP + 10 FP per fold × 5 folds = 50 TP + 50 FP.
S1 detect → P1+pool correct → V2 verdict → GPT-4o eval.
Resume-safe.

Usage:
    python run_p1pool_5fold.py --port 8003
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
QWEN32B_URL = "http://192.168.68.107:8090/v1/chat/completions"

key = None
for line in open(PROJECT_ROOT / ".env"):
    line = line.strip()
    if line.startswith("OPENAI_API_KEY=") and not line.startswith("#"):
        key = line.split("=", 1)[1]; break
from openai import OpenAI
gpt_client = OpenAI(api_key=key)
spending = {"calls": 0, "cost": 0.0}

PORT = 8003

DET_CONTRA = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

CHECK FOR CONTRADICTION: Does the answer state any fact that DIRECTLY CONFLICTS with the discharge notes?

For each key claim (medication names, dosages, procedures, diagnoses, lab values, dates):
1. Find the matching information in the notes
2. Does the answer say something DIFFERENT?

Only flag OPPOSING information. If you find a contradiction, explain what the answer says vs what the notes say."""

DET_QMIS = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

CHECK FOR QUESTION MISALIGNMENT: Does the answer address the WRONG thing?

Parse the question: which visit, what aspect, what time period?
Does the answer match? If misaligned, explain what the question asks vs what the answer discusses."""

DET_OMIS = """Discharge summary:
{note}

Question: {question}

Answer: {answer}

CHECK FOR CRITICAL OMISSION: Is there information in the notes ESSENTIAL to answer the question but COMPLETELY ABSENT?

Only flag omissions that would CHANGE the conclusion. Do NOT flag minor details."""

COR_P1_POOL = """Discharge summary:
{note}

Question: {question}

Your previous answer had this error:
ERROR TYPE: {error_type}
ERROR: {error_statement}
THE NOTES SAY: {correct_statement}

Here is an example of how a similar error was corrected in another clinical case:
  Original claim: "{pool_wrong}"
  Corrected to: "{pool_correct}"
  The correction approach: check the discharge notes for the exact information and use what the notes state.

Apply the same approach and re-answer the question.
Answer in 1-3 direct sentences."""

COR_P1_NOPOOL = """Discharge summary:
{note}

Question: {question}

Your previous answer had this error:
ERROR TYPE: {error_type}
ERROR: {error_statement}
THE NOTES SAY: {correct_statement}

Re-answer the question, fixing this specific error.
Answer in 1-3 direct sentences."""

V2_VERDICT = """Discharge summary:
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

EXTRACT_DET = """/nothink
Read this self-critique. Extract as JSON. Only INCORRECT if critical errors found.

TEXT:
{raw}

{{"verdict": "CORRECT" or "INCORRECT", "error_type": "CONTRADICTION" or "OMISSION" or "QUESTION_MISALIGNMENT" or "NONE", "error_statement": "the error as one sentence", "correct_statement": "what notes say"}}"""

EXTRACT_VERDICT = """/nothink
Which answer was picked?

TEXT:
{raw}

{{"pick": "A" or "B"}}"""

def build_chatml(s, u):
    return f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n"

def vllm_gen(prompt, max_tokens=2048, temperature=0.0):
    model = requests.get(f"http://localhost:{PORT}/v1/models", timeout=5).json()["data"][0]["id"]
    resp = requests.post(f"http://localhost:{PORT}/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": max_tokens,
              "temperature": temperature, "stop": ["<|im_end|>", "<|endoftext|>"]}, timeout=180)
    return resp.json()["choices"][0]["text"].strip()

def q32(raw, tmpl):
    try:
        resp = requests.post(QWEN32B_URL, json={
            "model": "Qwen/Qwen3-32B-MLX-bf16",
            "messages": [{"role": "system", "content": "Extract info. JSON only."},
                         {"role": "user", "content": tmpl.format(raw=raw[:2000])}],
            "max_tokens": 300, "temperature": 0.0,
        }, timeout=90)
        text = resp.json()["choices"][0]["message"]["content"].strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if m: return json.loads(m.group())
    except: pass
    return None

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

    # 10 TP + 10 FP per fold
    test_items = []
    for fold in range(5):
        fold_df = all_df[all_df["fold"] == fold]
        random.seed(fold * 100 + 42)
        wrong = fold_df[fold_df["binary_correct"]==0].sample(n=min(10, (fold_df["binary_correct"]==0).sum()), random_state=fold*100+42)
        correct = fold_df[fold_df["binary_correct"]==1].sample(n=min(10, (fold_df["binary_correct"]==1).sum()), random_state=fold*100+42)
        for _, row in wrong.iterrows():
            test_items.append({"idx": int(row["idx"]), "fold": fold, "label": "wrong", "row": row})
        for _, row in correct.iterrows():
            test_items.append({"idx": int(row["idx"]), "fold": fold, "label": "correct", "row": row})

    n_w = sum(1 for t in test_items if t["label"] == "wrong")
    n_c = sum(1 for t in test_items if t["label"] == "correct")

    save_file = OUTPUT_DIR / "p1pool_5fold_results.json"
    results = []
    done_keys = set()
    if save_file.exists():
        results = json.load(open(save_file))
        done_keys = {(r["fold"], r["idx"]) for r in results}
        print(f"Resuming: {len(done_keys)} done", flush=True)

    print(f"P1+pool 5-fold: {n_w} TP + {n_c} FP", flush=True)
    print("=" * 70, flush=True)

    for ti in test_items:
        if (ti["fold"], ti["idx"]) in done_keys: continue

        row = ti["row"]
        note = notes.get(str(row["patient_id"]), "")
        if not note: continue
        answer = str(row.get("openended_answer", row.get("model_answer", "")))
        gt = row["ground_truth"]
        eval_orig = int(row["binary_correct"])
        sys_det = "You are a strict medical expert checking clinical answers."

        # DETECT
        det = {}
        for dk, dp in [("contra", DET_CONTRA), ("qmis", DET_QMIS), ("omis", DET_OMIS)]:
            raw = vllm_gen(build_chatml(sys_det, dp.format(note=note, question=row["question"], answer=answer[:800])))
            obj = q32(raw, EXTRACT_DET) or {}
            det[dk] = {
                "verdict": str(obj.get("verdict", "UNCLEAR")).upper(),
                "error_type": str(obj.get("error_type", "NONE")).upper(),
                "error_statement": str(obj.get("error_statement", ""))[:250],
                "correct_statement": str(obj.get("correct_statement", ""))[:250],
            }

        detected_types = [k for k in det if det[k]["verdict"] == "INCORRECT"]

        if not detected_types:
            results.append({"idx": ti["idx"], "fold": ti["fold"], "label": ti["label"],
                            "eval_orig": eval_orig, "detected": False})
            with open(save_file, "w") as f: json.dump(results, f)
            if len(results) % 10 == 0:
                w_d = sum(1 for r in results if r["label"]=="wrong" and r.get("detected"))
                c_d = sum(1 for r in results if r["label"]=="correct" and r.get("detected"))
                w_t = sum(1 for r in results if r["label"]=="wrong")
                c_t = sum(1 for r in results if r["label"]=="correct")
                print(f"  [{len(results)}/{len(test_items)}] w={w_d}/{w_t} c={c_d}/{c_t} ${spending['cost']:.2f}", flush=True)
            continue

        # Most critical
        mc = None
        for dk in ["qmis", "contra", "omis"]:
            if det[dk]["verdict"] == "INCORRECT": mc = det[dk]; break

        # CORRECT P1+pool
        pool_ex = retrieve_pool(mc["error_statement"], mc["error_type"], ti["fold"])
        if pool_ex:
            msg = COR_P1_POOL.format(note=note, question=row["question"],
                error_type=mc["error_type"], error_statement=mc["error_statement"][:200],
                correct_statement=mc["correct_statement"][:200],
                pool_wrong=pool_ex["text_raw"][:150], pool_correct=pool_ex["gt_atom_raw"][:150])
        else:
            msg = COR_P1_NOPOOL.format(note=note, question=row["question"],
                error_type=mc["error_type"], error_statement=mc["error_statement"][:200],
                correct_statement=mc["correct_statement"][:200])
        corrected = vllm_gen(build_chatml("You are a medical expert.", msg), max_tokens=512, temperature=1.0)

        # VERDICT V2
        rng = random.Random(42 + hash(str(ti["idx"])))
        orig_is_a = rng.random() > 0.5
        ans_a = answer[:500] if orig_is_a else corrected[:500]
        ans_b = corrected[:500] if orig_is_a else answer[:500]
        v_raw = vllm_gen(build_chatml("You are a medical expert comparing answers.",
            V2_VERDICT.format(note=note, question=row["question"], answer_a=ans_a, answer_b=ans_b)),
            max_tokens=512, temperature=0.0)
        v_obj = q32(v_raw, EXTRACT_VERDICT) or {}
        pick = str(v_obj.get("pick", "UNCLEAR")).upper()
        accept = (pick == "B") if orig_is_a else (pick == "A")

        # GPT-4o eval (only corrected — original eval is known)
        ev_cor = gpt4o_eval(note, row["question"], gt, corrected)

        results.append({
            "idx": ti["idx"], "fold": ti["fold"], "label": ti["label"],
            "eval_orig": eval_orig, "detected": True, "det_types": detected_types,
            "error_type": mc["error_type"],
            "error_statement": mc["error_statement"][:200],
            "eval_corrected": ev_cor, "verdict_accept": accept,
            "pool_sim": pool_ex["sim"] if pool_ex else 0,
            "corrected": corrected[:200],
        })
        with open(save_file, "w") as f: json.dump(results, f)

        if len(results) % 5 == 0:
            w_d = sum(1 for r in results if r["label"]=="wrong" and r.get("detected"))
            c_d = sum(1 for r in results if r["label"]=="correct" and r.get("detected"))
            w_t = sum(1 for r in results if r["label"]=="wrong")
            c_t = sum(1 for r in results if r["label"]=="correct")
            print(f"  [{len(results)}/{len(test_items)}] w={w_d}/{w_t} c={c_d}/{c_t} ${spending['cost']:.2f}", flush=True)

    # SUMMARY
    with open(save_file, "w") as f: json.dump(results, f, indent=2)

    det_items = [r for r in results if r.get("detected")]
    tp = [r for r in det_items if r["label"] == "wrong"]
    fp = [r for r in det_items if r["label"] == "correct"]
    all_w = [r for r in results if r["label"] == "wrong"]
    all_c = [r for r in results if r["label"] == "correct"]

    print(f"\n{'='*70}", flush=True)
    print(f"P1+POOL 5-FOLD RESULTS: {len(all_w)} wrong + {len(all_c)} correct", flush=True)
    print(f"Detection: TP={len(tp)}/{len(all_w)} ({100*len(tp)/len(all_w):.0f}%) FP={len(fp)}/{len(all_c)} ({100*len(fp)/len(all_c):.0f}%)", flush=True)

    tp_fix_raw = sum(1 for r in tp if r["eval_corrected"] == 1)
    fp_brk_raw = sum(1 for r in fp if r["eval_corrected"] == 0)
    tp_fix_v2 = sum(1 for r in tp if r["verdict_accept"] and r["eval_corrected"] == 1)
    fp_brk_v2 = sum(1 for r in fp if r["verdict_accept"] and r["eval_corrected"] == 0)
    tp_rej = sum(1 for r in tp if not r["verdict_accept"])
    fp_rej = sum(1 for r in fp if not r["verdict_accept"])

    print(f"\nRaw: TP fix={tp_fix_raw}/{len(tp)} FP brk={fp_brk_raw}/{len(fp)} net={tp_fix_raw-fp_brk_raw:+d}", flush=True)
    print(f"+V2: TP fix={tp_fix_v2} rej={tp_rej} | FP brk={fp_brk_v2} rej={fp_rej} | net={tp_fix_v2-fp_brk_v2:+d}", flush=True)

    # Projection
    total=962; total_w=109; total_c=853
    base_acc = total_c/total
    det_w = len(tp)/max(len(all_w),1); det_c = len(fp)/max(len(all_c),1)
    fix_r = tp_fix_v2/max(len(tp),1); brk_r = fp_brk_v2/max(len(fp),1)
    pf = fix_r*det_w*total_w; pb = brk_r*det_c*total_c
    pa = base_acc + (pf-pb)/total
    print(f"\nProjected: fix={pf:.0f} brk={pb:.0f} net={pf-pb:+.0f} acc={100*pa:.2f}% ({100*(pa-base_acc):+.2f}pp)", flush=True)

    # Per fold
    print(f"\nPer fold:", flush=True)
    for fold in range(5):
        fw = [r for r in results if r["fold"]==fold and r["label"]=="wrong"]
        fc = [r for r in results if r["fold"]==fold and r["label"]=="correct"]
        fw_d = sum(1 for r in fw if r.get("detected"))
        fc_d = sum(1 for r in fc if r.get("detected"))
        fw_fix = sum(1 for r in fw if r.get("detected") and r.get("verdict_accept") and r.get("eval_corrected")==1)
        fc_brk = sum(1 for r in fc if r.get("detected") and r.get("verdict_accept") and r.get("eval_corrected")==0)
        print(f"  Fold {fold}: det w={fw_d}/{len(fw)} c={fc_d}/{len(fc)} | fix={fw_fix} brk={fc_brk}", flush=True)

    print(f"\nGPT-4o: {spending['calls']} calls, ${spending['cost']:.3f}", flush=True)
    print(f"Saved to {save_file}", flush=True)


if __name__ == "__main__":
    main()
