#!/usr/bin/env python3
"""
Audit one fix case from fullscale_results.json.
Re-executes the full pipeline (detect → correct → verdict → eval) for ONE item,
printing every input and output at every stage so we can verify parsing is correct.

Usage:
    python audit_one_case.py --fold 0 --idx 51
"""
import json, random, re, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer

# Reuse prompts + helpers from run_fullscale.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_fullscale import (
    DET_CONTRA, DET_QMIS, DET_OMIS,
    COR_CONTRADICTION, COR_OMISSION, COR_QMIS, COR_NOPOOL,
    V1_PROMPT, EXTRACT_DET, EXTRACT_VERDICT,
    build_chatml, load_notes, get_embedder,
    PROJECT_ROOT, OUTPUT_DIR, QWEN32B_URL, gpt_client, spending,
)
import run_fullscale  # to set PORT

SEP = "=" * 100
SUB = "-" * 100


def banner(label):
    print(f"\n{SEP}\n{label}\n{SEP}", flush=True)


def show(label, value, max_chars=2000):
    s = str(value)
    if len(s) > max_chars:
        s = s[:max_chars] + f"\n... [TRUNCATED, total {len(str(value))} chars]"
    print(f"\n[{label}]\n{s}", flush=True)


def vllm_gen_audit(prompt, max_tokens=2048, temperature=0.0, label=""):
    port = run_fullscale.PORT
    model = requests.get(f"http://localhost:{port}/v1/models", timeout=5).json()["data"][0]["id"]
    payload = {
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": temperature, "stop": ["<|im_end|>", "<|endoftext|>"],
    }
    print(f"\n>>> vLLM call [{label}] (model={model}, max_tokens={max_tokens}, temp={temperature})", flush=True)
    resp = requests.post(f"http://localhost:{port}/v1/completions", json=payload, timeout=180)
    text = resp.json()["choices"][0]["text"].strip()
    return text


def q32_audit(raw, tmpl, label=""):
    print(f"\n>>> Qwen3-32B extract [{label}]", flush=True)
    try:
        resp = requests.post(QWEN32B_URL, json={
            "model": "Qwen/Qwen3-32B-MLX-bf16",
            "messages": [
                {"role": "system", "content": "Extract info. JSON only."},
                {"role": "user", "content": tmpl.format(raw=raw[:2000])},
            ],
            "max_tokens": 300, "temperature": 0.0,
        }, timeout=90)
        text = resp.json()["choices"][0]["message"]["content"].strip()
        text_no_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        show("Qwen3-32B raw response", text)
        if text != text_no_think:
            show("Qwen3-32B (think stripped)", text_no_think)
        m = re.search(r'\{[^{}]*\}', text_no_think, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            show("Parsed JSON", json.dumps(obj, indent=2))
            return obj
        else:
            print("  !! No JSON match in response", flush=True)
    except Exception as e:
        print(f"  !! Qwen3-32B error: {e}", flush=True)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--idx", type=int, default=51)
    p.add_argument("--port", type=int, default=8003)
    p.add_argument("--no-gpt4o", action="store_true", help="skip GPT-4o eval")
    args = p.parse_args()
    run_fullscale.PORT = args.port

    banner(f"AUDIT: fold={args.fold} idx={args.idx}")

    # ----- Load source data -----
    f = PROJECT_ROOT / "output" / "step8" / "qwen2.5-7b-instruct" / f"fold_{args.fold}" / "zeroshot_evaluated_binary.csv"
    df = pd.read_csv(f)
    row = df[df["idx"] == args.idx].iloc[0]
    notes = load_notes()
    note = notes[str(row["patient_id"])]
    answer = str(row.get("openended_answer", row.get("model_answer", "")))
    gt = row["ground_truth"]
    eval_orig = int(row["binary_correct"])

    show("patient_id", row["patient_id"])
    show("question", row["question"])
    show("ground_truth", gt)
    show("original answer (Qwen2.5-7B zeroshot)", answer)
    show("eval_orig (1=correct, 0=wrong)", eval_orig)
    show("note", note, max_chars=1500)

    # ----- STAGE 1: DETECTION (3 sub-prompts) -----
    banner("STAGE 1: DETECTION (3 free-form sub-prompts)")
    sys_det = "You are a strict medical expert checking clinical answers."
    det = {}
    det_raws = {}
    for dk, dp in [("contra", DET_CONTRA), ("qmis", DET_QMIS), ("omis", DET_OMIS)]:
        print(f"\n{SUB}\nSUB-PROMPT: {dk}\n{SUB}", flush=True)
        full_user = dp.format(note=note, question=row["question"], answer=answer[:800])
        show(f"{dk} user prompt", full_user, max_chars=1500)
        prompt = build_chatml(sys_det, full_user)
        raw = vllm_gen_audit(prompt, label=f"detect-{dk}")
        det_raws[dk] = raw
        show(f"{dk} vLLM raw output", raw)
        obj = q32_audit(raw, EXTRACT_DET, label=f"extract-{dk}") or {}
        det[dk] = {
            "verdict": str(obj.get("verdict", "UNCLEAR")).upper(),
            "error_type": str(obj.get("error_type", "NONE")).upper(),
            "error_statement": str(obj.get("error_statement", ""))[:250],
            "correct_statement": str(obj.get("correct_statement", ""))[:250],
        }
        show(f"{dk} parsed det dict", json.dumps(det[dk], indent=2))

    # ----- AGGREGATE & PICK MOST CRITICAL -----
    banner("AGGREGATE: pick most critical error (priority qmis > contra > omis)")
    detected_types = [k for k in ["contra", "qmis", "omis"] if det[k]["verdict"] == "INCORRECT"]
    show("detected_types", detected_types)
    if not detected_types:
        print("\n!! No detection — pipeline would skip correction. STOP.", flush=True)
        return
    mc = None
    mc_type = "OMISSION"
    for dk in ["qmis", "contra", "omis"]:
        if det[dk]["verdict"] == "INCORRECT":
            mc = det[dk]
            mc_type = {"qmis": "QUESTION_MISALIGNMENT", "contra": "CONTRADICTION", "omis": "OMISSION"}[dk]
            break
    show("most-critical sub-prompt picked", mc_type)
    show("most-critical det dict", json.dumps(mc, indent=2))

    # ----- POOL RETRIEVAL -----
    banner("POOL RETRIEVAL (RA-ICL: similar BioMistral error from pool)")
    pool_f = PROJECT_ROOT / "workspace" / "self_critique" / "data" / "bm_atomic_pool" / f"fold_{args.fold}_atoms.json"
    emb_f = PROJECT_ROOT / "workspace" / "self_critique" / "data" / "bm_atomic_pool" / f"fold_{args.fold}_atom_embeddings.npy"
    show("pool file", str(pool_f))
    show("pool exists", pool_f.exists() and emb_f.exists())
    pool = json.load(open(pool_f))
    pool_emb = np.load(emb_f)
    type_map = {"CONTRADICTION": "factual_error", "OMISSION": "omission", "QUESTION_MISALIGNMENT": "factual_error"}
    target = type_map.get(mc_type, "factual_error")
    show("pool target type", target)
    indices = [i for i, a in enumerate(pool) if a.get("main_error_type") == target and a.get("gt_atom_raw")]
    show("# pool atoms matching target type", len(indices))
    if not indices:
        indices = [i for i, a in enumerate(pool) if a.get("gt_atom_raw")]
        show("# fallback (any type)", len(indices))
    query_emb = get_embedder().encode([mc["error_statement"]], normalize_embeddings=True)
    sims = np.dot(pool_emb[indices], query_emb.T).flatten()
    top = int(np.argsort(-sims)[0])
    atom = pool[indices[top]]
    pool_ex = {"text_raw": atom["text_raw"], "gt_atom_raw": atom["gt_atom_raw"], "sim": float(sims[top])}
    show("query (error_statement)", mc["error_statement"])
    show("retrieved pool example sim", pool_ex["sim"])
    show("retrieved pool wrong (text_raw)", pool_ex["text_raw"])
    show("retrieved pool correct (gt_atom_raw)", pool_ex["gt_atom_raw"])

    # ----- STAGE 2: CORRECTION -----
    banner("STAGE 2: CORRECTION (P1+pool, type-routed, temp=1.0)")
    if mc_type == "CONTRADICTION":
        msg = COR_CONTRADICTION
    elif mc_type == "QUESTION_MISALIGNMENT":
        msg = COR_QMIS
    else:
        msg = COR_OMISSION
    fmt = {
        "note": note, "question": row["question"],
        "error_statement": mc["error_statement"][:200],
        "correct_statement": mc["correct_statement"][:200],
        "pool_wrong": pool_ex["text_raw"][:150],
        "pool_correct": pool_ex["gt_atom_raw"][:150],
    }
    cor_user = msg.format(**fmt)
    show("correction user prompt", cor_user, max_chars=2500)
    corrected = vllm_gen_audit(build_chatml("You are a medical expert.", cor_user),
                               max_tokens=512, temperature=1.0, label="correction")
    show("CORRECTED ANSWER (vLLM raw)", corrected)

    # ----- STAGE 3: VERDICT V1 -----
    banner("STAGE 3: VERDICT V1 (contradiction-count A vs B)")
    rng = random.Random(42 + hash(str(args.idx)))
    orig_is_a = rng.random() > 0.5
    show("orig_is_a (random A/B placement)", orig_is_a)
    ans_a = answer[:500] if orig_is_a else corrected[:500]
    ans_b = corrected[:500] if orig_is_a else answer[:500]
    show("ANSWER A", ans_a)
    show("ANSWER B", ans_b)
    v_user = V1_PROMPT.format(note=note, question=row["question"], answer_a=ans_a, answer_b=ans_b)
    show("V1 user prompt", v_user, max_chars=2000)
    v_raw = vllm_gen_audit(build_chatml("You are a medical expert comparing answers.", v_user),
                           max_tokens=512, temperature=0.0, label="verdict-v1")
    show("V1 vLLM raw", v_raw)
    v_obj = q32_audit(v_raw, EXTRACT_VERDICT, label="extract-verdict") or {}
    pick = str(v_obj.get("pick", "UNCLEAR")).upper()
    show("V1 pick", pick)
    accept = (pick == "B") if orig_is_a else (pick == "A")
    show("V1 accept (corrected wins)", accept)

    # ----- STAGE 4: GPT-4o EVAL -----
    if accept and not args.no_gpt4o:
        banner("STAGE 4: GPT-4o EVAL of corrected answer")
        from run_fullscale import gpt4o_eval
        ev_cor = gpt4o_eval(note, row["question"], gt, corrected)
        show("GPT-4o eval (1=correct, 0=wrong, -1=parse fail)", ev_cor)
        final_eval = ev_cor
        action = "corrected"
    else:
        final_eval = eval_orig
        action = "kept_original"
    banner("FINAL")
    show("action", action)
    show("eval_orig", eval_orig)
    show("final_eval", final_eval)
    show("delta", final_eval - eval_orig)

    # Compare against stored result
    stored_path = OUTPUT_DIR / "fullscale_results.json"
    stored_all = json.load(open(stored_path))
    stored = next((x for x in stored_all if x["fold"] == args.fold and x["idx"] == args.idx), None)
    show("STORED entry from fullscale_results.json", json.dumps(stored, indent=2))


if __name__ == "__main__":
    main()
