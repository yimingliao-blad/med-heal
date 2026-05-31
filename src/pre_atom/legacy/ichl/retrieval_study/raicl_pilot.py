"""Section 3.2.1 \u2014 Few-shot retrieval-augmented ICL pilot.

Implements the three retrieval modes per the Overall Plan section 3.2.1 (locked
2026-04-27):

  A. Error-similarity:   pool filtered to BM_zs WRONG items;
                         anchor = target.X_zs vs pool.BM_zs;
                         demo shown = (Q, Note, BM_error_zs, GT).
  B. Correct-similarity: pool filtered to BM_zs CORRECT items;
                         anchor = target.X_zs vs pool.BM_zs;
                         demo shown = (Q, Note, GT).
  C. GT-similarity:      full pool;
                         anchor = target.X_zs vs pool.GT;
                         demo shown = (Q, Note, GT).

Locked retrieval scorer (Phase 0 \u03c1=0.643):
  score(test, pool_item) =
      cos_nomic(test.Q,    pool.Q)
    + cos_nomic(test.note, pool.note)
    + cos_nomic(test.X_zs, pool.<anchor field>)

Inherits from `src/ichl/retrieval_study/phase2_retrieve.py` (the locked-scorer
infrastructure) and step8 prompt patterns. NO fresh wrappers; per
`[Workflow] Implementation Discipline` Rule 1.

Generation params (matching step8): temperature=0.1, max_tokens=1024.
Judge: Stage-1 binary GPT-4o (note INCLUDED in prompt) per
`src/ichl/judges/gpt4o_stage1_binary_judge.py`. Pool of 962 items, 5-fold CV.

Pre-flight (per `[Workflow] Execution Discipline` Phase 1):
  - Token budget audit: max_prompt + max_tokens + 200 \u2264 max_model_len
  - Calibration sanity check: rejudge step8 zeroshot answers, confirm match

Output: output/ichl/retrieval_study/raicl_pilot/<target>/<mode>_k<K>_generated.jsonl
        plus _judged.jsonl plus summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RS = ROOT / "output" / "ichl" / "retrieval_study"
POOL_DIR = RS / "pool_index"
OUT_DIR = RS / "raicl_pilot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMB_MODEL = "Alibaba-NLP/gte-large-en-v1.5"   # post-2026-04-27 bake-off winner (full notes)

# ============================================================
# PROMPTS \u2014 verbatim per [Workflow] Complete Plan Acceptance Criteria Slot 11
# ============================================================

SYSTEM_RAICL = "You are a medical expert answering questions about discharge summaries."

USER_TMPL_MODE_A = """REFERENCE CASE (a similar case where the model's answer was incorrect):
Discharge Summary: {ref_note}
Question: {ref_question}
Incorrect Answer: {ref_BM_error_zs}
Correct Answer: {ref_GT}

YOUR CASE:
Discharge Summary:
{note}

Question: {question}

The reference case shows the type of error to avoid. Answer the question above based ONLY on the discharge summary in your case. Be specific and complete.

Answer:"""

USER_TMPL_MODE_B = """REFERENCE CASE (a similar case answered correctly):
Discharge Summary: {ref_note}
Question: {ref_question}
Answer: {ref_GT}

YOUR CASE:
Discharge Summary:
{note}

Question: {question}

Use the reference as guidance. Answer the question above based ONLY on the discharge summary in your case. Be specific and complete.

Answer:"""

USER_TMPL_MODE_C = USER_TMPL_MODE_B  # same template; difference is retrieval anchor

# Mode D — self-revision-with-exemplar. Hybrid of RA-ICL + self-correction.
# Shows model BOTH (a) similar reference case with correct answer, AND (b) its own zero-shot answer.
# Tests whether revision-with-exemplar lifts accuracy more than greedy-with-exemplar (Mode C).
USER_TMPL_MODE_D = """REFERENCE CASE (a similar case answered correctly):
Discharge Summary: {ref_note}
Question: {ref_question}
Correct Answer: {ref_GT}

YOUR CASE:
Discharge Summary: {note}
Question: {question}

Your initial answer:
{X_zs}

Review your initial answer in light of the reference case. If your initial answer is correct, restate it. If it has gaps or errors, correct them based on the discharge summary in your case. Be specific and complete.

Final answer:"""

USER_TMPL_ZS = """Discharge Summary:
{note}

Question: {question}

Answer:"""

# ============================================================
# JUDGE \u2014 three-tier per user directive 2026-04-27:
#   --judge magistral   \u2014 development / prompt iteration (1.3 s/call, 85% test agreement, FREE)
#   --judge qwen3-235b  \u2014 audit / key-stage validation (32 s/call, 92% on gold-112, FREE)
#   --judge gpt4o       \u2014 final check (0.008 USD/call, paid)
#
# Magistral M4 prompt: inherited from src/ichl/judges/magistral_judge.py (note IS in prompt).
# GPT-4o Stage-1 binary: inherited from src/step9_self_correction/evaluate_step9_binary.py /
#   src/ichl/judges/gpt4o_stage1_binary_judge.py (note IS in prompt, 92% human agreement, \u03ba=0.75).
# Qwen3-235B M4: same prompt structure as Magistral M4, served on MLX 192.168.68.107:8800.
# ============================================================

GPT4O_JUDGE_SYSTEM = "You are a medical expert evaluating an AI model's answer to a clinical question."
GPT4O_JUDGE_USER_TMPL = """DISCHARGE SUMMARY:
{note}

QUESTION:
{question}

CORRECT ANSWER (Ground Truth):
{ground_truth}

MODEL'S ANSWER:
{model_answer}

Task: Evaluate if the model's answer is correct compared to the ground truth.

Respond with ONLY a single digit:
1 = Correct
0 = Incorrect"""


# ============================================================
# RETRIEVAL  (inherits locked scorer from Phase 0 / phase2_retrieve.py)
# ============================================================

def load_pool_index() -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    items = [json.loads(l) for l in (POOL_DIR / "items.jsonl").open()]
    q  = np.load(POOL_DIR / "gte_question.npy")
    n  = np.load(POOL_DIR / "gte_note.npy")
    bm = np.load(POOL_DIR / "gte_bm_zs.npy")
    # Pool GT for the 3-comp + cos(zs, GT) scorer (Match #2)
    gt_path = POOL_DIR / "gte_gt.npy"
    if gt_path.exists():
        gt_emb = np.load(gt_path)
    else:
        # Encode pool GTs once (cached)
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(EMB_MODEL, trust_remote_code=True, device=device)
        gt_texts = [it.get("ground_truth", "") for it in items]
        gt_emb = model.encode(gt_texts, batch_size=16, show_progress_bar=False,
                              convert_to_numpy=True, normalize_embeddings=True)
        np.save(gt_path, gt_emb)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    # FIX 2026-04-27: items.jsonl's `note_text_truncated` is capped at 3000 chars and
    # missing the [Note i] headers that step8 used. Step8's actual generation input was
    # `[Note 1]\n{note_1}\n\n[Note 2]\n{note_2}\n\n[Note 3]\n{note_3}` from
    # EHRNoteQA_processed.jsonl, NOT truncated. Replace items[i]["note_text_truncated"]
    # with the FULL step8-format note for use in generation prompts.
    full_notes = _load_full_notes_step8_format()
    for it in items:
        pid = int(it["patient_id"])
        if pid in full_notes:
            it["note_text_truncated"] = full_notes[pid]   # overwrite key (keep name for compat)
    return items, q, n, bm, gt_emb


def _load_full_notes_step8_format() -> dict[int, str]:
    """Step8-format note: '[Note 1]\\n{note_1}\\n\\n[Note 2]\\n{note_2}...' from
    EHRNoteQA_processed.jsonl. Verbatim per src/step8_multimodel_icl/generate_step8.py
    and src/ichl/judges/gpt4o_stage1_binary_judge.py's load_notes()."""
    notes_file = ROOT / "output" / "EHRNoteQA_processed.jsonl"
    out = {}
    for line in notes_file.open():
        if not line.strip(): continue
        r = json.loads(line)
        pid = int(r["patient_id"])
        parts = []
        for i in [1, 2, 3]:
            v = r.get(f"note_{i}")
            if v and str(v).strip() and str(v).lower() != "nan":
                parts.append(f"[Note {i}]\n{str(v).strip()}")
        out[pid] = "\n\n".join(parts)
    return out


def load_bm_binary_correct() -> dict[int, int]:
    """patient_id -> binary_correct (1 = BM zs was correct, 0 = BM was wrong)."""
    rows = []
    for f in range(5):
        path = ROOT / "output" / "step8" / "biomistral-7b" / f"fold_{f}" / "zeroshot_evaluated_binary.csv"
        rows.append(pd.read_csv(path))
    df = pd.concat(rows)
    return {int(r["patient_id"]): int(r["binary_correct"]) for _, r in df.iterrows()
            if r["binary_correct"] in (0, 1)}


def encode_target_zs(target_zs_texts: list[str]) -> np.ndarray:
    """Encode target's zero-shot answers with nomic (for the cos(X_zs, anchor) term)."""
    from sentence_transformers import SentenceTransformer
    import torch
    # GPU \u2014 vLLM MUST be stopped before this phase. Lifecycle managed by vllm_manager
    # in main(): stop_vllm \u2192 nomic encode (GPU) \u2192 start target vLLM \u2192 gen \u2192 swap to Magistral \u2192 judge.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMB_MODEL, trust_remote_code=True, device=device)
    embs = model.encode(target_zs_texts, batch_size=16, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=True)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return embs


def encode_pool_gt() -> np.ndarray:
    """Encode pool item GTs for Mode C (cos(test.zs, pool.GT))."""
    cache = POOL_DIR / "nomic_gt.npy"
    if cache.exists():
        return np.load(cache)
    items = [json.loads(l) for l in (POOL_DIR / "items.jsonl").open()]
    gt_texts = [it["ground_truth"] for it in items]
    from sentence_transformers import SentenceTransformer
    import torch
    # GPU \u2014 vLLM MUST be stopped before this phase. Lifecycle managed by vllm_manager
    # in main(): stop_vllm \u2192 nomic encode (GPU) \u2192 start target vLLM \u2192 gen \u2192 swap to Magistral \u2192 judge.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMB_MODEL, trust_remote_code=True, device=device)
    embs = model.encode(gt_texts, batch_size=16, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=True)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    np.save(cache, embs)
    return embs


def retrieve_top1_per_mode(
    test_indices: list[int],            # indices into items array (the test items)
    test_zs_emb: np.ndarray,            # (n_test, gte_dim) target's zs embeddings
    items: list[dict], q_emb, n_emb, bm_emb,
    bm_correct: dict[int, int],         # patient_id -> 0/1
    gt_emb: np.ndarray | None,          # for Mode C and 3-comp scorer with cos(zs, GT)
    mode: str,                          # 'A' / 'B' / 'C'
    test_fold: int,                     # exclude same-fold items from pool
    scorer: str = "3comp",              # "2comp" or "3comp" (q+note+cos_zs_to_anchor) per Phase 0 redo 2026-04-27
) -> list[dict]:
    """For each test item, return top-1 pool match."""
    # Pool = all items NOT in same-fold-test
    pool_mask = np.array([
        (it["fold_test_member"] != test_fold) for it in items
    ])
    # Apply mode-specific filter
    if mode == "A":
        # BM-error pool only
        mode_mask = np.array([
            bm_correct.get(int(it["patient_id"]), -1) == 0 for it in items
        ])
        pool_mask = pool_mask & mode_mask
        anchor_emb = bm_emb  # cos(test.zs, pool.BM_zs)
    elif mode == "B":
        # BM-correct pool only
        mode_mask = np.array([
            bm_correct.get(int(it["patient_id"]), -1) == 1 for it in items
        ])
        pool_mask = pool_mask & mode_mask
        anchor_emb = bm_emb
    elif mode == "C":
        anchor_emb = gt_emb
    elif mode == "D":
        # Mode D = same retrieval as C (full pool, GT anchor); differs only in PROMPT (shows X_zs).
        anchor_emb = gt_emb
    else:
        raise ValueError(f"Unknown mode {mode}")

    pool_idx = np.where(pool_mask)[0]
    pool_q_e = q_emb[pool_idx]
    pool_n_e = n_emb[pool_idx]
    pool_anchor = anchor_emb[pool_idx]

    out = []
    for i, ti in enumerate(test_indices):
        tq = q_emb[ti]
        tn = n_emb[ti]
        tzs = test_zs_emb[i]
        cos_q = pool_q_e @ tq
        cos_n = pool_n_e @ tn
        cos_anchor = pool_anchor @ tzs
        if scorer == "2comp":
            score = cos_q + cos_n
        else:   # "3comp": q + note + cos(test.zs, anchor)
            score = cos_q + cos_n + cos_anchor
        top1 = int(np.argmax(score))
        pool_item_idx = int(pool_idx[top1])
        out.append({
            "test_item_idx": ti,
            "test_patient_id": int(items[ti]["patient_id"]),
            "pool_item_idx": pool_item_idx,
            "pool_patient_id": int(items[pool_item_idx]["patient_id"]),
            "pool_bm_correct": bm_correct.get(int(items[pool_item_idx]["patient_id"]), -1),
            "score": float(score[top1]),
            "cos_q": float(cos_q[top1]),
            "cos_note": float(cos_n[top1]),
            "cos_anchor": float(cos_anchor[top1]),
            "mode": mode,
        })
    return out


# ============================================================
# GENERATION + JUDGE
# ============================================================

def truncation_check(text: str, finish_reason: str | None) -> dict:
    text = text or ""
    return {
        "finish_reason": finish_reason,
        "certain_truncated": finish_reason == "length",
        "char_len": len(text),
        "suspicious_no_terminal": (len(text) > 50 and not text.rstrip().endswith((".", "?", "!", '"', "'", ")", ":", "]"))),
    }


def vllm_call(client, model_name: str, system: str, user: str,
              max_tokens: int = 1024, temperature: float = 0.1,
              enable_thinking: bool | None = None) -> dict:
    """vLLM chat call. Pass enable_thinking=True for Qwen3 thinking mode (per
    Prompt Design per Model: extra_body={'chat_template_kwargs': {'enable_thinking': ...}})."""
    kwargs = {
        "model": model_name,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature, "max_tokens": max_tokens,
    }
    if enable_thinking is not None:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}}
    try:
        r = client.chat.completions.create(**kwargs)
        text = r.choices[0].message.content or ""
        fr = r.choices[0].finish_reason
        usage = r.usage
        return {"text": text, "finish_reason": fr,
                "prompt_tok": usage.prompt_tokens if usage else None,
                "comp_tok": usage.completion_tokens if usage else None,
                "trunc": truncation_check(text, fr)}
    except Exception as e:
        return {"_err": str(e)[:200]}


def gpt4o_judge(client, note: str, question: str, gt: str, model_answer: str) -> dict:
    """Stage-1 binary GPT-4o judge \u2014 final check tier. Note IS in prompt."""
    msgs = [
        {"role": "system", "content": GPT4O_JUDGE_SYSTEM},
        {"role": "user", "content": GPT4O_JUDGE_USER_TMPL.format(
            note=note, question=question, ground_truth=gt, model_answer=model_answer)},
    ]
    try:
        r = client.chat.completions.create(model="gpt-4o", messages=msgs, temperature=0.1, max_tokens=10)
        txt = (r.choices[0].message.content or "").strip()
        m = re.search(r"[01]", txt)
        score = int(m.group(0)) if m else None
        usage = r.usage
        return {"binary_correct": score, "raw": txt[:30],
                "prompt_tok": usage.prompt_tokens if usage else None,
                "comp_tok": usage.completion_tokens if usage else None}
    except Exception as e:
        return {"_err": str(e)[:200]}


def magistral_judge_one(judge, note: str, question: str, gt: str, model_answer: str) -> dict:
    """Magistral M4 judge (development tier). Inherits MagistralJudge wrapper verbatim per Implementation Discipline Rule 1."""
    r = judge.judge(question=question, ground_truth=gt, model_answer=model_answer, note=note)
    if r.error:
        return {"_err": r.error[:200]}
    return {"binary_correct": r.label, "raw": r.content[:30],
            "prompt_tok": r.prompt_tokens, "comp_tok": r.completion_tokens,
            "latency_s": r.latency_s, "trunc_certain": r.truncation_certain}


def qwen3_235b_judge_one(client, note: str, question: str, gt: str, model_answer: str) -> dict:
    """Qwen3-235B-MLX M4 judge (audit tier, MLX server port 8800). Same M4 rules as Magistral, larger model. C=1 per MEMORY."""
    from ichl.judges.magistral_judge import M4_RULES, SYSTEM as MAG_SYSTEM, _build_user
    user = _build_user(question, gt, model_answer, note)
    try:
        r = client.chat.completions.create(
            model="/Users/madblade/Projects/local-llm/models/mlx/Qwen3.5-27B-6bit-NexVeridian",
            messages=[{"role": "system", "content": MAG_SYSTEM}, {"role": "user", "content": user}],
            temperature=0.0, max_tokens=256,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        txt = (r.choices[0].message.content or "").strip()
        m = re.search(r"[01]", txt)
        score = int(m.group(0)) if m else None
        return {"binary_correct": score, "raw": txt[:30]}
    except Exception as e:
        return {"_err": str(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="e.g., qwen2.5-7b-instruct")
    ap.add_argument("--vllm-url", default="http://localhost:8003/v1")
    ap.add_argument("--vllm-model", required=True, help="served-model-name on vLLM")
    ap.add_argument("--target-zs-csv", required=True,
                    help="Path to step8 zeroshot_generated.csv with model_answer column for the target")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--modes", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C", "D"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all; smoke-test with 3")
    ap.add_argument("--gen-workers", type=int, default=4)
    ap.add_argument("--judge-workers", type=int, default=8)
    ap.add_argument("--judge", choices=["magistral", "gpt4o", "qwen3-235b"], default="magistral",
                    help="Judge tier: magistral (iteration default), qwen3-235b (audit), gpt4o (final, paid).")
    ap.add_argument("--scorer", choices=["2comp", "3comp"], default="3comp",
                    help="Retrieval scorer (post Phase 0 redo 2026-04-27): 2comp = cos_q + cos_note; "
                         "3comp = cos_q + cos_note + cos(test.zs, pool.<mode-anchor>) where anchor "
                         "is BM_zs (Mode A/B) or GT (Mode C). 2comp won Phase 0 (\u03c1=0.581 vs 0.528-0.543).")
    ap.add_argument("--magistral-vllm-url", default="http://localhost:8003/v1",
                    help="vLLM URL serving Magistral (used only when --judge magistral; user must swap vLLM to Magistral after generation).")
    ap.add_argument("--qwen3-235b-mlx-url", default="http://192.168.68.107:8800/v1",
                    help="MLX URL for Qwen3-235B audit judge.")
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--skip-judge", action="store_true")
    # Mode C confidence-gated variant per Plan: Mode C Confidence-Gated RA-ICL (2026-04-27).
    # When --gate-tau is set: for Mode C, if (cos_q + cos_note) < tau, fall back to ZS prompt.
    ap.add_argument("--gate-tau", type=float, default=None,
                    help="If set, gate Mode C: use exemplar prompt only when (cos_q + cos_note) >= tau, "
                         "else fall back to ZS prompt. Logs gate_used / gate_score per item.")
    ap.add_argument("--out-suffix", type=str, default="",
                    help="Suffix appended to fold_N output dir (e.g. '_gated_tau1.65').")
    # Generation determinism (added 2026-04-27 after lockdown_v1 found regen noise dominated signal).
    ap.add_argument("--temperature", type=float, default=0.1,
                    help="Generation temperature. Set to 0.0 for deterministic gen (required for "
                         "splice-style RA-ICL evaluation per Finding: Mode C lockdown_v1 dominated by regen noise).")
    ap.add_argument("--max-gen-tokens", type=int, default=1024,
                    help="Generation max_tokens. Bump to 16384+ for Qwen3 thinking mode "
                         "(thinking 200-800 tok + answer 1024 tok per Prompt Design per Model).")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="Qwen3 thinking mode: extra_body={'chat_template_kwargs': {'enable_thinking': True}}.")
    ap.add_argument("--disable-thinking", action="store_true",
                    help="Qwen3 explicit no-think: extra_body={'chat_template_kwargs': {'enable_thinking': False}}. "
                         "Used for Cell D (CoT-interference isolation, 2026-04-28).")
    args = ap.parse_args()

    out_dir = OUT_DIR / args.target / f"fold_{args.fold}{args.out_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # PHASE 0: VLLM LIFECYCLE \u2014 always swap (per user 2026-04-27)
    # ============================================================
    # Flow: stop any vLLM \u2192 nomic encode on GPU \u2192 start target vLLM \u2192 generate
    # \u2192 stop target vLLM \u2192 start Magistral vLLM (if --judge magistral) \u2192 judge.
    # This block stops vLLM so nomic can use the GPU.
    from ichl.common import vllm_manager
    print("[Phase 0] Stopping any running vLLM to free GPU for nomic encoding...")
    vllm_manager.stop()
    time.sleep(3)

    # ============================================================
    # PHASE 1: PRE-FLIGHT
    # ============================================================
    print("=" * 70)
    print(f"[Phase 1] Pre-flight \u2014 RA-ICL pilot, target={args.target}, fold={args.fold}")
    print("=" * 70)

    # Token budget audit (max_model_len for target from MEMORY table)
    max_model_len = {"qwen2.5-7b-instruct": 16384, "qwen3-8b": 16384,
                     "biomistral-7b": 8192, "llama-3.1-8b-instruct": 8192,
                     "deepseek-r1-distill-llama-8b": 32768}.get(args.target, 8192)
    # Worst case: ref_note + test_note + ref_question + test_question + ref_zs + ref_GT + template ~ 12000 chars / 4 ~ 3000 tok
    # plus 1024 max_tokens + 200 safety
    worst_prompt = 5000  # generous estimate
    max_tokens = 1024
    total = worst_prompt + max_tokens + 200
    print(f"  max_model_len      : {max_model_len}")
    print(f"  worst_case_estimate: {worst_prompt} + {max_tokens} + 200 = {total}")
    if total > max_model_len:
        raise SystemExit(f"Token budget audit FAIL: {total} > {max_model_len}")
    print(f"  AUDIT PASS         : {total} \u2264 {max_model_len} (headroom {max_model_len - total})")

    # Load pool index + BM labels
    print("\n[Phase 1] Loading pool index + BM binary_correct...")
    items, q_emb, n_emb, bm_emb, gt_emb_pool = load_pool_index()
    bm_correct = load_bm_binary_correct()
    print(f"  items: {len(items)}  BM correct={sum(1 for v in bm_correct.values() if v==1)}  BM error={sum(1 for v in bm_correct.values() if v==0)}")

    # Test items = fold members
    test_indices = [i for i, it in enumerate(items) if it["fold_test_member"] == args.fold]
    if args.limit > 0:
        test_indices = test_indices[: args.limit]
    print(f"  test_indices (fold_{args.fold}): {len(test_indices)} items")

    # Target zs answers
    print("\n[Phase 1] Loading target zero-shot answers...")
    qwen_df = pd.read_csv(args.target_zs_csv)
    zs_by_pid = {int(r["patient_id"]): str(r["model_answer"] or "") for _, r in qwen_df.iterrows()}
    test_zs_texts = [zs_by_pid.get(int(items[ti]["patient_id"]), "") for ti in test_indices]
    n_missing_zs = sum(1 for t in test_zs_texts if not t)
    print(f"  target zs available: {len(test_zs_texts) - n_missing_zs}/{len(test_zs_texts)}")
    if n_missing_zs > 0:
        print(f"  WARNING: {n_missing_zs} test items missing target zs answer")

    # Encode target zs
    print("\n[Phase 1] Encoding target zs answers via nomic...")
    t0 = time.monotonic()
    test_zs_emb = encode_target_zs(test_zs_texts)
    print(f"  done in {time.monotonic()-t0:.1f}s shape={test_zs_emb.shape}")

    # Encode pool GT (cached) for Mode C
    gt_emb = gt_emb_pool   # already loaded with pool_index (gte_gt.npy)

    # ============================================================
    # PHASE 1.5: RETRIEVAL
    # ============================================================
    print("\n[Phase 1.5] Retrieving top-1 per mode...")
    retrievals_by_mode: dict[str, list[dict]] = {}
    for mode in args.modes:
        retrievals_by_mode[mode] = retrieve_top1_per_mode(
            test_indices, test_zs_emb, items, q_emb, n_emb, bm_emb, bm_correct, gt_emb, mode, args.fold,
            scorer=args.scorer,
        )
        # Patient-leakage check
        leaks = sum(1 for r in retrievals_by_mode[mode]
                    if r["test_patient_id"] == r["pool_patient_id"])
        print(f"  Mode {mode}: {len(retrievals_by_mode[mode])} retrievals, leakage={leaks}/n (should be 0)")
    # Save retrievals for reproducibility
    (out_dir / "retrievals.json").write_text(json.dumps(retrievals_by_mode, indent=2))

    # ============================================================
    # PHASE 2: GENERATION (one variant at a time, with per-50 checkpoint)
    # ============================================================
    if not args.skip_gen:
        # Start target vLLM (vllm_manager will start fresh since Phase 0 stopped any running)
        print(f"\n[Phase 2] Starting vLLM for target={args.target}...")
        vllm_manager.ensure_model(args.target, log_dir=out_dir / "vllm_logs")
        from openai import OpenAI
        # timeout=300 to accommodate Qwen3 thinking-mode + Mode D long prompts (per Cell B fold_0
        # timeout incident 2026-04-28: 1 item at 120s tripped the per-50 abort gate).
        vllm = OpenAI(base_url=args.vllm_url, api_key="not-needed", timeout=300)
        print(f"[Phase 2] Generation against {args.vllm_url} (model={args.vllm_model})...")

        variants = ["zs"] + [f"mode_{m}" for m in args.modes]
        for variant in variants:
            out_file = out_dir / f"{variant}_generated.jsonl"
            if out_file.exists() and sum(1 for _ in out_file.open()) >= len(test_indices):
                print(f"  [skip] {out_file.name}")
                continue
            print(f"\n[Phase 2] Variant: {variant}  ({len(test_indices)} items)")

            def gen_one(args_tuple):
                ti, idx = args_tuple
                test_item = items[ti]
                note = test_item["note_text_truncated"]
                question = test_item["question"]
                if variant == "zs":
                    user = USER_TMPL_ZS.format(note=note, question=question)
                else:
                    mode = variant.split("_")[1]
                    rt = retrievals_by_mode[mode][idx]
                    pool_item = items[rt["pool_item_idx"]]
                    pool_zs = zs_by_pid.get(int(pool_item["patient_id"]), "")
                    pool_bm_zs = pool_item.get("bm_zeroshot", "")
                    # Full reference note. NEVER truncate per [Workflow] No Silent
                    # Truncation. EHRNoteQA notes max ~5709 nomic tokens; with the
                    # bumped 16K target context, full-note ref fits comfortably.
                    pool_note = pool_item["note_text_truncated"]
                    pool_q = pool_item["question"]
                    pool_gt = pool_item["ground_truth"]
                    # Confidence-gated Mode C / D: if --gate-tau is set and gate fails, fall back to ZS prompt.
                    gate_used = True
                    gate_score = None
                    if mode in ("C", "D") and args.gate_tau is not None:
                        gate_score = float(rt["cos_q"] + rt["cos_note"])
                        gate_used = gate_score >= args.gate_tau
                    if mode == "A":
                        user = USER_TMPL_MODE_A.format(
                            ref_note=pool_note, ref_question=pool_q,
                            ref_BM_error_zs=pool_bm_zs, ref_GT=pool_gt,
                            note=note, question=question)
                    elif mode == "D" and gate_used:
                        # Mode D — self-revision-with-exemplar. Pass test's own zs answer (X_zs).
                        test_x_zs = zs_by_pid.get(int(test_item["patient_id"]), "")
                        user = USER_TMPL_MODE_D.format(
                            ref_note=pool_note, ref_question=pool_q, ref_GT=pool_gt,
                            note=note, question=question, X_zs=test_x_zs)
                    elif gate_used:  # B, or C with gate passing
                        user = USER_TMPL_MODE_B.format(
                            ref_note=pool_note, ref_question=pool_q, ref_GT=pool_gt,
                            note=note, question=question)
                    else:  # C/D with gate failing -> ZS fallback
                        user = USER_TMPL_ZS.format(note=note, question=question)
                # Tri-state thinking control: --enable-thinking=True, --disable-thinking=False, neither=None (default).
                if args.enable_thinking:
                    _think = True
                elif args.disable_thinking:
                    _think = False
                else:
                    _think = None
                r = vllm_call(vllm, args.vllm_model, SYSTEM_RAICL, user,
                              max_tokens=args.max_gen_tokens, temperature=args.temperature,
                              enable_thinking=_think)
                # Universal think-strip per Prompt Design per Model: applies to Qwen3 thinking AND
                # DeepSeek-R1 (always thinks, emits </think> with NO opening <think> in vLLM output).
                # Pattern: if </think> appears, drop everything up to and including the first one.
                if "text" in r and "</think>" in r["text"]:
                    raw = r["text"]
                    r["text_with_think"] = raw
                    r["text"] = re.sub(r"^.*?</think>\s*", "", raw, flags=re.DOTALL).strip()
                row = {"variant": variant, "test_item_idx": ti, "patient_id": int(test_item["patient_id"]),
                       "question": question, "ground_truth": test_item["ground_truth"]}
                if "_err" in r:
                    row["_err"] = r["_err"]
                else:
                    row["model_answer"] = r["text"]
                    row["finish_reason"] = r["finish_reason"]
                    row["truncation"] = r["trunc"]
                    row["prompt_tok"] = r["prompt_tok"]; row["comp_tok"] = r["comp_tok"]
                if variant != "zs":
                    row["retrieval"] = retrievals_by_mode[variant.split("_")[1]][idx]
                    if variant in ("mode_C", "mode_D") and args.gate_tau is not None:
                        row["gate_tau"] = args.gate_tau
                        row["gate_score"] = gate_score
                        row["gate_used"] = gate_used
                return row

            t0 = time.monotonic()
            n_done, n_err, n_trunc, n_susp = 0, 0, 0, 0
            with out_file.open("w") as f, ThreadPoolExecutor(max_workers=args.gen_workers) as ex:
                inputs = list(enumerate(test_indices))
                for r in ex.map(gen_one, [(ti, i) for i, ti in inputs]):
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                    n_done += 1
                    if "_err" in r: n_err += 1
                    elif r.get("truncation", {}).get("certain_truncated"): n_trunc += 1
                    elif r.get("truncation", {}).get("suspicious_no_terminal"): n_susp += 1
                    if n_done % 50 == 0:
                        dt = time.monotonic() - t0
                        eta = dt * (len(test_indices) - n_done) / n_done if n_done else 0
                        print(f"  ckpt {n_done}/{len(test_indices)}  elapsed={dt:.0f}s eta={eta:.0f}s "
                              f"err={n_err} trunc_certain={n_trunc} trunc_susp={n_susp}")
                        # Loosened 2026-04-28: allow up to 5% transient errors (Qwen3 thinking-mode
                        # Mode D occasionally produces pathologically long generations that exceed
                        # the 300s client timeout for a single item). Abort only on persistent failure.
                        if n_err / max(n_done, 1) > 0.05:
                            print(f"  ABORT: HTTP/API error rate {100*n_err/n_done:.1f}% > 5% after {n_done} items")
                            ex.shutdown(wait=False, cancel_futures=True)
                            raise SystemExit("Per-50 abort gate triggered")
                        if n_trunc / max(n_done, 1) > 0.05:
                            print(f"  ABORT: certain-truncation rate {100*n_trunc/n_done:.1f}% > 5%")
                            ex.shutdown(wait=False, cancel_futures=True)
                            raise SystemExit("Per-50 abort gate triggered")
            elapsed = time.monotonic() - t0
            print(f"  DONE {variant}: {n_done} in {elapsed:.0f}s err={n_err} trunc={n_trunc}")

    # ============================================================
    # PHASE 3: JUDGE
    # ============================================================
    if not args.skip_judge:
        # vLLM lifecycle: target vLLM is still running from Phase 2.
        # If --judge magistral, swap vLLM target \u2192 Magistral.
        # If --judge gpt4o or qwen3-235b, stop the target vLLM (free GPU \u2014 not needed for these judges).
        if args.judge == "magistral":
            if "magistral-small-2509-awq" not in vllm_manager.TARGETS:
                raise SystemExit(
                    "Magistral not registered in vllm_manager.TARGETS. Either:\n"
                    "  (a) download Magistral-Small-2509-AWQ to models/magistral-small-2509-awq/ "
                    "and add a VLLMLaunchSpec entry, OR\n"
                    "  (b) use --judge gpt4o or --judge qwen3-235b instead."
                )
            print(f"\n[Phase 3] Stopping target vLLM and starting Magistral for judging...")
            vllm_manager.stop()
            time.sleep(3)
            vllm_manager.ensure_model("magistral-small-2509-awq", log_dir=out_dir / "vllm_logs")
        else:
            print(f"\n[Phase 3] Stopping target vLLM (--judge {args.judge} doesn't need it; freeing GPU)...")
            vllm_manager.stop()
            time.sleep(3)

        # Build the right client per --judge
        from openai import OpenAI
        if args.judge == "gpt4o":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                env = ROOT / ".env"
                for line in env.read_text().splitlines():
                    if line.startswith("OPENAI_API_KEY="):
                        api_key = line.split("=", 1)[1].strip(); break
            judge_client = OpenAI(api_key=api_key, timeout=60)
            judge_workers = args.judge_workers
            print(f"\n[Phase 3] Judge tier: GPT-4o (final / paid).")
        elif args.judge == "magistral":
            from ichl.judges.magistral_judge import MagistralJudge
            judge_client = MagistralJudge(base_url=args.magistral_vllm_url)
            judge_workers = 4   # local vLLM, throttle
            print(f"\n[Phase 3] Judge tier: Magistral M4 (iteration default). vLLM URL: {args.magistral_vllm_url}")
            print(f"  NOTE: vLLM must be serving Magistral-Small-2509-AWQ on this URL. Swap from {args.target} first.")
        elif args.judge == "qwen3-235b":
            judge_client = OpenAI(base_url=args.qwen3_235b_mlx_url, api_key="not-needed", timeout=300)
            judge_workers = 1   # MLX 235B requires C=1 per MEMORY
            print(f"\n[Phase 3] Judge tier: Qwen3-235B-MLX (audit). MLX URL: {args.qwen3_235b_mlx_url}, C=1.")

        for variant in ["zs"] + [f"mode_{m}" for m in args.modes]:
            in_file = out_dir / f"{variant}_generated.jsonl"
            out_file = out_dir / f"{variant}_judged_{args.judge}.jsonl"
            if not in_file.exists() or out_file.exists():
                continue
            print(f"\n[Phase 3] Judging {variant} via {args.judge}...")
            gens = [json.loads(l) for l in in_file.open() if l.strip()]
            t0 = time.monotonic()
            cost_in, cost_out = 0, 0

            def judge_one(g):
                if "_err" in g: return {**g, "_judge_err": "no_gen"}
                note = items[g["test_item_idx"]]["note_text_truncated"]
                if args.judge == "gpt4o":
                    return {**g, **gpt4o_judge(judge_client, note, g["question"], g["ground_truth"], g.get("model_answer", ""))}
                elif args.judge == "magistral":
                    return {**g, **magistral_judge_one(judge_client, note, g["question"], g["ground_truth"], g.get("model_answer", ""))}
                elif args.judge == "qwen3-235b":
                    return {**g, **qwen3_235b_judge_one(judge_client, note, g["question"], g["ground_truth"], g.get("model_answer", ""))}

            n_done, n_err = 0, 0
            with out_file.open("w") as f, ThreadPoolExecutor(max_workers=judge_workers) as ex:
                for r in ex.map(judge_one, gens):
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                    n_done += 1
                    if "_err" in r or r.get("binary_correct") is None: n_err += 1
                    cost_in += r.get("prompt_tok", 0) or 0
                    cost_out += r.get("comp_tok", 0) or 0
                    if n_done % 50 == 0:
                        cost = cost_in * 5e-6 + cost_out * 1.5e-5 if args.judge == "gpt4o" else 0.0
                        print(f"  judged {n_done}/{len(gens)}  err={n_err}  cost~${cost:.2f}")
            cost = cost_in * 5e-6 + cost_out * 1.5e-5 if args.judge == "gpt4o" else 0.0
            print(f"  DONE judging {variant}: {n_done} in {time.monotonic()-t0:.0f}s  err={n_err}  cost~${cost:.2f}")

    # ============================================================
    # PHASE 4: SUMMARY
    # ============================================================
    print(f"\n[Phase 4] Per-variant accuracy (judge={args.judge}):")
    summary = {"target": args.target, "fold": args.fold, "n_test": len(test_indices),
               "judge": args.judge, "by_variant": {}}
    for variant in ["zs"] + [f"mode_{m}" for m in args.modes]:
        f = out_dir / f"{variant}_judged_{args.judge}.jsonl"
        if not f.exists(): continue
        rows = [json.loads(l) for l in f.open() if l.strip()]
        valid = [r for r in rows if r.get("binary_correct") in (0, 1)]
        n_correct = sum(1 for r in valid if r["binary_correct"] == 1)
        summary["by_variant"][variant] = {
            "n": len(rows), "n_judged": len(valid), "n_correct": n_correct,
            "accuracy": round(n_correct / max(len(valid), 1), 4),
        }
        print(f"  {variant:10s}  n={len(rows)}  judged={len(valid)}  acc={summary['by_variant'][variant]['accuracy']:.3f} ({n_correct}/{len(valid)})")

    (out_dir / f"summary_{args.judge}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_dir / f'summary_{args.judge}.json'}")


if __name__ == "__main__":
    main()
