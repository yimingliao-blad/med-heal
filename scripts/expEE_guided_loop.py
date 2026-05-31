#!/usr/bin/env python3
"""Controlled single-case study: does step-by-step guidance make the correction converge?

Loop on one familiar case (nodule: gold='had grown', Qwen tends to say 'stable'):
  round 0: Qwen CoT answer from section-index spans
  each round: GPT checker (WITH the gold) points out the SPECIFIC wrong spot ->
              Qwen re-reasons (CoT) on that spot from the spans -> new answer -> judge
Repeat up to N rounds. Prints every step so we can see if guidance helps or it stays random.
"""
import sys, json
sys.path.insert(0, "scripts"); sys.path.insert(0, "src/pre_atom/legacy/step9_self_correction/v2")
import phase2b_extract_compare_detection as P2
from note_span_index import get_embedder
from expZ_section_qa import retrieve

emb = get_embedder()
rows = {(r["fold"], r["idx"]): r for r in P2.load_rows(40, 20, 42)}
looked = {(r["fold"], r["idx"]): r.get("lookup", "") for r in (json.loads(l) for l in open("runs/expO_decompose_locate/qwen25_nw40_nc20_seed42/records.jsonl"))}
row = next(r for r in rows.values() if "lung nodule" in r["question"])
q = row["question"]; gold = row["ground_truth"]
items = [ln.strip(" -*0123456789.").strip() for ln in looked[(row["fold"], row["idx"])].splitlines() if ln.strip()][:14]
ctx = "\n".join(f"[Adm#{n} {d} | {h}] {s[:150]}" for n, d, h, s in retrieve(row["note"], [q] + items, emb))

print("Q:", q); print("GOLD:", gold[:130]); print("\nSPANS:\n", ctx[:900], "\n")

COT_SYS = "You are a clinical expert answering using only the provided excerpts (tagged admission/date)."
ans = P2.vllm_chat(COT_SYS, f"Excerpts:\n{ctx}\n\nQuestion:\n{q}\n\nReason step by step over the excerpts, then give the answer.", 8003, 400, 0.0, tag="r0")
print("=" * 80); print("ROUND 0 answer:", ans.strip()[:300])
print("   judged:", "CORRECT" if P2.judge(row, ans).get("label") == 1 else "wrong")

CHK_SYS = "You point out the single specific spot where an answer is wrong, given the true answer and note excerpts. Be precise and brief."
REF_SYS = "You are a clinical expert revising your answer using only the provided excerpts."
for rnd in range(1, 4):
    guide = P2.gpt("gpt-4o", CHK_SYS, f"Question: {q}\nTrue answer: {gold}\nExcerpts:\n{ctx[:4000]}\nModel's answer: {ans[:600]}\n\nIn one or two sentences, point out the SPECIFIC spot where the model's answer is wrong and what it should be per the excerpts. If it is already correct, say CORRECT.", 120, 0.0, False, f"chk{rnd}")
    print("=" * 80); print(f"ROUND {rnd} guidance (GPT, gold-aware):", guide.strip()[:220])
    if "CORRECT" in guide.upper()[:15]:
        print("   -> checker says already correct; stop."); break
    ans = P2.vllm_chat(REF_SYS, f"Excerpts:\n{ctx}\n\nQuestion:\n{q}\n\nYour previous answer:\n{ans[:600]}\n\nReviewer feedback on what is wrong:\n{guide}\n\nRe-examine the excerpts for that specific point, reason step by step, then give the corrected answer.", 8003, 400, 0.0, tag=f"r{rnd}")
    print(f"ROUND {rnd} revised answer:", ans.strip()[:300])
    print("   judged:", "CORRECT" if P2.judge(row, ans).get("label") == 1 else "wrong")
