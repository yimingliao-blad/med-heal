#!/usr/bin/env python3
"""Step-replacement teacher forcing: fix a wrong CoT step in place, let the model CONTINUE from it.

Model emits numbered steps. We check step-by-step (GPT, gold-aware); when a step is wrong we REWRITE
that step with the correct text and ask the model to CONTINUE from the next step (keeping the corrected
prefix). Repeat. Tests whether guiding the reasoning trajectory step-by-step makes it converge.
"""
import sys, json, re
sys.path.insert(0, "scripts"); sys.path.insert(0, "src/pre_atom/legacy/step9_self_correction/v2")
import phase2b_extract_compare_detection as P2
from note_span_index import get_embedder
from expZ_section_qa import retrieve

emb = get_embedder()
rows = {(r["fold"], r["idx"]): r for r in P2.load_rows(40, 20, 42)}
looked = {(r["fold"], r["idx"]): r.get("lookup", "") for r in (json.loads(l) for l in open("runs/expO_decompose_locate/qwen25_nw40_nc20_seed42/records.jsonl"))}
row = next(r for r in rows.values() if "lung nodule" in r["question"])
q, gold = row["question"], row["ground_truth"]
items = [ln.strip(" -*0123456789.").strip() for ln in looked[(row["fold"], row["idx"])].splitlines() if ln.strip()][:14]
ctx = "\n".join(f"[Adm#{n} {d} | {h}] {s[:150]}" for n, d, h, s in retrieve(row["note"], [q] + items, emb))
print("Q:", q, "\nGOLD:", gold[:120], "\n")

SYS = "You are a clinical expert. Use ONLY the excerpts (tagged admission/date)."
START = f"Excerpts:\n{ctx}\n\nQuestion:\n{q}\n\nReason in explicit numbered steps (Step 1:, Step 2:, ...), each a short claim grounded in the excerpts. After the steps write 'ANSWER:' and the final answer."


def steps_of(text):
    parts = re.split(r"(?=Step\s*\d+\s*:)", text)
    return [p.strip() for p in parts if re.match(r"Step\s*\d+", p.strip())]


def final_of(text):
    m = re.search(r"ANSWER\s*:\s*(.+)", text, re.S)
    return m.group(1).strip()[:300] if m else text[-200:]


cot = P2.vllm_chat(SYS, START, 8003, 450, 0.0, tag="cot0")
print("ROUND 0 CoT:\n", cot.strip()[:600])
print("judged:", "CORRECT" if P2.judge(row, final_of(cot)).get("label") == 1 else "wrong")

for rnd in range(1, 5):
    steps = steps_of(cot)
    if not steps:
        print("(no parseable steps; stop)"); break
    chk = P2.gpt("gpt-4o", "You audit reasoning steps against the true answer and excerpts.",
                 f"Question: {q}\nTrue answer: {gold}\nExcerpts:\n{ctx[:3500]}\nSteps:\n" + "\n".join(steps) +
                 "\n\nWhich is the FIRST wrong step number, and what should that step say (grounded in the excerpts)? "
                 'Reply JSON: {"wrong_step": N or 0, "corrected": "Step N: ..."}', 150, 0.0, True, f"chk{rnd}")
    try:
        a = json.loads(chk)
    except Exception:
        a = {"wrong_step": 0}
    ws = int(a.get("wrong_step", 0) or 0)
    print("=" * 80, f"\nROUND {rnd}: first wrong step = {ws}")
    if ws == 0:
        print("  all steps correct per checker; stop."); break
    print("  corrected:", a.get("corrected", "")[:160])
    # prefix = steps 1..ws-1 + corrected step ws ; ask to continue
    prefix = "\n".join(steps[:ws - 1] + [a.get("corrected", "")])
    cont = P2.vllm_chat(SYS, START + "\n\nHere is the reasoning so far (a correction has been applied to the last step). CONTINUE from the next step, then give 'ANSWER:'.\n\n" + prefix, 8003, 400, 0.0, tag=f"cont{rnd}")
    cot = prefix + "\n" + cont
    print("  continued ->", cont.strip()[:300])
    print("  judged:", "CORRECT" if P2.judge(row, final_of(cot)).get("label") == 1 else "wrong")
