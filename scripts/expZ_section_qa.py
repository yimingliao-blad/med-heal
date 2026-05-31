#!/usr/bin/env python3
"""Section-index retrieval -> focused QA -> judge. The principle: match coarse (section), return fine
(sentence), dedup, carry provenance; whitelist headers so lists stay intact.

retrieve(): per case ->
  - admissions (provenance, skip dateless phantom) -> WHITELISTED sections (lists intact)
  - SECTION INDEX: embed sections, take top-N, EXPAND each to its sentences (container completeness)
  - SENTENCE UNION: top-k sentence-level matches (value precision), provenance-tagged
  - dedup -> [Adm#n date | header] sentence
QA: restrictive prompt (won the A/B) over the focused, provenance-tagged context. Judge fix/break.

--limit N prints per-case detail (smoke). No limit = full 40 wrong + 20 correct.
Usage: python scripts/expZ_section_qa.py --limit 5      # smoke
       python scripts/expZ_section_qa.py                 # full
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src/pre_atom/legacy/step9_self_correction/v2"))
import phase2b_extract_compare_detection as P2  # noqa
from note_span_index import get_embedder, split_sentences  # noqa
from note_chunker import split_admissions, split_sections_clean  # noqa
from llm_audit import set_ledger  # noqa

QA_SYS = "You answer a clinical question using ONLY the provided note excerpts. Each excerpt is tagged with the admission and date it came from; if the question specifies a date, answer from the matching admission. Do not use outside knowledge."
QA_USER = "Note excerpts:\n{ctx}\n\nQuestion:\n{question}\n\nAnswer using only these excerpts. Be specific and quote values exactly."


def retrieve(note, queries, emb, n_sec=6, k_sent=6):
    adms = [a for a in split_admissions(note) if a["chartdate"] != "9999-99-99"]
    secs, sent_pool = [], []
    for a in adms:
        for h, c in split_sections_clean(a["text"]):
            secs.append((a["n"], a["chartdate"], h, c))
            for s in split_sentences(c):
                sent_pool.append((a["n"], a["chartdate"], h, s))
    if not secs:
        return []
    qv = emb.encode(queries, normalize_embeddings=True, show_progress_bar=False)
    out, seen = [], set()
    # SECTION INDEX -> expand to sentences (completeness)
    sv = emb.encode([f"{h}: {c}"[:500] for _, _, h, c in secs], normalize_embeddings=True, show_progress_bar=False)
    for i in np.argsort(-(sv @ qv.T).max(axis=1))[:n_sec]:
        n, d, h, c = secs[i]
        for s in split_sentences(c):
            if s[:60] not in seen:
                seen.add(s[:60]); out.append((n, d, h, s))
    # SENTENCE UNION (value precision)
    if sent_pool:
        ev = emb.encode([x[3] for x in sent_pool], normalize_embeddings=True, show_progress_bar=False)
        for i in np.argsort(-(ev @ qv.T).max(axis=1))[:k_sent]:
            n, d, h, s = sent_pool[i]
            if s[:60] not in seen:
                seen.add(s[:60]); out.append((n, d, h, s))
    return out


def process_one(row, looked, emb, port):
    q = row["question"]
    items = [ln.strip(" -*0123456789.").strip() for ln in looked.splitlines() if ln.strip()][:14]
    tagged = retrieve(row["note"], [q] + items, emb)
    ctx = "\n".join(f"[Adm#{n} {d} | {h}] {s[:150]}" for n, d, h, s in tagged)
    ans = P2.vllm_chat(QA_SYS, QA_USER.format(ctx=ctx, question=q), port, 350, 0.0, tag="zqa")
    return {"fold": row["fold"], "idx": row["idx"], "stored_label": row["stored_label"],
            "correct": P2.judge(row, ans).get("label"), "answer": ans, "n_ctx": len(tagged), "q": q,
            "gold": row["ground_truth"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--expO", default=str(PROJECT_ROOT / "runs/expO_decompose_locate/qwen25_nw40_nc20_seed42/records.jsonl"))
    args = ap.parse_args()
    looked = {(r["fold"], r["idx"]): r.get("lookup", "") for r in (json.loads(l) for l in open(args.expO))}
    rows = {(r["fold"], r["idx"]): r for r in P2.load_rows(40, 20, 42)}
    keys = list(looked)
    if args.limit:
        keys = [k for k in keys if rows[k]["stored_label"] == 0][:args.limit]  # smoke = wrong cases
    emb = get_embedder()
    out_dir = PROJECT_ROOT / "runs" / "expZ_section_qa"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expZ_section_qa", served=P2.served_model_id(args.port))
    print(f"section-index QA on {len(keys)} cases" + (" (SMOKE)" if args.limit else " (full 40w+20c)"), flush=True)
    recs = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, rows[key], looked[key], emb, args.port) for key in keys]
        for i, fut in enumerate(as_completed(futs), 1):
            recs.append(fut.result())
            if not args.limit and (i % 10 == 0 or i == len(keys)):
                print(f"  {i}/{len(keys)}", flush=True)
    (out_dir / "records.jsonl").write_text("\n".join(json.dumps({k: r[k] for k in ["fold", "idx", "stored_label", "correct", "n_ctx"]}) for r in recs))
    if args.limit:
        for r in recs:
            print("=" * 90)
            print("Q:", r["q"][:110]); print("GOLD:", r["gold"][:130])
            print(f"CTX sentences: {r['n_ctx']}   judged: {'CORRECT' if r['correct']==1 else 'wrong'}")
            print("ANSWER:", r["answer"].strip()[:260])
    else:
        W = [r for r in recs if r["stored_label"] == 0]; C = [r for r in recs if r["stored_label"] == 1]
        fixes = sum(1 for r in W if r["correct"] == 1); breaks = sum(1 for r in C if r["correct"] == 0)
        print(f"\n=== section-index focused QA (vs raw-span Qwen 28/45) ===")
        print(f"FIX   {fixes}/{len(W)}={fixes/max(1,len(W))*100:.0f}%   BREAK {breaks}/{len(C)}={breaks/max(1,len(C))*100:.0f}%")
        print(f"avg context sentences: {sum(r['n_ctx'] for r in recs)/len(recs):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
