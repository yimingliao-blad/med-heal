#!/usr/bin/env python3
"""Prove the section-routing method across cases: gold-fact recall, section vs whole-note vs embed.

Claim to prove: feeding the model FOCUSED sections (machinery split + route) recovers gold facts that
the whole-note path omits (long-context omission) and that sentence-embedding misses (fuzzy).

Per wrong case, three extraction arms produce a fact list:
  WHOLE   : model extracts from the whole 24k note.
  SECTION : machinery splits the note into sections (header string-split), routes by embedding the
            question vs section text (top-k sections), model extracts from the focused sections.
  EMBED   : top-k GTR sentences for the items, model extracts (the earlier method).

PROOF METRIC — gold-fact recall (judge-free): parse the gold answer into atomic facts (gpt-4o-mini,
parsing not judging), then a gold fact is COVERED if it appears in the extracted list by EITHER
GTR cosine > 0.72 OR its key term is a substring. Report mean recall per arm. SECTION > WHOLE proves it.

Usage: python scripts/expS_section_proof.py --topk 10 --concurrency 4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src/pre_atom/legacy/step9_self_correction/v2"))
import phase2b_extract_compare_detection as P2  # noqa
from note_span_index import topk_spans, get_embedder, split_sentences  # noqa
from llm_audit import set_ledger  # noqa

HDR = re.compile(r"^([A-Z][A-Za-z /]{2,45}):\s*(.*)")
EX_SYS = "You list facts from a clinical note, using only the provided text. Keep all details and exact values; do not omit."
EX_USER = "{ctx}\n\nList all the distinct facts in the text above that relate to: {q}\nOne fact per line, exact values."
GF_SYS = "You list the distinct factual claims an answer makes."
GF_USER = "Answer:\n{gold}\n\nList each distinct fact or entity this answer asserts, one short item per line."


def sectionize(note):
    secs, cur = [], None
    for ln in note.splitlines():
        m = HDR.match(ln.strip())
        if m:
            if cur:
                secs.append(cur)
            cur = [m.group(1), m.group(2)]
        elif cur is not None:
            cur[1] += " " + ln.strip()
    if cur:
        secs.append(cur)
    # dedup by content
    seen, out = set(), []
    for h, c in secs:
        key = (h + c.strip()[:50])
        if key not in seen and c.strip():
            seen.add(key); out.append((h, c.strip()))
    return out


def route_sections(secs, query, emb, topk):
    texts = [f"{h}: {c}"[:300] for h, c in secs]
    if not texts:
        return ""
    sv = emb.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    qv = emb.encode([query], normalize_embeddings=True, show_progress_bar=False)
    sims = (sv @ qv.T)[:, 0]
    order = np.argsort(-sims)[:topk]
    return "\n".join(f"{secs[i][0]}: {secs[i][1][:300]}" for i in order)


def covered(fact, lines_emb, lines_text, emb):
    fv = emb.encode([fact], normalize_embeddings=True, show_progress_bar=False)
    sim = float((lines_emb @ fv.T).max()) if len(lines_emb) else 0.0
    key = max(re.findall(r"[A-Za-z]{5,}", fact.lower()), key=len, default="")
    strhit = bool(key) and key in lines_text.lower()
    return sim > 0.72 or strhit


def process_one(row, looked, topk, port, emb):
    q = row["question"]
    note = row["note"]
    secs = sectionize(note)
    focused = route_sections(secs, q, emb, topk)
    items = [ln.strip(" -*0123456789.").strip() for ln in looked.splitlines() if ln.strip()][:12]
    embspans = "\n".join(f"- {s['sentence'][:150]}" for s in topk_spans(note, items, k=topk))
    lst = {}
    lst["WHOLE"] = P2.vllm_chat(EX_SYS, EX_USER.format(ctx="Note:\n" + note[:24000], q=q), port, 350, 0.0, tag="ex.whole")
    lst["SECTION"] = P2.vllm_chat(EX_SYS, EX_USER.format(ctx="Relevant note sections:\n" + focused, q=q), port, 350, 0.0, tag="ex.section")
    lst["EMBED"] = P2.vllm_chat(EX_SYS, EX_USER.format(ctx="Retrieved note sentences:\n" + embspans, q=q), port, 350, 0.0, tag="ex.embed")
    gf = P2.gpt("gpt-4o-mini", GF_SYS, GF_USER.format(gold=row["ground_truth"][:600]), 200, 0.0, False, "goldfacts")
    golds = [g.strip(" -*0123456789.").strip() for g in gf.splitlines() if len(g.strip()) > 3]
    out = {"fold": row["fold"], "idx": row["idx"], "n_gold": len(golds)}
    for arm, text in lst.items():
        lines = [l for l in text.splitlines() if l.strip()]
        le = emb.encode(lines, normalize_embeddings=True, show_progress_bar=False) if lines else np.zeros((0, 768))
        cov = sum(1 for g in golds if covered(g, le, text, emb))
        out[arm] = cov / max(1, len(golds))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--expO", default=str(PROJECT_ROOT / "runs/expO_decompose_locate/qwen25_nw40_nc20_seed42/records.jsonl"))
    args = ap.parse_args()
    looked = {(r["fold"], r["idx"]): r.get("lookup", "") for r in (json.loads(l) for l in open(args.expO))}
    rows = {(r["fold"], r["idx"]): r for r in P2.load_rows(40, 20, 42)}
    wrong = [k for k in looked if rows[k]["stored_label"] == 0]
    emb = get_embedder()
    out_dir = PROJECT_ROOT / "runs" / "expS_section_proof" / f"k{args.topk}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expS_section_proof", served=P2.served_model_id(args.port))
    print(f"section-routing proof on {len(wrong)} wrong cases, topk={args.topk}", flush=True)
    recs = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, rows[k], looked[k], args.topk, args.port, emb) for k in wrong]
        for i, fut in enumerate(as_completed(futs), 1):
            recs.append(fut.result())
            if i % 10 == 0 or i == len(wrong):
                print(f"  {i}/{len(wrong)}", flush=True)
    (out_dir / "records.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    print("\n=== GOLD-FACT RECALL (how much of the gold answer each extraction recovers) ===")
    for arm in ["WHOLE", "SECTION", "EMBED"]:
        m = sum(r[arm] for r in recs) / len(recs)
        print(f"  {arm:8} {m*100:.0f}%")
    print("\nSECTION > WHOLE proves machinery-focus reduces omission; SECTION > EMBED proves sections beat fuzzy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
