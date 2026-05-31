#!/usr/bin/env python3
"""Comparative verdict as the gate: original (ZS) vs corrected (A/B), grounded in the section spans.

Correction = Qwen CoT from section-index spans. Then a COMPARATIVE verdict (not absolute) picks
original vs corrected, grounded in the spans (slots randomized; parsed by the GPT-4o-mini semantic judge).
Final = whichever the verdict picks. We measure the gate's real break-catch / fix-keep and the
final fix/break vs always-take-correction.

Usage: python scripts/expDD_comparative_verdict.py --concurrency 4
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src/pre_atom/legacy/step9_self_correction/v2"))
import phase2b_extract_compare_detection as P2  # noqa
import expK_cascade_collect as K  # noqa  (llm_verdict, parse_verdict_letter)
from note_span_index import get_embedder  # noqa
from expZ_section_qa import retrieve  # noqa
from llm_audit import set_ledger  # noqa

COT_SYS = "You are a clinical expert answering a question about a patient using only the provided note excerpts (tagged admission/date)."
COT_USER = "Excerpts:\n{ctx}\n\nQuestion:\n{question}\n\nReason through the excerpts, then give a clear, complete answer."

V_SYS = "You decide which of two answers is better and more correct for the question, using only the note excerpts."
V_USER = """Note excerpts:
{ctx}

Question:
{q}

Answer A:
{a}

Answer B:
{b}

Step by step: which answer is better supported by the excerpts and more complete for the question? Consider wrong values, missing required facts, and which admission/date applies. On the very last line write exactly 'FINAL: A' or 'FINAL: B'."""


def process_one(row, looked, emb, port):
    q = row["question"]
    items = [ln.strip(" -*0123456789.").strip() for ln in looked.splitlines() if ln.strip()][:14]
    tagged = retrieve(row["note"], [q] + items, emb)
    ctx = "\n".join(f"[Adm#{n} {d} | {h}] {s[:150]}" for n, d, h, s in tagged)
    orig = row["original_answer"]
    corr = P2.vllm_chat(COT_SYS, COT_USER.format(ctx=ctx, question=q), port, 450, 0.0, tag="corr")
    rng = random.Random(42 + (row["fold"] << 16) + row["idx"])
    orig_a = rng.random() > 0.5
    a, b = (orig, corr) if orig_a else (corr, orig)
    corr_slot = "B" if orig_a else "A"
    raw = P2.vllm_chat(V_SYS, V_USER.format(ctx=ctx, q=q, a=a[:1400], b=b[:1400]), port, 500, 0.0, tag="cmp_verdict")
    pick = K.llm_verdict(raw, K.parse_verdict_letter(raw), "cmp")
    accept = (pick == corr_slot)
    final = corr if accept else orig
    return {"fold": row["fold"], "idx": row["idx"], "stored_label": row["stored_label"],
            "orig_ok": P2.judge(row, orig).get("label"), "corr_ok": P2.judge(row, corr).get("label"),
            "accept": accept, "final_ok": P2.judge(row, final).get("label")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--expO", default=str(PROJECT_ROOT / "runs/expO_decompose_locate/qwen25_nw40_nc20_seed42/records.jsonl"))
    args = ap.parse_args()
    looked = {(r["fold"], r["idx"]): r.get("lookup", "") for r in (json.loads(l) for l in open(args.expO))}
    rows = {(r["fold"], r["idx"]): r for r in P2.load_rows(40, 20, 42)}
    keys = list(looked)
    emb = get_embedder()
    out_dir = PROJECT_ROOT / "runs" / "expDD_comparative_verdict"; out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expDD_comparative_verdict", served=P2.served_model_id(args.port))
    print(f"comparative verdict gate on {len(keys)} cases", flush=True)
    recs = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, rows[k], looked[k], emb, args.port) for k in keys]
        for i, fut in enumerate(as_completed(futs), 1):
            recs.append(fut.result())
            if i % 10 == 0 or i == len(keys):
                print(f"  {i}/{len(keys)}", flush=True)
    (out_dir / "records.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    W = [r for r in recs if r["stored_label"] == 0]; C = [r for r in recs if r["stored_label"] == 1]
    # gate quality: break-risk = correct-stratum cases where correction is WRONG -> did verdict keep original?
    risk = [r for r in C if r["corr_ok"] == 0]
    caught = [r for r in risk if not r["accept"]]
    # fix available = wrong-stratum where correction CORRECT -> did verdict accept?
    favail = [r for r in W if r["corr_ok"] == 1]
    fkept = [r for r in favail if r["accept"]]
    print("\n=== comparative verdict gate (original vs corrected, grounded in section spans) ===")
    print(f"break-catch: of {len(risk)} break-risk cases (corr wrong on correct-stratum), verdict kept original on {len(caught)} = {len(caught)/max(1,len(risk))*100:.0f}%")
    print(f"fix-keep:    of {len(favail)} available fixes (corr right on wrong-stratum), verdict accepted {len(fkept)} = {len(fkept)/max(1,len(favail))*100:.0f}%")
    fx = sum(1 for r in W if r["final_ok"] == 1); bk = sum(1 for r in C if r["final_ok"] == 0)
    fx_nv = sum(1 for r in W if r["corr_ok"] == 1); bk_nv = sum(1 for r in C if r["corr_ok"] == 0)
    print(f"\nFINAL with verdict gate:   FIX {fx}/{len(W)}={fx/len(W)*100:.0f}%   BREAK {bk}/{len(C)}={bk/len(C)*100:.0f}%")
    print(f"vs always-take-correction: FIX {fx_nv}/{len(W)}={fx_nv/len(W)*100:.0f}%   BREAK {bk_nv}/{len(C)}={bk_nv/len(C)*100:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
