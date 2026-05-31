#!/usr/bin/env python3
"""Post-correction verification: can we tell TRUE FIX from TRUE BREAK using the span material?

Correction = Qwen CoT answer from the section-index spans (ground-truth material). Then two verifiers
try to predict whether that correction is correct (keep) or wrong (reject -> a break):
  A GROUNDING : decompose the correction's claims and check each against the spans (supported?).
  B VERDICT   : a verdict prompt — given the spans, ACCEPT or REJECT the answer.
Ground truth = P2.judge(correction). We measure whether each verifier keeps the correct corrections
and rejects the wrong ones (break-catch), and the resulting net vs keeping everything.

Usage: python scripts/expCC_verify.py --concurrency 4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src/pre_atom/legacy/step9_self_correction/v2"))
import phase2b_extract_compare_detection as P2  # noqa
from note_span_index import get_embedder  # noqa
from expZ_section_qa import retrieve  # noqa
from llm_audit import set_ledger  # noqa

COT_SYS = "You are a clinical expert answering a question about a patient using only the provided note excerpts (tagged admission/date)."
COT_USER = "Excerpts:\n{ctx}\n\nQuestion:\n{question}\n\nReason through the excerpts, then give a clear, complete answer."

# A — grounding: claim-by-claim check against the spans
GR_SYS = "You check whether every claim in an answer is supported by note excerpts."
GR_USER = "Excerpts:\n{ctx}\n\nAnswer:\n{ans}\n\nCheck each claim in the answer against the excerpts. Is EVERY claim supported (no unsupported or invented facts)? Reply SUPPORTED or NOT-SUPPORTED."

# B — verdict
VD_SYS = "You decide whether an answer is correct and complete for a question, given note excerpts."
VD_USER = "Excerpts:\n{ctx}\n\nQuestion:\n{question}\n\nAnswer:\n{ans}\n\nIs this answer correct and complete for the question, supported by the excerpts? Reply ACCEPT or REJECT."


def process_one(row, looked, emb, port):
    q = row["question"]
    items = [ln.strip(" -*0123456789.").strip() for ln in looked.splitlines() if ln.strip()][:14]
    tagged = retrieve(row["note"], [q] + items, emb)
    ctx = "\n".join(f"[Adm#{n} {d} | {h}] {s[:150]}" for n, d, h, s in tagged)
    corr = P2.vllm_chat(COT_SYS, COT_USER.format(ctx=ctx, question=q), port, 450, 0.0, tag="corr")
    truth = P2.judge(row, corr).get("label")
    gr = P2.vllm_chat(GR_SYS, GR_USER.format(ctx=ctx[:6000], ans=corr[:1200]), port, 8, 0.0, tag="ground").upper()
    vd = P2.vllm_chat(VD_SYS, VD_USER.format(ctx=ctx[:6000], question=q, ans=corr[:1200]), port, 8, 0.0, tag="verdict").upper()
    return {"fold": row["fold"], "idx": row["idx"], "stored_label": row["stored_label"], "truth": truth,
            "grounding_keep": ("NOT" not in gr and "SUPPORT" in gr), "verdict_keep": ("ACCEPT" in vd)}


def evalv(recs, keyflag):
    # keep = verifier says good. Compare to truth (1=correct).
    kept_correct = sum(1 for r in recs if r[keyflag] and r["truth"] == 1)
    kept_wrong = sum(1 for r in recs if r[keyflag] and r["truth"] == 0)      # these are realized breaks
    rej_correct = sum(1 for r in recs if not r[keyflag] and r["truth"] == 1)  # lost good answers
    rej_wrong = sum(1 for r in recs if not r[keyflag] and r["truth"] == 0)    # caught breaks
    catch = rej_wrong / max(1, rej_wrong + kept_wrong)
    keepgood = kept_correct / max(1, kept_correct + rej_correct)
    return kept_correct, kept_wrong, rej_correct, rej_wrong, catch, keepgood


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
    out_dir = PROJECT_ROOT / "runs" / "expCC_verify"; out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expCC_verify", served=P2.served_model_id(args.port))
    print(f"post-correction verification on {len(keys)} cases", flush=True)
    recs = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, rows[k], looked[k], emb, args.port) for k in keys]
        for i, fut in enumerate(as_completed(futs), 1):
            recs.append(fut.result())
            if i % 10 == 0 or i == len(keys):
                print(f"  {i}/{len(keys)}", flush=True)
    (out_dir / "records.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    ncorr = sum(1 for r in recs if r["truth"] == 1); nwrong = sum(1 for r in recs if r["truth"] == 0)
    print(f"\ncorrections: {ncorr} correct, {nwrong} wrong (the wrong ones are the breaks/no-fixes to catch)")
    for v in ["grounding_keep", "verdict_keep"]:
        kc, kw, rc, rw, catch, kg = evalv(recs, v)
        print(f"\n{v}: keeps {kc} correct + {kw} wrong | rejects {rc} correct + {rw} wrong")
        print(f"   break-catch (rejected wrong / all wrong) = {catch*100:.0f}%   keep-good (kept correct / all correct) = {kg*100:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
