#!/usr/bin/env python3
"""Diagnose the gap (we're ~35-40% fix vs oracle ~60%): is it retrieval or the model's use of spans?

On the SAME section-index spans, three arms:
  BASE        : Qwen restrictive QA (the current ~35-40% fix).
  VERIFY      : Qwen self-verifies its BASE answer via CoT against the spans, correcting unsupported claims.
  GPT+ORACLE  : GPT synthesizes the spans honestly + the oracle-guided material -> the ceiling. If this
                hits ~oracle level, the spans are SUFFICIENT and the gap is synthesis/guidance, not retrieval.

Usage: python scripts/expAA_diagnose.py --concurrency 4
"""
from __future__ import annotations

import argparse
import json
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

QA_SYS = "You answer a clinical question using ONLY the provided note excerpts (tagged with admission and date). Do not use outside knowledge."
QA_USER = "Note excerpts:\n{ctx}\n\nQuestion:\n{question}\n\nAnswer using only these excerpts. Be specific and quote values exactly."

VERIFY_SYS = "You verify and correct a draft answer using ONLY the provided note excerpts."
VERIFY_USER = """Note excerpts:\n{ctx}\n\nQuestion:\n{question}\n\nDraft answer:\n{draft}\n\nCheck each claim in the draft against the excerpts step by step. Remove or fix any claim not supported by the excerpts, and add anything the excerpts show that the question requires and the draft missed. Then give the FINAL corrected answer after 'FINAL ANSWER:'."""

ORA_SYS = "You answer a clinical question. You are given note excerpts (tagged with admission/date) and expert guidance about what a wrong answer got wrong. Use the excerpts as the source of truth; use the guidance to focus. Answer honestly from the excerpts."
ORA_USER = "Note excerpts:\n{ctx}\n\nExpert guidance on the likely error:\n{oracle}\n\nQuestion:\n{question}\n\nGive the correct answer, grounded in the excerpts."


def process_one(row, looked, emb, port):
    q = row["question"]
    items = [ln.strip(" -*0123456789.").strip() for ln in looked.splitlines() if ln.strip()][:14]
    tagged = retrieve(row["note"], [q] + items, emb)
    ctx = "\n".join(f"[Adm#{n} {d} | {h}] {s[:150]}" for n, d, h, s in tagged)
    base = P2.vllm_chat(QA_SYS, QA_USER.format(ctx=ctx, question=q), port, 350, 0.0, tag="base")
    ver = P2.vllm_chat(VERIFY_SYS, VERIFY_USER.format(ctx=ctx, question=q, draft=base[:1200]), port, 500, 0.0, tag="verify")
    ver_ans = ver.split("FINAL ANSWER:")[-1].strip() if "FINAL ANSWER:" in ver else ver
    out = {"fold": row["fold"], "idx": row["idx"], "stored_label": row["stored_label"],
           "base": P2.judge(row, base).get("label"), "verify": P2.judge(row, ver_ans).get("label")}
    if row["stored_label"] == 0:  # oracle ceiling only meaningful on wrong cases
        g = P2.gpt("gpt-4o", ORA_SYS, ORA_USER.format(ctx=ctx, oracle=row.get("oracle_error_description", "")[:800], question=q), 350, 0.0, False, "gptora")
        out["gpt_oracle"] = P2.judge(row, g).get("label")
    return out


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
    out_dir = PROJECT_ROOT / "runs" / "expAA_diagnose"; out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expAA_diagnose", served=P2.served_model_id(args.port))
    print(f"diagnose on {len(keys)} cases", flush=True)
    recs = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, rows[k], looked[k], emb, args.port) for k in keys]
        for i, fut in enumerate(as_completed(futs), 1):
            recs.append(fut.result())
            if i % 10 == 0 or i == len(keys):
                print(f"  {i}/{len(keys)}", flush=True)
    (out_dir / "records.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    W = [r for r in recs if r["stored_label"] == 0]; C = [r for r in recs if r["stored_label"] == 1]
    def fb(arm, onW=True):
        f = sum(1 for r in W if r.get(arm) == 1)
        b = sum(1 for r in C if r.get(arm) == 0) if onW else None
        return f, b
    print("\n=== diagnose (section-index spans) ===")
    for arm in ["base", "verify"]:
        f, b = fb(arm)
        print(f"  {arm:11} FIX {f}/{len(W)}={f/len(W)*100:.0f}%   BREAK {b}/{len(C)}={b/len(C)*100:.0f}%")
    go = sum(1 for r in W if r.get("gpt_oracle") == 1)
    print(f"  GPT+ORACLE  FIX {go}/{len(W)}={go/len(W)*100:.0f}%  (ceiling; oracle correction ~60%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
