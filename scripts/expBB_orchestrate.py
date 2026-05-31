#!/usr/bin/env python3
"""Division of labor: GPT orchestrates the retrieved spans (honestly, no gold); QWEN answers from it.

Tests whether Qwen's gap (38% vs 78% ceiling) is SYNTHESIS (organizing messy spans) — which we can
offload to GPT — or ANSWERING. If Qwen-on-GPT-orchestrated >> Qwen-on-raw-spans, synthesis was the issue.

Arms on the SAME section-index spans:
  BASE        : Qwen answers from raw spans.
  QWEN_ON_ORCH: GPT organizes the spans into clean relevant material (no answer, no gold); Qwen answers from it.

Usage: python scripts/expBB_orchestrate.py --concurrency 4
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

QA_SYS = "You answer a clinical question using ONLY the provided note excerpts (tagged with admission/date). Do not use outside knowledge."
QA_USER = "Note excerpts:\n{ctx}\n\nQuestion:\n{question}\n\nAnswer using only these excerpts. Be specific and quote values exactly."

# GPT orchestrates the spans into clean material — honestly, from the excerpts only, NO answer, NO gold
ORCH_SYS = "You read note excerpts and organize the facts relevant to a question into clean, clear material. You do not answer the question; you only present the organized, true facts from the excerpts."
ORCH_USER = "Note excerpts (tagged with admission and date):\n{ctx}\n\nQuestion:\n{question}\n\nFrom the excerpts only, organize the facts relevant to this question into a clean, clear summary. Keep exact values, dates, and which admission each fact is from. Group related facts. Do NOT answer the question — just present the organized relevant facts."

# Qwen answers from GPT's organized material
QAM_SYS = "You answer a clinical question using ONLY the provided organized facts. Do not use outside knowledge."
QAM_USER = "Organized facts from the patient's note:\n{material}\n\nQuestion:\n{question}\n\nAnswer using only these facts. Be specific and quote values exactly."


def process_one(row, looked, emb, port):
    q = row["question"]
    items = [ln.strip(" -*0123456789.").strip() for ln in looked.splitlines() if ln.strip()][:14]
    tagged = retrieve(row["note"], [q] + items, emb)
    ctx = "\n".join(f"[Adm#{n} {d} | {h}] {s[:150]}" for n, d, h, s in tagged)
    base = P2.vllm_chat(QA_SYS, QA_USER.format(ctx=ctx, question=q), port, 350, 0.0, tag="base")
    material = P2.gpt("gpt-4o", ORCH_SYS, ORCH_USER.format(ctx=ctx, question=q), 500, 0.0, False, "orch")
    qom = P2.vllm_chat(QAM_SYS, QAM_USER.format(material=material[:3500], question=q), port, 350, 0.0, tag="qwen_on_orch")
    return {"fold": row["fold"], "idx": row["idx"], "stored_label": row["stored_label"],
            "base": P2.judge(row, base).get("label"), "qwen_on_orch": P2.judge(row, qom).get("label")}


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
    out_dir = PROJECT_ROOT / "runs" / "expBB_orchestrate"; out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expBB_orchestrate", served=P2.served_model_id(args.port))
    print(f"orchestrate (GPT organizes -> Qwen answers) on {len(keys)} cases", flush=True)
    recs = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, rows[k], looked[k], emb, args.port) for k in keys]
        for i, fut in enumerate(as_completed(futs), 1):
            recs.append(fut.result())
            if i % 10 == 0 or i == len(keys):
                print(f"  {i}/{len(keys)}", flush=True)
    (out_dir / "records.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    W = [r for r in recs if r["stored_label"] == 0]; C = [r for r in recs if r["stored_label"] == 1]
    print("\n=== GPT-orchestrates -> Qwen-answers (section-index spans) ===")
    for arm in ["base", "qwen_on_orch"]:
        f = sum(1 for r in W if r.get(arm) == 1); b = sum(1 for r in C if r.get(arm) == 0)
        print(f"  {arm:13} FIX {f}/{len(W)}={f/len(W)*100:.0f}%   BREAK {b}/{len(C)}={b/len(C)*100:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
