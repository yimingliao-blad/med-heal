#!/usr/bin/env python3
"""Old regen+count, plus K=3@T=1 + verdict sharpening, plus retrieval-augmented regen.

Faithful to Step9 regen_pilot: REGEN a fresh answer, then COUNT-COMPARE verdict (count claims that
contradict the note in A vs B, keep fewer errors). Arms:
  regen1     : K=1 regen @ T=0, count-verdict vs original (the established baseline).
  regenK     : K=3 regens @ T=1, verdict tournament over {original, r1, r2, r3} (user hypothesis: sharpen).
  regenK_ra  : K=3 regens @ T=1 from SECTION-INDEX spans (retrieval-augmented), verdict tournament.

Judge fix/break on 40 wrong + 20 correct. --limit N = smoke (first N wrong cases, printed).
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

REGEN_SYS = "You are a medical expert."
REGEN_NOTE = "Discharge note:\n{src}\n\nQuestion: {question}\n\nAnswer the question using only information from the discharge note. Be specific and complete. If the question asks about multiple visits, conditions, or events, cover all of them."
CMP_SYS = "You are a strict medical expert."
CMP_TMPL = "Discharge summary:\n{src}\n\nQuestion: {question}\n\nANSWER A:\n{a}\n\nANSWER B:\n{b}\n\nCount how many factual claims in each answer contradict the discharge notes. Different wording for the same fact is NOT a contradiction.\n\nA_ERRORS: <number>\nB_ERRORS: <number>"


def parse_err(raw):
    a = re.search(r"A_ERRORS\s*:\s*(\d+)", raw or "", re.I)
    b = re.search(r"B_ERRORS\s*:\s*(\d+)", raw or "", re.I)
    return (int(a.group(1)) if a else 99), (int(b.group(1)) if b else 99)


def regen(src, q, port, temp):
    return P2.vllm_chat(REGEN_SYS, REGEN_NOTE.format(src=src[:18000], question=q), port, 700, temp, tag="regen")


def tournament(src, q, original, cands, port):
    champ = original
    for c in cands:
        ae, be = parse_err(P2.vllm_chat(CMP_SYS, CMP_TMPL.format(src=src[:18000], question=q, a=champ[:1500], b=c[:1500]), port, 256, 0.0, tag="cmp"))
        if be < ae:
            champ = c
    return champ


def process_one(row, looked, emb, port):
    q = row["question"]; note = row["note"]; orig = row["original_answer"]
    items = [ln.strip(" -*0123456789.").strip() for ln in looked.splitlines() if ln.strip()][:14]
    spans_ctx = "\n".join(f"[Adm#{n} {d} | {h}] {s[:150]}" for n, d, h, s in retrieve(note, [q] + items, emb))
    # regen1
    r0 = regen(note, q, port, 0.0)
    f1 = tournament(note, q, orig, [r0], port)
    # regenK @ T=1
    rk = [regen(note, q, port, 1.0) for _ in range(3)]
    fk = tournament(note, q, orig, rk, port)
    # regenK_ra @ T=1 from section spans
    rra = [regen(spans_ctx, q, port, 1.0) for _ in range(3)]
    fra = tournament(spans_ctx, q, orig, rra, port)
    return {"fold": row["fold"], "idx": row["idx"], "stored_label": row["stored_label"],
            "regen1": P2.judge(row, f1).get("label"), "regenK": P2.judge(row, fk).get("label"),
            "regenK_ra": P2.judge(row, fra).get("label"), "q": q, "gold": row["ground_truth"], "fra": fra}


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
        keys = [k for k in keys if rows[k]["stored_label"] == 0][:args.limit]
    emb = get_embedder()
    out_dir = PROJECT_ROOT / "runs" / "expGG_regen_verdict"; out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expGG_regen_verdict", served=P2.served_model_id(args.port))
    print(f"regen+verdict on {len(keys)} cases" + (" (SMOKE)" if args.limit else " (full)"), flush=True)
    recs = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, rows[k], looked[k], emb, args.port) for k in keys]
        for i, fut in enumerate(as_completed(futs), 1):
            recs.append(fut.result())
            if not args.limit and (i % 10 == 0 or i == len(keys)):
                print(f"  {i}/{len(keys)}", flush=True)
    (out_dir / "records.jsonl").write_text("\n".join(json.dumps({k: r[k] for k in ["fold", "idx", "stored_label", "regen1", "regenK", "regenK_ra"]}) for r in recs))
    if args.limit:
        for r in recs:
            print("=" * 80); print("Q:", r["q"][:90]); print("GOLD:", r["gold"][:110])
            print(f"  regen1={r['regen1']} regenK={r['regenK']} regenK_ra={r['regenK_ra']}")
            print("  regenK_ra answer:", r["fra"].strip()[:160])
    else:
        W = [r for r in recs if r["stored_label"] == 0]; C = [r for r in recs if r["stored_label"] == 1]
        print(f"\n=== regen+verdict (cached Qwen2.5 regen+count 40-case was +3) ===")
        for arm in ["regen1", "regenK", "regenK_ra"]:
            f = sum(1 for r in W if r[arm] == 1); b = sum(1 for r in C if r[arm] == 0)
            print(f"  {arm:10} FIX {f}/{len(W)}={f/len(W)*100:.0f}%  BREAK {b}/{len(C)}={b/len(C)*100:.0f}%  net(sample) +{f-b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
