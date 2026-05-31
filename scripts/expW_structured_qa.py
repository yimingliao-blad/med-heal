#!/usr/bin/env python3
"""Structured-scaffold focused QA: machinery chunks -> routed sections (with provenance) -> QA -> judge.

Builds on note_chunker: split the note into admissions (chrono-numbered, dated) and sections, route the
relevant sections to the question (embedding over section text), and present a FOCUSED, provenance-tagged
context (Admission #k (date): SECTION: content) to the QA model. Then answer + judge vs gold.

Compares Qwen and GPT, vs the earlier raw-embedding-span focused QA (Qwen 28/45, GPT 35/30).
Usage: python scripts/expW_structured_qa.py --topk 12 --concurrency 4
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
from note_span_index import get_embedder  # noqa
from note_chunker import structured  # noqa
from llm_audit import set_ledger  # noqa

QA_SYS = "You answer a clinical question using ONLY the provided note sections. Each fact is tagged with the admission and date it came from; use that for any temporal comparison. Do not use outside knowledge."
QA_USER = """Note sections (organized by admission):
{ctx}

Question:
{question}

Answer using only these sections. Be specific, quote values exactly, and respect which admission each fact is from."""


def build_context(note, query, emb, topk):
    """Route sections (across admissions) to the query, keep provenance, drop junk headers."""
    DROP = {"Patient ID", "Name", "Date of Birth", "Allergies", "Attending", "Admission Date", "Service", "Unit No"}
    rows = []  # (adm_n, date, header, content)
    for a in structured(note):
        for h, c in a["fields"].items():
            if h in DROP or not c.strip():
                continue
            rows.append((a["n"], a["chartdate"], h, c))
    if not rows:
        return ""
    texts = [f"{h}: {c}"[:250] for _, _, h, c in rows]
    sv = emb.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    qv = emb.encode([query], normalize_embeddings=True, show_progress_bar=False)
    order = np.argsort(-(sv @ qv.T)[:, 0])[:topk]
    keep = sorted([rows[i] for i in order], key=lambda r: (r[0], r[2]))
    lines, cur = [], None
    for n, date, h, c in keep:
        if n != cur:
            lines.append(f"\nAdmission #{n} ({date}):"); cur = n
        lines.append(f"  {h}: {c[:220]}")
    return "\n".join(lines)


def process_one(row, k, port, emb):
    q = row["question"]
    ctx = build_context(row["note"], q, emb, k)
    user = QA_USER.format(ctx=ctx, question=q)
    qwen = P2.vllm_chat(QA_SYS, user, port, 350, 0.0, tag="sqa.qwen")
    gpt = P2.gpt("gpt-4o", QA_SYS, user, 350, 0.0, False, "sqa.gpt")
    return {"fold": row["fold"], "idx": row["idx"], "stored_label": row["stored_label"],
            "qwen_correct": P2.judge(row, qwen).get("label"),
            "gpt_correct": P2.judge(row, gpt).get("label")}


def report(recs, model):
    W = [r for r in recs if r["stored_label"] == 0]; C = [r for r in recs if r["stored_label"] == 1]
    key = f"{model}_correct"
    fixes = sum(1 for r in W if r[key] == 1); breaks = sum(1 for r in C if r[key] == 0)
    print(f"  {model:5} FIX {fixes}/{len(W)}={fixes/max(1,len(W))*100:.0f}%   BREAK {breaks}/{len(C)}={breaks/max(1,len(C))*100:.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    rows = {(r["fold"], r["idx"]): r for r in P2.load_rows(40, 20, 42)}
    keys = list(rows)
    emb = get_embedder()
    out_dir = PROJECT_ROOT / "runs" / "expW_structured_qa" / f"k{args.topk}"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_ledger(out_dir / "llm_calls.jsonl", script="expW_structured_qa", served=P2.served_model_id(args.port))
    print(f"structured-scaffold QA on {len(keys)} cases, topk sections={args.topk}", flush=True)
    recs = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, rows[key], args.topk, args.port, emb) for key in keys]
        for i, fut in enumerate(as_completed(futs), 1):
            recs.append(fut.result())
            if i % 10 == 0 or i == len(keys):
                print(f"  {i}/{len(keys)}", flush=True)
    (out_dir / "records.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    print("\n=== structured-scaffold focused QA (vs raw-span: Qwen 28/45, GPT 35/30) ===")
    report(recs, "qwen")
    report(recs, "gpt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
