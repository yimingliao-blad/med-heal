#!/usr/bin/env python3
"""
Retriever bake-off (Step B).

Compares note-span retrievers on the wrong-and-detected items from the
existing D2+V2 pilot audit log:

  R1   single-query embedding (legacy: error_statement only, max scoring)
  R2   multi-query embedding with agreement scoring (the K=5 reasons + question)
  R3   LLM cite-by-number (Qwen2.5, K=5, vote)

For each item we run all three retrievers and ask GPT-4o (offline eval, not
in the runtime pipeline) to grade whether each retriever's top-3 spans
contain enough evidence to answer the question correctly. GPT-4o sees
question + ground_truth + the retrieved spans only — not the original
discharge note — so its judgment is purely about whether the retrieved
spans suffice.

Output: output/step9_v2/retriever_bakeoff.json + a summary in
        output/step9_v2/retriever_bakeoff.md

GPT-4o usage here is OFFLINE EVALUATION only, not runtime gating. This is
the same role GPT-4o plays as judge_orig / judge_corrected.
"""
from __future__ import annotations

import os
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

SOURCE_ROOT = Path(os.environ.get("PRE_ATOM_SOURCE_REPO_ROOT", Path(__file__).resolve().parents[5]))
RUN_ROOT = Path(os.environ.get("PRE_ATOM_PROJECT_ROOT", SOURCE_ROOT))
PROJECT_ROOT = SOURCE_ROOT
OUT_DIR = RUN_ROOT / "output" / "step9_v2"

sys.path.insert(0, str(Path(__file__).parent))
from correction import _collect_d2_queries
from llm_span_retrieval import llm_topk_spans
from note_span_index import topk_spans
from judge import client


GRADE_SYS = (
    "You are a medical expert grading whether a small set of retrieved sentences "
    "from a discharge note is sufficient to answer a clinical question correctly."
)

GRADE_USER_TMPL = """QUESTION:
{question}

GROUND TRUTH ANSWER:
{ground_truth}

RETRIEVED SENTENCES (from a discharge note):
{spans_block}

Decide whether the retrieved sentences contain ENOUGH evidence to construct
the ground-truth answer above. The retrieved sentences do NOT need to literally
spell out the ground truth; they need to give a careful reader the necessary
facts to derive it.

Reply with EXACTLY two lines:
SUFFICIENT: yes  or  no
WHY: <one short sentence>"""


def _grade_spans(question: str, ground_truth: str, spans: list[dict]) -> dict:
    spans_block = "\n".join(f"  [{i+1}] \"{s['sentence']}\""
                            for i, s in enumerate(spans))
    if not spans_block:
        spans_block = "  (none retrieved)"
    user = GRADE_USER_TMPL.format(question=question, ground_truth=ground_truth,
                                  spans_block=spans_block)
    for attempt in range(3):
        try:
            r = client().chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": GRADE_SYS},
                          {"role": "user", "content": user}],
                max_tokens=80,
                temperature=0.0,
            )
            text = r.choices[0].message.content.strip()
            sufficient = None
            why = ""
            for line in text.splitlines():
                u = line.strip().upper()
                if u.startswith("SUFFICIENT:"):
                    after = line.split(":", 1)[1].strip().lower()
                    sufficient = "yes" if after.startswith("yes") else (
                        "no" if after.startswith("no") else None)
                elif u.startswith("WHY:"):
                    why = line.split(":", 1)[1].strip()
            return {"sufficient": sufficient, "why": why, "raw": text}
        except Exception as e:
            print(f"  grade retry {attempt+1}/3: {e}", flush=True)
            time.sleep(5)
    return {"sufficient": None, "why": "", "raw": ""}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8003)
    p.add_argument("--log", type=Path, default=OUT_DIR / "pilot_audit_log.jsonl")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    if not args.log.exists():
        print(f"!! audit log not found: {args.log}")
        return 1

    recs = [json.loads(l) for l in open(args.log)]
    # Wrong + detected items: those where pipeline fired and item was actually wrong
    targets = [r for r in recs
               if (r.get("detection") or {}).get("fired")
               and (r.get("judge_orig") or {}).get("label") == 0]
    if args.limit:
        targets = targets[:args.limit]
    print(f"Bake-off on {len(targets)} wrong+detected items", flush=True)

    results = []
    for i, r in enumerate(targets, 1):
        item = r["item"]
        det = r["detection"]
        note = item["note"]
        question = item["question"]
        gt = item["ground_truth"]
        es = det.get("error_statement", "")

        # ---- R1: legacy single-query embedding (error_statement only, max) ----
        r1 = topk_spans(note, [es] if es else [question], k=3, scoring="max")

        # ---- R2: multi-query embedding agreement, top-5 ----
        queries = _collect_d2_queries(det, question)
        r2 = topk_spans(note, queries, k=5, scoring="agreement")

        # ---- R3: LLM cite-by-number, question-only, top-5 ----
        try:
            r3_full = llm_topk_spans(note, question, "", port=args.port, k=5,
                                     max_per_sample=5, max_topk=5)
            r3 = [{
                "sentence": t["sentence"],
                "similarity": t["votes"] / 5,
                "sentence_number": t["sentence_number"],
            } for t in r3_full["top_sentences"]]
        except Exception as e:
            print(f"  R3 error on ({r['fold']},{r['idx']}): {e}", flush=True)
            r3 = []

        # ---- R4: union of R2 and R3, deduped, capped to 5 ----
        r4: list[dict] = []
        seen_sents: set[str] = set()
        # Interleave: take 1 from R3, 1 from R2, 1 from R3, ... so each
        # retriever gets equal representation in the top-5 union
        i_r2 = 0; i_r3 = 0
        while len(r4) < 5 and (i_r2 < len(r2) or i_r3 < len(r3)):
            if i_r3 < len(r3):
                cand = r3[i_r3]; i_r3 += 1
                key = cand["sentence"][:80]
                if key not in seen_sents:
                    seen_sents.add(key)
                    r4.append({**cand, "source": "R3"})
            if len(r4) >= 5:
                break
            if i_r2 < len(r2):
                cand = r2[i_r2]; i_r2 += 1
                key = cand["sentence"][:80]
                if key not in seen_sents:
                    seen_sents.add(key)
                    r4.append({**cand, "source": "R2"})

        # ---- Grade each via GPT-4o ----
        g1 = _grade_spans(question, gt, r1)
        time.sleep(0.5)
        g2 = _grade_spans(question, gt, r2)
        time.sleep(0.5)
        g3 = _grade_spans(question, gt, r3)
        time.sleep(0.5)
        g4 = _grade_spans(question, gt, r4)
        time.sleep(0.5)

        results.append({
            "fold": r["fold"], "idx": r["idx"],
            "question": question[:200],
            "ground_truth": gt[:300],
            "error_statement": es[:200],
            "R1": {"spans": r1, "grade": g1},
            "R2": {"spans": r2, "grade": g2},
            "R3": {"spans": r3, "grade": g3},
            "R4": {"spans": r4, "grade": g4},
        })
        print(f"  [{i}/{len(targets)}] R1={g1.get('sufficient')} R2={g2.get('sufficient')} R3={g3.get('sufficient')} R4={g4.get('sufficient')}", flush=True)

    # Tally
    def yes_rate(key):
        ys = sum(1 for r in results if r[key]["grade"].get("sufficient") == "yes")
        return ys, ys / len(results) if results else 0

    y1, p1 = yes_rate("R1")
    y2, p2 = yes_rate("R2")
    y3, p3 = yes_rate("R3")
    y4, p4 = yes_rate("R4")

    print()
    print("=" * 60)
    print(f"RETRIEVER BAKE-OFF — N={len(results)} wrong+detected items")
    print("=" * 60)
    print(f"  R1 (legacy embed, top-3, error_stmt only):    {y1}/{len(results)} = {100*p1:.0f}%")
    print(f"  R2 (multi-query embed agreement, top-5):       {y2}/{len(results)} = {100*p2:.0f}%")
    print(f"  R3 (Qwen2.5 cite-by-number K=5, q-only, top-5): {y3}/{len(results)} = {100*p3:.0f}%")
    print(f"  R4 (R3 ∪ R2 union deduped, top-5):             {y4}/{len(results)} = {100*p4:.0f}%")

    out_path = OUT_DIR / "retriever_bakeoff.json"
    out_path.write_text(json.dumps({
        "n": len(results),
        "R1_sufficient": y1, "R1_rate": p1,
        "R2_sufficient": y2, "R2_rate": p2,
        "R3_sufficient": y3, "R3_rate": p3,
        "R4_sufficient": y4, "R4_rate": p4,
        "items": results,
    }, indent=2, default=str))

    md_path = OUT_DIR / "retriever_bakeoff.md"
    md = [
        "# Retriever Bake-off (Step B)",
        "",
        f"N = {len(results)} wrong+detected items from the D2+V2 pilot audit log.",
        "",
        "GPT-4o (temp=0, OFFLINE eval only) graded each retriever's spans on",
        "whether they contain enough evidence to construct the ground-truth answer.",
        "",
        "| Retriever | Sufficient |",
        "|---|---:|",
        f"| R1 — single-query embedding (error_statement, top-3) | {y1}/{len(results)} = {100*p1:.0f}% |",
        f"| R2 — multi-query embed + agreement scoring (top-5) | {y2}/{len(results)} = {100*p2:.0f}% |",
        f"| R3 — Qwen2.5 cite-by-number K=5, question-only (top-5) | {y3}/{len(results)} = {100*p3:.0f}% |",
        f"| R4 — Union(R3, R2) deduped (top-5) | {y4}/{len(results)} = {100*p4:.0f}% |",
        "",
    ]
    md_path.write_text("\n".join(md))
    print(f"\nWrote {out_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
