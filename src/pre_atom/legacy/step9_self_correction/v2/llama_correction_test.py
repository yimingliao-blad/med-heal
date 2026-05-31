#!/usr/bin/env python3
"""
Llama correction-step test: edit-from-previous vs CoVe vs vanilla regen.

Tests 3 different correction prompt strategies on the SAME 10 wrong items
from a Llama audit log. Each strategy is followed directly by the GPT-4o
oracle judge — NO verdict step. We measure raw correction quality (fix rate)
to identify what the regen step itself can achieve.

Strategies:

  R0  vanilla regen (no signal, no original answer in context)
      — current "regen+count" approach, anchored to nothing
      — control / baseline

  EDIT  edit-from-previous
      — original answer SHOWN in the prompt; model is asked to verify each
        claim against the note and produce a revised answer that preserves
        unchanged content and fixes the wrong parts.
      — designed to stop Llama's "fresh-context drift" failure mode

  COVE  true Chain-of-Verification (Dhuliawala et al. ACL 2024)
      — step 1: extract a list of factual claims from the original answer
      — step 2: for EACH claim, verify against the note (yes/no/unclear)
      — step 3: rewrite the answer dropping unverified claims and adding
        any missing facts identified in the verification
      — implemented as 3 sequential vLLM calls per item (more compute)

We use the same 10 wrong items the user requested, drawn from the Llama
phase 2 V3 audit log.

Output: output/step9_v2/multi_model/llama_correction_test.json + table.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
from detection_format_bakeoff import served_model_id, vllm_chat
from judge import judge as judge_call

OUT_DIR = PROJECT_ROOT / "output" / "step9_v2" / "multi_model"


# ---------------- Strategy R0: vanilla regen (control) ----------------

R0_SYS = "You are a medical expert."
R0_USER = """Discharge note:
{note}

Question: {question}

Answer the question using only information from the discharge note. Be specific
and complete. If the question asks about multiple visits, conditions, or events,
cover all of them."""


def run_r0(item: dict, port: int) -> str:
    user = R0_USER.format(note=item["note"][:18000], question=item["question"])
    return vllm_chat(R0_SYS, user, port, max_tokens=600, temperature=0.0)


# ---------------- Strategy EDIT: edit-from-previous ----------------

EDIT_SYS = "You are a medical expert auditing a previous answer for errors against discharge notes."

EDIT_USER = """Discharge note:
{note}

Question: {question}

Below is a previous answer to the question. It MAY contain errors but it
also contains correct content that should be preserved.

PREVIOUS ANSWER:
{original_answer}

Your task: produce a REVISED version of the previous answer.

Rules:
1. Keep every claim that is supported by the discharge note exactly as it was.
2. For any claim that contradicts the note, replace it with the correct fact
   from the note (quoting the relevant evidence is helpful).
3. If the previous answer omits a fact essential to the question, add it.
4. Do NOT delete supported claims just to be concise. If the previous answer
   covered multiple facts and they are all correct, keep all of them.
5. Do NOT introduce new facts that are not in the note.

Reply with the REVISED answer in 1-5 sentences."""


def run_edit(item: dict, port: int) -> str:
    user = EDIT_USER.format(
        note=item["note"][:18000],
        question=item["question"],
        original_answer=item["original_answer"][:1500],
    )
    return vllm_chat(EDIT_SYS, user, port, max_tokens=700, temperature=0.0)


# ---------------- Strategy COVE: true chain-of-verification ----------------

COVE_EXTRACT_SYS = "You are a medical expert breaking down an answer into atomic factual claims."

COVE_EXTRACT_USER = """Below is an answer to a clinical question. Break the answer into a list of
atomic factual claims (one fact per line). Do not include opinions or filler;
include only verifiable factual claims.

ANSWER:
{original_answer}

Reply with each claim on its own line, prefixed with a dash. Example format:
- claim 1
- claim 2
- claim 3"""


COVE_VERIFY_SYS = "You are a strict medical expert verifying a single factual claim against discharge notes."

COVE_VERIFY_USER = """Discharge note:
{note}

CLAIM: {claim}

Is this claim supported by the discharge note above?

Reply on the FIRST line with exactly one word: yes  or  no
On the SECOND line, give one short sentence saying which sentence in the note
supports the claim (or "no support found")."""


COVE_REVISE_SYS = "You are a medical expert producing a final answer grounded in verified evidence."

COVE_REVISE_USER = """Discharge note:
{note}

Question: {question}

You previously gave this answer:
{original_answer}

Your individual claims have been verified against the note. Here is the
verification result for each claim:

{verification_block}

Now produce a REVISED answer to the question. Use ONLY the verified-yes claims.
If a verified-no claim is essential to the question, replace it with the correct
fact from the note. Be specific and complete in 1-5 sentences."""


_BULLET_RE = re.compile(r"^\s*[-*\u2022]\s*(.+?)\s*$", re.MULTILINE)


def parse_claims(text: str) -> list[str]:
    out = []
    for m in _BULLET_RE.finditer(text):
        c = m.group(1).strip()
        if c and len(c) >= 8:
            out.append(c)
    if not out:
        # fallback: split lines
        for line in text.splitlines():
            c = line.strip().lstrip("0123456789.) ").strip()
            if c and len(c) >= 8 and not c.startswith("#"):
                out.append(c)
    return out[:8]  # cap at 8 claims


def parse_yesno(text: str) -> str:
    if not text:
        return "unclear"
    first = text.strip().splitlines()[0].lower() if text.strip() else ""
    if first.startswith("yes"):
        return "yes"
    if first.startswith("no"):
        return "no"
    return "unclear"


def run_cove(item: dict, port: int) -> dict:
    """Three-step CoVe. Returns dict with intermediate steps + final answer."""
    # Step 1: extract claims
    claims_raw = vllm_chat(
        COVE_EXTRACT_SYS,
        COVE_EXTRACT_USER.format(original_answer=item["original_answer"][:1500]),
        port, max_tokens=400, temperature=0.0,
    )
    claims = parse_claims(claims_raw)
    if not claims:
        return {"claims": [], "verifications": [], "revised": "",
                "error": "no claims extracted"}

    # Step 2: verify each claim
    verifications = []
    for c in claims:
        v_raw = vllm_chat(
            COVE_VERIFY_SYS,
            COVE_VERIFY_USER.format(note=item["note"][:18000], claim=c),
            port, max_tokens=80, temperature=0.0,
        )
        verdict = parse_yesno(v_raw)
        verifications.append({"claim": c, "verdict": verdict, "raw": v_raw[:200]})

    # Step 3: revise
    verification_block = "\n".join(
        f"  - [{v['verdict']}] {v['claim']}" for v in verifications
    )
    revised = vllm_chat(
        COVE_REVISE_SYS,
        COVE_REVISE_USER.format(
            note=item["note"][:18000],
            question=item["question"],
            original_answer=item["original_answer"][:1500],
            verification_block=verification_block,
        ),
        port, max_tokens=700, temperature=0.0,
    )
    return {"claims": claims, "verifications": verifications,
            "claims_raw": claims_raw, "revised": revised}


# ---------------- Driver ----------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source-log", default="llama-3.1-8b-instruct/regen_p2_v3.jsonl")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--port", type=int, default=8003)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    served = served_model_id(args.port)
    print(f"vLLM serving: {served}")
    print(f"Source log: {args.source_log}")
    print()

    log_path = OUT_DIR / args.source_log
    recs = [json.loads(l) for l in open(log_path)]
    wrong = [r for r in recs if (r.get("judge_orig") or {}).get("label") == 0]
    rng = random.Random(args.seed)
    sample = rng.sample(wrong, min(args.n, len(wrong)))
    print(f"Wrong items: {len(wrong)}, sampled {len(sample)}")
    print()

    results = []
    for i, r in enumerate(sample, 1):
        item = r["item"]
        print(f"[{i}/{len(sample)}] fold={r['fold']} idx={r['idx']}", flush=True)

        # R0
        try:
            r0_out = run_r0(item, args.port)
        except Exception as e:
            r0_out = ""
            print(f"  R0 err: {e}")
        time.sleep(0.3)
        r0_judge = judge_call(item["note"], item["question"], item["ground_truth"],
                              r0_out, n=1, temperature=0.0) if r0_out else {"label": None}
        time.sleep(0.5)

        # EDIT
        try:
            edit_out = run_edit(item, args.port)
        except Exception as e:
            edit_out = ""
            print(f"  EDIT err: {e}")
        edit_judge = judge_call(item["note"], item["question"], item["ground_truth"],
                                edit_out, n=1, temperature=0.0) if edit_out else {"label": None}
        time.sleep(0.5)

        # COVE
        try:
            cove_result = run_cove(item, args.port)
            cove_out = cove_result.get("revised", "")
        except Exception as e:
            cove_out = ""
            cove_result = {"error": str(e)}
            print(f"  COVE err: {e}")
        cove_judge = judge_call(item["note"], item["question"], item["ground_truth"],
                                cove_out, n=1, temperature=0.0) if cove_out else {"label": None}
        time.sleep(0.5)

        results.append({
            "fold": r["fold"], "idx": r["idx"],
            "question": item["question"][:200],
            "ground_truth": item["ground_truth"][:200],
            "original_answer": item["original_answer"][:300],
            "R0_out": r0_out[:400],
            "R0_judge": r0_judge["label"],
            "EDIT_out": edit_out[:400],
            "EDIT_judge": edit_judge["label"],
            "COVE_out": cove_out[:400],
            "COVE_judge": cove_judge["label"],
            "COVE_n_claims": len(cove_result.get("claims", [])),
            "COVE_n_verified_yes": sum(1 for v in cove_result.get("verifications", [])
                                       if v.get("verdict") == "yes"),
        })
        labels = [r0_judge.get("label"), edit_judge.get("label"), cove_judge.get("label")]
        print(f"  R0={labels[0]} EDIT={labels[1]} COVE={labels[2]}", flush=True)

    # Tally
    print()
    print("=" * 70)
    print(f"LLAMA CORRECTION TEST  (target: {served}, N={len(results)} wrong items)")
    print("=" * 70)
    for strat in ("R0", "EDIT", "COVE"):
        fixes = sum(1 for r in results if r[f"{strat}_judge"] == 1)
        wrong_still = sum(1 for r in results if r[f"{strat}_judge"] == 0)
        none = sum(1 for r in results if r[f"{strat}_judge"] is None)
        rate = 100 * fixes / max(1, fixes + wrong_still)
        print(f"  {strat:<6} fix={fixes}/{len(results)} ({rate:.0f}%)  still_wrong={wrong_still}  none={none}")

    print()
    print("Per-item:")
    print(f"  {'fold,idx':<10} {'R0':>4} {'EDIT':>6} {'COVE':>6}  question")
    for r in results:
        def fmt(label):
            if label == 1: return "FIX"
            if label == 0: return "..."
            return "?"
        print(f"  {r['fold']},{r['idx']:<8} {fmt(r['R0_judge']):>4} {fmt(r['EDIT_judge']):>6} {fmt(r['COVE_judge']):>6}  {r['question'][:80]}")

    out_path = OUT_DIR / "llama_correction_test.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
