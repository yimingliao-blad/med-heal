#!/usr/bin/env python3
"""
RA-ICL pilot (fresh QA with retrieved example).

The standard RA-ICL definition: take the clinical question, retrieve a
similar (question, correct-answer) example from the BM contrast pool, prepend
it as a single few-shot demonstration, ask the target model to answer the
new question. NO verdict step, NO parsing — GPT-4o judges the resulting
answer directly. Fix/break is computed by comparing the GPT-4o oracle
verdict on the RA-ICL answer to the verdict on the original zero-shot
answer (already in the source log).

This is "zero-shot + similar question example", not "correct this previous
wrong answer". The original wrong answer is NOT shown to the model.

Sampling: 100 random items per model (random-per-fold), matching the phase 1
regen+count pilot. The pool retrieval uses fold-disjoint cross-validation.

Usage:
    python raicl_test.py --source-log llama-3.1-8b-instruct/regen_p2_v3.jsonl \
                         --random-per-fold 20 --port 8003
"""
from __future__ import annotations

import os
import argparse
import json
import random
import sys
import time
from pathlib import Path

SOURCE_ROOT = Path(os.environ.get("PRE_ATOM_SOURCE_REPO_ROOT", Path(__file__).resolve().parents[5]))
RUN_ROOT = Path(os.environ.get("PRE_ATOM_PROJECT_ROOT", SOURCE_ROOT))
PROJECT_ROOT = SOURCE_ROOT
sys.path.insert(0, str(Path(__file__).parent))
from correction import retrieve_contrast_example
from detection_format_bakeoff import served_model_id, set_default_chat_template_kwargs, vllm_chat
from judge import judge as judge_call

OUT_DIR = PROJECT_ROOT / "output" / "step9_v2" / "multi_model"


# RA-ICL prompt: one (question, correct-answer) worked example, then fresh QA.
# The model sees the example and the NEW note+question — NOT the original answer.
RAICL_SYS = "You are a medical expert answering clinical questions grounded in discharge notes."

RAICL_USER_TMPL = """Here is an example of a well-answered clinical question grounded in a discharge note.

=== EXAMPLE (different patient) ===
Question: {ex_question}

Correct answer:
{ex_correct_answer}

Key evidence from the notes:
{ex_evidence_block}
=== END EXAMPLE ===

Now answer THIS question about a different patient.

Discharge note:
{note}

Question: {question}

Answer the question using only information from the discharge note. Be specific
and complete. Ground every claim in the note. Reply in 1-3 sentences."""


def build_raicl_prompt(item: dict, contrast_ex: dict) -> str:
    evidence = contrast_ex.get("evidence_from_notes", []) or []
    evidence_block = "\n".join(f'  - "{q}"' for q in evidence) if evidence else "  (none)"
    return RAICL_USER_TMPL.format(
        ex_question=contrast_ex.get("question", ""),
        ex_correct_answer=(contrast_ex.get("ground_truth", "") or "")[:400],
        ex_evidence_block=evidence_block,
        note=item["note"][:18000],
        question=item["question"],
    )


def sample_random_per_fold(recs: list[dict], per_fold: int, seed: int) -> list[dict]:
    """Sample `per_fold` random items from each fold."""
    by_fold: dict[int, list[dict]] = {}
    for r in recs:
        by_fold.setdefault(r["fold"], []).append(r)
    rng = random.Random(seed)
    out = []
    for fold in sorted(by_fold):
        pool = by_fold[fold]
        k = min(per_fold, len(pool))
        out.extend(rng.sample(pool, k))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source-log", required=True,
                   help="audit log inside output/step9_v2/multi_model/ (e.g. llama-3.1-8b-instruct/regen_p2_v3.jsonl)")
    p.add_argument("--random-per-fold", type=int, default=20,
                   help="random items per fold (default 20 → 100 total)")
    p.add_argument("--port", type=int, default=8003)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-think", action="store_true",
                   help="disable Qwen3 thinking mode via chat_template_kwargs")
    p.add_argument("--audit-name", default=None,
                   help="output filename (default: raicl_pilot.jsonl under model dir)")
    args = p.parse_args()

    if args.no_think:
        set_default_chat_template_kwargs({"enable_thinking": False})

    served = served_model_id(args.port)
    print(f"vLLM serving: {served}")
    print(f"Source log: {args.source_log}")
    print()

    log_path = OUT_DIR / args.source_log
    if not log_path.exists():
        print(f"!! log not found: {log_path}")
        return 1
    recs = [json.loads(l) for l in open(log_path)]
    print(f"Total items in log: {len(recs)}")

    # Sample random items across all folds (not just wrong ones)
    sample = sample_random_per_fold(recs, args.random_per_fold, args.seed)
    n_wrong = sum(1 for r in sample if (r.get("judge_orig") or {}).get("label") == 0)
    n_correct = sum(1 for r in sample if (r.get("judge_orig") or {}).get("label") == 1)
    print(f"Sampled {len(sample)} items ({n_wrong}W + {n_correct}C) for RA-ICL pilot")
    print()

    # Determine output path
    model_dir = Path(args.source_log).parts[0]  # e.g. "llama-3.1-8b-instruct"
    out_dir = OUT_DIR / model_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_name = args.audit_name or "raicl_pilot.jsonl"
    out_path = out_dir / audit_name

    # Resume support: skip already-done items
    done_keys = set()
    if out_path.exists():
        for line in open(out_path):
            rec = json.loads(line)
            done_keys.add((rec["fold"], rec["idx"]))
        print(f"Resuming: {len(done_keys)} already done")

    fh = open(out_path, "a")
    stats = {"fix": 0, "brk": 0, "keep_c": 0, "keep_w": 0, "skip": 0}

    for i, r in enumerate(sample, 1):
        if (r["fold"], r["idx"]) in done_keys:
            continue

        item = r["item"]
        j_orig = (r.get("judge_orig") or {}).get("label")

        # Retrieve contrast example from BM contrast pool (fold-disjoint)
        contrast_ex = retrieve_contrast_example(r["fold"], item["question"])
        if not contrast_ex:
            print(f"  [{i}/{len(sample)}] no contrast example for fold={r['fold']}")
            rec = {"fold": r["fold"], "idx": r["idx"], "judge_orig": j_orig,
                   "skipped": "no_contrast_ex"}
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            stats["skip"] += 1
            continue

        # Build fresh QA prompt with one example and call vLLM
        prompt = build_raicl_prompt(item, contrast_ex)
        try:
            answer = vllm_chat(RAICL_SYS, prompt, args.port,
                               max_tokens=600, temperature=0.0)
        except Exception as e:
            print(f"  [{i}/{len(sample)}] regen err: {e}")
            rec = {"fold": r["fold"], "idx": r["idx"], "judge_orig": j_orig,
                   "skipped": f"regen_err:{e}"}
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            stats["skip"] += 1
            continue

        # Judge RA-ICL answer with GPT-4o oracle
        judge = judge_call(item["note"], item["question"], item["ground_truth"],
                           answer, n=1, temperature=0.0)
        j_raicl = judge["label"]
        time.sleep(0.5)

        # Compute outcome: fix / break / keep
        if j_orig == 0 and j_raicl == 1:
            outcome = "fix"; stats["fix"] += 1
        elif j_orig == 1 and j_raicl == 0:
            outcome = "break"; stats["brk"] += 1
        elif j_orig == 1 and j_raicl == 1:
            outcome = "keep_correct"; stats["keep_c"] += 1
        else:
            outcome = "keep_wrong"; stats["keep_w"] += 1

        rec = {
            "fold": r["fold"], "idx": r["idx"],
            "judge_orig": j_orig,
            "judge_raicl": j_raicl,
            "outcome": outcome,
            "question": item["question"][:200],
            "ground_truth": item["ground_truth"][:200],
            "original_answer": item["original_answer"][:300],
            "raicl_answer": answer[:600],
            "contrast_ex_question": contrast_ex.get("question", "")[:200],
            "contrast_retrieval_sim": contrast_ex.get("retrieval_sim", 0.0),
        }
        fh.write(json.dumps(rec, default=str) + "\n")
        fh.flush()

        print(f"  [{i}/{len(sample)}] fold={r['fold']} idx={r['idx']} orig={j_orig} raicl={j_raicl} → {outcome}", flush=True)

        if i % 5 == 0:
            total_done = stats["fix"] + stats["brk"] + stats["keep_c"] + stats["keep_w"]
            print(f"    progress: {total_done} done, fix={stats['fix']} brk={stats['brk']} skip={stats['skip']}")

    fh.close()

    # Final tally
    total = stats["fix"] + stats["brk"] + stats["keep_c"] + stats["keep_w"]
    print()
    print("=" * 70)
    print(f"RA-ICL PILOT  (target model: {served})")
    print("=" * 70)
    print(f"  N sampled: {len(sample)}")
    print(f"  Skipped: {stats['skip']}")
    print(f"  Processed: {total}")
    print(f"  FIX (wrong→correct):    {stats['fix']}")
    print(f"  BREAK (correct→wrong):  {stats['brk']}")
    print(f"  Keep correct:           {stats['keep_c']}")
    print(f"  Keep wrong:             {stats['keep_w']}")
    if total > 0:
        acc_orig = (stats["keep_c"] + stats["brk"]) / total * 100
        acc_raicl = (stats["keep_c"] + stats["fix"]) / total * 100
        print(f"  acc_orig={acc_orig:.1f}%  acc_raicl={acc_raicl:.1f}%  delta={acc_raicl-acc_orig:+.1f}pp")
        print(f"  net = fix-brk = {stats['fix']-stats['brk']:+d}")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
