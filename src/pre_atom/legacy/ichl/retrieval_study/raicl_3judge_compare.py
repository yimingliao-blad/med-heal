"""3-way judge comparison: GPT-4o (Tier 3 final), Magistral M4 (Tier 1 dev),
Qwen3-235B-MLX M4 (Tier 2 audit). Operates on existing GPT-4o-judged smoke
outputs in raicl_pilot/<target>/<fold>/<variant>_judged_gpt4o.jsonl.

Adds Magistral and Qwen3-235B labels per item, prints agreement table.

Magistral: vLLM on localhost:8003 (must be Magistral, not target).
Qwen3-235B: MLX on 192.168.68.107:8800, Qwen3.5-27B-6bit-NexVeridian, C=1.

Usage:
    PYTHONPATH=src .venv/bin/python -m ichl.retrieval_study.raicl_3judge_compare \
        --target qwen2.5-7b-instruct --fold 0
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RS = ROOT / "output" / "ichl" / "retrieval_study"
POOL_DIR = RS / "pool_index"


def load_full_notes() -> dict[int, str]:
    """Step8-format full note. Same as raicl_pilot._load_full_notes_step8_format."""
    notes_file = ROOT / "output" / "EHRNoteQA_processed.jsonl"
    out = {}
    for line in notes_file.open():
        if not line.strip(): continue
        r = json.loads(line)
        pid = int(r["patient_id"])
        parts = []
        for i in [1, 2, 3]:
            v = r.get(f"note_{i}")
            if v and str(v).strip() and str(v).lower() != "nan":
                parts.append(f"[Note {i}]\n{str(v).strip()}")
        out[pid] = "\n\n".join(parts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    out_dir = RS / "raicl_pilot" / args.target / f"fold_{args.fold}"
    notes = load_full_notes()

    # Variants to judge — assume zs + mode_A + mode_B + mode_C exist
    from ichl.judges.magistral_judge import MagistralJudge, M4_RULES, SYSTEM as MAG_SYSTEM, _build_user
    from openai import OpenAI
    mag_judge = MagistralJudge(base_url="http://localhost:8003/v1", model="Magistral-Small-2509-AWQ")
    qwen3_client = OpenAI(base_url="http://192.168.68.107:8800/v1", api_key="not-needed", timeout=300)

    rows_all: list[dict] = []
    for variant in ["zs", "mode_A", "mode_B", "mode_C"]:
        f = out_dir / f"{variant}_judged_gpt4o.jsonl"
        if not f.exists():
            print(f"  MISSING: {f}")
            continue
        for r in (json.loads(l) for l in f.open() if l.strip()):
            r["variant"] = variant
            rows_all.append(r)

    print(f"Loaded {len(rows_all)} items across {len(set(r['variant'] for r in rows_all))} variants.\n")

    # ===== Magistral =====
    print("=" * 60)
    print("Phase 1 — Magistral M4 (local vLLM, fast)")
    print("=" * 60)
    t0 = time.monotonic()
    for r in rows_all:
        note = notes[r["patient_id"]]
        m = mag_judge.judge(question=r["question"], ground_truth=r["ground_truth"],
                            model_answer=r.get("model_answer", ""), note=note)
        r["mag_label"] = m.label
        r["mag_latency_s"] = m.latency_s
        r["mag_raw"] = m.content[:30]
    print(f"  done in {time.monotonic()-t0:.0f}s")

    # ===== Qwen3-235B-MLX =====
    print("\n" + "=" * 60)
    print("Phase 2 — Qwen3.5-27B-6bit (MLX 8800, audit, C=1)")
    print("=" * 60)
    t0 = time.monotonic()
    qwen_model = "/Users/madblade/Projects/local-llm/models/mlx/Qwen3.5-27B-6bit-NexVeridian"
    for i, r in enumerate(rows_all, 1):
        note = notes[r["patient_id"]]
        user = _build_user(r["question"], r["ground_truth"], r.get("model_answer", ""), note)
        try:
            resp = qwen3_client.chat.completions.create(
                model=qwen_model,
                messages=[{"role": "system", "content": MAG_SYSTEM}, {"role": "user", "content": user}],
                temperature=0.0, max_tokens=256,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            txt = (resp.choices[0].message.content or "").strip()
            m = re.search(r"[01]", txt)
            r["q235_label"] = int(m.group(0)) if m else None
            r["q235_raw"] = txt[:30]
        except Exception as e:
            r["q235_label"] = None
            r["q235_err"] = str(e)[:100]
        if i % 5 == 0:
            print(f"  {i}/{len(rows_all)} ({time.monotonic()-t0:.0f}s)")
    print(f"  done in {time.monotonic()-t0:.0f}s")

    # ===== Comparison =====
    print("\n" + "=" * 60)
    print("3-way per-variant agreement")
    print("=" * 60)
    print(f"{'variant':10s} {'pid':<10s} {'gpt4o':<5s} {'mag':<5s} {'q235':<5s} {'all_agree':<10s}")
    by_variant: dict[str, list[dict]] = {}
    for r in rows_all:
        by_variant.setdefault(r["variant"], []).append(r)
        all_agree = (r.get("binary_correct") == r.get("mag_label") == r.get("q235_label"))
        print(f"{r['variant']:10s} {r['patient_id']:<10d} {r.get('binary_correct', '?'):<5} "
              f"{r.get('mag_label', '?'):<5} {r.get('q235_label', '?'):<5} {'Y' if all_agree else 'X':<10}")

    print("\n" + "=" * 60)
    print("Per-variant accuracy across the 3 judges")
    print("=" * 60)
    print(f"{'variant':10s} {'gpt4o':<8s} {'mag':<8s} {'q235':<8s}")
    for variant, rs in by_variant.items():
        n = len(rs)
        gpt = sum(1 for r in rs if r.get("binary_correct") == 1)
        mag = sum(1 for r in rs if r.get("mag_label") == 1)
        q235 = sum(1 for r in rs if r.get("q235_label") == 1)
        print(f"{variant:10s} {gpt}/{n}     {mag}/{n}     {q235}/{n}")

    # Pairwise agreement
    print("\n" + "=" * 60)
    print("Pairwise agreement")
    print("=" * 60)
    pairs = [("gpt4o vs magistral", "binary_correct", "mag_label"),
             ("gpt4o vs q235", "binary_correct", "q235_label"),
             ("magistral vs q235", "mag_label", "q235_label")]
    for label, a, b in pairs:
        agree = sum(1 for r in rows_all if r.get(a) == r.get(b))
        print(f"  {label:24s}: {agree}/{len(rows_all)} = {100*agree/len(rows_all):.1f}%")

    # Save
    save_path = out_dir / "judges_3way.jsonl"
    with save_path.open("w") as f:
        for r in rows_all:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
