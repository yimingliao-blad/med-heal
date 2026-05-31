"""Phase B.2 — extract decision rules from GPT-4o using verbatim Stage-1 binary prompt.

Per-case two-pass extraction:
  1. For N cases, feed GPT-4o the ORIGINAL Stage-1 binary system + user prompt,
     but with a reasoning-out-loud appendix requesting RULE + VERDICT.
  2. Verify returned VERDICT matches gold label; discard mismatches.
  3. Call GPT-4o once more to aggregate per-case rules into a unified rule-set.

Output: output/ichl/mlx_judge/rules/
  - per_case_rules.jsonl (10 per-case rationales)
  - unified_rules.md (aggregated 8-12 rule-set, ready to paste into Qwen3 prompt)
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEV_JSONL = PROJECT_ROOT / "output" / "ichl" / "mlx_judge" / "splits" / "dev.jsonl"
OUT_DIR = PROJECT_ROOT / "output" / "ichl" / "mlx_judge" / "rules"

SEED = 42
N_CASES = 10
TARGETS = ["qwen2.5-7b-instruct", "qwen3-8b", "llama-3.1-8b-instruct", "deepseek-r1-distill-llama-8b"]

# Verbatim Stage-1 binary system prompt
STAGE1_SYSTEM = "You are a medical expert evaluating an AI model's answer to a clinical question."

# Verbatim Stage-1 binary user template + reasoning-out-loud appendix
RULE_EXTRACTION_USER = """DISCHARGE SUMMARY:
{note}

QUESTION:
{question}

CORRECT ANSWER (Ground Truth):
{ground_truth}

MODEL'S ANSWER:
{model_answer}

Task: Evaluate if the model's answer is correct compared to the ground truth.

Before responding with the final digit, think aloud: articulate the specific decision rule or heuristic you apply to THIS case. What do you look for? What tips the verdict toward 1 or 0? Which details matter and which don't? Be concrete to this case — not generic.

Then output in this exact format:
RULE: <one declarative sentence, ≤ 25 words, describing the specific rule applied>
VERDICT: <1 or 0>
"""

AGGREGATION_SYSTEM = "You are a medical expert evaluating an AI model's answer to a clinical question."

AGGREGATION_USER = """I have run the Stage-1 binary judge (you, with the same prompt) on {n} specific cases and asked for the per-case decision rule. Below are the {n} rules you articulated, each paired with the ground truth label.

Consolidate these into a unified rule-set a smaller judge can follow to replicate your labels.

Requirements:
- 8-12 rules
- Each rule = one declarative sentence, ≤ 25 words
- Non-overlapping where possible
- Concrete, not generic (avoid "be accurate", "be thorough", etc.)
- Focus on edges: paraphrasing, partial answers, extra detail, hedging, wrong specifics
- Cover both directions (what MAKES correct, what MAKES incorrect)

Output format:

### Unified rule-set

- <rule 1>
- <rule 2>
...

=== CASE-LEVEL RULES ===

{case_block}
"""


def sample_cases(rng, n=N_CASES):
    """Stratified: 2-3 per target × 1-2 per label. Seed=42."""
    rows = [json.loads(line) for line in DEV_JSONL.open() if line.strip()]
    by_strat = {}
    for r in rows:
        k = (r["target"], r["binary_correct"])
        by_strat.setdefault(k, []).append(r)
    for v in by_strat.values():
        rng.shuffle(v)

    # 10 cases: try 1 per (target x label) = 8; add 2 more for variety
    picked = []
    strata = [(t, lbl) for t in TARGETS for lbl in [0, 1]]
    rng.shuffle(strata)
    for (t, lbl) in strata:
        if len(picked) >= n:
            break
        if by_strat.get((t, lbl)):
            picked.append(by_strat[(t, lbl)].pop(0))

    # Pad remainder from any stratum
    for (t, lbl) in strata:
        if len(picked) >= n:
            break
        if by_strat.get((t, lbl)):
            picked.append(by_strat[(t, lbl)].pop(0))

    return picked[:n]


def load_key():
    env = Path(PROJECT_ROOT / ".env").read_text().splitlines()
    for line in env:
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip()
    return os.environ.get("OPENAI_API_KEY")


def call_gpt4o(client, system, user, max_tokens=500, max_retries=5):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip(), resp.usage
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                raise


def parse_rule_verdict(response_text):
    rule, verdict = None, None
    for line in response_text.splitlines():
        line = line.strip()
        if line.upper().startswith("RULE:"):
            rule = line[5:].strip()
        if line.upper().startswith("VERDICT:"):
            v = line[8:].strip()
            if "1" in v and "0" not in v:
                verdict = 1
            elif "0" in v:
                verdict = 0
    return rule, verdict


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    cases = sample_cases(rng)
    print(f"Sampled {len(cases)} cases:")
    for c in cases:
        print(f"  {c['target']:30s} label={c['binary_correct']}  pid={c['patient_id']}")

    client = OpenAI(api_key=load_key())
    per_case = []
    total_pt, total_ct = 0, 0
    for i, c in enumerate(cases, 1):
        user = RULE_EXTRACTION_USER.format(
            note=c["question"] and c.get("note", "") or "",  # we'll need note; dev.jsonl may lack note column, check
            question=c["question"],
            ground_truth=c["ground_truth"],
            model_answer=c["model_answer"],
        )
        # dev.jsonl may not contain `note` column — rebuild from EHRNoteQA if missing
        if "note" not in c or not c.get("note"):
            # Reload note from EHRNoteQA_processed.jsonl on the fly
            import pandas as pd
            notes_file = PROJECT_ROOT / "output" / "EHRNoteQA_processed.jsonl"
            notes_df = pd.read_json(notes_file, lines=True)
            nrow = notes_df[notes_df["patient_id"].astype(str) == str(c["patient_id"])]
            note_text = ""
            if len(nrow) > 0:
                for j in [1, 2, 3]:
                    col = f"note_{j}"
                    if col in nrow.columns:
                        val = nrow.iloc[0].get(col)
                        if pd.notna(val) and str(val).strip() and str(val).lower() != "nan":
                            note_text += f"[Note {j}]\n{val}\n\n"
            c["note"] = note_text.strip()
            user = RULE_EXTRACTION_USER.format(
                note=c["note"], question=c["question"],
                ground_truth=c["ground_truth"], model_answer=c["model_answer"],
            )

        print(f"\n--- Case {i}/{len(cases)}  target={c['target']}  gold={c['binary_correct']} ---")
        text, usage = call_gpt4o(client, STAGE1_SYSTEM, user, max_tokens=400)
        rule, verdict = parse_rule_verdict(text)
        pt = usage.prompt_tokens
        ct = usage.completion_tokens
        total_pt += pt
        total_ct += ct
        match = verdict == c["binary_correct"]
        print(f"  RULE: {rule!r}")
        print(f"  VERDICT: {verdict}  (gold={c['binary_correct']}  match={match})  tokens={pt}+{ct}")
        per_case.append({
            "target": c["target"], "patient_id": c["patient_id"], "fold_id": c["fold_id"],
            "gold_label": c["binary_correct"], "gpt_verdict": verdict, "match_gold": match,
            "rule": rule, "raw_response": text,
            "prompt_tokens": pt, "completion_tokens": ct,
        })

    # Save per-case rules
    per_case_path = OUT_DIR / "per_case_rules.jsonl"
    with per_case_path.open("w") as f:
        for r in per_case:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\nSaved: {per_case_path}")

    # Aggregation call — only include cases where verdict matched gold
    valid = [r for r in per_case if r["match_gold"] and r["rule"]]
    print(f"\n{len(valid)}/{len(per_case)} cases had VERDICT matching gold and a parseable RULE")

    case_block_lines = []
    for i, r in enumerate(valid, 1):
        case_block_lines.append(f"Case {i} (target={r['target']}, gold={r['gold_label']}):\n  RULE: {r['rule']}")
    case_block = "\n\n".join(case_block_lines)
    agg_user = AGGREGATION_USER.format(n=len(valid), case_block=case_block)
    print("\n--- Aggregation call ---")
    agg_text, agg_usage = call_gpt4o(client, AGGREGATION_SYSTEM, agg_user, max_tokens=1000)
    total_pt += agg_usage.prompt_tokens
    total_ct += agg_usage.completion_tokens
    print(agg_text)

    unified_path = OUT_DIR / "unified_rules.md"
    unified_path.write_text(
        f"# Unified rule-set (GPT-4o derived from {len(valid)} verified dev cases)\n\n"
        f"Source: {per_case_path.name}\n"
        f"Seed: {SEED}\n"
        f"Extraction cost: {total_pt} prompt tokens + {total_ct} completion tokens "
        f"≈ ${(total_pt * 2.5 + total_ct * 10) / 1_000_000:.3f}\n\n"
        f"---\n\n{agg_text}\n"
    )
    print(f"\nSaved: {unified_path}")
    print(f"\nTotal cost: ${(total_pt * 2.5 + total_ct * 10) / 1_000_000:.3f}")


if __name__ == "__main__":
    main()
