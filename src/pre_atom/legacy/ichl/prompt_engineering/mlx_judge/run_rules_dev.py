"""Phase B.2 dev run: Qwen3-30B judge with GPT-4o-derived rules in the prompt.

Compares against V0 zero-shot baseline (66.7% agreement, κ=0.334).
"""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEV_JSONL = PROJECT_ROOT / "output" / "ichl" / "mlx_judge" / "splits" / "dev.jsonl"
RULES_MD = PROJECT_ROOT / "output" / "ichl" / "mlx_judge" / "rules" / "unified_rules.md"
OUT_DIR = PROJECT_ROOT / "output" / "ichl" / "mlx_judge" / "phase_b"
OUT_FILE = OUT_DIR / "vrules_dev.jsonl"

LM_STUDIO_URL = "http://192.168.68.107:1234/v1"
LM_STUDIO_MODEL = "qwen3-30b-a3b-instruct-2507-mlx"


def load_rules():
    md = RULES_MD.read_text()
    # Extract bulleted rules between "### Unified rule-set" and next section
    lines = md.splitlines()
    rules = []
    in_block = False
    for line in lines:
        if line.startswith("### Unified rule-set"):
            in_block = True
            continue
        if in_block:
            if line.startswith("###") or line.startswith("---") or line.startswith("==="):
                break
            if line.strip().startswith("-"):
                rules.append(line.strip()[1:].strip())
    return rules


SYSTEM = "You are a medical expert evaluating an AI model's answer to a clinical question."

PROMPT_TEMPLATE = """When judging, apply the following rules (derived from previous expert-labeled cases):

{rules_block}

DISCHARGE SUMMARY:
{note}

QUESTION:
{question}

CORRECT ANSWER (Ground Truth):
{ground_truth}

MODEL'S ANSWER:
{model_answer}

Task: Evaluate if the model's answer is correct compared to the ground truth.

Respond with ONLY a single digit:
1 = Correct
0 = Incorrect"""


def parse_01(text):
    t = (text or "").strip()
    # Find first occurrence of 0 or 1
    m = re.search(r"[01]", t)
    if m:
        return int(m.group(0))
    return None


def build_note_lookup():
    import pandas as pd
    notes_file = PROJECT_ROOT / "output" / "EHRNoteQA_processed.jsonl"
    df = pd.read_json(notes_file, lines=True)
    lookup = {}
    for _, r in df.iterrows():
        pid = str(r.get("patient_id", ""))
        parts = []
        for i in [1, 2, 3]:
            col = f"note_{i}"
            if col in r.index:
                v = r.get(col)
                if pd.notna(v) and str(v).strip() and str(v).lower() != "nan":
                    parts.append(f"[Note {i}]\n{v}")
        lookup[pid] = "\n\n".join(parts)
    return lookup


def call_lm_studio(client, system, user, max_tokens=16, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=LM_STUDIO_MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content, resp.usage
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 + attempt * 2)
            else:
                print(f"  ERROR: {e}")
                return None, None


def judge_one(client, rules_block, note, question, gt, answer):
    t0 = time.monotonic()
    user = PROMPT_TEMPLATE.format(
        rules_block=rules_block,
        note=note, question=question, ground_truth=gt, model_answer=answer,
    )
    text, usage = call_lm_studio(client, SYSTEM, user, max_tokens=16)
    lat = time.monotonic() - t0
    label = parse_01(text) if text else None
    pt = usage.prompt_tokens if usage else 0
    ct = usage.completion_tokens if usage else 0
    return label, text, lat, pt, ct


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading rules…")
    rules = load_rules()
    print(f"  loaded {len(rules)} rules")
    rules_block = "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1))

    print("Loading dev set…")
    dev = [json.loads(line) for line in DEV_JSONL.open() if line.strip()]
    print(f"  dev={len(dev)}")

    print("Building note lookup…")
    notes = build_note_lookup()
    print(f"  notes loaded={len(notes)}")

    client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

    def task(row):
        note = notes.get(str(row["patient_id"]), "")
        label, text, lat, pt, ct = judge_one(
            client, rules_block, note, row["question"], row["ground_truth"], row["model_answer"],
        )
        out = {
            "target": row["target"], "patient_id": row["patient_id"], "fold_id": row["fold_id"],
            "gold_label": int(row["binary_correct"]), "mlx_label": label, "raw": text,
            "latency_s": round(lat, 2), "prompt_tokens": pt, "completion_tokens": ct,
        }
        return out

    t_start = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        for i, r in enumerate(ex.map(task, dev), 1):
            results.append(r)
            if i % 25 == 0:
                elapsed = time.monotonic() - t_start
                eta = elapsed * (len(dev) - i) / i
                print(f"  {i}/{len(dev)}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    elapsed = time.monotonic() - t_start
    print(f"DONE in {elapsed:.0f}s")

    with OUT_FILE.open("w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"Saved: {OUT_FILE}")

    # Metrics
    n = len(results)
    correct = sum(1 for r in results if r["mlx_label"] == r["gold_label"])
    none_cnt = sum(1 for r in results if r["mlx_label"] is None)
    print(f"\n=== Vrules result ===")
    print(f"  n={n}  agreement={correct}/{n} = {100*correct/n:.1f}%  None={none_cnt}")
    try:
        from sklearn.metrics import cohen_kappa_score
        parsed = [(r["gold_label"], r["mlx_label"]) for r in results if r["mlx_label"] is not None]
        if parsed:
            k = cohen_kappa_score([p[0] for p in parsed], [p[1] for p in parsed])
            print(f"  Cohen's κ = {k:.3f}")
    except Exception as e:
        print(f"  κ failed: {e}")

    from collections import defaultdict
    conf = defaultdict(int)
    per_tgt = defaultdict(lambda: {"n": 0, "ok": 0})
    for r in results:
        conf[(r["gold_label"], r["mlx_label"])] += 1
        per_tgt[r["target"]]["n"] += 1
        if r["mlx_label"] == r["gold_label"]:
            per_tgt[r["target"]]["ok"] += 1
    print(f"  confusion (gold→mlx): (0,0)={conf.get((0,0),0)}  (0,1)={conf.get((0,1),0)}  (1,0)={conf.get((1,0),0)}  (1,1)={conf.get((1,1),0)}")
    print("  per-target:")
    for t, s in sorted(per_tgt.items()):
        print(f"    {t:30s}  {s['ok']/s['n']*100:.1f}%  ({s['ok']}/{s['n']})")


if __name__ == "__main__":
    main()
