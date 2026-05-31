"""Phase C Round 1: 4 candidate rule-sets evaluated on dev (300 items each).

Baselines:
  V0 (no rules, no ICL):                 66.7%  κ=0.334
  V4 (no rules, 4-shot ICL):             67.7%  κ=0.355
  Vrules (10 rules, no ICL):             73.3%  κ=0.464   ← current leader

Candidates (all zero-shot, different rule phrasings):
  C1 Strict        — emphasis on rejection conditions, "0 if ANY"
  C2 Balanced      — each rule expresses both directions (1 if X; 0 if Y)
  C3 Compact       — 5 tight rules
  C4 GPT-polish    — GPT-4o's rewrite targeting permissive bias (Vrules had 48% permissive-error rate)
"""
from __future__ import annotations
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[4]
DEV_JSONL = ROOT / "output" / "ichl" / "mlx_judge" / "splits" / "dev.jsonl"
OUT_DIR = ROOT / "output" / "ichl" / "mlx_judge" / "phase_c"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LM_STUDIO_URL = "http://192.168.68.107:1234/v1"
LM_STUDIO_MODEL = "qwen3-30b-a3b-instruct-2507-mlx"
SYSTEM = "You are a medical expert evaluating an AI model's answer to a clinical question."

# ------------------------------------------------------------------
# 4 candidate rule-sets
# ------------------------------------------------------------------
C1_STRICT = [
    "Output 0 if the model's answer contradicts the ground truth on any specific fact (medication name, dose, diagnosis, timing, procedure).",
    "Output 0 if the answer omits a fact that the question specifically asks about.",
    "Output 0 if the answer adds details not supported by the discharge summary.",
    "Output 0 if the answer confuses the sequence or timing of medical events.",
    "Output 0 if the answer misinterprets lab values or test results.",
    "Output 0 if the answer addresses a different aspect than the question asks.",
    "Output 0 if the answer hedges with multiple uncommitted options.",
    "Output 1 ONLY if the answer is consistent with the ground truth on every specific claim.",
    "Output 1 ONLY if the answer directly addresses the exact question asked.",
    "Output 1 if paraphrasing differs but all ground-truth facts are preserved.",
]

C2_BALANCED = [
    "Output 1 if medication names, doses, diagnoses, and procedures MATCH the ground truth; output 0 if ANY is wrong.",
    "Output 1 if the sequence of events matches the ground truth; output 0 if timing or context is confused.",
    "Output 1 if extra detail is consistent with the discharge summary; output 0 if extra detail contradicts or fabricates.",
    "Output 1 if the answer directly addresses the question asked; output 0 if it addresses a different aspect.",
    "Output 1 if the answer is committal; output 0 if it hedges with several uncommitted options.",
    "Output 1 if paraphrasing preserves all ground-truth facts; output 0 if key facts drift or are missing.",
    "Output 1 if lab values and interpretations match the ground truth; output 0 if any is misread.",
    "Output 1 if the cause of symptoms matches the ground truth; output 0 if a different cause is implied.",
]

C3_COMPACT = [
    "Output 0 if any specific fact (medication, dose, diagnosis, timing, procedure, lab value) contradicts the ground truth.",
    "Output 0 if the answer omits a fact that the question specifically asks about.",
    "Output 0 if the answer addresses a different aspect than the question asked.",
    "Output 0 if the answer hedges with multiple uncommitted options.",
    "Output 1 ONLY if all specific claims align with the ground truth AND the answer directly addresses the question.",
]

# C4 = GPT-4o polish (loaded from file; generated in prior step)
def load_c4():
    md = (ROOT / "output" / "ichl" / "mlx_judge" / "rules" / "phase_c_rules_C4.md").read_text()
    # Extract bulleted rules from the "### Unified rule-set" section
    lines = md.splitlines()
    rules = []
    in_block = False
    for line in lines:
        if line.startswith("### Unified rule-set"):
            in_block = True
            continue
        if in_block:
            if line.strip().startswith("-"):
                rules.append(line.strip()[1:].strip())
            elif line.startswith("###") or line.startswith("---"):
                break
    return rules

C4_GPT_POLISH = load_c4()

CANDIDATES = {
    "C1_strict": C1_STRICT,
    "C2_balanced": C2_BALANCED,
    "C3_compact": C3_COMPACT,
    "C4_gpt_polish": C4_GPT_POLISH,
}


PROMPT_TEMPLATE = """When judging, apply the following rules:

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
    m = re.search(r"[01]", t)
    return int(m.group(0)) if m else None


def build_note_lookup():
    import pandas as pd
    notes_file = ROOT / "output" / "EHRNoteQA_processed.jsonl"
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


def call(client, system, user, max_tokens=16, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=LM_STUDIO_MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0, max_tokens=max_tokens,
            )
            return resp.choices[0].message.content, resp.usage
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 + attempt * 2)
            else:
                print(f"  ERROR: {e}")
                return None, None


def judge_one(args):
    client, rules_block, row, note = args
    t0 = time.monotonic()
    user = PROMPT_TEMPLATE.format(
        rules_block=rules_block,
        note=note, question=row["question"],
        ground_truth=row["ground_truth"], model_answer=row["model_answer"],
    )
    text, usage = call(client, SYSTEM, user, max_tokens=16)
    lat = time.monotonic() - t0
    label = parse_01(text) if text else None
    pt = usage.prompt_tokens if usage else 0
    ct = usage.completion_tokens if usage else 0
    return {
        "target": row["target"], "patient_id": row["patient_id"], "fold_id": row["fold_id"],
        "gold_label": int(row["binary_correct"]), "mlx_label": label, "raw": text,
        "latency_s": round(lat, 2), "prompt_tokens": pt, "completion_tokens": ct,
    }


def run_candidate(name, rules, client, dev, notes):
    rules_block = "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1))
    tasks = [(client, rules_block, row, notes.get(str(row["patient_id"]), "")) for row in dev]
    results = []
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as ex:
        for i, r in enumerate(ex.map(judge_one, tasks), 1):
            results.append(r)
            if i % 50 == 0:
                dt = time.monotonic() - t0
                print(f"  [{name}] {i}/{len(dev)}  elapsed={dt:.0f}s  eta={dt*(len(dev)-i)/i:.0f}s")
    dt = time.monotonic() - t0
    print(f"  [{name}] DONE in {dt:.0f}s")
    return results, dt


def metrics(results, name):
    from sklearn.metrics import cohen_kappa_score
    from collections import defaultdict
    n = len(results)
    correct = sum(1 for r in results if r["mlx_label"] == r["gold_label"])
    none_cnt = sum(1 for r in results if r["mlx_label"] is None)
    parsed = [(r["gold_label"], r["mlx_label"]) for r in results if r["mlx_label"] is not None]
    k = cohen_kappa_score([p[0] for p in parsed], [p[1] for p in parsed]) if parsed else None
    conf = defaultdict(int)
    for r in results:
        conf[(r["gold_label"], r["mlx_label"])] += 1
    per_tgt = defaultdict(lambda: {"n": 0, "ok": 0})
    for r in results:
        per_tgt[r["target"]]["n"] += 1
        if r["mlx_label"] == r["gold_label"]: per_tgt[r["target"]]["ok"] += 1
    return {
        "name": name, "n": n, "agreement": correct / n, "kappa": k, "none": none_cnt,
        "conf_00": conf.get((0, 0), 0), "conf_01": conf.get((0, 1), 0),
        "conf_10": conf.get((1, 0), 0), "conf_11": conf.get((1, 1), 0),
        "per_target": {t: s["ok"] / s["n"] for t, s in per_tgt.items()},
    }


def main():
    print("Loading dev + notes…")
    dev = [json.loads(l) for l in DEV_JSONL.open() if l.strip()]
    notes = build_note_lookup()
    print(f"  dev={len(dev)}  notes={len(notes)}")

    client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
    all_summary = {}

    for name, rules in CANDIDATES.items():
        print(f"\n=== Running candidate: {name}  ({len(rules)} rules) ===")
        results, wall = run_candidate(name, rules, client, dev, notes)
        out_path = OUT_DIR / f"{name}_dev.jsonl"
        with out_path.open("w") as f:
            for r in results:
                f.write(json.dumps(r, default=str) + "\n")
        m = metrics(results, name)
        m["wall_s"] = round(wall, 1)
        m["n_rules"] = len(rules)
        all_summary[name] = m
        print(f"  agreement={m['agreement']*100:.1f}%  κ={m['kappa']:.3f}  "
              f"conf=(0,0)={m['conf_00']} (0,1)={m['conf_01']} (1,0)={m['conf_10']} (1,1)={m['conf_11']}  "
              f"None={m['none']}")

    # Summary table
    summary_path = OUT_DIR / "round1_summary.json"
    summary_path.write_text(json.dumps(all_summary, indent=2, default=str))
    print(f"\nSaved summary: {summary_path}")

    print("\n=== Round 1 Summary (vs baselines V0=66.7% Vrules=73.3%) ===")
    print(f"{'candidate':18s}  {'rules':>5}  {'agree':>6}  {'κ':>6}  {'(0,0)':>5}  {'(0,1)':>5}  {'(1,0)':>5}  {'(1,1)':>5}  {'wall_s':>6}")
    for name, m in all_summary.items():
        print(f"{name:18s}  {m['n_rules']:>5}  {m['agreement']*100:>5.1f}%  {m['kappa']:>6.3f}  "
              f"{m['conf_00']:>5}  {m['conf_01']:>5}  {m['conf_10']:>5}  {m['conf_11']:>5}  {m['wall_s']:>6.0f}")


if __name__ == "__main__":
    main()
