"""Phase C Round 2: 5 refinements of C3 (Round 1 winner, 78.0% / κ=0.559).

Candidates:
  C3a Minimal           — 3 rules (drop hedging + addresses-different)
  C3b Expanded          — 6 rules (add explicit "extra detail" edge rule)
  C3c Reordered         — acceptance condition first, rejections after
  C3d SystemInjection   — rules moved into system message instead of user
  C3e GPTPolish         — 7-rule rewrite from GPT-4o targeting C3's 66 dev disagreements
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
SYSTEM_DEFAULT = "You are a medical expert evaluating an AI model's answer to a clinical question."

# ------------------------------------------------------------------
# Candidates
# ------------------------------------------------------------------
C3a_MINIMAL = [
    "Output 0 if any specific fact (medication, dose, diagnosis, timing, procedure, lab value) contradicts the ground truth.",
    "Output 0 if the answer omits a fact that the question specifically asks about.",
    "Output 1 ONLY if all specific claims align with the ground truth AND the answer directly addresses the question.",
]

C3b_EXPANDED = [
    "Output 0 if any specific fact (medication, dose, diagnosis, timing, procedure, lab value) contradicts the ground truth.",
    "Output 0 if the answer omits a fact that the question specifically asks about.",
    "Output 0 if the answer addresses a different aspect than the question asked.",
    "Output 0 if the answer hedges with multiple uncommitted options.",
    "Output 0 if the answer adds details not supported by the discharge summary, even if the ground-truth fact is also mentioned.",
    "Output 1 ONLY if all specific claims align with the ground truth AND the answer directly addresses the question.",
]

# Same 5 rules as C3 but reorder: acceptance first, rejections after
C3c_REORDERED = [
    "Output 1 ONLY if all specific claims align with the ground truth AND the answer directly addresses the question.",
    "Output 0 if any specific fact (medication, dose, diagnosis, timing, procedure, lab value) contradicts the ground truth.",
    "Output 0 if the answer omits a fact that the question specifically asks about.",
    "Output 0 if the answer addresses a different aspect than the question asked.",
    "Output 0 if the answer hedges with multiple uncommitted options.",
]

# C3d: rules in SYSTEM message
C3d_SYS_RULES = [
    "Output 0 if any specific fact (medication, dose, diagnosis, timing, procedure, lab value) contradicts the ground truth.",
    "Output 0 if the answer omits a fact that the question specifically asks about.",
    "Output 0 if the answer addresses a different aspect than the question asked.",
    "Output 0 if the answer hedges with multiple uncommitted options.",
    "Output 1 ONLY if all specific claims align with the ground truth AND the answer directly addresses the question.",
]

C3e_GPT_POLISH = [
    "Output 0 if any specific fact (medication, dose, diagnosis, timing, procedure, lab value) contradicts the ground truth.",
    "Output 0 if the answer omits a fact that the question specifically asks about.",
    "Output 0 if the answer addresses a different aspect than the question asked.",
    "Output 0 if the answer hedges with multiple uncommitted options.",
    "Output 1 only if all specific claims align with the ground truth and the answer directly addresses the question.",
    "Output 0 if the answer includes additional, incorrect information not present in the ground truth.",
    "Output 1 if the answer provides correct additional context that does not contradict the ground truth.",
]

CANDIDATES = {
    "C3a_minimal": ("user", C3a_MINIMAL),
    "C3b_expanded": ("user", C3b_EXPANDED),
    "C3c_reordered": ("user", C3c_REORDERED),
    "C3d_sysrules": ("system", C3d_SYS_RULES),
    "C3e_gptpolish": ("user", C3e_GPT_POLISH),
}

# ------------------------------------------------------------------
# Prompt assembly per location
# ------------------------------------------------------------------
USER_TEMPLATE_WITH_RULES = """When judging, apply the following rules:

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

USER_TEMPLATE_NO_RULES = """DISCHARGE SUMMARY:
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

SYS_TEMPLATE_WITH_RULES = """You are a medical expert evaluating an AI model's answer to a clinical question. When judging, apply these rules:

{rules_block}"""


def parse_01(text):
    t = (text or "").strip()
    m = re.search(r"[01]", t)
    return int(m.group(0)) if m else None


def build_note_lookup():
    import pandas as pd
    df = pd.read_json(ROOT / "output" / "EHRNoteQA_processed.jsonl", lines=True)
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
    client, system, user_template, rules_block, row, note = args
    t0 = time.monotonic()
    if rules_block and "{rules_block}" in user_template:
        user = user_template.format(rules_block=rules_block, note=note,
                                     question=row["question"], ground_truth=row["ground_truth"],
                                     model_answer=row["model_answer"])
    else:
        user = user_template.format(note=note, question=row["question"],
                                     ground_truth=row["ground_truth"], model_answer=row["model_answer"])
    text, usage = call(client, system, user, max_tokens=16)
    lat = time.monotonic() - t0
    label = parse_01(text) if text else None
    pt = usage.prompt_tokens if usage else 0
    ct = usage.completion_tokens if usage else 0
    return {
        "target": row["target"], "patient_id": row["patient_id"], "fold_id": row["fold_id"],
        "gold_label": int(row["binary_correct"]), "mlx_label": label, "raw": text,
        "latency_s": round(lat, 2), "prompt_tokens": pt, "completion_tokens": ct,
    }


def run_candidate(name, location, rules, client, dev, notes):
    rules_block = "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1))
    if location == "system":
        system = SYS_TEMPLATE_WITH_RULES.format(rules_block=rules_block)
        user_template = USER_TEMPLATE_NO_RULES
        rb = None  # consumed in system
    else:
        system = SYSTEM_DEFAULT
        user_template = USER_TEMPLATE_WITH_RULES
        rb = rules_block
    tasks = [(client, system, user_template, rb, row, notes.get(str(row["patient_id"]), "")) for row in dev]
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

    for name, (location, rules) in CANDIDATES.items():
        print(f"\n=== {name}  (location={location}, n_rules={len(rules)}) ===")
        results, wall = run_candidate(name, location, rules, client, dev, notes)
        out_path = OUT_DIR / f"{name}_dev.jsonl"
        with out_path.open("w") as f:
            for r in results:
                f.write(json.dumps(r, default=str) + "\n")
        m = metrics(results, name)
        m["wall_s"] = round(wall, 1)
        m["n_rules"] = len(rules)
        m["location"] = location
        all_summary[name] = m
        print(f"  agreement={m['agreement']*100:.1f}%  κ={m['kappa']:.3f}  "
              f"conf=(0,0)={m['conf_00']} (0,1)={m['conf_01']} (1,0)={m['conf_10']} (1,1)={m['conf_11']}")

    summary_path = OUT_DIR / "round2_summary.json"
    summary_path.write_text(json.dumps(all_summary, indent=2, default=str))

    print("\n=== Round 2 Summary (vs C3=78.0%) ===")
    print(f"{'candidate':18s}  {'loc':8s}  {'rules':>5}  {'agree':>6}  {'κ':>6}  {'(0,0)':>5}  {'(0,1)':>5}  {'(1,0)':>5}  {'(1,1)':>5}")
    for name, m in all_summary.items():
        print(f"{name:18s}  {m['location']:8s}  {m['n_rules']:>5}  "
              f"{m['agreement']*100:>5.1f}%  {m['kappa']:>6.3f}  "
              f"{m['conf_00']:>5}  {m['conf_01']:>5}  {m['conf_10']:>5}  {m['conf_11']:>5}")


if __name__ == "__main__":
    main()
