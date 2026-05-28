"""Qwen3 Judge — model & thinking-mode comparison on full dev with C3e rules.

Three configs, run sequentially on LM Studio (port 1234):
  1. Qwen3-32B  /no_think              max_tokens=32     (~40 min)
  2. Qwen3-32B  thinking enabled       max_tokens=4096   (~5 h)
  3. Qwen3.5-27B thinking enabled       max_tokens=4096   (~6 h)

Parser: digit MUST come from `content`. If `content` empty (thinking truncated
before final answer), mark verdict as None — DO NOT parse from
`reasoning_content` (would be unreliable).

Resume safety: if output JSONL has N rows for a config, skip first N items.
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
SYSTEM = "You are a medical expert evaluating an AI model's answer to a clinical question."

C3E_RULES = [
    "Output 0 if any specific fact (medication, dose, diagnosis, timing, procedure, lab value) contradicts the ground truth.",
    "Output 0 if the answer omits a fact that the question specifically asks about.",
    "Output 0 if the answer addresses a different aspect than the question asked.",
    "Output 0 if the answer hedges with multiple uncommitted options.",
    "Output 1 only if all specific claims align with the ground truth and the answer directly addresses the question.",
    "Output 0 if the answer includes additional, incorrect information not present in the ground truth.",
    "Output 1 if the answer provides correct additional context that does not contradict the ground truth.",
]

CONFIGS = [
    # (run_name, model_id, thinking_enabled, max_tokens, est_per_call_seconds)
    ("Q32B_nothink",   "qwen/qwen3-32b",  False, 32,   8),
    ("Q32B_think",     "qwen/qwen3-32b",  True,  4096, 60),
    ("Q35_27B_think",  "qwen3.5-27b",     True,  4096, 70),
]


def build_user_prompt(item, note, thinking_enabled):
    rules_block = "\n".join(f"{i}. {r}" for i, r in enumerate(C3E_RULES, 1))
    prefix = "" if thinking_enabled else "/no_think\n"
    return f"""{prefix}When judging, apply the following rules:

{rules_block}

DISCHARGE SUMMARY:
{note}

QUESTION:
{item["question"]}

CORRECT ANSWER (Ground Truth):
{item["ground_truth"]}

MODEL'S ANSWER:
{item["model_answer"]}

Task: Evaluate if the model's answer is correct compared to the ground truth.

Respond with ONLY a single digit:
1 = Correct
0 = Incorrect"""


def parse_content(content):
    """Strict parse: digit from content. Returns None if content empty/missing."""
    if not content:
        return None
    s = content.strip()
    if not s:
        return None
    m = re.search(r"[01]", s)
    if m:
        return int(m.group(0))
    return None


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


def call_lm(client, model, system, user, max_tokens, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return resp
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 + attempt * 3)
            else:
                print(f"    ERROR: {e}")
                return None
    return None


def judge_item(args):
    client, model, thinking, max_tokens, item, note = args
    user = build_user_prompt(item, note, thinking)
    t0 = time.monotonic()
    resp = call_lm(client, model, SYSTEM, user, max_tokens)
    lat = time.monotonic() - t0
    if resp is None:
        return {
            "target": item["target"], "patient_id": item["patient_id"], "fold_id": item["fold_id"],
            "gold_label": int(item["binary_correct"]), "mlx_label": None,
            "content": None, "reasoning_token_count": None, "completion_tokens": None,
            "prompt_tokens": None, "latency_s": round(lat, 2), "finish_reason": "ERROR",
            "truncated": True,
        }
    msg = resp.choices[0].message
    content = getattr(msg, "content", None) or ""
    reasoning_tokens = None
    if resp.usage and getattr(resp.usage, "completion_tokens_details", None):
        reasoning_tokens = getattr(resp.usage.completion_tokens_details, "reasoning_tokens", None)
    finish = resp.choices[0].finish_reason
    truncated = (finish == "length") or (thinking and not content.strip())
    label = parse_content(content)
    return {
        "target": item["target"], "patient_id": item["patient_id"], "fold_id": item["fold_id"],
        "gold_label": int(item["binary_correct"]), "mlx_label": label,
        "content": content, "reasoning_token_count": reasoning_tokens,
        "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
        "latency_s": round(lat, 2), "finish_reason": finish, "truncated": truncated,
    }


def run_config(name, model, thinking, max_tokens, dev, notes, client):
    out_path = OUT_DIR / f"{name}_dev.jsonl"
    # Resume: read existing and skip
    done_keys = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                r = json.loads(line)
                done_keys.add((r["target"], r["patient_id"], r["fold_id"]))
            except Exception:
                continue
    pending = [r for r in dev if (r["target"], r["patient_id"], r["fold_id"]) not in done_keys]
    if not pending:
        print(f"  [{name}] already complete ({len(dev)}/{len(dev)})")
    else:
        print(f"  [{name}] starting: {len(pending)}/{len(dev)} pending  (model={model} think={thinking} max={max_tokens})")

    tasks = [(client, model, thinking, max_tokens, item, notes.get(str(item["patient_id"]), "")) for item in pending]

    t0 = time.monotonic()
    workers = 1 if thinking else 2  # avoid contention on heavy thinking
    with out_path.open("a") as fout, ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(judge_item, tasks), 1):
            fout.write(json.dumps(r, default=str) + "\n")
            fout.flush()
            if i % 10 == 0:
                dt = time.monotonic() - t0
                eta = dt * (len(pending) - i) / i
                print(f"  [{name}] {i}/{len(pending)}  elapsed={dt:.0f}s  eta={eta:.0f}s  "
                      f"trunc={sum(1 for _ in [None] if r['truncated'])}")
    print(f"  [{name}] DONE in {time.monotonic()-t0:.0f}s")


def metrics(out_path, name):
    from sklearn.metrics import cohen_kappa_score
    from collections import defaultdict
    rows = [json.loads(l) for l in out_path.open() if l.strip()]
    n = len(rows)
    correct = sum(1 for r in rows if r["mlx_label"] == r["gold_label"])
    none_cnt = sum(1 for r in rows if r["mlx_label"] is None)
    truncated = sum(1 for r in rows if r.get("truncated"))
    parsed = [(r["gold_label"], r["mlx_label"]) for r in rows if r["mlx_label"] is not None]
    k = cohen_kappa_score([p[0] for p in parsed], [p[1] for p in parsed]) if parsed else None
    conf = defaultdict(int)
    for r in rows:
        conf[(r["gold_label"], r["mlx_label"])] += 1
    per_tgt = defaultdict(lambda: {"n": 0, "ok": 0})
    for r in rows:
        per_tgt[r["target"]]["n"] += 1
        if r["mlx_label"] == r["gold_label"]: per_tgt[r["target"]]["ok"] += 1
    mean_lat = sum(r.get("latency_s", 0) for r in rows) / n if n else 0
    return {
        "name": name, "n": n,
        "agreement_overall": correct / n,
        "agreement_excluding_none": correct / (n - none_cnt) if (n - none_cnt) > 0 else None,
        "kappa": k,
        "n_none": none_cnt,
        "n_truncated": truncated,
        "conf_00": conf.get((0, 0), 0), "conf_01": conf.get((0, 1), 0),
        "conf_10": conf.get((1, 0), 0), "conf_11": conf.get((1, 1), 0),
        "conf_None_gold0": conf.get((0, None), 0), "conf_None_gold1": conf.get((1, None), 0),
        "per_target": {t: s["ok"] / s["n"] for t, s in per_tgt.items()},
        "mean_latency_s": round(mean_lat, 2),
    }


def main():
    print("Loading dev + notes…")
    dev = [json.loads(l) for l in DEV_JSONL.open() if l.strip()]
    notes = build_note_lookup()
    print(f"  dev={len(dev)}  notes={len(notes)}")

    client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

    summary = {}
    for name, model, thinking, max_tokens, _eta in CONFIGS:
        print(f"\n{'='*70}\n=== {name}  ({model}, think={thinking}, max_tokens={max_tokens}) ===\n{'='*70}")
        run_config(name, model, thinking, max_tokens, dev, notes, client)
        out_path = OUT_DIR / f"{name}_dev.jsonl"
        m = metrics(out_path, name)
        summary[name] = m
        print(f"  agreement={m['agreement_overall']*100:.1f}% (excl-None: "
              f"{m['agreement_excluding_none']*100 if m['agreement_excluding_none'] else float('nan'):.1f}%)  "
              f"κ={m['kappa']:.3f}  None={m['n_none']}  trunc={m['n_truncated']}  "
              f"conf=(0,0)={m['conf_00']} (0,1)={m['conf_01']} (1,0)={m['conf_10']} (1,1)={m['conf_11']}  "
              f"mean_lat={m['mean_latency_s']}s")

    summary_path = OUT_DIR / "model_compare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved: {summary_path}")
    print("\n=== FINAL TABLE ===")
    print(f"{'config':18s}  {'n':>4} {'agree':>6} {'κ':>6} {'None':>4} {'trunc':>5}  {'mean_lat':>9}")
    for name, m in summary.items():
        print(f"{name:18s}  {m['n']:>4} {m['agreement_overall']*100:>5.1f}% {m['kappa']:>6.3f} "
              f"{m['n_none']:>4} {m['n_truncated']:>5}  {m['mean_latency_s']:>8.1f}s")


if __name__ == "__main__":
    main()
