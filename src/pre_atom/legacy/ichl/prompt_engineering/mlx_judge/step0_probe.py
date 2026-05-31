"""Step 0 token-budget probe for the Qwen3 Judge.

Per user spec (2026-04-25):
  Phase A — 5 unique-patient items with the LONGEST combined notes (worst-case probe).
  Phase B — 30 unique-patient items stratified per note-count tier (10 each: 1, 2, 3 notes).

Probe runs C3e rules + max_tokens=16384 (very generous; should never truncate).
Captures per-call: completion_tokens, reasoning_tokens, finish_reason, latency.
Output JSONL per (config, item).

Configs probed (skipping Q32B_nothink — already verified 0 trunc at max_tokens=32):
  - qwen/qwen3-32b      thinking ON
  - qwen3.5-27b         thinking ON
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from ichl.prompt_engineering.correction.truncation_detector import detect_truncation

ROOT = Path(__file__).resolve().parents[4]
DEV_JSONL = ROOT / "output" / "ichl" / "mlx_judge" / "splits" / "dev.jsonl"
OUT_DIR = ROOT / "output" / "ichl" / "mlx_judge" / "step0_probe"
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
    ("Q32B_think", "qwen/qwen3-32b"),
    ("Q35_27B_think", "qwen3.5-27b"),
]
PROBE_MAX_TOKENS = 16384


def build_user_prompt(item, note):
    rules_block = "\n".join(f"{i}. {r}" for i, r in enumerate(C3E_RULES, 1))
    return f"""When judging, apply the following rules:

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


def build_note_lookup_with_count():
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
        text = "\n\n".join(parts)
        lookup[pid] = {"text": text, "n_notes": len(parts), "total_chars": sum(len(p) for p in parts)}
    return lookup


def select_probe_items(dev, lookup):
    """Phase A: 5 unique patients with longest combined notes.
    Phase B: 10 unique patients per note-count tier (1, 2, 3)."""
    seen = set()
    by_pid = []
    for d in dev:
        pid = str(d["patient_id"])
        if pid in seen: continue
        seen.add(pid)
        info = lookup.get(pid, {})
        by_pid.append({**d, **info})

    # Phase A: top-5 by total_chars
    sorted_long = sorted(by_pid, key=lambda x: -(x.get("total_chars") or 0))
    phase_a = sorted_long[:5]

    # Phase B: stratify by n_notes
    phase_a_pids = {it["patient_id"] for it in phase_a}
    rest = [it for it in by_pid if it["patient_id"] not in phase_a_pids]
    by_count = {1: [], 2: [], 3: []}
    for it in rest:
        n = it.get("n_notes", 0)
        if n in by_count:
            by_count[n].append(it)
    # Within each tier, deterministic mid-spread sample (every k-th)
    phase_b = []
    for nc in [1, 2, 3]:
        pool = sorted(by_count[nc], key=lambda x: -(x.get("total_chars") or 0))
        if len(pool) <= 10:
            phase_b.extend(pool)
        else:
            step = len(pool) // 10
            picks = [pool[i*step] for i in range(10)]
            phase_b.extend(picks)
    return phase_a, phase_b


def call_lm(client, model, system, user, max_tokens, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0, max_tokens=max_tokens,
            )
            return resp
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 + attempt * 3)
            else:
                print(f"  ERROR: {e}")
                return None
    return None


def probe_one(args):
    client, model, item = args
    user = build_user_prompt(item, item.get("text", ""))
    t0 = time.monotonic()
    resp = call_lm(client, model, SYSTEM, user, PROBE_MAX_TOKENS)
    lat = time.monotonic() - t0
    if resp is None:
        return {**{k: item[k] for k in ["target", "patient_id", "fold_id", "n_notes", "total_chars"]},
                "model": model, "completion_tokens": None, "reasoning_tokens": None,
                "prompt_tokens": None, "finish_reason": "ERROR", "latency_s": round(lat, 2)}
    msg = resp.choices[0].message
    content = getattr(msg, "content", None) or ""
    reasoning_content = getattr(msg, "reasoning_content", None) or ""
    fin = resp.choices[0].finish_reason
    usage = resp.usage
    reason_tok = None
    if usage and getattr(usage, "completion_tokens_details", None):
        reason_tok = getattr(usage.completion_tokens_details, "reasoning_tokens", None)
    # Apply truncation detector (per Claude: Principle: Truncation Detection on Every LLM Output)
    raw_for_detector = (reasoning_content + "\n" + content) if reasoning_content else content
    report = detect_truncation(
        raw_response=raw_for_detector,
        text_clean=content,
        finish_reason=fin,
        usage={"completion_tokens": usage.completion_tokens if usage else None,
               "prompt_tokens": usage.prompt_tokens if usage else None},
        max_tokens=PROBE_MAX_TOKENS,
        target=model,
        sub_variant="think",
    )
    return {
        "model": model, "patient_id": item["patient_id"], "fold_id": item["fold_id"],
        "target": item["target"], "n_notes": item.get("n_notes"), "total_chars": item.get("total_chars"),
        "gold_label": int(item["binary_correct"]),
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "reasoning_tokens": reason_tok,
        "content": content,
        "reasoning_content_tail": reasoning_content[-300:] if reasoning_content else "",
        "finish_reason": fin, "latency_s": round(lat, 2),
        "truncation_report": report.as_dict(),
    }


def run_probe(name, model, items, client):
    out_path = OUT_DIR / f"{name}_probe.jsonl"
    print(f"\n=== {name}  ({model})  n={len(items)}  max_tokens={PROBE_MAX_TOKENS} ===")
    tasks = [(client, model, it) for it in items]
    t0 = time.monotonic()
    with out_path.open("w") as f, ThreadPoolExecutor(max_workers=2) as ex:
        for i, r in enumerate(ex.map(probe_one, tasks), 1):
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            elapsed = time.monotonic() - t0
            print(f"  [{name}] {i}/{len(items)}  pid={r['patient_id']} n_notes={r['n_notes']} "
                  f"chars={r.get('total_chars')} prompt_tok={r.get('prompt_tokens')} "
                  f"comp_tok={r.get('completion_tokens')} reason_tok={r.get('reasoning_tokens')} "
                  f"finish={r.get('finish_reason')} lat={r['latency_s']:.0f}s  "
                  f"elapsed={elapsed:.0f}s")
    print(f"  [{name}] DONE in {time.monotonic()-t0:.0f}s  → {out_path}")


def report(name):
    out_path = OUT_DIR / f"{name}_probe.jsonl"
    rows = [json.loads(l) for l in out_path.open() if l.strip()]
    if not rows: return
    comps = [r["completion_tokens"] for r in rows if r.get("completion_tokens") is not None]
    reasons = [r["reasoning_tokens"] for r in rows if r.get("reasoning_tokens") is not None]
    fins = [r["finish_reason"] for r in rows]
    n = len(rows)
    n_length = sum(1 for f in fins if f == "length")
    n_certain = sum(1 for r in rows if (r.get("truncation_report") or {}).get("is_truncated_certain"))
    n_likely = sum(1 for r in rows if (r.get("truncation_report") or {}).get("is_truncated_likely"))
    print(f"\n  {name} report (n={n}):")
    if comps:
        comps_sorted = sorted(comps)
        p50 = comps_sorted[n//2]
        p75 = comps_sorted[3*n//4]
        p95 = comps_sorted[max(0, n*95//100 - 1)]
        max_c = max(comps)
        print(f"    completion_tokens \u2014 max={max_c}  p95={p95}  p75={p75}  p50={p50}")
        if reasons:
            r_sorted = sorted([r for r in reasons if r is not None])
            if r_sorted:
                print(f"    reasoning_tokens \u2014 max={max(r_sorted)}  p95={r_sorted[max(0,len(r_sorted)*95//100-1)]}  p50={r_sorted[len(r_sorted)//2]}")
        print(f"    finish=length count: {n_length}/{n}")
        print(f"    truncation_detector: certain={n_certain}/{n}  likely={n_likely}/{n}")
        # Recommended cap = 2 * max observed (safety margin), per Step 0 convention
        rec = max(256, 2 * max_c)
        print(f"    RECOMMENDED max_tokens = 2 \u00d7 max_observed = {rec}")
        # Show fired-signal counts
        from collections import Counter
        sig_counter = Counter()
        for r in rows:
            tr = r.get("truncation_report") or {}
            for sig, fired in (tr.get("signals") or {}).items():
                if fired:
                    sig_counter[sig] += 1
        if sig_counter:
            print(f"    fired signals: {dict(sig_counter)}")


def main():
    print("Loading dev + notes…")
    dev = [json.loads(l) for l in DEV_JSONL.open() if l.strip()]
    lookup = build_note_lookup_with_count()
    print(f"  dev={len(dev)}, lookup={len(lookup)}")

    phase_a, phase_b = select_probe_items(dev, lookup)
    print(f"\nPhase A (5 longest): pids={[it['patient_id'] for it in phase_a]}")
    print(f"  total_chars: {[it['total_chars'] for it in phase_a]}")
    print(f"\nPhase B (30 stratified by n_notes): "
          f"1-note: {sum(1 for it in phase_b if it['n_notes']==1)}, "
          f"2-note: {sum(1 for it in phase_b if it['n_notes']==2)}, "
          f"3-note: {sum(1 for it in phase_b if it['n_notes']==3)}")

    client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

    for name, model in CONFIGS:
        # Phase A: 5 longest
        run_probe(f"{name}_phaseA", model, phase_a, client)
        report(f"{name}_phaseA")
        # Phase B: stratified
        run_probe(f"{name}_phaseB", model, phase_b, client)
        report(f"{name}_phaseB")


if __name__ == "__main__":
    main()
