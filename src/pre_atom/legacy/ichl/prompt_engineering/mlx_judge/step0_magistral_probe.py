"""Step-0 token-budget probe for Magistral-Small-2509-AWQ on local vLLM.

Phase A only: 5 longest-notes items, max_tokens=8192, with truncation_detector
applied per Claude: Principle: Truncation Detection on Every LLM Output.

Goal: confirm prior 300-dev run at max_tokens=4096 (0% certain truncation) was
not undercut. If detector fires nothing at 8192 either, 4096 was correctly sized
and we can proceed to prompt iteration without re-running dev.
"""
from __future__ import annotations
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from ichl.prompt_engineering.correction.truncation_detector import detect_truncation

ROOT = Path(__file__).resolve().parents[4]
DEV_JSONL = ROOT / "output" / "ichl" / "mlx_judge" / "splits" / "dev.jsonl"
OUT_DIR = ROOT / "output" / "ichl" / "mlx_judge" / "step0_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VLLM_URL = "http://localhost:8003/v1"
MODEL = "Magistral-Small-2509-AWQ"
PROBE_MAX_TOKENS = 8192

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


def build_user(item, note):
    rules = "\n".join(f"{i}. {r}" for i, r in enumerate(C3E_RULES, 1))
    return f"""When judging, apply the following rules:

{rules}

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


def select_longest(dev, lookup, k=5):
    seen = set()
    by_pid = []
    for d in dev:
        pid = str(d["patient_id"])
        if pid in seen:
            continue
        seen.add(pid)
        info = lookup.get(pid, {})
        by_pid.append({**d, **info})
    return sorted(by_pid, key=lambda x: -(x.get("total_chars") or 0))[:k]


def probe_one(args):
    client, item = args
    user = build_user(item, item.get("text", ""))
    t0 = time.monotonic()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            temperature=0.0, max_tokens=PROBE_MAX_TOKENS,
        )
    except Exception as e:
        return {"patient_id": item["patient_id"], "error": str(e), "latency_s": time.monotonic() - t0}
    lat = time.monotonic() - t0
    msg = resp.choices[0].message
    content = getattr(msg, "content", None) or ""
    reasoning_content = getattr(msg, "reasoning_content", None) or ""
    fin = resp.choices[0].finish_reason
    usage = resp.usage
    raw_for_detector = (reasoning_content + "\n" + content) if reasoning_content else content
    report = detect_truncation(
        raw_response=raw_for_detector, text_clean=content,
        finish_reason=fin,
        usage={"completion_tokens": usage.completion_tokens if usage else None,
               "prompt_tokens": usage.prompt_tokens if usage else None},
        max_tokens=PROBE_MAX_TOKENS, target=MODEL, sub_variant="vllm",
    )
    return {
        "patient_id": item["patient_id"], "fold_id": item["fold_id"], "target": item["target"],
        "n_notes": item.get("n_notes"), "total_chars": item.get("total_chars"),
        "gold_label": int(item["binary_correct"]),
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "content": content, "reasoning_tail": reasoning_content[-200:] if reasoning_content else "",
        "finish_reason": fin, "latency_s": round(lat, 2),
        "truncation_report": report.as_dict(),
    }


def main():
    print("Loading dev + notes…")
    dev = [json.loads(l) for l in DEV_JSONL.open() if l.strip()]
    lookup = build_note_lookup_with_count()
    items = select_longest(dev, lookup, k=5)
    print(f"  Phase A: 5 longest-notes items, total_chars={[it['total_chars'] for it in items]}")

    client = OpenAI(base_url=VLLM_URL, api_key="not-needed")
    out_path = OUT_DIR / "magistral_phaseA.jsonl"

    t0 = time.monotonic()
    with out_path.open("w") as f, ThreadPoolExecutor(max_workers=2) as ex:
        for i, r in enumerate(ex.map(probe_one, [(client, it) for it in items]), 1):
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            tr = r.get("truncation_report") or {}
            print(f"  [{i}/5] pid={r.get('patient_id')} comp_tok={r.get('completion_tokens')} "
                  f"finish={r.get('finish_reason')} certain={tr.get('is_truncated_certain')} "
                  f"likely={tr.get('is_truncated_likely')} signals={(tr.get('signals') or {})} "
                  f"lat={r.get('latency_s')}s")
    elapsed = time.monotonic() - t0
    print(f"\nDONE in {elapsed:.0f}s → {out_path}")

    # Summary
    rows = [json.loads(l) for l in out_path.open() if l.strip()]
    n_cert = sum(1 for r in rows if (r.get("truncation_report") or {}).get("is_truncated_certain"))
    n_like = sum(1 for r in rows if (r.get("truncation_report") or {}).get("is_truncated_likely"))
    max_comp = max((r.get("completion_tokens") or 0) for r in rows)
    print(f"\nSummary: certain={n_cert}/{len(rows)}  likely={n_like}/{len(rows)}  max_comp_tokens={max_comp}")
    print(f"  ⇒ recommended max_tokens = max(256, 2*max_obs) = {max(256, 2*max_comp)}")
    print(f"  ⇒ prior run used 4096 → {'OK' if max_comp < 4096 else 'UNDERCUT, need to re-run'}")


if __name__ == "__main__":
    main()
