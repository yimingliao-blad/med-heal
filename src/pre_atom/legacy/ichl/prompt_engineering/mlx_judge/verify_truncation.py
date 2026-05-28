"""Two-stage truncation verifier (per Claude: Principle: Truncation Detection on Every LLM Output § Two-stage verification).

Takes a per-item record from a previous judge run that was flagged by the detector,
re-runs the same (note, question, GT, model_answer) at 2× max_tokens, applies the
detector, and classifies:

  VERIFIED_SAME       — retry verdict == original verdict; original was effectively complete.
                        Original truncation was cosmetic (e.g., explanation cutoff after digit).
  VERIFIED_DIFFERENT  — retry verdict != original verdict (and original is parseable).
                        Real truncation changed the verdict; retry is the correct one.
  RECOVERED_FROM_NONE — original was None (truncated, no digit); retry produced a digit.
                        Use retry verdict.
  STILL_TRUNCATED     — retry also truncated; cannot resolve. Escalate to user.

Inputs: a per-item record with at minimum: target, patient_id, fold_id, gold_label,
        plus the original `mlx_label`, `truncation_report` (or finish_reason/completion_tokens).
Outputs: a verification record per flagged item.
"""
from __future__ import annotations
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from ichl.prompt_engineering.correction.truncation_detector import detect_truncation

ROOT = Path(__file__).resolve().parents[4]
DEV_JSONL = ROOT / "output" / "ichl" / "mlx_judge" / "splits" / "dev.jsonl"
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


def build_prompt(item, note, thinking_enabled, with_no_think_prefix=False):
    rules_block = "\n".join(f"{i}. {r}" for i, r in enumerate(C3E_RULES, 1))
    prefix = "/no_think\n" if with_no_think_prefix else ""
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


def parse_digit(text):
    if not text: return None
    s = text.strip()
    if not s: return None
    m = re.search(r"[01]", s)
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
                return None
    return None


def verify_one(args):
    """Re-run a flagged item at 2× max_tokens; classify outcome."""
    client, model, with_no_think, retry_max_tokens, original_record, dev_item, note = args
    user = build_prompt(dev_item, note, thinking_enabled=not with_no_think, with_no_think_prefix=with_no_think)
    t0 = time.monotonic()
    resp = call_lm(client, model, SYSTEM, user, retry_max_tokens)
    lat = time.monotonic() - t0
    if resp is None:
        return {**original_record, "verify_status": "API_ERROR", "verify_latency_s": round(lat, 2)}
    msg = resp.choices[0].message
    content = getattr(msg, "content", None) or ""
    reasoning_content = getattr(msg, "reasoning_content", None) or ""
    fin = resp.choices[0].finish_reason
    usage = resp.usage
    new_verdict = parse_digit(content)
    raw_for_detector = (reasoning_content + "\n" + content) if reasoning_content else content
    new_report = detect_truncation(
        raw_response=raw_for_detector, text_clean=content,
        finish_reason=fin,
        usage={"completion_tokens": usage.completion_tokens if usage else None,
               "prompt_tokens": usage.prompt_tokens if usage else None},
        max_tokens=retry_max_tokens, target=model, sub_variant="think" if not with_no_think else "",
    )
    original_verdict = original_record.get("mlx_label")
    if new_report.is_truncated_certain and new_verdict is None:
        status = "STILL_TRUNCATED"
    elif original_verdict is None and new_verdict is not None:
        status = "RECOVERED_FROM_NONE"
    elif original_verdict is not None and new_verdict is not None and original_verdict == new_verdict:
        status = "VERIFIED_SAME"
    elif original_verdict is not None and new_verdict is not None and original_verdict != new_verdict:
        status = "VERIFIED_DIFFERENT"
    elif original_verdict is None and new_verdict is None:
        status = "STILL_TRUNCATED"  # both None
    else:
        status = "UNCLEAR"
    return {
        **original_record,
        "verify_status": status,
        "verify_max_tokens": retry_max_tokens,
        "verify_verdict": new_verdict,
        "verify_completion_tokens": usage.completion_tokens if usage else None,
        "verify_reasoning_tokens": getattr(usage.completion_tokens_details, "reasoning_tokens", None) if (usage and getattr(usage, "completion_tokens_details", None)) else None,
        "verify_finish_reason": fin,
        "verify_latency_s": round(lat, 2),
        "verify_truncation_report": new_report.as_dict(),
        "verify_content_first50": content[:50],
        "verify_reasoning_tail": reasoning_content[-300:] if reasoning_content else "",
    }


def filter_flagged(rows, original_max_tokens, model_id, sub_var):
    """Apply detector to rows, return list of flagged + classification reason."""
    flagged = []
    for r in rows:
        raw = r.get("raw") or r.get("content") or ""
        text = r.get("content") or r.get("raw") or ""
        report = detect_truncation(
            raw_response=raw, text_clean=text,
            finish_reason=r.get("finish_reason"),
            usage={"completion_tokens": r.get("completion_tokens"), "prompt_tokens": r.get("prompt_tokens")},
            max_tokens=original_max_tokens, target=model_id, sub_variant=sub_var,
        )
        # For binary judge: label-critical if no parseable digit AND certain truncation
        label_critical = report.is_truncated_certain and r.get("mlx_label") is None
        cosmetic = report.is_truncated_likely and not label_critical
        if label_critical or cosmetic:
            flagged.append({**r, "_detector_signals": report.fired_signals(),
                            "_label_critical": label_critical, "_cosmetic": cosmetic})
    return flagged


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="One of: Q32B_think, Q35_27B_think, C-series-sample, V0/V4/etc")
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--model", required=True, help="LM Studio model id")
    ap.add_argument("--original-max-tokens", type=int, required=True)
    ap.add_argument("--retry-max-tokens", type=int, required=True)
    ap.add_argument("--no-think", action="store_true", help="Use /no_think prefix (for nothink configs)")
    ap.add_argument("--cosmetic-sample", type=int, default=0, help="Subsample N cosmetic-flagged for verification (else all)")
    args = ap.parse_args()

    print(f"Loading {args.input_jsonl}…")
    rows = [json.loads(l) for l in Path(args.input_jsonl).open() if l.strip()]
    print(f"  {len(rows)} rows total")

    sub_var = "think" if not args.no_think else ""
    flagged = filter_flagged(rows, args.original_max_tokens, args.model, sub_var)
    crits = [f for f in flagged if f["_label_critical"]]
    cosms = [f for f in flagged if f["_cosmetic"]]
    print(f"  flagged: label_critical={len(crits)}  cosmetic={len(cosms)}")

    # Sample cosmetics if requested
    if args.cosmetic_sample > 0 and len(cosms) > args.cosmetic_sample:
        import random
        random.Random(42).shuffle(cosms)
        cosms = cosms[:args.cosmetic_sample]
        print(f"  cosmetic subsampled to {len(cosms)}")
    to_verify = crits + cosms
    print(f"  total to verify: {len(to_verify)}")

    if not to_verify:
        print("  nothing to verify; writing empty output")
        Path(args.output_jsonl).touch()
        return

    print("Loading dev set + notes…")
    dev = [json.loads(l) for l in DEV_JSONL.open() if l.strip()]
    dev_lookup = {(r["target"], r["patient_id"], r["fold_id"]): r for r in dev}
    notes = build_note_lookup()

    client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

    tasks = []
    for r in to_verify:
        key = (r["target"], r["patient_id"], r["fold_id"])
        dev_item = dev_lookup.get(key)
        if dev_item is None:
            continue
        note = notes.get(str(r["patient_id"]), "")
        tasks.append((client, args.model, args.no_think, args.retry_max_tokens, r, dev_item, note))

    print(f"\nVerifying {len(tasks)} items at retry_max_tokens={args.retry_max_tokens}…")
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counter = {"VERIFIED_SAME": 0, "VERIFIED_DIFFERENT": 0, "RECOVERED_FROM_NONE": 0,
               "STILL_TRUNCATED": 0, "UNCLEAR": 0, "API_ERROR": 0}
    workers = 1 if not args.no_think else 2  # thinking-mode = serial to avoid contention
    t0 = time.monotonic()
    with out_path.open("w") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(verify_one, tasks), 1):
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            status = r.get("verify_status", "?")
            counter[status] = counter.get(status, 0) + 1
            if i % 5 == 0 or i == len(tasks):
                elapsed = time.monotonic() - t0
                print(f"  {i}/{len(tasks)}  elapsed={elapsed:.0f}s  counts={counter}")
    elapsed = time.monotonic() - t0
    print(f"\nDONE in {elapsed:.0f}s.  Final counts:")
    for k, v in counter.items():
        if v > 0:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
