"""Stage III Verdict pipeline.

Steps (selected via --step):
  probe       : Step 0 — token-budget probe (5 items at probe_max_tokens)
  pilot       : Step 2 — run V-prompt on stratified pilot (40 items: 10 per cell)
  lockdown    : Step 3 — run winning V-prompt on full eligible pairs
  cross_model : Step 4 — apply winner V to alt verdict models (Qwen3 / DS-R1 / Llama)
  audit       : Step 5 — bias audit (Pick-A, length-r, Pick-first)
  report      : Step 6 — pool + summary

All LLM calls inherit vllm_call from error_location pipeline (auto-retry, truncation report).
DS-R1 messy output handled via inherited regex parser; LLM-arbiter @ port 8803 (Qwen3.5-27B)
on disagreement per `Use MLX as External Validator`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from openai import OpenAI

from ichl.error_location.pipeline import vllm_call
from ichl.verdict.cells import build_pairs, stratified_pilot
from ichl.verdict.prompts import VERDICT_VERSIONS, VERDICT_SYSTEM


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output" / "ichl" / "verdict" / "round0"
OUT.mkdir(parents=True, exist_ok=True)

# Verdict-model table (vllm_manager keys)
VERDICT_MODELS = {
    "qwen2.5-7b-instruct": {"vllm_model": "Qwen/Qwen2.5-7B-Instruct", "max_model_len": 16384,
                             "max_gen_tokens": 50, "enable_thinking": None},
    "qwen3-8b": {"vllm_model": "Qwen/Qwen3-8B", "max_model_len": 32768,
                  "max_gen_tokens": 50, "enable_thinking": False},
    "deepseek-r1-distill-llama-8b": {"vllm_model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                                       "max_model_len": 32768, "max_gen_tokens": 4096,
                                       "enable_thinking": None},
    "llama-3.1-8b-instruct": {"vllm_model": "meta-llama/Llama-3.1-8B-Instruct",
                                "max_model_len": 16384, "max_gen_tokens": 50,
                                "enable_thinking": None},
}

# Parser: regex first; if UNKNOWN, can fall back to LLM-arbiter (Qwen3.5-27B @ 8803)
_PICK_RE = re.compile(r"PICK:\s*(A|B|UNCERTAIN|NEITHER|EITHER)", re.IGNORECASE)
_FALLBACK_A_RE = re.compile(r"\b(answer\s*A|candidate\s*A|^\s*A\b)", re.IGNORECASE)
_FALLBACK_B_RE = re.compile(r"\b(answer\s*B|candidate\s*B|^\s*B\b)", re.IGNORECASE)


def parse_pick(text: str) -> dict:
    """Parse verdict output. Returns {pick: 'A'|'B'|'UNCERTAIN'|None, parser: str}."""
    if not text:
        return {"pick": None, "parser": "empty"}
    # Strip <think>...</think> first (DS-R1, Qwen3 think)
    if "</think>" in text:
        text = re.sub(r"^.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    m = _PICK_RE.search(text)
    if m:
        v = m.group(1).upper()
        if v in ("EITHER", "NEITHER"): v = "UNCERTAIN"
        return {"pick": v, "parser": "regex_pick"}
    # Fallback: look for A/B mention near "answer" or "candidate"
    a = _FALLBACK_A_RE.search(text); b = _FALLBACK_B_RE.search(text)
    if a and not b: return {"pick": "A", "parser": "regex_fallback_A"}
    if b and not a: return {"pick": "B", "parser": "regex_fallback_B"}
    return {"pick": None, "parser": "regex_failed"}


def llm_arbiter_pick(text: str, base_url: str = "http://192.168.68.107:8803/v1") -> dict:
    """Qwen3.5-27B @ 8803 acts as LLM-parser-arbiter when regex fails."""
    try:
        client = OpenAI(base_url=base_url, api_key="not-needed", timeout=60)
        prompt = f"""The following text is a verdict response that should pick A or B. Extract the pick.

TEXT:
{text[:2000]}

Output exactly one of:
PICK: A
PICK: B
PICK: UNCERTAIN"""
        models = client.models.list().data
        model = next((m.id for m in models if "27B" in m.id or "Qwen3" in m.id), models[0].id)
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=20,
        )
        out = r.choices[0].message.content or ""
        m = _PICK_RE.search(out)
        if m:
            v = m.group(1).upper()
            if v in ("EITHER", "NEITHER"): v = "UNCERTAIN"
            return {"pick": v, "parser": "llm_arbiter"}
    except Exception as e:
        return {"pick": None, "parser": f"llm_arbiter_err:{str(e)[:50]}"}
    return {"pick": None, "parser": "llm_arbiter_failed"}


def is_correct_pick(pick: str | None, gold: str) -> int | None:
    """Map (pick, gold_pick) to 1/0/None.
    gold ∈ {'A', 'B', 'EITHER', 'NEITHER'}. Verdict output ∈ {'A', 'B', 'UNCERTAIN', None}.
    """
    if pick is None: return None
    if gold == "EITHER":
        return 1 if pick in ("A", "B", "UNCERTAIN") else 0
    if gold == "NEITHER":
        return 1 if pick == "UNCERTAIN" else 0
    # gold is 'A' or 'B'
    return 1 if pick == gold else 0


def run_verdict(args):
    """Run a verdict V-prompt on a set of pairs."""
    pair_type = args.pair_type
    n_per_cell = args.n_per_cell
    verdict_version = args.verdict_version
    verdict_model_key = args.verdict_model

    v_entry = VERDICT_VERSIONS[verdict_version]
    if len(v_entry) == 3:
        sys_p, tmpl, v_max_gen = v_entry
    else:
        sys_p, tmpl = v_entry; v_max_gen = None
    cfg = VERDICT_MODELS[verdict_model_key]
    # DS-R1 always-think requires the model-level (large) budget regardless of V;
    # other models follow the per-V budget when set.
    if cfg.get("enable_thinking") is None and "deepseek" in verdict_model_key.lower():
        max_gen = cfg["max_gen_tokens"]
    elif v_max_gen is not None:
        max_gen = max(v_max_gen, 50)
    else:
        max_gen = cfg["max_gen_tokens"]
    print(f"  max_gen for {verdict_version}/{verdict_model_key} = {max_gen}")

    pairs = build_pairs(pair_type=pair_type)
    print(f"Pairs available: {len(pairs)} (pair_type={pair_type})")

    if args.step == "pilot":
        items = stratified_pilot(pairs, n_per_cell=n_per_cell)
        scope = "pilot"
    else:  # lockdown / cross_model — full eligible pairs
        items = pairs
        # Add detail_cell label even for full set
        for p in items:
            zs_label = p["primary_zs_label"]
            alt_label = p["B_label"] if "::zs" in p["A_source"] else p["A_label"]
            if zs_label == 0 and alt_label == 1: p["detail_cell"] = "FIX"
            elif zs_label == 1 and alt_label == 0: p["detail_cell"] = "BREAK"
            elif zs_label == 1 and alt_label == 1: p["detail_cell"] = "stay_right"
            else: p["detail_cell"] = "stay_wrong"
        scope = args.step

    print(f"  scope={scope}, items={len(items)}, cells={Counter(p['detail_cell'] for p in items)}")

    out_dir = OUT / scope / pair_type / verdict_version / verdict_model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "predictions.jsonl"

    from ichl.common import vllm_manager
    vllm_manager.stop()
    time.sleep(2)
    vllm_manager.ensure_model(verdict_model_key, log_dir=out_dir / "vllm_logs")
    vllm = OpenAI(base_url="http://localhost:8003/v1", api_key="not-needed", timeout=600)

    results = []
    n_done = n_err = n_trunc = n_arbiter = 0

    with out_file.open("w") as fout:
        for it in items:
            user = tmpl.format(
                note=it["note"], question=it["question"],
                answer_a=it["candidate_A"], answer_b=it["candidate_B"],
            )
            r = vllm_call(vllm, cfg["vllm_model"], sys_p, user,
                          max_tokens=max_gen, temperature=0.0,
                          target=verdict_model_key,
                          enable_thinking=cfg.get("enable_thinking"),
                          max_model_len=cfg["max_model_len"], max_retries=2)
            n_done += 1
            if r["_err"]: n_err += 1
            elif r["truncation_report"]["is_truncated_certain"]: n_trunc += 1

            parsed = parse_pick(r["text"])
            if parsed["pick"] is None and not r["_err"] and r["text"]:
                # Regex failed — try LLM arbiter (Qwen3.5-27B @ 8803)
                parsed = llm_arbiter_pick(r["text"])
                if parsed["pick"] is not None:
                    n_arbiter += 1

            correct = is_correct_pick(parsed["pick"], it["gold_pick"])
            rec = {
                **{k: it[k] for k in (
                    "patient_id", "fold_id", "detail_cell",
                    "A_source", "B_source", "A_label", "B_label",
                    "A_length", "B_length", "gold_pick",
                )},
                "verdict_version": verdict_version,
                "verdict_model": verdict_model_key,
                "verdict_text": r["text"][:300],
                "picked": parsed["pick"],
                "parser_used": parsed["parser"],
                "correct": correct,
                "truncation_report": r["truncation_report"],
                "_err": r["_err"],
            }
            results.append(rec)
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if n_done % 25 == 0:
                acc = sum(1 for r_ in results if r_["correct"] == 1) / n_done
                print(f"  done {n_done}/{len(items)} err={n_err} trunc={n_trunc} arbiter={n_arbiter} running_acc={acc:.3f}")
            # In-flight abort gate per Per-50 Sanity Checkpoint (here: per-25)
            if n_done >= 10 and (n_trunc / n_done) > 0.10:
                print(f"  [ABORT] truncation rate {n_trunc}/{n_done} = {n_trunc/n_done:.2%} > 10% threshold")
                raise SystemExit(f"truncation gate: raise max_gen for {verdict_version}/{verdict_model_key}")

    # Summary stats
    valid = [r_ for r_ in results if r_["correct"] in (0, 1)]
    correct = sum(1 for r_ in valid if r_["correct"] == 1)
    by_cell = defaultdict(lambda: {"n": 0, "correct": 0})
    for r_ in valid:
        c = r_["detail_cell"]
        by_cell[c]["n"] += 1
        if r_["correct"]: by_cell[c]["correct"] += 1

    summary = {
        "scope": scope, "pair_type": pair_type,
        "verdict_version": verdict_version, "verdict_model": verdict_model_key,
        "n_total": len(results), "n_valid": len(valid),
        "n_err": n_err, "n_trunc": n_trunc,
        "n_arbiter_used": n_arbiter,
        "n_correct": correct,
        "accuracy": round(correct / max(len(valid), 1), 4),
        "by_cell": {c: {**v, "rate": round(v["correct"]/max(v["n"],1), 4)} for c, v in by_cell.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[verdict] Saved {out_dir}")
    print(f"  acc: {correct}/{len(valid)} = {summary['accuracy']:.3f}")
    for c, v in by_cell.items():
        rate = v["correct"] / max(v["n"], 1)
        print(f"  {c}: {v['correct']}/{v['n']} = {rate:.3f}")


def step_audit(args):
    """Bias audit on existing predictions.jsonl."""
    pair_type = args.pair_type; ver = args.verdict_version; mdl = args.verdict_model
    scope = args.scope or "pilot"
    in_file = OUT / scope / pair_type / ver / mdl / "predictions.jsonl"
    if not in_file.exists():
        raise SystemExit(f"missing {in_file}")
    rows = [json.loads(l) for l in in_file.open()]
    valid = [r for r in rows if r.get("picked") in ("A", "B")]
    if not valid:
        print("no valid picks for audit")
        return

    # Pick-A bias
    pick_a = sum(1 for r in valid if r["picked"] == "A") / len(valid)

    # Length bias: correlation of (A_length - B_length) with (picked == "A")
    diffs = [(r["A_length"] - r["B_length"], 1 if r["picked"] == "A" else 0) for r in valid]
    if len(diffs) >= 5:
        x = np.array([d[0] for d in diffs])
        y = np.array([d[1] for d in diffs])
        if x.std() > 0:
            r_corr = float(np.corrcoef(x, y)[0, 1])
        else:
            r_corr = 0.0
    else:
        r_corr = 0.0

    flags = []
    if abs(pick_a - 0.5) > 0.05: flags.append(f"pick_a_bias={pick_a:.3f}")
    if abs(r_corr) > 0.3: flags.append(f"length_corr={r_corr:.3f}")

    audit = {
        "n_valid": len(valid),
        "pick_a_rate": round(pick_a, 4),
        "length_correlation": round(r_corr, 4),
        "bias_flags": flags,
    }
    print(json.dumps(audit, indent=2))
    out_path = in_file.parent / "audit.json"
    out_path.write_text(json.dumps(audit, indent=2))
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True,
                    choices=["probe", "pilot", "lockdown", "cross_model", "audit", "report"])
    ap.add_argument("--pair-type", default="cross_zs",
                    choices=["zs_vs_t0a", "zs_vs_v4", "cross_zs"])
    ap.add_argument("--verdict-version", default="v1", choices=list(VERDICT_VERSIONS.keys()))
    ap.add_argument("--verdict-model", default="qwen2.5-7b-instruct",
                    choices=list(VERDICT_MODELS.keys()))
    ap.add_argument("--n-per-cell", type=int, default=10)
    ap.add_argument("--scope", default="")
    args = ap.parse_args()

    if args.step in ("probe", "pilot", "lockdown", "cross_model"):
        run_verdict(args)
    elif args.step == "audit":
        step_audit(args)
    else:
        raise SystemExit(f"step {args.step} not yet implemented")


if __name__ == "__main__":
    main()
