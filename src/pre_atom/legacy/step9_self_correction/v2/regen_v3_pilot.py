#!/usr/bin/env python3
"""
Regen V3 + Verdict V3 + 2-round CoVe pilot.

Combines all the user-driven design choices from the multi-model V2/regen
post-mortem:

REGEN V3 PROMPT (1, 2, 5):
  - explicitly pushes the model through three checks BEFORE writing the answer:
      contradiction check / question alignment check / reasoning link check
  - EVIDENCE: ... / ANSWER: ... structured output (1-3 sentences)
  - temperature = 0.0, K=1 (deterministic)

2-ROUND COVE-STYLE (3):
  - round 1: regen V3 from the question + note (no original answer shown — factored CoVe)
  - critique: ask the model to list any claim in round-1 answer that is NOT supported
              by a verbatim quote from the note (returns a numbered list, possibly empty)
  - round 2: regen again, instructed to drop the unsupported claims from round 1

VERDICT V3 (4):
  - per-criterion COUNTS instead of vague pairwise judgment:
      A_CONTRADICTIONS, A_UNADDRESSED, A_UNSUPPORTED  (and B_*)
  - WINNER = the answer with the lower TOTAL of the three counts
  - blind A/B placement (deterministic seed by item idx)
  - robust parsing chain:
      1. strict regex on the count lines
      2. fuzzy regex (loose number extraction near keyword)
      3. Qwen3-32B fallback parser (extracts WINNER text)
      4. consistency sanity-check: if winner contradicts the count totals, flag it

ROBUST OUTPUT ANALYSIS (1):
  - every parsed result is validated:
      - counts are non-negative integers
      - winner is consistent with counts (lower-total wins)
      - if any check fails, the parse_path is recorded and the verdict step
        defaults to the conservative "keep original"
  - the audit log persists every raw output, parsed value, parse_path,
    and a per-item parse_warnings list

Per target model the same model is used for regen, critique, and verdict
(self-correction). vLLM lifecycle is managed by the caller.

Output: output/step9_v2/multi_model/{model_dir}/regen_v3_audit.jsonl
"""
from __future__ import annotations

import os
import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

SOURCE_ROOT = Path(os.environ.get("PRE_ATOM_SOURCE_REPO_ROOT", Path(__file__).resolve().parents[5]))
RUN_ROOT = Path(os.environ.get("PRE_ATOM_PROJECT_ROOT", SOURCE_ROOT))
PROJECT_ROOT = SOURCE_ROOT
sys.path.insert(0, str(Path(__file__).parent))
from audit_log import AuditLog, make_record
from detection_format_bakeoff import served_model_id, vllm_chat, set_default_chat_template_kwargs
from judge import _load_notes_lookup, judge as judge_call
from multi_model_pilot import MODELS, sample_test_items, sample_random_per_fold
from correction_prompt_v2 import parse_premises_output, verify_evidence_quotes

OUT_DIR = PROJECT_ROOT / "output" / "step9_v2" / "multi_model"
QWEN32B_URL = "http://192.168.68.107:8090/v1/chat/completions"


# ---------------------------------------------------------------------------
# Regen V3 prompt
# ---------------------------------------------------------------------------

REGEN_SYS = "You are a medical expert."

REGEN_V3_TMPL = """Discharge note:
{note}

Question: {question}

Answer the question. Before writing your final answer, do these three checks:

1. CONTRADICTION CHECK: which factual claims would contradict the discharge
   notes? Avoid those claims.
2. QUESTION ALIGNMENT CHECK: re-read the question. Are you answering EXACTLY
   what is asked — the right visit, the right time period, the right body
   part, ALL parts of a multi-part question?
3. REASONING LINK CHECK: every claim in your answer must be supported by a
   sentence in the note. If a claim has no supporting sentence, drop it.

After the three checks, write your answer in this EXACT format:

EVIDENCE:
- "<verbatim quote from the note>"
- "<verbatim quote from the note>"
- "<verbatim quote from the note>"
(2 to 5 quotes total, covering ALL the facts your answer relies on)

ANSWER:
<your final answer in 1-3 sentences, supported only by the evidence above>

Rules:
- Each EVIDENCE line must be a VERBATIM quote from the note in double quotes.
- The ANSWER must be supported by your quoted evidence.
- If a question asks about multiple visits or events, cover ALL of them.
- If the note does not contain the information, say so explicitly in the ANSWER."""


# ---------------------------------------------------------------------------
# CoVe critique prompt (round 1.5)
# ---------------------------------------------------------------------------

CRITIQUE_SYS = "You are a strict medical expert auditing an answer for unsupported claims."

CRITIQUE_TMPL = """Discharge note:
{note}

Question: {question}

The following answer was written by an AI model:

{round1_answer}

For each factual claim in this answer, decide whether it is supported by a
verbatim sentence in the discharge note above. List ONLY the claims that are
NOT supported by any sentence in the note (i.e. fabricated or unsupported).

Reply in this exact format:

UNSUPPORTED_CLAIMS:
- "<the unsupported claim, quoted>"
- "<the unsupported claim, quoted>"
(if there are no unsupported claims, write: NONE)
"""


# ---------------------------------------------------------------------------
# Regen V3 round 2 — drop the unsupported claims
# ---------------------------------------------------------------------------

REGEN_ROUND2_TMPL = """Discharge note:
{note}

Question: {question}

A previous answer to this question contained the following unsupported claims:

{unsupported_block}

Re-answer the question. Drop those unsupported claims entirely. Use the same
format as before:

EVIDENCE:
- "<verbatim quote from the note>"
- "<verbatim quote from the note>"
(2 to 5 verbatim quotes)

ANSWER:
<your final answer in 1-3 sentences, supported only by the evidence above>

Make sure every claim in your final answer is backed by a verbatim quote you
listed."""


# ---------------------------------------------------------------------------
# Verdict V3 — 3-count comparison with robust parsing
# ---------------------------------------------------------------------------

VERDICT_SYS = "You are a strict medical expert comparing two clinical answers against discharge notes."

VERDICT_V3_TMPL = """Discharge summary:
{note}

Question: {question}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

For each answer, count three things SEPARATELY:

1. CONTRADICTIONS: factual claims that contradict the discharge notes.
   Different wording for the same fact is NOT a contradiction.

2. UNADDRESSED_PARTS: parts of the question that the answer does NOT address.
   A multi-part question (e.g. "left knee AND right knee", "first visit AND
   second visit") counts each missing part separately.

3. UNSUPPORTED_CLAIMS: claims in the answer that have no supporting sentence
   in the discharge notes (i.e. fabricated or made-up content).

Then declare the winner: the answer with the lower TOTAL of the three counts.
If both answers have the same total, pick A.

Reply in this EXACT format. Do not deviate.

A_CONTRADICTIONS: <integer>
A_UNADDRESSED:    <integer>
A_UNSUPPORTED:    <integer>
B_CONTRADICTIONS: <integer>
B_UNADDRESSED:    <integer>
B_UNSUPPORTED:    <integer>
WINNER: A or B
"""


# ---------------------------------------------------------------------------
# Robust parsing
# ---------------------------------------------------------------------------

# Strict line-keyed integer regex
_KEY_INT_RE = {
    key: re.compile(rf"\b{key}\s*:\s*\*?\*?\s*(\d+)", re.IGNORECASE)
    for key in [
        "A_CONTRADICTIONS", "A_UNADDRESSED", "A_UNSUPPORTED",
        "B_CONTRADICTIONS", "B_UNADDRESSED", "B_UNSUPPORTED",
    ]
}
_WINNER_RE = re.compile(r"\bWINNER\s*:?\s*\*?\*?\s*([AB])\b", re.IGNORECASE)
# Fuzzy: any "answer X has N ..."
_FUZZY_NUM_RE = re.compile(r"(\d+)")


def _strip_md(text: str) -> str:
    return text.replace("**", "").replace("`", "")


def parse_verdict_v3(raw: str) -> dict:
    """Strict-regex-only parser for VERDICT_V3 output.

    This is one half of the dual-parse stack. The other half (Qwen3-32B
    fallback parser) is called separately by run_verdict_v3 on EVERY item,
    not just when this parser fails — so we can compute the agreement rate
    between regex and Qwen3-32B and decide whether the regex layer is even
    needed.

    Returns a dict with:
      counts: {A: {contradictions, unaddressed, unsupported}, B: ...}
      totals: {A: int, B: int}
      winner: 'A' | 'B' | None
      is_tie: bool                      ← TRUE iff totals[A] == totals[B]
      parse_path: 'strict' | 'unparseable'
      warnings: list of strings (consistency issues)
    """
    out = {
        "counts": {"A": {"contradictions": None, "unaddressed": None, "unsupported": None},
                    "B": {"contradictions": None, "unaddressed": None, "unsupported": None}},
        "totals": {"A": None, "B": None},
        "winner": None,
        "is_tie": False,
        "parse_path": "unparseable",
        "warnings": [],
    }
    if not raw:
        out["warnings"].append("empty raw")
        return out
    text = _strip_md(raw)

    # Strict pass: try line-keyed integers
    found = {}
    for key, rx in _KEY_INT_RE.items():
        m = rx.search(text)
        if m:
            try:
                v = int(m.group(1))
                if v < 0 or v > 100:  # absurdly large counts → reject
                    raise ValueError
                found[key] = v
            except (ValueError, OverflowError):
                pass

    if len(found) == 6:
        out["counts"]["A"] = {
            "contradictions": found["A_CONTRADICTIONS"],
            "unaddressed":    found["A_UNADDRESSED"],
            "unsupported":    found["A_UNSUPPORTED"],
        }
        out["counts"]["B"] = {
            "contradictions": found["B_CONTRADICTIONS"],
            "unaddressed":    found["B_UNADDRESSED"],
            "unsupported":    found["B_UNSUPPORTED"],
        }
        out["totals"]["A"] = sum(out["counts"]["A"].values())
        out["totals"]["B"] = sum(out["counts"]["B"].values())
        out["parse_path"] = "strict"
    else:
        # Partial — fall back further
        out["warnings"].append(f"strict regex matched {len(found)}/6 fields")

    # Try to extract winner from the WINNER: line (still part of the strict
    # regex layer — Qwen3 fallback is run separately by run_verdict_v3)
    wm = _WINNER_RE.search(text)
    if wm:
        out["winner"] = wm.group(1).upper()
    elif "WINNER" in text.upper():
        # Look for an A or B token after the keyword
        idx = text.upper().find("WINNER")
        tail = text[idx:idx + 80].upper()
        for ch in tail:
            if ch in "AB":
                out["winner"] = ch
                break

    # Tie detection + consistency check
    tA = out["totals"]["A"]
    tB = out["totals"]["B"]
    if tA is not None and tB is not None:
        if tA == tB:
            out["is_tie"] = True
            out["winner"] = None  # caller must treat as keep-original
            out["warnings"].append(f"tie A={tA} B={tB} → keep-original")
        else:
            derived_winner = "A" if tA < tB else "B"
            if out["winner"] is None:
                out["winner"] = derived_winner
            elif out["winner"] != derived_winner:
                out["warnings"].append(
                    f"winner '{out['winner']}' contradicts totals A={tA} B={tB}; "
                    f"using derived winner '{derived_winner}'"
                )
                out["winner"] = derived_winner

    return out


def _q32_fallback_winner(analysis: str) -> str | None:
    """Use Qwen3-32B to extract a WINNER pick from a free-form verdict
    output. Returns 'A', 'B', or None."""
    user = f"""A medical expert produced an analysis comparing two answers (A and B). Here is their output:

---
{analysis[:1500]}
---

Did they pick A or B as the winner? If unclear, reply UNCLEAR.

Reply with exactly one word: A, B, or UNCLEAR.
/no_think"""
    for attempt in range(3):
        try:
            r = requests.post(QWEN32B_URL, json={
                "model": "Qwen/Qwen3-32B-MLX-bf16",
                "messages": [
                    {"role": "system", "content": "Extract a one-letter answer."},
                    {"role": "user", "content": user},
                ],
                "max_tokens": 16, "temperature": 0.0,
            }, timeout=60)
            text = r.json()["choices"][0]["message"]["content"].strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip().upper()
            if text.startswith("A"):
                return "A"
            if text.startswith("B"):
                return "B"
            return None
        except Exception as e:
            print(f"  q32 winner retry {attempt+1}/3: {e}", flush=True)
            time.sleep(3)
    return None


# ---------------------------------------------------------------------------
# CoVe critique parser
# ---------------------------------------------------------------------------

def parse_critique(raw: str) -> list[str]:
    """Extract the unsupported-claim quotes. Returns [] if none."""
    if not raw:
        return []
    text = _strip_md(raw)
    if "NONE" in text.upper().splitlines()[0:3] if text.upper().splitlines() else False:
        return []
    # Match bullet-quoted lines
    out: list[str] = []
    for m in re.finditer(r'^\s*[-*\u2022]\s*["\u201c](.+?)["\u201d]\s*$',
                         text, re.MULTILINE):
        q = m.group(1).strip()
        if q and q.lower() != "none":
            out.append(q)
    return out


# ---------------------------------------------------------------------------
# Two-round CoVe regen
# ---------------------------------------------------------------------------

def cove_two_round_regen(note: str, question: str, port: int) -> dict:
    """Round 1: regen with V3 prompt. Round 1.5: critique unsupported claims.
    Round 2: regen again, dropping unsupported claims.

    Returns:
      {
        round1: {raw, parsed, n_verified, parse_ok, evidence_verified},
        critique: {raw, unsupported_claims},
        round2: {raw, parsed, n_verified, parse_ok, evidence_verified},
        final: parsed answer text from round2 (or round1 if round2 failed)
      }
    """
    # ---- Round 1 ----
    user1 = REGEN_V3_TMPL.format(note=note, question=question)
    raw1 = vllm_chat(REGEN_SYS, user1, port, max_tokens=1024, temperature=0.0)
    parsed1 = parse_premises_output(raw1)
    parsed1["evidence_verified"] = verify_evidence_quotes(parsed1["evidence_quotes"], note)
    parsed1["n_verified"] = sum(1 for v in parsed1["evidence_verified"] if v)

    # ---- Critique ----
    if parsed1.get("answer"):
        crit_user = CRITIQUE_TMPL.format(
            note=note, question=question,
            round1_answer=parsed1["answer"][:1500],
        )
        crit_raw = vllm_chat(CRITIQUE_SYS, crit_user, port,
                             max_tokens=512, temperature=0.0)
    else:
        crit_raw = ""
    unsupported = parse_critique(crit_raw)

    # ---- Round 2 ----
    if unsupported:
        unsupported_block = "\n".join(f'  - "{u[:200]}"' for u in unsupported[:5])
        user2 = REGEN_ROUND2_TMPL.format(
            note=note, question=question,
            unsupported_block=unsupported_block,
        )
        raw2 = vllm_chat(REGEN_SYS, user2, port, max_tokens=1024, temperature=0.0)
        parsed2 = parse_premises_output(raw2)
        parsed2["evidence_verified"] = verify_evidence_quotes(parsed2["evidence_quotes"], note)
        parsed2["n_verified"] = sum(1 for v in parsed2["evidence_verified"] if v)
    else:
        # Nothing to drop — round 2 = round 1
        raw2 = raw1
        parsed2 = parsed1

    # Final = round 2 answer if it parsed, else round 1
    if parsed2.get("answer"):
        final_text = parsed2["answer"]
    elif parsed1.get("answer"):
        final_text = parsed1["answer"]
    else:
        final_text = raw2 or raw1

    return {
        "round1": {"raw": raw1, "parsed": parsed1, "n_verified": parsed1["n_verified"],
                   "parse_ok": parsed1["parse_ok"]},
        "critique": {"raw": crit_raw, "unsupported_claims": unsupported},
        "round2": {"raw": raw2, "parsed": parsed2, "n_verified": parsed2["n_verified"],
                   "parse_ok": parsed2["parse_ok"]},
        "final": final_text,
    }


# ---------------------------------------------------------------------------
# Verdict V3 wrapper
# ---------------------------------------------------------------------------

def run_verdict_v3(fold: int, idx: int, note: str, question: str,
                   answer_orig: str, answer_corrected: str,
                   *, port: int) -> dict:
    """Single-call verdict V3 with DUAL PARSING for the parser bake-off.

    On every item we run BOTH:
      1. parse_verdict_v3 (strict regex on the structured 6-count + WINNER format)
      2. _q32_fallback_winner (Qwen3-32B free-form winner extraction)

    The audit log records both results and an `agreement` flag, so after the
    pilot we can compute:
      - regex success rate
      - Qwen3-32B success rate
      - agreement rate when both succeed
      - which one was correct on disagreements (using GT later)

    The accept_correction decision uses regex if available (strict-or-tie),
    falling back to qwen3 only when regex is unparseable. Tie → keep original.
    """
    rng = random.Random(42 + (fold << 16) + idx)
    orig_in_a = rng.random() > 0.5
    ans_a = answer_orig if orig_in_a else answer_corrected
    ans_b = answer_corrected if orig_in_a else answer_orig
    user = VERDICT_V3_TMPL.format(
        note=note, question=question,
        answer_a=ans_a[:1500], answer_b=ans_b[:1500],
    )
    raw = vllm_chat(VERDICT_SYS, user, port, max_tokens=400, temperature=0.0)

    # ---- Parser 1: strict regex ----
    regex_parsed = parse_verdict_v3(raw)
    regex_winner = regex_parsed.get("winner")  # may be None on tie or fail
    # "regex_success" means we got a usable A/B winner (regardless of whether
    # all 6 count fields parsed)
    regex_success = regex_winner in ("A", "B")
    regex_full_parse = regex_parsed["parse_path"] == "strict"  # all 6 counts

    # ---- Parser 2: Qwen3-32B (run UNCONDITIONALLY for the bake-off) ----
    q3_winner = _q32_fallback_winner(raw)
    q3_success = q3_winner in ("A", "B")

    # ---- Agreement flag ----
    # Comparable iff both parsers produced a non-tie A/B winner.
    if regex_success and q3_success and not regex_parsed.get("is_tie"):
        agreement = (regex_winner == q3_winner)
    else:
        agreement = None  # not comparable (one or both failed, or tie)

    # ---- Final accept decision (per-user direction Apr 2026):
    #   - tie                           → keep original
    #   - regex/q3 disagreement         → keep original (cannot tell which the
    #                                     model "really meant"; the model
    #                                     probably hallucinated/self-contradicted;
    #                                     log it for pattern analysis)
    #   - regex success and agreement   → use regex winner
    #   - regex unparseable, q3 success → use q3 winner
    #   - both fail                     → keep original
    corrected_slot = "B" if orig_in_a else "A"
    final_winner = None
    final_source = "none"

    if regex_parsed.get("is_tie"):
        accept = False
        final_source = "regex_tie"
    elif (regex_winner in ("A", "B") and q3_winner in ("A", "B")
          and regex_winner != q3_winner):
        accept = False
        final_source = "disagreement_keep_original"
    elif regex_winner in ("A", "B"):
        final_winner = regex_winner
        final_source = "regex"
        accept = (final_winner == corrected_slot)
    elif q3_winner in ("A", "B"):
        final_winner = q3_winner
        final_source = "q32_fallback"
        accept = (final_winner == corrected_slot)
    else:
        accept = False
        final_source = "none"

    return {
        "variant": "verdict_v3_3count_dualparse",
        "orig_in_slot_A": orig_in_a,
        "raw": raw,
        "regex_parsed": regex_parsed,
        "regex_winner": regex_winner,
        "regex_success": regex_success,
        "regex_full_parse": regex_full_parse,
        "q3_winner": q3_winner,
        "q3_success": q3_success,
        "agreement": agreement,  # True / False / None (not comparable)
        "final_winner": final_winner,
        "final_source": final_source,
        "corrected_slot": corrected_slot,
        "accept_correction": accept,
    }


# ---------------------------------------------------------------------------
# Per-item runner
# ---------------------------------------------------------------------------

def run_one(item: dict, notes: dict, *, port: int, args, log: AuditLog) -> None:
    fold, idx = item["fold"], item["idx"]
    if log.has(fold, idx) and not args.force:
        return
    note = notes.get(str(item["patient_id"]), "")
    if not note:
        return

    rec = make_record(fold, idx, item={
        "question": item["question"],
        "patient_id": item["patient_id"],
        "ground_truth": item["ground_truth"],
        "note": note,
        "original_answer": item["model_answer"],
        "label": item["label"],
    })

    j_orig = judge_call(note, item["question"], item["ground_truth"],
                        item["model_answer"], n=1, temperature=0.0)
    rec["judge_orig"] = {
        "label": j_orig["label"],
        "raw": j_orig["raws"][0] if j_orig["raws"] else "",
        "label_T01_legacy": int(item["label"]),
    }
    eval_orig = j_orig["label"] if j_orig["label"] is not None else int(item["label"])

    # ---- 2-round CoVe regen ----
    try:
        cove = cove_two_round_regen(note, item["question"], port)
    except Exception as e:
        print(f"  cove err ({fold},{idx}): {e}", flush=True)
        rec["correction"] = {"skipped_reason": "cove_failed", "error": str(e)}
        rec["verdict"] = None
        rec["judge_corrected"] = None
        rec["outcome"] = {"action": "keep", "delta": 0, "final_eval": eval_orig}
        log.write(rec)
        return

    rec["correction"] = {
        "skipped_reason": None,
        "method": "cove_2round_regen_v3",
        "round1_parse_ok": cove["round1"]["parse_ok"],
        "round1_n_verified": cove["round1"]["n_verified"],
        "round1_raw": cove["round1"]["raw"][:1500],
        "critique_unsupported": cove["critique"]["unsupported_claims"],
        "critique_raw": cove["critique"]["raw"][:1000],
        "round2_parse_ok": cove["round2"]["parse_ok"],
        "round2_n_verified": cove["round2"]["n_verified"],
        "round2_raw": cove["round2"]["raw"][:1500],
        "proposed": cove["final"],
    }

    # ---- Verdict V3 ----
    try:
        v = run_verdict_v3(fold, idx, note, item["question"],
                           item["model_answer"], cove["final"], port=port)
    except Exception as e:
        print(f"  verdict err ({fold},{idx}): {e}", flush=True)
        v = None

    rec["verdict"] = v

    if not v or not v.get("accept_correction"):
        rec["judge_corrected"] = None
        rec["outcome"] = {"action": "kept_original", "delta": 0, "final_eval": eval_orig}
        log.write(rec)
        return

    # ---- judge_corrected (oracle) ----
    time.sleep(0.5)
    j_cor = judge_call(note, item["question"], item["ground_truth"], cove["final"],
                       n=1, temperature=0.0)
    rec["judge_corrected"] = {
        "label": j_cor["label"],
        "raw": j_cor["raws"][0] if j_cor["raws"] else "",
    }
    eval_cor = j_cor["label"] if j_cor["label"] is not None else eval_orig
    delta = (1 if eval_cor == 1 and eval_orig == 0
             else (-1 if eval_cor == 0 and eval_orig == 1 else 0))
    rec["outcome"] = {"action": "corrected", "delta": delta, "final_eval": eval_cor}
    log.write(rec)


# ---------------------------------------------------------------------------
# Per-model driver
# ---------------------------------------------------------------------------

def run_model(model_alias: str, *, args, notes: dict) -> dict:
    cfg = MODELS[model_alias]
    print(f"\n{'=' * 70}")
    print(f"REGEN V3 + COVE 2-ROUND PILOT — {model_alias}")
    print(f"{'=' * 70}")
    served = served_model_id(args.port).lower()
    print(f"  vLLM serving: {served}")
    if cfg["expected_id_substring"] not in served:
        print(f"  ❌ wrong model loaded ({cfg['expected_id_substring']} not in {served}); skipping")
        return {"model": model_alias, "skipped": True}

    # Apply per-model chat_template_kwargs (e.g. Qwen3 enable_thinking=False)
    ctk = cfg.get("chat_template_kwargs")
    set_default_chat_template_kwargs(ctk)
    if ctk:
        print(f"  chat_template_kwargs: {ctk}")

    if args.random_per_fold:
        items = sample_random_per_fold(cfg["step8_dir"],
                                       n_per_fold=args.random_per_fold,
                                       seed=args.seed)
        n_w = sum(1 for i in items if i["label"] == 0)
        n_c = sum(1 for i in items if i["label"] == 1)
        print(f"  Random sample: {len(items)} items ({n_w}W + {n_c}C)")
    else:
        items = sample_test_items(cfg["step8_dir"], args.n_wrong, args.n_correct,
                                  seed=args.seed)
        print(f"  Stratified sample: {len(items)} items")

    model_dir = OUT_DIR / cfg["step8_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_dir / args.audit_name
    log = AuditLog(log_path)
    print(f"  Audit log: {log_path}")
    print(f"  Already done: {len(log.all())}")

    for i, item in enumerate(items, 1):
        try:
            run_one(item, notes, port=args.port, args=args, log=log)
        except Exception as e:
            print(f"  ❌ ({item['fold']},{item['idx']}): {e}", flush=True)
            continue
        if i % 5 == 0:
            done = log.all()
            actions = Counter((r.get("outcome") or {}).get("action", "?") for r in done)
            fixes = sum(1 for r in done
                        if (r.get("outcome") or {}).get("action") == "corrected"
                        and (r.get("outcome") or {}).get("delta") == 1)
            brks = sum(1 for r in done
                       if (r.get("outcome") or {}).get("action") == "corrected"
                       and (r.get("outcome") or {}).get("delta") == -1)
            print(f"  [{i}/{len(items)}] log={len(done)} {dict(actions)} fixes={fixes} brk={brks}",
                  flush=True)

    done = log.all()
    actions = Counter((r.get("outcome") or {}).get("action", "?") for r in done)
    fixes = sum(1 for r in done
                if (r.get("outcome") or {}).get("action") == "corrected"
                and (r.get("outcome") or {}).get("delta") == 1)
    brks = sum(1 for r in done
               if (r.get("outcome") or {}).get("action") == "corrected"
               and (r.get("outcome") or {}).get("delta") == -1)
    return {
        "model": model_alias,
        "n_items": len(done),
        "actions": dict(actions),
        "fixes": fixes,
        "breaks": brks,
        "log_path": str(log_path),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="qwen2.5,llama3,deepseek,qwen3")
    p.add_argument("--n-wrong", type=int, default=10)
    p.add_argument("--n-correct", type=int, default=30)
    p.add_argument("--random-per-fold", type=int, default=None,
                   help="if set, sample N random items per fold (overrides n-wrong/n-correct)")
    p.add_argument("--audit-name", default="regen_v3_audit.jsonl",
                   help="filename for the per-model audit log")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--port", type=int, default=8003)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    selected = [m.strip() for m in args.models.split(",") if m.strip() in MODELS]
    if not selected:
        print(f"!! no valid models")
        return 1

    notes = _load_notes_lookup()
    summaries = []
    for m in selected:
        try:
            s = run_model(m, args=args, notes=notes)
        except Exception as e:
            print(f"❌ {m}: {e}", flush=True)
            s = {"model": m, "error": str(e)}
        summaries.append(s)

    print(f"\n{'=' * 70}")
    print(f"REGEN V3 PILOT SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Model':<14} {'N':>5} {'fix':>5} {'brk':>5} {'corr':>5} {'kept':>5} {'keep':>5}")
    for s in summaries:
        if s.get("skipped") or s.get("error"):
            print(f"{s['model']:<14} SKIPPED/ERROR")
            continue
        a = s["actions"]
        kept = a.get("kept_original", 0)
        keep = a.get("keep", 0)
        corr = a.get("corrected", 0)
        print(f"{s['model']:<14} {s['n_items']:>5} {s['fixes']:>5} {s['breaks']:>5} {corr:>5} {kept:>5} {keep:>5}")

    out_path = OUT_DIR / "regen_v3_summary.json"
    out_path.write_text(json.dumps(summaries, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
