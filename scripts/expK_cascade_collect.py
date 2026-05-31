#!/usr/bin/env python3
"""Experiment K — CASCADE collector. One shared collection; project any pipeline after.

User cascade design: run gate-on-ALL and diagnoser-on-ALL (cheap, projectable), run
correction for the candidate diagnosers, run all verdicts on all corrections. Then any
(gate x diagnoser x correction x verdict) combination is a post-hoc projection on the
SAME cases — no re-running, statistically comparable, multiple candidates carried.

Per case, this collector produces:
  GATES (flag/block): union(>=1), majority(>=2), all(==3), plain_confirm, positive_confirm, none
  DIAGNOSERS (error stmt + GPT-localization): blind_plain, blind_cot, blind_cot_clean
     (round-1 paraphrase shared; round-2 = blind consistency in 3 styles)
  CORRECTION (per diagnoser, 2 arms): source_led (tight WRONG/CORRECT/EVIDENCE), raicl (retrieved fix example)
  VERDICT (per corrected, 2 variants): C3_cot, C3_strict   (adaptive computed in projection)
  JUDGE: original + every corrected (GPT-4o)

Roles honored: gate = protect absolutely-right (block-precision); diagnoser = reliable source;
correction = apply source +/- retrieved cases; verdict = shut off bad fixes.

Resumable (append per-case jsonl; skips done fold/idx). Real notes (24k). c=8, ledger.
Output: runs/expK_cascade/qwen25_nw{NW}_nc{NC}/{records.jsonl, meta.json}
Project with scripts/expK_project.py afterward.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPO = Path(os.environ.get("MED_HEAL_SOURCE_REPO", PROJECT_ROOT.parent / "llm-ehr-hallucination"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import phase2b_extract_compare_detection as P2  # noqa: E402
import expA_detection_feedback as EA  # noqa: E402  (detect_k3_union)
from llm_audit import set_ledger, log_call  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "runs" / "expK_cascade"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
POOL_DIR = SOURCE_REPO / "workspace" / "self_critique" / "data" / "bm_contrast_pool"

DIAGNOSERS = ["blind_plain", "blind_cot", "blind_cot_clean"]
CORR_ARMS = ["source_led", "raicl"]
VERDICTS = ["C3_cot", "C3_strict"]

FLAG_TAIL = "\n\nAfter all of the above, output one FINAL line exactly in this form and nothing after it:\nFLAG: YES   (if there is a real error/inconsistency/unconfirmed required fact)\nFLAG: NO    (if the answer is fine and correctly answers the question)"


def parse_flag(raw: str) -> bool:
    """Robust flag parse: key on the explicit final FLAG: YES/NO token (last occurrence)."""
    flags = re.findall(r"FLAG\s*:\s*(YES|NO)", raw or "", re.I)
    if flags:
        return flags[-1].upper() == "YES"
    return False  # no explicit flag token -> treat as not flagged (was the over-flag bug source)


def parse_verdict_letter(raw: str) -> str:
    """Prefer the explicit final-answer token (last occurrence); fall back to last-line letter.
    Fixes a 10% mis-parse where the model writes '**Final Answer: B**' then a trailing line
    like 'A or B (both correct)' that the naive last-line heuristic wrongly grabbed."""
    toks = re.findall(r"(?:FINAL\s*ANSWER|FINAL|ANSWER|CHOICE|VERDICT)\s*:?\s*\*{0,2}\s*([AB])\b", raw or "", re.I)
    if toks:
        return toks[-1].upper()
    for ln in reversed([l for l in (raw or "").splitlines() if l.strip()]):
        m = re.search(r"\b([AB])\b", ln.upper())
        if m:
            return m.group(1)
    return "A"


# ===================== GPT-4o-mini SEMANTIC JUDGE (authoritative parse) =====================
# The judge READS THE SEMANTICS of the raw and EXTRACTS the reviewer's own final decision —
# it does NOT re-judge by its own standard. Framing the raw as "the reviewer's complete
# response" (not "an evaluation to interpret") is what makes it read a bare 'A' correctly and
# follow a reviewer's final FLAG line instead of over-flagging silence. Validated on real
# outputs: verdict 18/18 on bare-letter+clean; flag clean YES 6/6, NO 6/6, self-contradiction
# 9/10 follow the reviewer. Regex is a logged cross-check (parse_divergence), NOT the authority.

PARSE_MODEL = "gpt-4o-mini"

LVERDICT_SYS = ("A reviewer compared two options labeled A and B and indicated which one they chose. "
                "You read the reviewer's COMPLETE response and report the single letter they ultimately chose.")
LVERDICT_USER = """The reviewer was instructed to choose option A or option B. Their complete response is between the markers:

<<<RESPONSE
{analysis}
RESPONSE>>>

Their response may be just a single letter, or it may reason first and end with a choice like "FINAL: A" or "Final Answer: B". Identify the reviewer's FINAL chosen option. Reply with exactly one character: A or B. If they truly state no choice, reply U."""

LFLAG_SYS = ("You report the FINAL CONCLUSION reached by a reviewer who checked a clinical answer against a "
             "discharge note. You extract THEIR decision; you do not re-judge the answer yourself.")
LFLAG_USER = """The reviewer's complete analysis is between the markers:

<<<ANALYSIS
{analysis}
ANALYSIS>>>

What did the REVIEWER ultimately decide? Defer to their own final conclusion (e.g. a closing 'FLAG: YES/NO' line, or 'INCONSISTENT: none', or 'WRONG: none'). If the reviewer noted minor concerns but still concluded the answer is acceptable, report NO. Only report YES if the reviewer concluded the answer needs a correction.
Reply with one word: YES (reviewer wants a correction), NO (reviewer concluded it is fine), or UNCLEAR (no conclusion stated)."""


def llm_verdict(raw: str, regex_candidate: str, stage: str = "") -> str:
    """Returns 'A'/'B', or 'U' when the reviewer stated no conclusion (truncated/no final line)."""
    j = P2.gpt(PARSE_MODEL, LVERDICT_SYS, LVERDICT_USER.format(analysis=(raw or "")[:6000]), 4, 0.0, False, f"parse.verdict.{stage}")
    up = (j or "").upper()
    if re.search(r"\bU\b", up) and not re.search(r"\b[AB]\b", up):
        return "U"
    m = re.search(r"\b([AB])\b", up)
    if m:
        return m.group(1)
    return regex_candidate if j is not None else regex_candidate  # API hiccup -> regex fallback


def llm_flag(raw: str, regex_candidate: bool, stage: str = "") -> bool:
    j = P2.gpt(PARSE_MODEL, LFLAG_SYS, LFLAG_USER.format(analysis=(raw or "")[:6000]), 6, 0.0, False, f"parse.flag.{stage}")
    up = (j or "").upper()
    if "UNCLEAR" in up:
        return False  # no conclusion -> not flagged (precision-favoring; avoids over-flag)
    if re.search(r"\bYES\b", up):
        return True
    if re.search(r"\bNO\b", up):
        return False
    return regex_candidate  # API hiccup -> regex fallback

# ===================== GATES =====================

PLAIN_CONFIRM_SYS = "You verify a clinical answer against the discharge note, confirming what is clearly supported."
PLAIN_CONFIRM = """Discharge note:
{note}

Question:
{question}

Answer to verify:
{answer}

Go through the answer claim by claim. End with two sections:
SUPPORTED: claims the note clearly supports.
UNCONFIRMED: claims the note does NOT clearly support or that may be wrong (write "none" if every claim is supported and the answer correctly answers the question).""" + FLAG_TAIL

POSITIVE_CONFIRM_SYS = "You decide whether a clinical answer is definitely, fully correct for the question."
POSITIVE_CONFIRM = """Discharge note:
{note}

Question:
{question}

Answer:
{answer}

Is this answer DEFINITELY and FULLY correct for the exact question, with every claim supported by the note? Answer YES only if you are confident there is no error and nothing required is missing; otherwise NO.
Reply only YES or NO."""


def gate_plain_confirm(row, port) -> bool:
    raw = P2.vllm_chat(PLAIN_CONFIRM_SYS, PLAIN_CONFIRM.format(note=row["note"][:24000], question=row["question"], answer=row["original_answer"][:1500]), port, 700, 0.0, tag="gate.plain_confirm")
    return llm_flag(raw, parse_flag(raw), "gate_plain")  # GPT-4o-mini final decision (regex is first pass)


def gate_positive_confirm(row, port) -> bool:
    raw = P2.vllm_chat(POSITIVE_CONFIRM_SYS, POSITIVE_CONFIRM.format(note=row["note"][:24000], question=row["question"], answer=row["original_answer"][:1500]), port, 8, 0.0, tag="gate.positive_confirm")
    yes = bool(re.search(r"\bYES\b", (raw or "").upper()))
    return not yes  # flagged = NOT definitely-correct


# ===================== DIAGNOSERS (blind two-round) =====================

R1_SYS = "You restate a clinical answer as a plain list of the factual claims it makes."
R1 = """Question:
{question}

Answer:
{answer}

Restate this answer as a numbered list of the distinct factual claims it makes in response to the question. Just rephrase each claim plainly. Do not judge correctness."""

R2_PLAIN_SYS = "You check whether a set of claims is consistent with a discharge note. You are NOT shown the original answer; judge the claims on their own."
R2_PLAIN = """Discharge note:
{note}

Question:
{question}

Claims:
{claims}

Check each claim against the note: CONSISTENT or INCONSISTENT (contradicted, unsupported, or off-topic).
End with: INCONSISTENT: each inconsistent claim and what the note says instead ("none" if all consistent).""" + FLAG_TAIL

R2_COT_SYS = "You check whether a set of claims is consistent with a discharge note, reasoning step by step. You are NOT shown the original answer."
R2_COT = """Discharge note:
{note}

Question:
{question}

Claims:
{claims}

For EACH claim, reason step by step: find what the note says, then decide SUPPORTS / CONTRADICTS / SILENT. Treat SILENT as consistent unless a required fact is clearly stated differently.
End with: INCONSISTENT: each clearly-contradicted claim and what the note says instead ("none" if all consistent).""" + FLAG_TAIL

R2_COT_CLEAN_SYS = "You check claims against a discharge note step by step and write a precise correction instruction. You are NOT shown the original answer."
R2_COT_CLEAN = """Discharge note:
{note}

Question:
{question}

Claims:
{claims}

For EACH claim, reason step by step against the note (SUPPORTS / CONTRADICTS / SILENT; SILENT = consistent). Then, for the single most important contradicted claim (if any), write exactly:
WRONG: what is wrong or missing in the answer.
CORRECT: the note-supported correct fact.
EVIDENCE: the exact note sentence(s).
If every claim is consistent, write: WRONG: none""" + FLAG_TAIL


def round1(row, port) -> str:
    return P2.vllm_chat(R1_SYS, R1.format(question=row["question"], answer=row["original_answer"][:1500]), port, 500, 0.0, tag="r1.paraphrase")


def _inconsistent(raw):
    # error text = the INCONSISTENT section (for correction handoff); flag decided by GPT-4o-mini below
    m = re.search(r"INCONSISTENT\s*:?\s*(.+?)(?:\nFLAG\s*:|$)", raw or "", re.I | re.S)
    inc = (m.group(1).strip() if m else "").strip()
    return inc[:1800]


def diagnose(name, row, claims, port) -> dict[str, Any]:
    note = row["note"][:24000]
    if name == "blind_plain":
        raw = P2.vllm_chat(R2_PLAIN_SYS, R2_PLAIN.format(note=note, question=row["question"], claims=claims[:3000]), port, 700, 0.0, tag="diag.blind_plain")
        inc = _inconsistent(raw)
        rc = parse_flag(raw)
        return {"error": inc, "flagged": llm_flag(raw, rc, "blind_plain"), "flag_regex": rc, "raw": raw}
    if name == "blind_cot":
        raw = P2.vllm_chat(R2_COT_SYS, R2_COT.format(note=note, question=row["question"], claims=claims[:3000]), port, 900, 0.0, tag="diag.blind_cot")
        inc = _inconsistent(raw)
        rc = parse_flag(raw)
        return {"error": inc, "flagged": llm_flag(raw, rc, "blind_cot"), "flag_regex": rc, "raw": raw}
    # blind_cot_clean -> WRONG/CORRECT/EVIDENCE
    raw = P2.vllm_chat(R2_COT_CLEAN_SYS, R2_COT_CLEAN.format(note=note, question=row["question"], claims=claims[:3000]), port, 900, 0.0, tag="diag.blind_cot_clean")
    flag_regex = parse_flag(raw)  # explicit FLAG token, not the messy WRONG: none inline parse
    # the WRONG/CORRECT/EVIDENCE block (up to the FLAG line) is the error stmt
    block = raw
    m2 = re.search(r"(WRONG\s*:[\s\S]*?)(?:\nFLAG\s*:|$)", raw or "", re.I)
    if m2:
        block = m2.group(1)
    return {"error": block[:1800], "flagged": llm_flag(raw, flag_regex, "blind_cot_clean"), "flag_regex": flag_regex, "raw": raw}


def gpt_localize(row, error) -> bool:
    user = (f"Discharge note:\n{row['note'][:18000]}\n\nQuestion:\n{row['question']}\n\nAnswer:\n{row['original_answer'][:1500]}\n\n"
            f"Correct answer (gold):\n{row['ground_truth'][:800]}\n\nA check identified this problem:\n{error[:1500]}\n\n"
            'Does this correctly identify the real error in the answer? Return JSON: {"matches_real_error":true/false}')
    raw = P2.gpt("gpt-4o", "Judge whether a finding matches the real error. Return JSON only.", user, max_tokens=60, temperature=0.0, json_mode=True, tag="gpt.localize")
    m = re.search(r"\{[\s\S]*\}", raw or "")
    try:
        return bool(json.loads(m.group()).get("matches_real_error")) if m else False
    except Exception:
        return False


# ===================== CORRECTION =====================

CORR_SYS = ("You are a careful clinical QA assistant. Fix the previous answer using the provided error/source, "
            "grounded in the note. Do not add facts not in the note. Return only the final answer.")

_pool_cache: dict[int, Any] = {}


def load_pool(fold):
    if fold not in _pool_cache:
        import numpy as np
        pool = json.loads((POOL_DIR / f"fold_{fold}_pool.json").read_text())
        emb = np.load(POOL_DIR / f"fold_{fold}_question_embeddings.npy")
        _pool_cache[fold] = (pool, emb)
    return _pool_cache[fold]


_gtr = None


def gtr():
    global _gtr
    if _gtr is None:
        from sentence_transformers import SentenceTransformer
        _gtr = SentenceTransformer("sentence-transformers/gtr-t5-base", device="cpu")
    return _gtr


def retrieve_fix_example(row) -> str:
    import numpy as np
    pool, emb = load_pool(row["fold"])
    q = gtr().encode([row["question"] + " " + row["original_answer"][:300]], normalize_embeddings=True)[0]
    embn = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    i = int(np.argmax(embn @ q))
    e = pool[i]
    return (f"Example of a similar correction:\nQuestion: {e.get('question','')}\n"
            f"What was wrong: {e.get('what_was_wrong','')}\nCorrected answer: {e.get('ground_truth','')}")


def correct(arm, row, error, port) -> str:
    if arm == "raicl":
        ex = retrieve_fix_example(row)
        extra = f"\n\n{ex}\n"
    else:
        extra = ""
    user = (f"Question:\n{row['question']}\n\nPrevious answer:\n{row['original_answer']}\n\n"
            f"Identified error / source to fix:\n{error[:2500]}{extra}\n\n"
            "Fix the previous answer using the error/source, grounded in the note evidence it cites. If no real error is shown, keep the previous answer. Return only the final answer.")
    return P2.vllm_chat(CORR_SYS, user, port, 700, 0.0, tag=f"corr.{arm}")


# ===================== VERDICT =====================

VSYS = {"C3_cot": "You compare two clinical answers carefully, step by step, before deciding.",
        "C3_strict": "You decide which of two answers is better for a clinical question. Reply with only A or B."}
VTMPL = {
    "C3_cot": """Discharge note:
{note}

Question:
{question}

Answer A:
{a}

Answer B:
{b}

Step by step: (1) what does the question require? (2) is each claim in A supported? (3) is each claim in B supported? (4) any unsupported facts / dropped detail / wrong focus? (5) which is more correct and complete?
On the very last line write exactly 'FINAL: A' or 'FINAL: B' (one letter, nothing after it).""",
    "C3_strict": """Discharge note:
{note}

Question:
{question}

Answer A:
{a}

Answer B:
{b}

Which answer is better and more correct for the question? Reply only A or B.""",
}


def verdict_pick(name, row, original, corrected, port) -> bool:
    rng = random.Random(42 + (row["fold"] << 16) + row["idx"])
    orig_a = rng.random() > 0.5
    a, b = (original, corrected) if orig_a else (corrected, original)
    corrected_slot = "B" if orig_a else "A"
    mt = VTMPL[name].format(note=row["note"][:24000], question=row["question"], a=a[:1500], b=b[:1500])
    raw = P2.vllm_chat(VSYS[name], mt, port, 900 if name == "C3_cot" else 8, 0.0, tag=f"verdict.{name}")
    pick = llm_verdict(raw, parse_verdict_letter(raw), name)  # semantic judge (regex = logged cross-check)
    if pick == "U":
        return False  # reviewer stated no conclusion -> reject the correction (keep original, safe)
    return pick == corrected_slot  # accept correction


# ===================== orchestration =====================

def process_one(row, port, parser) -> dict[str, Any]:
    out = {k: row[k] for k in ["fold", "idx", "patient_id", "question", "ground_truth", "original_answer", "stored_label"]}
    try:
        out["judge_original"] = P2.judge(row, row["original_answer"])
        # GATES
        det = EA.detect_k3_union(row, port, parser)  # gives 3 verdicts -> union/majority/all
        ninc = sum(1 for v in det["verdicts"] if v == "INCORRECT")
        out["gates"] = {
            "union": ninc >= 1, "majority": ninc >= 2, "all": ninc >= 3,
            "plain_confirm": gate_plain_confirm(row, port),
            "positive_confirm": gate_positive_confirm(row, port),
            "none": True,
        }
        # DIAGNOSERS (round1 shared)
        claims = round1(row, port)
        out["paraphrase_chars"] = len(claims)
        diags = {}
        corrected = {}
        verdicts = {}
        for d in DIAGNOSERS:
            dd = diagnose(d, row, claims, port)
            dd["localized"] = gpt_localize(row, dd["error"]) if dd["flagged"] else False
            diags[d] = {"flagged": dd["flagged"], "error": dd["error"], "localized": dd["localized"], "raw": dd["raw"]}
            # CORRECTION per arm (only if this diagnoser flagged)
            if dd["flagged"]:
                for arm in CORR_ARMS:
                    c = correct(arm, row, dd["error"], port)
                    key = f"{d}|{arm}"
                    jc = P2.judge(row, c)
                    rec = {"corrected": c, "judge_corrected": jc}
                    # VERDICTS on this correction
                    rec["verdict"] = {v: verdict_pick(v, row, row["original_answer"], c, port) for v in VERDICTS}
                    corrected[key] = rec
        out["diagnosers"] = diags
        out["corrections"] = corrected
        return out
    except Exception as e:
        out["error"] = str(e)
        return out


def done_keys(path: Path) -> set[tuple[int, int]]:
    s = set()
    if path.exists():
        for line in path.open():
            try:
                r = json.loads(line)
                s.add((r["fold"], r["idx"]))
            except Exception:
                pass
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--n-wrong", type=int, default=-1)
    ap.add_argument("--n-correct", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--parser", choices=["gpt4o-mini", "helper-v2", "qwen35"], default="helper-v2")
    args = ap.parse_args()
    served = P2.served_model_id(args.port)
    if "qwen2" not in served.lower():
        raise RuntimeError(f"expected Qwen2.5, found {served}")
    sample = P2.load_rows(args.n_wrong, args.n_correct, args.seed)
    # GUARD against the empty-note bug: abort if any sampled case has an empty note.
    empty = [(r["fold"], r["idx"]) for r in sample if not (r.get("note") or "").strip()]
    if empty:
        raise RuntimeError(f"ABORT: {len(empty)} sampled cases have EMPTY notes (e.g. {empty[:3]}). Note loader is broken.")
    mean_chars = sum(len(r["note"]) for r in sample) // max(1, len(sample))
    print(f"NOTE GUARD OK: all {len(sample)} notes non-empty, mean {mean_chars} chars, max {max(len(r['note']) for r in sample)}", flush=True)
    out_dir = OUT_ROOT / f"qwen25_nw{args.n_wrong}_nc{args.n_correct}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / "records.jsonl"
    set_ledger(out_dir / "llm_calls.jsonl", script="expK_cascade", served=served, args=vars(args))
    done = done_keys(rec_path)
    todo = [r for r in sample if (r["fold"], r["idx"]) not in done]
    (out_dir / "meta.json").write_text(json.dumps({"diagnosers": DIAGNOSERS, "corr_arms": CORR_ARMS, "verdicts": VERDICTS, "n_total": len(sample), "n_done": len(done)}, indent=2))
    print(f"sample={len(sample)} todo={len(todo)} (resume {len(done)}) c={args.concurrency} out={out_dir}", flush=True)
    if todo:
        P2.topk_spans(todo[0]["note"], [todo[0]["question"]], k=1, scoring="agreement")
    import threading
    lock = threading.Lock()
    f = rec_path.open("a")
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(process_one, r, args.port, args.parser) for r in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            with lock:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
                f.flush()
            if i % 5 == 0 or i == len(futs):
                print(f"collected {i}/{len(todo)}", flush=True)
    f.close()
    print("CASCADE COLLECTION DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
