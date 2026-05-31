#!/usr/bin/env python3
"""
Module 3 — Stronger correction signal.

Inputs (from a successful detection step):
  detection_final = {verdict, error_type, error_statement, correct_statement}
  the patient's discharge note
  the original answer
  the original question

Pipeline:
  1. Pull top-3 sentence-level spans from THIS patient's note that match
     the detection's `error_statement` and/or `correct_statement`
     (note_span_index.topk_spans).
  2. If no span has similarity >= τ → REFUSE to correct (no signal, no action).
  3. Build a rule-based correction prompt for the detected error type and
     slot in the retrieved spans + (optionally) one BM-pool fix-example
     demonstrating the wrong→corrected pattern.
  4. Generate K candidate corrections at temp=0.7 (multi-sample).
  5. Return all candidates + which one to pick (best-of-K is decided in the
     verdict module).

The BM pool slot is gated by the Module 3a audit decision: if the audit
verdict is "DROP", the few-shot slot is empty.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import requests

from note_span_index import topk_spans

SOURCE_ROOT = Path(os.environ.get("PRE_ATOM_SOURCE_REPO_ROOT", Path(__file__).resolve().parents[5]))
RUN_ROOT = Path(os.environ.get("PRE_ATOM_PROJECT_ROOT", SOURCE_ROOT))
PROJECT_ROOT = SOURCE_ROOT
POOL_DIR = PROJECT_ROOT / "workspace" / "self_critique" / "data" / "bm_atomic_pool"
CONTRAST_POOL_DIR = PROJECT_ROOT / "workspace" / "self_critique" / "data" / "bm_contrast_pool"
POOL_AUDIT_PATH = PROJECT_ROOT / "output" / "step9_v2" / "bm_pool_audit.json"

# ---------- Rule templates ----------

COR_CONTRADICTION = """Discharge summary:
{note}

Question: {question}

YOUR PREVIOUS ANSWER (which contained an error):
{original_answer}

You made a CONTRADICTION error. Specifically:
- you claimed: "{error_statement}"
- the notes actually state: "{correct_statement}"

Relevant note spans (verbatim quotes from THIS patient's notes):
{spans_block}
{pool_block}
Re-answer the question using ONLY information from the notes. Be direct and
correct the contradicted claim. Reply in 1-3 sentences."""

COR_OMISSION = """Discharge summary:
{note}

Question: {question}

YOUR PREVIOUS ANSWER (which omitted critical information):
{original_answer}

You made an OMISSION error. Specifically:
- the missing fact is: "{correct_statement}"
- you said nothing about it: "{error_statement}"

Relevant note spans (verbatim quotes from THIS patient's notes):
{spans_block}
{pool_block}
Re-answer the question, this time INCLUDING the missing information from the
notes. Reply in 1-3 sentences."""

COR_QMIS = """Discharge summary:
{note}

Question: {question}

YOUR PREVIOUS ANSWER (which addressed the wrong aspect of the question):
{original_answer}

You made a QUESTION_MISALIGNMENT error. Specifically:
- the issue: "{error_statement}"
- the right focus: "{correct_statement}"

Relevant note spans (verbatim quotes from THIS patient's notes):
{spans_block}
{pool_block}
Re-answer the question, this time addressing the right aspect. Reply in
1-3 sentences."""

COR_BY_TYPE = {
    "CONTRADICTION": COR_CONTRADICTION,
    "OMISSION": COR_OMISSION,
    "QUESTION_MISALIGNMENT": COR_QMIS,
}

# ---------- Pool fix-example retrieval ----------

_pool_cache: dict[int, list[dict]] = {}


def _load_pool(fold: int) -> list[dict]:
    if fold in _pool_cache:
        return _pool_cache[fold]
    f = POOL_DIR / f"fold_{fold}_atoms.json"
    if not f.exists():
        return []
    pool = json.loads(f.read_text())
    _pool_cache[fold] = pool
    return pool


def pool_audit_says_keep() -> bool:
    """If the BM pool audit ran and verdict was DROP, return False."""
    if not POOL_AUDIT_PATH.exists():
        return True  # default to enabled until audit runs
    try:
        d = json.loads(POOL_AUDIT_PATH.read_text())
        return d.get("verdict", "KEEP") == "KEEP"
    except Exception:
        return True


_contrast_pool_cache: dict[int, tuple[list[dict], np.ndarray]] = {}


def _load_contrast_pool(fold: int) -> tuple[list[dict], np.ndarray] | None:
    """Load fold-disjoint BM contrast pool (built by Step C)."""
    if fold in _contrast_pool_cache:
        return _contrast_pool_cache[fold]
    pool_f = CONTRAST_POOL_DIR / f"fold_{fold}_pool.json"
    emb_f = CONTRAST_POOL_DIR / f"fold_{fold}_question_embeddings.npy"
    if not pool_f.exists() or not emb_f.exists():
        return None
    pool = json.loads(pool_f.read_text())
    embs = np.load(emb_f)
    _contrast_pool_cache[fold] = (pool, embs)
    return pool, embs


def retrieve_contrast_example(fold: int, question: str) -> dict | None:
    """Retrieve the most question-similar entry from the BM contrast pool
    (excluding the same fold). Returns the full entry dict (question, ground_truth,
    wrong_answer, what_was_wrong, evidence_from_notes, ...) or None.

    Question-text similarity is the right metric for this pool — we want a
    similar clinical question, not a similar wrong-statement surface.
    """
    loaded = _load_contrast_pool(fold)
    if not loaded:
        return None
    pool, embs = loaded
    if not pool:
        return None
    # Lazy embedder import to avoid loading GTR-T5 unless we use it
    from sentence_transformers import SentenceTransformer
    global _question_embedder
    try:
        _question_embedder
    except NameError:
        _question_embedder = SentenceTransformer(
            "sentence-transformers/gtr-t5-base", device="cpu")
    qemb = _question_embedder.encode([question],
                                      normalize_embeddings=True,
                                      show_progress_bar=False)
    sims = (embs @ qemb.T).flatten()
    top = int(np.argmax(sims))
    entry = dict(pool[top])
    entry["retrieval_sim"] = float(sims[top])
    return entry


def retrieve_pool_fix_example(fold: int, error_type: str, error_statement: str) -> dict | None:
    """Pull ONE BM pool atom whose (text_raw, gt_atom_raw) pair is intended as
    a worked example of "wrong → corrected". Filter by main_error_type, then
    rank by simple lexical similarity to the query (no embedding model needed
    here — semantic similarity is not the point, the point is to show the
    *form* of an atomic correction)."""
    if not pool_audit_says_keep():
        return None
    pool = _load_pool(fold)
    if not pool:
        return None
    type_map = {
        "CONTRADICTION": "factual_error",
        "QUESTION_MISALIGNMENT": "factual_error",
        "OMISSION": "omission",
    }
    target = type_map.get(error_type, "factual_error")
    candidates = [a for a in pool
                  if a.get("main_error_type") == target
                  and (a.get("gt_atom_raw") or "").strip()
                  and (a.get("text_raw") or "").strip()]
    if not candidates:
        return None
    # Quick token-overlap score against the error_statement (no embedding,
    # cheap & deterministic). The audit gates correctness of the pair, the
    # ranking just needs to surface a plausible one.
    es = set(re.findall(r"\w+", (error_statement or "").lower()))
    if not es:
        atom = candidates[0]
    else:
        def score(a):
            words = set(re.findall(r"\w+", (a.get("text_raw") or "").lower()))
            return len(es & words)
        atom = max(candidates, key=score)
    return {"text_raw": atom["text_raw"], "gt_atom_raw": atom["gt_atom_raw"]}


# ---------- Spans → human-readable block ----------

def render_spans_block(spans: list[dict]) -> str:
    if not spans:
        return "(no relevant spans found)"
    lines = []
    for i, s in enumerate(spans, 1):
        lines.append(f"  {i}. \"{s['sentence']}\"  (sim={s['similarity']:.2f})")
    return "\n".join(lines)


def render_pool_block(pool_ex: dict | None) -> str:
    if not pool_ex:
        return ""
    return (
        "\nWorked example of how a similar atomic error was fixed in another patient's case "
        "(use only as a stylistic guide, not as factual content):\n"
        f"  wrong claim:    \"{pool_ex['text_raw']}\"\n"
        f"  corrected claim: \"{pool_ex['gt_atom_raw']}\"\n"
    )


# ---------- Top-level: build prompt + multi-sample call ----------

def _collect_d2_queries(detection_final: dict, question: str) -> list[str]:
    """Build the multi-query list for note-span retrieval.

    Sources, in order:
      - the question itself (always specific to the medical content)
      - every distinct reason text from the K=5 contradiction "yes" samples
      - every distinct reason text from the K=5 qmis "no" samples
      - the legacy single-string error_statement / correct_statement (for
        backwards compatibility with non-D2 callers)
    """
    queries: list[str] = []
    if question and question.strip():
        queries.append(question.strip())

    # D2-specific: pull every yes-vote sample's reason from the contradiction
    # check, and every no-vote sample's reason from the qmis check
    contra = (detection_final.get("contradiction") or {})
    qmis = (detection_final.get("qmis") or {})
    for sample in contra.get("samples", []):
        if sample.get("first_line") == "yes":
            r = (sample.get("reason") or "").strip()
            if r and r.lower() != "none":
                queries.append(r)
    for sample in qmis.get("samples", []):
        if sample.get("first_line") == "no":
            r = (sample.get("reason") or "").strip()
            if r and r.lower() not in ("none", "addresses fully"):
                queries.append(r)

    # Legacy F1-style fields (kept so non-D2 callers still work)
    cs = (detection_final.get("correct_statement") or "").strip()
    es = (detection_final.get("error_statement") or "").strip()
    if cs:
        queries.append(cs)
    if es and es not in queries:
        queries.append(es)

    # Dedup while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        key = q.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


def build_correction_prompt(detection_final: dict, note: str, question: str,
                            original_answer: str, fold: int,
                            similarity_threshold: float = 0.45,
                            *, port: int = 8003,
                            retriever: str = "union",
                            llm_k: int = 5,
                            llm_min_votes: int = 3) -> dict:
    """Returns:
        {
          'skipped_reason': None | 'unknown_error_type' | 'low_evidence_refusal',
          'error_type': str,
          'queries': [...],     # multi-query list used for retrieval
          'spans': [...],
          'pool_ex': dict | None,
          'prompt': str | None,
        }
    """
    et = (detection_final.get("error_type") or "").upper()
    if et not in COR_BY_TYPE:
        return {
            "skipped_reason": "unknown_error_type",
            "error_type": et,
            "queries": [],
            "spans": [], "pool_ex": None, "prompt": None,
        }

    queries = _collect_d2_queries(detection_final, question)

    # ---- retrieval ----
    # Default strategy "union": combine R3 (LLM cite-by-number, question-only)
    # and R2 (multi-query embedding with agreement scoring), take top-3 from
    # each, dedup. This is the bake-off-validated combination — R2 wins on
    # numeric / surface-token questions, R3 wins on inferential ones (like
    # idx=51), and the union gives the correction step ~6 sentences total.
    #
    # Other strategies kept for ablation:
    #   "embed_only"  R2 only
    #   "llm_only"    R3 only

    spans: list[dict] = []
    retriever_used = "none"
    llm_result: dict | None = None

    # R2: embedding agreement, top-5
    r2_spans: list[dict] = []
    if retriever in ("union", "embed_only"):
        r2_spans = topk_spans(note, queries, k=5, agreement_floor=0.40,
                              scoring="agreement")
        for s in r2_spans:
            s["source"] = "R2_embed_agreement"

    # R3: Qwen2.5 cite-by-number, question-only, K=5, top-5
    r3_spans: list[dict] = []
    if retriever in ("union", "llm_only", "llm_then_embed"):
        from llm_span_retrieval import llm_topk_spans
        llm_result = llm_topk_spans(note, question, "",
                                    port=port, k=llm_k,
                                    max_per_sample=8, max_topk=5)
        confident = [t for t in llm_result["top_sentences"]
                     if t["votes"] >= llm_min_votes]
        for t in confident:
            r3_spans.append({
                "sentence": t["sentence"],
                "score": float(t["votes"]) / llm_k,
                "max_sim": float(t["votes"]) / llm_k,
                "similarity": float(t["votes"]) / llm_k,
                "n_supporting": t["votes"],
                "query_hits": [],
                "source": "R3_llm_cite",
                "sentence_number": t["sentence_number"],
            })

    # Combine according to the chosen strategy
    if retriever == "embed_only":
        spans = r2_spans
        retriever_used = "R2_embed_agreement"
    elif retriever == "llm_only":
        spans = r3_spans
        retriever_used = "R3_llm_cite"
    elif retriever == "llm_then_embed":
        spans = r3_spans if r3_spans else r2_spans
        retriever_used = "R3_llm_cite" if r3_spans else "R2_embed_agreement"
    else:  # "union" (default)
        seen: set[str] = set()
        # R3 first because it tends to surface inferential matches that R2
        # misses; then R2 fills in numeric/surface-token cases.
        for cand in r3_spans + r2_spans:
            key = cand["sentence"][:80].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            spans.append(cand)
        retriever_used = "union(R3,R2)"

    # The "best similarity" semantics differ across retrievers; for the LLM
    # citation case the value is vote_fraction (0..1). The same threshold
    # default of 0.45 still makes sense as a sanity floor.
    best_sim = max((s.get("max_sim", s.get("similarity", 0.0))
                    for s in spans), default=0.0)
    if best_sim < similarity_threshold:
        return {
            "skipped_reason": "low_evidence_refusal",
            "error_type": et,
            "queries": queries,
            "spans": spans,
            "retriever_used": retriever_used,
            "llm_retrieval": llm_result,
            "pool_ex": None,
            "prompt": None,
            "best_sim": best_sim,
        }

    # ---- contrast example from rebuilt BM contrast pool (Step C) ----
    contrast_ex = retrieve_contrast_example(fold, question)

    # ---- correction prompt (factored CoVe + premises/conclusion) ----
    # The original answer is intentionally NOT included in the prompt — this
    # is the factored-CoVe principle (Dhuliawala et al. 2024). The previous
    # answer biases the regeneration toward paraphrase; clean context lets
    # the model re-answer cold from the retrieved evidence.
    from correction_prompt_v2 import build_v2_prompt
    prompt = build_v2_prompt(note=note, question=question,
                              spans=spans, contrast_example=contrast_ex)
    return {
        "skipped_reason": None,
        "error_type": et,
        "queries": queries,
        "spans": spans,
        "retriever_used": retriever_used,
        "llm_retrieval": llm_result,
        "contrast_ex": contrast_ex,
        "pool_ex": None,  # legacy v1 atom pool slot, unused
        "prompt": prompt,
        "best_sim": best_sim,
    }


def generate_corrections(prompt: str, *, port: int, k: int = 3,
                          temperature: float = 0.7, max_tokens: int = 1024) -> list[str]:
    """Generate K candidate corrections via vLLM at the given temperature.
    Uses chat-completions so it works across model families (Qwen, Llama,
    DeepSeek, Qwen3) — each model's tokenizer applies the right template."""
    from detection_format_bakeoff import vllm_chat
    sys = "You are a medical expert."
    out = []
    for _ in range(k):
        text = vllm_chat(sys, prompt, port,
                         max_tokens=max_tokens, temperature=temperature)
        out.append(text)
    return out


# ---------- Self-test on idx=51 ----------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from judge import _load_notes_lookup
    notes = _load_notes_lookup()

    # Use a synthetic detection_final to exercise the full path
    fake_detection = {
        "verdict": "INCORRECT",
        "error_type": "CONTRADICTION",
        "error_statement": "the answer says no postoperative complications occurred",
        "correct_statement": "the patient experienced urinary retention and required catheterization",
    }
    note = notes["12152580"]  # idx=51 patient
    question = "What was the main postop complication from the infraclavicular first rib resection procedure?"
    answer = ("Based on the discharge summary provided, there does not appear to be any "
              "mention of a postoperative complication from the infraclavicular first rib "
              "resection procedure. The patient was described as tolerating the procedure well "
              "and was discharged without any significant issues noted.")
    plan = build_correction_prompt(fake_detection, note, question, answer, fold=0)
    print(f"skipped_reason: {plan['skipped_reason']}")
    print(f"error_type: {plan['error_type']}")
    print(f"best_sim: {plan.get('best_sim', 0):.3f}")
    print(f"spans:")
    for s in plan["spans"]:
        print(f"  ({s['similarity']:.3f}) {s['sentence']}")
    print(f"pool_ex: {plan['pool_ex']}")
    print()
    print("=== correction prompt (first 2000 chars) ===")
    print((plan["prompt"] or "")[:2000])
