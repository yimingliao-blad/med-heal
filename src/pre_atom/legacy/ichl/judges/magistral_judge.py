"""Magistral-Small-2509-AWQ binary-correctness judge (M4 production prompt).

Result of the prompt-engineering iteration completed 2026-04-25. M4 was the
GPT-4o-revised version of the M1 charitable rules — see Notion finding
"Claude: Finding: Magistral M4 Judge — Local LLM as Secondary Judge".

Performance (frozen prompt, frozen model, frozen parser):
    Dev (n=298, stratified 50/50)   : 81.5%   κ=0.631   TPR=81.3%   TNR=81.8%
    Test (n=415, natural 252/163)   : 85.1%   κ=0.691   TPR=84.9%   TNR=85.3%
    Latency on local vLLM (24 GB)   : ~1.3 s/call  (~4 s with longer prompts)
    Cost                            : free (local)

Use this judge for:
    - prompt-engineering iteration where ~85% agreement is good enough
    - batch sweeps over many target models / many conditions
    - quick correctness screening during development

Do NOT use for high-stakes final evaluation — keep GPT-4o Stage-1 for that
(92% human agreement, κ=0.75, ~$0.008/call).

Server requirements:
    Magistral-Small-2509-AWQ-4bit served by vLLM at http://localhost:8003
    (or wherever; pass `base_url` to the constructor). The cyankiwi AWQ-4bit
    weights work with vLLM 0.12 + `--tokenizer-mode mistral`. Boot command:

        .venv/bin/python -m vllm.entrypoints.openai.api_server \\
          --model cyankiwi/Magistral-Small-2509-AWQ-4bit \\
          --served-model-name Magistral-Small-2509-AWQ \\
          --tokenizer-mode mistral --max-model-len 14336 \\
          --gpu-memory-utilization 0.93 --port 8003 --dtype auto

Truncation detection is applied per Claude: Principle: Truncation Detection
on Every LLM Output.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from ichl.prompt_engineering.correction.truncation_detector import detect_truncation

DEFAULT_BASE_URL = "http://localhost:8003/v1"
DEFAULT_MODEL = "Magistral-Small-2509-AWQ"
DEFAULT_MAX_TOKENS = 256        # M4 outputs ≤2 tokens; 256 is wildly safe (verified on 415-item test)

SYSTEM = "You are a medical expert evaluating an AI model's answer to a clinical question."

# === LOCKED M4 PRODUCTION RULES ===
# Source: GPT-4o critique of M1 (2026-04-25), light edit. DO NOT modify without re-validating.
M4_RULES = [
    "Output 0 if any specific fact (medication, dose, diagnosis, timing, procedure, lab value) "
    "contradicts the ground truth, unless the discrepancy is clinically insignificant.",
    "Output 0 if the answer omits a fact that the question specifically asks about.",
    "Output 0 if the answer addresses a different aspect than the question asked.",
    "Output 0 if the answer hedges with multiple uncommitted options.",
    "Output 1 if all specific claims align with the ground truth and the answer addresses the "
    "question — paraphrases, synonyms, different orderings, or different units that convey the "
    "same clinical fact all count as alignment. Correct additional context or detail that aligns "
    "with the clinical scenario should be considered positively.",
    "Output 0 if the answer includes additional, incorrect information that introduces clinically "
    "significant errors or contradictions not present in the ground truth.",
    "Be charitable on form, strict on content: do not penalize an answer for using different "
    "wording, abbreviations, generic vs. brand names, or restated phrasing — penalize only when "
    "the underlying clinical fact differs from the ground truth. Correctness in clinical "
    "interpretation takes precedence over verbatim matching.",
]


@dataclass
class JudgeResult:
    label: int | None
    content: str
    finish_reason: str | None
    completion_tokens: int | None
    prompt_tokens: int | None
    latency_s: float
    truncation_certain: bool
    truncation_signals: list[str]
    error: str | None = None


def _build_user(question: str, ground_truth: str, model_answer: str, note: str) -> str:
    rules_block = "\n".join(f"{i}. {r}" for i, r in enumerate(M4_RULES, 1))
    return f"""When judging, apply the following rules:

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


def _parse_digit(text: str) -> int | None:
    if not text:
        return None
    s = text.strip()
    if not s:
        return None
    m = re.search(r"[01]", s)
    return int(m.group(0)) if m else None


class MagistralJudge:
    """Local-vLLM Magistral M4 binary judge.

    Example
    -------
        judge = MagistralJudge()
        r = judge.judge(
            question="What was the patient's discharge medication?",
            ground_truth="Aspirin 81 mg PO daily",
            model_answer="The patient was discharged on 81 mg aspirin once a day.",
            note="...full discharge summary here...",
        )
        print(r.label, r.latency_s)   # 1, 1.4
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL,
                 model: str = DEFAULT_MODEL,
                 default_max_tokens: int = DEFAULT_MAX_TOKENS,
                 timeout: int = 120,
                 max_retries: int = 3):
        self.base_url = base_url
        self.model = model
        self.default_max_tokens = default_max_tokens
        self.max_retries = max_retries
        self.client = OpenAI(base_url=base_url, api_key="not-needed", timeout=timeout)

    def judge(self, *, question: str, ground_truth: str, model_answer: str,
              note: str = "", max_tokens: int | None = None) -> JudgeResult:
        """Score a single (Q, GT, MA, note) tuple. Returns JudgeResult."""
        mt = max_tokens or self.default_max_tokens
        user = _build_user(question, ground_truth, model_answer, note)
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            t0 = time.monotonic()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": user}],
                    temperature=0.0, max_tokens=mt,
                )
                lat = time.monotonic() - t0
                msg = resp.choices[0].message
                content = getattr(msg, "content", None) or ""
                fin = resp.choices[0].finish_reason
                usage = resp.usage
                report = detect_truncation(
                    raw_response=content, text_clean=content,
                    finish_reason=fin,
                    usage={"completion_tokens": usage.completion_tokens if usage else None,
                           "prompt_tokens": usage.prompt_tokens if usage else None},
                    max_tokens=mt, target=self.model, sub_variant="M4",
                )
                return JudgeResult(
                    label=_parse_digit(content),
                    content=content,
                    finish_reason=fin,
                    completion_tokens=usage.completion_tokens if usage else None,
                    prompt_tokens=usage.prompt_tokens if usage else None,
                    latency_s=round(lat, 2),
                    truncation_certain=report.is_truncated_certain,
                    truncation_signals=report.fired_signals(),
                )
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 + attempt * 3)
        return JudgeResult(
            label=None, content="", finish_reason="ERROR",
            completion_tokens=None, prompt_tokens=None,
            latency_s=-1.0, truncation_certain=False, truncation_signals=[],
            error=str(last_err)[:300] if last_err else "unknown",
        )

    def judge_batch(self, items: list[dict[str, Any]], *,
                    note_lookup: dict[str, str] | None = None,
                    max_workers: int = 4,
                    progress_every: int = 25) -> list[JudgeResult]:
        """Batch-judge a list of dicts.

        Each item dict needs: question, ground_truth, model_answer.
        If `note_lookup` is provided, item['patient_id'] is used as key to
        look up the discharge summary; otherwise item['note'] is used directly.
        """
        from concurrent.futures import ThreadPoolExecutor

        def _one(it: dict[str, Any]) -> JudgeResult:
            note = ""
            if note_lookup is not None:
                note = note_lookup.get(str(it.get("patient_id", "")), "")
            elif "note" in it:
                note = it["note"]
            return self.judge(
                question=it["question"],
                ground_truth=it["ground_truth"],
                model_answer=it["model_answer"],
                note=note,
            )

        results: list[JudgeResult] = []
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for i, r in enumerate(ex.map(_one, items), 1):
                results.append(r)
                if progress_every and i % progress_every == 0:
                    dt = time.monotonic() - t0
                    eta = dt * (len(items) - i) / i
                    print(f"  judge_batch: {i}/{len(items)}  elapsed={dt:.0f}s  eta={eta:.0f}s")
        return results
