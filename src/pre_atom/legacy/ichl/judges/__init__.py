"""Production binary-correctness judges for EHRNoteQA evaluation.

Each judge takes (question, ground_truth, model_answer, note) and returns a
dict with at least:
  - label: int in {0, 1} or None
  - content: raw model text
  - latency_s, completion_tokens, prompt_tokens
  - finish_reason
  - truncation_certain: bool (per `truncation_detector`)

Available judges:
  - GPT4oJudge      — Stage-1 binary prompt; 92% human agreement, κ=0.75. High-stakes.
  - MagistralJudge  — Local vLLM, M4 prompt; 85.1% test agreement, κ=0.691. Free, fast.

Use MagistralJudge for development-iteration sweeps where ~85% agreement is
good enough; keep GPT4oJudge for final stage-1 evaluation.
"""
from ichl.judges.magistral_judge import MagistralJudge

__all__ = ["MagistralJudge"]
