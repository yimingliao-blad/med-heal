"""Error Location pipeline — produce contradiction narratives blind from (note, question, zs_answer).

Per `Claude: Plan: Error Location (Contradiction Narrative) — Qwen2.5` (2026-04-28):
- Step 0: token-budget probe (max_tokens via p95 formula)
- Step 1: GPT-4o gold narrative generation (one-time labeling)
- Step 2: 5-item calibration anchor (hand-verify golds)
- Step 3: format smoke + Qwen2.5 locator iteration on fold_0
- Step 4: 5-fold lockdown
- Step 5: GPT-4o spot-check vs Qwen3-235B comparator (10-item)
- Step 6: report

All LLM calls pass through `truncation_detector.detect_truncation` per
`[Workflow] Truncation Detection on Every LLM Output`.

Inheritance manifest (per Implementation Discipline Rule 1):
- vllm_call           ← src/ichl/retrieval_study/raicl_pilot.py (extended for full truncation_report)
- detect_truncation   ← src/ichl/prompt_engineering/correction/truncation_detector.py
- MLXOpenAIClient     ← src/ichl/clients/mlx_openai_client.py (Qwen3-235B comparator at C=1)
- GPT-4o client       ← OpenAI() construction pattern from gpt4o_stage1_binary_judge.py (prompts new)
- vllm_manager        ← src/ichl/common/vllm_manager.py (lifecycle for Qwen2.5 target)
"""
