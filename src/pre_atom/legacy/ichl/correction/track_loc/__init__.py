"""Stage IV Track-Loc — Locator-Guided Correction.

Plan: https://app.notion.com/p/3516be46cf3c818495a7f3ed974c78d1
Two-phase: A=corrector iteration with GPT-4o gold narratives (ceiling),
           B=plug winning corrector into Qwen3 v4 locator output (pipeline).
"""
