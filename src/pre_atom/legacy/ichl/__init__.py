"""ichl — In-Context Iterative Correction / Learning framework.

Top-level package for the refactored EHRNoteQA work. Organized by ICL step:
  - detection
  - verdict
  - regeneration
  - error location
  - error correction

Plus shared infrastructure:
  - clients/      LLM client wrappers (OpenAI, MLX, vLLM)
  - common/       config loader, shared utilities
  - prompt_engineering/   reusable prompt-optimization loop

See Notion database "EHRNoteQA ICL Plan" for principles, plans, and findings.
"""
__version__ = "0.1.0"
