"""Stage III Verdict — pick the better candidate (A vs B) given (note, question).

Plan: https://app.notion.com/p/3526be46cf3c81b5bdb1cbfaf39d8b18
Master: Three-Stage Design plan, Stage III.

Per user 2026-04-30:
- Cells from existing 5-target step8 + 4corrector judgments (no new compute).
- Bare PICK: A | B | UNCERTAIN output (no reasoning).
- Position randomized (seed=42), bias monitored.
- DS-R1 messy output handled by inherited regex+LLM-arbiter parser.
"""
