"""T0 Anchor correction — Stage II of EHRNoteQA self-correction project.

Five regen sub-variants on detection-flagged items, all outputs retained.
Stage III (verdict design) consumes T0 outputs.

Modules:
    sub_variants : 5 prompt templates (a..e) + thinking-mode rules per target
    data_loader  : join zeroshot CSV with detection JSONL; filter INCORRECT
    runner       : per-item correction runner with template substitution
    step0_probe  : Step-0 token-budget probe (5 items x 5 sub-variants)
"""
