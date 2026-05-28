"""Prompt Engineering module — reusable across all ICL steps.

Upstream module that produces a ranked shortlist of candidate prompts at pilot
scale (40 items). Downstream ICL steps (detection, verdict, regeneration,
error location, error correction) consume the top-N candidates and run them
at real scale (e.g., 962 items).

See Notion page "Claude: Module: Prompt Engineering Iteration" for design,
conventions, and output layout.

Usage sketch (not yet end-to-end):
    from ichl.prompt_engineering import optimizer
    result = optimizer.run(
        step='detection',
        base_prompt_pool='prompts/detection/seeds.yaml',
        variation_pool='variations/general.yaml',
        tool_model='gpt-4o',
        metric='selectivity',
        pilot_data='data/pilots/detection_pilot_40.jsonl',
        top_n_candidates=3,
        max_rounds=10,
        epsilon=0.02,
        out_dir='output/ichl/detection/runs/20260422_pe/',
    )
"""
