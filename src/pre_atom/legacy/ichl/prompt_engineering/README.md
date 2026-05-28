# Prompt Engineering Module

Upstream module that produces ranked candidate prompts at pilot scale (40
items). Downstream ICL steps (detection, verdict, regeneration, error
location, error correction) consume the top-N and run them at real scale.

**Notion design doc**: `Claude: Module: Prompt Engineering Iteration`

## Layout

```
prompt_engineering/
├── pool.py               # load seeds + variation primitives from YAML
├── mutator/
│   ├── rules.py          # bone — structural mutations
│   └── llm.py            # flesh — LLM polish
├── evaluator.py          # run a variant vs metric on pilot
├── optimizer.py          # Round 0 → k loop
├── metrics/
│   └── base.py           # Metric registry; concrete metrics added per step
├── prompts/              # one dir per step; starter seeds go here
│   ├── detection/
│   ├── verdict/
│   ├── regeneration/
│   ├── location/
│   └── correction/
├── variations/
│   └── general.yaml      # starter variation primitives
└── runs/                 # empty — run artifacts land under output/ichl/.../runs/
```

## Usage (sketch — not yet end-to-end wired)

```python
from ichl.prompt_engineering import optimizer

result = optimizer.run(
    step='detection',
    base_prompt_pool='src/ichl/prompt_engineering/prompts/detection/seeds.yaml',
    variation_pool='src/ichl/prompt_engineering/variations/general.yaml',
    tool_model='gpt-4o',
    metric='selectivity',
    pilot_data='data/pilots/detection_pilot_40.jsonl',
    target_client_name='qwen3-8b',
    top_n_candidates=3,
    max_rounds=10,
    epsilon=0.02,
    out_dir='output/ichl/detection/runs/20260422_pe/',
)
# result.top_candidates -> list[Candidate] (handed to downstream)
```

## Current status — STUBS ONLY

- Clients (`ichl.clients`) are usable now.
- `rule_based_mutate` is a `_noop` pass-through. Real rules register via
  `@register_rule(name)` in `mutator/rules.py`.
- No concrete metrics implemented yet. Add one when starting the first
  optimization run (first target: `selectivity` for detection).
- No seed prompts in `prompts/` yet. Add them as starter seeds for each step.

## Related Notion pages

- `Claude: Module: Prompt Engineering Iteration` — full design
- `Principle: Training Prompt Engineering` — the iterative approach
- `Claude: Principle: Regex Parser Unreliability` — parser co-design rule
- `Claude: Principle: Prompt Design per Model` — per-model constraints
- `Claude: Principle: Use MLX as External Validator` — MLX role
- `Claude: Principle: Experiment Audit Guidelines` — audit checklist
