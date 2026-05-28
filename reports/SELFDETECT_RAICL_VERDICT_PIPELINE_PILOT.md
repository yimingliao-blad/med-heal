# Self-Detection RA-ICL Correction Verdict Pipeline Pilot

Status: first executable full-pipeline test for the self-detection -> RA-ICL correction -> verdict gate path. This is separate from regen+verdict.

## Pipeline

```text
zero-shot answer
  -> p5 self-detection payload
  -> regex parse, with gpt-4o-mini fallback available for malformed output
  -> GTR same-patient evidence retrieval
  -> BM contrast-pool RA-ICL example retrieval
  -> correction generation
  -> pairwise verdict gate
  -> final GPT-4o judge
```

## Settings

- Model: `Qwen/Qwen2.5-7B-Instruct` on vLLM port `8003`.
- Local vLLM concurrency: `8`.
- Detection: `p5_retrieval_payload`, temperature `0.7`.
- Correction: evidence + RA-ICL example, temperature `0.0`.
- Verdict: pairwise A/B gate, `k=3`, temperature `0.7`; unclear/tie keeps original.
- Final judge: fixed old Stage 1 GPT-4o prompt, temperature `0.1`, sequential.

## Results

| Run | N | Detected | Accepted | Fix | Break | Net | Parse fallback | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| smoke 2 wrong + 2 correct | 4 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| pilot 20 wrong + 20 correct | 40 | 11 | 8 | 4 | 1 | 3 | 0 | 0 |

## Interpretation

The 40-case pilot is positive but small: `fix=4`, `break=1`, `net=+3`. The verdict gate filtered 3 of 11 detected/corrected attempts before final judging. All detection outputs in these Qwen2.5 runs were regex-parseable, so the `gpt-4o-mini` fallback did not fire; it remains in the runner for models that do not follow format, such as DeepSeek or BioMistral.

This is a ready-to-go full-pipeline test path, but it is not enough to confirm a multirun method. Next step should be the same runner on all 109 Qwen2.5 wrong cases plus 109 matched correct cases, then compare against the separate regen+verdict path.

## Outputs

- Smoke summary: `refactor/pre_atom_pipeline/output/selfdetect_raicl_verdict/qwen25_nw2_nc2_seed42_detp5_raicl_vgate/summary.json`
- Pilot summary: `refactor/pre_atom_pipeline/output/selfdetect_raicl_verdict/qwen25_nw20_nc20_seed42_detp5_raicl_vgate/summary.json`
- Pilot judged rows: `refactor/pre_atom_pipeline/output/selfdetect_raicl_verdict/qwen25_nw20_nc20_seed42_detp5_raicl_vgate/judged_outputs.jsonl`
- Runner: `refactor/pre_atom_pipeline/scripts/run_selfdetect_raicl_verdict.py`
