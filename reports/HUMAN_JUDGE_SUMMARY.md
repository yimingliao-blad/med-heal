# Human Judge Source Summary

## Status

Provisional evidence only. Do not treat these artifacts as the final human-judge conclusion; they are being used now to confirm the LLM judge prompt.

## Selected Sources

- Raw latest human export: `datasets/external/all_users_openended_BioMistral-7B_1775740232208.csv`
- Final 100-case human/gold artifact: `output/step9_v2/sample100_gold_seed42.csv`
- Extended 200-case artifact: `output/step9_v2/sample200_with_gold.csv`
- Binary mapping: `Answer Quality == 5` means correct (`1`); all other values mean incorrect (`0`).

## Raw Human Export

- Rows after `(User Name, Patient ID)` de-duplication: 788
- Unique patients: 493

| Reviewer | Rows |
|---|---:|
| Sara Saif | 328 |
| Jose E. Lizarraga Mazab | 310 |
| Caitlin Schwanke | 100 |
| Kushali darak | 50 |

## Sara/Jose Agreement Baseline

- Sara/Jose shared patients: 145
- Sara/Jose agreed patients: 112
- Sara/Jose disagreed patients: 33
- Agreed-label distribution: {'0': 21, '1': 91}

Use these 112 agreed patients as the clean baseline for selecting the GPT judge prompt that best aligns with humans.

## Final 100-Case Set

- Cases: 100
- Rater-count distribution: {'1': 12, '2': 53, '3': 35}
- Gold-label distribution: {'0': 32, '1': 68}
- GPT-label distribution currently attached: {'0': 32, '1': 68}
- Reviewer coverage: {'Sara Saif': 81, 'Jose E. Lizarraga Mazab': 92, 'Caitlin Schwanke': 50, 'Kushali darak': 23}
- Sara/Jose jointly reviewed inside sample: 73
- Sara/Jose agree/disagree inside sample: 56/17

This is the broader human set to keep for final reporting because it includes Caitlin coverage and resolved `gold` labels for 100 cases.

## Provisional Working Decision

- Human-judge source: `datasets/external/all_users_openended_BioMistral-7B_1775740232208.csv`.
- Human-gold baseline for GPT judge prompt selection: Sara/Jose A∩B agreement subset, N=112.
- Broader final human adjudication artifact: `output/step9_v2/sample100_gold_seed42.csv`.
