# Correction-Oriented Error Taxonomy

## Three Error Types (by how to fix them)

### 1. CONTRADICTION (64% of Qwen2.5 errors)
**What**: The answer states a fact that CONFLICTS with the discharge notes.
**Includes**: misread medication/dosage, wrong procedure, confused visits, fabricated details.
**Fix action**: UPDATE the wrong fact/evidence to match what the notes say.
**Example**: Answer says "prescribed Metoprolol 50mg" → Notes say "Lisinopril 10mg" → Update claim.

### 2. QUESTION_MISALIGNMENT (20% of Qwen2.5 errors)
**What**: The answer addresses the wrong visit, time period, or clinical aspect.
**Includes**: answering about visit 1 when asked about visit 2, including irrelevant medications, hedging with multiple answers.
**Fix action**: REFOCUS the answer on what the question specifically asks.
**Example**: Q asks "second admission medications" → Answer lists first admission → Refocus on second.

### 3. OMISSION (16% of Qwen2.5 errors)
**What**: Critical information is IN the notes and NEEDED for the answer, but MISSING from the answer.
**Includes**: missing a key procedure, missing a medication change, not mentioning a diagnosis.
**Fix action**: ADD the missing fact/evidence from the notes to the reasoning.
**Example**: Q asks "medication changes" → Answer mentions 1 of 3 changes → Add the other 2.

## Mapping from GPT-4o labels
| GPT-4o label | → Correction type | Rationale |
|-------------|-------------------|-----------|
| MISREADING | CONTRADICTION | Wrong fact → update it |
| FABRICATION | CONTRADICTION | Invented fact → remove/replace it |
| OMISSION | OMISSION | Missing fact → add it |
| QUESTION_MISALIGNMENT | QUESTION_MISALIGNMENT | Wrong focus → refocus |
| HEDGING | QUESTION_MISALIGNMENT | Multiple answers → commit to one |

## Distribution (Qwen2.5-7B, 109 wrong answers)
- CONTRADICTION: 70 (64%)
- QUESTION_MISALIGNMENT: 22 (20%)
- OMISSION: 17 (16%)

## Distribution comparison (BioMistral-7B, 212 human annotations)
- OMISSION: ~120 (57%)
- CONTRADICTION: ~65 (31%)
- QUESTION_MISALIGNMENT: ~10 (5%)

Key difference: Qwen2.5 makes more CONTRADICTION errors (misreads facts), BioMistral makes more OMISSION errors (misses facts). Different models, different error profiles.

## Pipeline Design

```
Detection → find error + classify type + locate error
         ↓
Correction → route by type:
  CONTRADICTION → point out wrong claim + notes evidence → re-answer
  OMISSION → point out missing info + notes evidence → re-answer  
  QUESTION_MISALIGNMENT → point out correct focus → re-answer
         ↓
Verdict → compare original vs corrected using detection principles
```

## Detection Requirements
The detection prompt must output:
1. Whether there IS an error (verdict)
2. What TYPE of error (contradiction/omission/question_misalignment)
3. WHERE the error is (specific wrong claim or missing info)
4. What the NOTES say (evidence for correction)

## Files
- `correction_oriented_annotations.json` — 109 Qwen2.5 items with correction_type
- `phase1_wrong_gpt4o.json` — original GPT-4o annotations
- `phase1_correct_gpt4o.json` — 50 correct answer analyses
