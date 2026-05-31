# Substudy 7A Scope Detail

Date: 2026-05-29
Status: pre-decision detail. No code changes proposed in this file. The purpose is to give you the concrete numbers needed to pick the 7A bake-off scope.

User direction this turn: *"7A scope: I need more details so I know what to make decision with."*

## What 7A is asking

The yesterday-pilot's `taxonomy_evidence` arm reached **net +27** on 218 Qwen2.5 cases. That arm uses a **pre-computed offline oracle**: `phase1_wrong_gpt4o.json` (109 GPT-4o-audited entries for Qwen2.5 wrong cases) which provides `PRIMARY_ERROR`, `ERROR_DESCRIPTION`, `QUESTION_FOCUS`, `MODEL_CLAIMS`. That oracle is not available at inference time on any new case.

7A asks two related questions:

1. **What is the oracle ACTUALLY worth?** If the oracle hint were replaced by a live detection step's `error_type`, how much of the +27 do we recover?
2. **Where is the gain coming from?** Spans alone, hint alone, or only when both are present?

The first answers "is this method deployable on a new patient." The second answers "what should we keep when we describe the method in the paper."

## Hard fact about the taxonomy coverage

```
items in phase1_wrong_gpt4o.json: 109
coverage of Qwen2.5 wrong cases: 109/109 (100%)
coverage of other model wrong cases: NONE (taxonomy is Qwen2.5-specific)

PRIMARY_ERROR distribution:
  MISREADING            63
  QUESTION_MISALIGNMENT 21
  OMISSION              17
  FABRICATION            7
  HEDGING                1
```

Implication:

- The +27 result is Qwen2.5-only. It cannot transfer to the other 4 models without either rebuilding the taxonomy per model (~$100-150 GPT-4o oracle cost) or substituting a live detector.
- For the paper, 7A-oracle is a **Qwen2.5 ablation** that bounds the oracle ceiling. 7A-live (using the natural pipeline detection) is the **deployable** method that runs on ALL models.

## The full ablation grid (3 hint sources × 2 span states = 6 cells)

| # | Cell | Hint source | Spans | Existing? | Pipeline shape |
|---|---|---|---|---|---|
| 1 | (none, no spans) | none | no | Use existing `regen_fullscale` (net -6) or zero-shot | 0 LLM stages |
| 2 | (none, spans) | none | yes | YES — yesterday's `evidence_only` (net +20) | 1-stage correction |
| 3 | (oracle, no spans) | offline taxonomy | no | NEW | 1-stage correction |
| 4 | (oracle, spans) | offline taxonomy | yes | YES — yesterday's `taxonomy_evidence` (net +27) | 1-stage correction |
| 5 | (live, no spans) | detection.error_type + wrong_claim + correct_or_missing_info | no | NEW | detection + 1-stage correction |
| 6 | (live, spans) | detection (same as 5) | yes | NEW | detection + 1-stage correction |

What each cell tells us:

| Comparison | Answers |
|---|---|
| 4 vs 2 (+27 vs +20) | What does the oracle hint add when spans are present? Already known: +7. |
| 4 vs 3 | Do retrieved spans add on top of an oracle hint? |
| 4 vs 6 | How much of the oracle ceiling can the live detector recover? |
| 6 vs 5 | Do retrieved spans add on top of a live detector hint? |
| 6 vs 2 | Is a live detector hint worth adding to spans-only correction? |
| 3 vs 5 | How much oracle quality leaks through to a hint-only setup? |
| 2 vs 1 | What does retrieval alone buy over zero-shot/regen? Already known: +20 vs -6 = +26 (Qwen2.5). |

The single most important comparison for **method deployability** is **6 vs 4** — that delta IS the oracle premium. If 6 lands near 4, the natural-pipeline detection is essentially as good as the offline taxonomy for this purpose. If 6 falls back near 2, the oracle is doing most of the work and the method needs a stronger live signal.

## What each new arm costs on Qwen2.5

Sample size: 218 cases (109 wrong + 109 correct, matches yesterday's pilot for direct comparability). Could shrink to 50 for an initial screen and rerun at 218 after narrowing.

Per-arm cost at 218 cases:

| Cell | Qwen2.5 vLLM calls | GPT-4o-mini calls | GPT-4o judge calls | Vllm wall (c=8) | Oracle cost |
|---|---:|---:|---:|---:|---:|
| 3 (oracle, no spans) | 218 × 1 = 218 | 0 | 218 | ~5 min | $1.74 |
| 5 (live, no spans) | 218 × ~3 (plan, confirm, correction) = 654 | 218 (parser) | 218 | ~25 min | $1.78 |
| 6 (live, spans) | 218 × ~3 = 654 | 218 (parser) | 218 | ~25 min | $1.78 |

Total for the 3 new arms at 218 cases: ~55 min Qwen2.5 + ~$5.30 oracle.

At 50 cases (smaller initial screen):
Total: ~13 min Qwen2.5 + ~$1.30 oracle.

For the existing cells:
- Cell 2 (`evidence_only` 218) and Cell 4 (`taxonomy_evidence` 218) are already on disk — no rerun unless we need fresh judging.
- Cell 1 (regen baseline) — existing `regen_fullscale.jsonl` Qwen2.5 has 962 rows, net -6. Filter to the same 218 cases for matched comparison: ~3 min Python.

## The hard architectural choice — what happens when detection misses

For cells 5 and 6 ("live" hint), the natural detection step runs first. From the 200-case Qwen3-8B and 40-case Qwen2.5 screens we know detection recall is below 100%.

Two options for what to do on a wrong case that detection labels CORRECT:

| Option | Behavior | Effect on comparison |
|---|---|---|
| (A) Strict gate | Detection-CORRECT → final = zero-shot answer (skip correction). | The 7A-live cell can only fix as many cases as detection routes. Ceiling = detection recall × correction precision. Reflects real deployment. |
| (B) Force correction | Detection-CORRECT → still run correction with whatever fields detection produced (likely thin). | Matches the per-case sample of the oracle arms (every case enters correction). Comparable to 7A-oracle on the same N, but does NOT reflect deployment safety. |

These give different numbers. The oracle arms (3, 4) run correction on EVERY case because the oracle is always available. To make 5, 6 directly comparable to 3, 4 on the same denominator we either:
- Pick (B) and accept that the result overstates real-world performance, OR
- Pick (A) and report two denominators side by side.

A clean third option: report BOTH (A) and (B) for cells 5 and 6, name them `5A`/`5B`/`6A`/`6B`. That's 5 new arm runs instead of 3, adding ~15 min vLLM and ~$1.50 oracle. This is the cleanest pattern for the paper.

## Wrong-case error-type distribution constrains the ceiling

From the +27 arm's per-error-type breakdown:

| Error type | Wrong-case count | Fixed | Recall |
|---|---:|---:|---:|
| MISREADING | 63 | 16 | 25% |
| QUESTION_MISALIGNMENT | 21 | 10 | 48% |
| OMISSION | 17 | 3 | 18% |
| FABRICATION | 7 | 2 | 29% |
| HEDGING | 1 | 1 | 100% |

The largest unfixed pool is MISREADING. 7A's per-cell deltas will be dominated by MISREADING behavior. If the goal is to pick the cell that maximizes net, focus on MISREADING fix-rate, not aggregate.

For the natural-pipeline detection step, the error_type categories are {CONTRADICTION, OMISSION, QUESTION_MISALIGNMENT, NONE, UNCLEAR}. There's no MISREADING bucket. The mapping is approximately:

| Offline taxonomy | Likely natural-detection mapping |
|---|---|
| MISREADING | CONTRADICTION (when note disagrees) or QUESTION_MISALIGNMENT (when the wrong fact is from a different visit) |
| QUESTION_MISALIGNMENT | QUESTION_MISALIGNMENT |
| OMISSION | OMISSION |
| FABRICATION | CONTRADICTION (when fabricated content is wrong) or NONE (when LLM defers) |
| HEDGING | NONE or UNCLEAR |

So 7A-live's hint will be COARSER than 7A-oracle's hint. The 4 vs 6 delta partly measures this granularity gap, which is also informative for the paper's methods section.

## Three concrete options for the 7A scope

I propose three discrete packages so you can pick one rather than tune all dimensions.

### Option 7A-MIN (minimum useful): 2 new arms, 1 size

- Arms: cell 5A (live + no spans, strict gate), cell 6A (live + spans, strict gate).
- Sample: 218 cases (matches existing).
- Cells covered: 5A, 6A. Comparators: 2 (existing +20), 4 (existing +27), 1 (regen -6).
- Cost: ~50 min Qwen2.5 + $3.50.
- Answers: deployable method's gain over zero-shot, and oracle premium. Does NOT separate hint-vs-spans contribution.

### Option 7A-STD (standard): 4 new arms, 1 size

- Arms: 3 (oracle no-spans), 5A (live no-spans, strict), 5B (live no-spans, force), 6A (live spans, strict), 6B (live spans, force).
- Sample: 218.
- Cells covered: 3, 5A, 5B, 6A, 6B. Comparators: 2, 4, 1.
- Cost: ~75 min Qwen2.5 + $6.80.
- Answers: deployable method's gain, oracle premium, strict-vs-force gap, hint-vs-spans decomposition (3 vs 4 vs 2).

### Option 7A-MAX (full ablation): all 5 new arms, two sizes

- Round 1 small screen: 50 cases, all 5 new arms — confirm shapes, ~13 min + $1.30.
- Round 2 full: 218 cases, the 2-3 most informative arms — ~30-40 min + $2.50.
- Total ~55 min Qwen2.5 + $4.00.

## Things this scope does NOT decide

- Cross-model transfer. The taxonomy doesn't exist for the other 4 models, so 7A-oracle cannot run there. 7A-live can; it inherits whatever per-model detection-recall is observed in Gate 7's per-model setting tuning.
- Multirun K. All 7A arms can start at K=1 deterministic; multirun decision (Gate 1) can be applied later if a single arm wins.
- Verdict gate. Yesterday's pilot used NO verdict. 7A reuses that — every correction is kept. The verdict-gate option from the natural pipeline is orthogonal and can be layered later.

## Decisions waiting

Pick exactly one of MIN / STD / MAX, plus one architectural choice:

1. Scope: MIN, STD, or MAX?
2. Detection-miss handling for live arms: A (strict gate, real deployment) / B (force, comparable denominator) / BOTH (run both flavors)?
3. Sample size for the new arms: 218 (matched), 100, or 50 (small screen first)?
4. Add a `detect → correction skipped` count to the summary, so the strict-gate denominator is explicit? Y/N.

Once those are chosen, the bakeoff scripts and prompt definitions can be written without further iteration.
