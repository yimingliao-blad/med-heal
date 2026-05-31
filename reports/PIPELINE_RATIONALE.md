# Med-Heal Self-Correction — Consolidated Plan (single source of truth)

This is THE plan. Companion: DETECTION_DESIGN_SPACE_CHECKLIST.md is the design-space appendix. If anything here and there conflicts, this wins.

## Core thesis — WHY the pipeline has this shape
**The ZS error is a long-context problem, not a reasoning problem.** Qwen2.5 is already ~89% correct; on a 10k+ char discharge note it simply doesn't attend to all the right facts, or misreads them. Two confirmations: (a) when the model re-checks its OWN answer it rationalizes — 26/27 misses concluded "the answer is fine" (expM); (b) the errors are concrete value-mistakes and omissions, not bad logic.

**Therefore the fix is retrieval-augmented FOCUSED QA:** gather the true relevant facts into a short, focused context and **re-answer the question** from it. Right facts in a short context → right answer. No error-diagnosis required.

NOTE TO SELF (anti-drift): the post-retrieval step is **extract + normalize true facts**, NOT "diagnose what's wrong / rebuild the oracle hint." Diagnosing makes the model rationalize the ZS again (that mistake = expQ). The ZS answer is NOT shown to the fact-gathering or QA steps (anchoring → rationalization).

## How we got here — the process narrative (do not lose this chain)
The architecture was not assumed; it was derived step by step:

1. **CoT showed how the ZS model thinks, and named its weakness.** We ran a single CoT that does the whole check and analyzed *where* it breaks (expM). It revealed that the model, reasoning as it does at zero-shot, fails in two specific ways: **OMISSION** (it validates the claims it made — all "supported" — but never checks the facts the question *required* that it left out) and **MISREADING** (it accepts an approximate or wrong value, e.g. "the note supports this but doesn't give the exact number," as correct). 26/27 misses had the model concluding "the answer is fine." → The weakness is attention/reading on a long note, not logic.

2. **We proved that reading the note + comparing to the ZS can LOCATE those mistakes.** Decompose → blind note-read → compare to the ZS (expO) recovered **68% recall** (2× the model's own self-check). So the omissions and misreadings the model made are findable by reading the note fresh and comparing — *if the reader does not see the answer as its own.*

3. **We confirmed the strategy: retrieve the GENUINE information from the discharge notes.** An LLM free-reading a 24k note can hallucinate the very facts we check, so we moved to machinery retrieval. GTR embedding + string retrieval (expP) pull REAL note sentences and reach **~95% of the oracle (gold-query) ceiling** — we can get the genuine information out of the note without trusting an LLM paraphrase.

4. **The strategy that follows: use the extracted genuine information to BUILD the correct answer** (focused QA) — because the ZS failure is long-context focus, the same model with the right facts in a short context answers correctly.

5. **WE ARE HERE — the last open stage: how to get the information OUT** (extract + normalize the retrieved real quotes into clean true facts) before feeding correction. Detour avoided: making this step *diagnose the error* (expQ) makes the model rationalize the ZS again — so this step only extracts/normalizes; QA re-answers.

## The pipeline

### Main path — correction = focused QA
1. **DECOMPOSE** (Q + ZS → look-up items): open question-driven items + specific answer-claim items, one merged prompt. = the retrieval queries / "what to look up".
2. **RETRIEVE** (items → real note quotes): GTR embedding (`topk_spans`) + string search. Machinery on the REAL note, not LLM free-reading. Proven ~95% of oracle ceiling.
3. **EXTRACT + NORMALIZE** (raw quotes → clean true facts): tidy the retrieved spans into clear, normalized factual statements relevant to the question. No judging the ZS, ZS not shown.
4. **QA** (answer from focused facts): re-answer the question using only the normalized true facts.

### Separate concern — whether to SWAP the ZS for the QA answer (quality control)
- **DETECTION**: is the ZS actually wrong? (decompose → blind-locate → compare; recall 68% / over-flag 45%, expO). Decides whether to replace ZS with the QA answer.
- **GATE / VERDICT**: stop bad swaps — false positives (pre) and breaks (post).
Detection is NOT the spine; the spine is focused QA. Detection just routes whether to keep ZS or the new answer.

## What's proven (judge-free where it matters)
- **expM** — self-check rationalizes (26/27 misses concluded "fine"). → isolate ZS; root cause is focus, not reasoning.
- **expO** — decompose→blind-locate→compare detection: **recall 68%** (2× blind's 32%), **over-flag 45%** (compare over-fires on concise correct answers).
- **expP** — GTR retrieval from decompose look-ups = **95% of oracle (gold-query) ceiling**; **string ≈ GTR** (keep string for exact terms/numbers); 46% exact-span overlap (gets the region, ~half the precise sentences).
- **expQ — the MISTAKE, kept as a lesson.** Built the post-retrieval step as error-diagnosis ("write a correction guideline, what's WRONG/MISSING, else NO CORRECTION NEEDED") → it rationalized the ZS and said "no correction needed" on 55-68% of WRONG cases. DROPPED. Lesson: the post-retrieval step must just extract/normalize facts; QA re-answers; do NOT diagnose.

## Focused-QA exploration findings (2026-05-31)
Extensive single-case + 40-case investigation of the "retrieve focused facts → re-answer" path:
- **Focused QA from top-10 embedding spans (no whole note):** Qwen FIX 28% / BREAK 45% / GROUNDED 92%; GPT-4o on the SAME spans FIX 35% / BREAK 30%. → grounding works (focus prevents hallucination), but it is **both model- and retrieval-limited**: GPT uses the spans better (+10pp, e.g. cerclage was retrieved at rank 2 but Qwen omitted it), yet even GPT is capped because **top-10 is lossy** (misses facts → can't fix, breaks correct answers via incompleteness).
- **The gold-fact-recall metric was BROKEN** — it scored clearly-correct answers 0 ("urinary retention"≠"voiding", "210"≠"below 300"). Use the project correctness judge, not embedding/string overlap. (Section-routing "lost" to whole-note only under that broken metric.)
- **Embedding retrieval misses narrative-embedded facts**; deterministic **section routing recovers structured fields** (the surgeries omission — hysterectomy/cystoscopy/Spigelian — was retrieved by header-routing but missed by embedding). Sentence-embedding favors narrative prose over structured `HEADER: value` lines.
- **Chunk patterns (key→value decomposition):** (1) header-labeled fields, (2) lab codes `LABEL-value`, (3) meds `drug dose route freq` → all **naive/regex extractable**; (4) **narrative prose** ("collection was < 300", "nodule has grown") → key→value embedded in language, **needs the LLM**; (5) junk (IDs, vitals) → filter.
- **Temporal / multi-admission:** one item has different values across admissions (nodule "grown" vs "stable"; "second vs third admission") → decomposition is **key→(value, admission, date)**. This is a **machinery fix**: deterministic admission split on `Patient ID/Chartdate`, sort by date, number chronologically; the question usually carries the temporal cue.

### Converged decomposition design
- **Machinery (deterministic):** admission split (+date, chrono #) → section split → structured key→value parse (patterns 1–3). Handles provenance, temporal, and the structured majority — no hallucination.
- **LLM (only where machinery can't reach):** narrative prose → embedded key→value (few-shot / instruction-guided); and final QA reasoning over the assembled, provenance-tagged facts.
- A capable model uses focused facts better (GPT>Qwen), so for the small model the structuring matters more.

## ★ ROOT CAUSE (2026-05-31): INFORMATION RETRIEVAL is the bottleneck — needs dedicated design
Tracing every focused-QA failure to its root, the limiting stage is **getting the right facts out of the note reliably** — not the model's reasoning, not the correction mechanism, not detection. Evidence:
- Focused QA breaks ~half of CORRECT cases because the retrieved context is **incomplete vs the whole note the ZS already had** (method-independent: raw spans, sections, structured all break 45–55%).
- GPT-4o on the SAME spans only modestly beats Qwen (35/30 vs 28/45) and is **still capped** — so even an oracle model can't fix what retrieval doesn't surface.
- Break/error root causes, traced on real cases, are ALL retrieval-shaped:
  - **Too many ZS-anchored items crowd/dilute retrieval** (case 2: 14 items buried the core diagnosis; focused query brought it back). Items must be **question-core, de-anchored from the ZS**.
  - **Embedding fragments structured fields** — `Discharge Diagnosis:\nMetastatic Gastric\nAdenocarcinoma` is never a clean sentence; the value lives in a **section field**, not a loose sentence. Treatment in a **med list** doesn't embed near "treatment".
  - **Provenance / temporal**: multi-admission notes → same item differs by admission; date-specific questions need **admission+date tags** on every retrieved fact (now correctly implemented via per-admission sentence indexing). Sentence-embedding alone loses this.
  - Embedding favors **narrative prose over structured `HEADER: value`** lines (the surgeries omission).
- The chunker already represents the hard cases correctly (clean `Discharge Diagnosis`/`Discharge Medications` section fields + admission/date provenance) — so the fix is to **retrieve over the chunker's provenance-tagged section fields + question-core items**, not loose embedding sentences.

**Conclusion:** information retrieval is a real, non-trivial design problem (structured-vs-narrative representation, provenance/temporal, item-crowding, fragmentation, fuzzy-vs-exact). It is THE bottleneck and **needs dedicated design time** before more end-to-end runs. The downstream (QA, correction, gate) is comparatively solved/understood.

## Principles (anti-drift rules)
1. **Make each step SIMPLE, prove it, THEN integrate.** Don't merge stages prematurely — signals break.
2. **Never show the ZS to the fact-gathering / QA steps.** Seeing its own answer → rationalization.
3. **Prove on judge-free metrics vs GROUND TRUTH** (recall; QA-correct vs gold), not LLM-judge opinions. (Burned repeatedly by lenient/ noisy judges and regex-on-prose.)
4. **Missing (false negative) is worse than false positive** — a miss can't be rescued; an over-flag can (correction from real facts keeps a correct answer correct). Lean recall, keep both low.
5. **Parsing of any flag/verdict = GPT-4o-mini semantic judge as authority** (regex first pass), per project-flag-token-parsing.

## Next steps (the plan, in order)
1. **Build EXTRACT+NORMALIZE (simple) + QA. Prove it:** does QA from the normalized focused facts answer correctly vs gold? On wrong cases = fix-rate; on correct cases = break-rate. (Judge-free vs gold.)
2. **Compare retrieval feeding QA:** GTR vs string vs union; sweep k.
3. **Integrate detection + gate:** decide when to swap ZS → QA answer (use detection recall + gates to protect correct answers).
4. **Scale:** 5-fold mean±std, then cross-model, reporting lift over zeroshot at the TRUE base rate (962 = 109 wrong + 853 correct; correct weighted ×4.27).

## Metrics
- **Headline:** QA correctness vs gold — fix-rate on wrong, break-rate on correct.
- Retrieval coverage (proven ~95%).
- Detection recall / over-flag (for the gate only).
- End-to-end net over zeroshot at true base rate.
