# Self-Correction Design-Space Checklist (design-space APPENDIX — not the plan)

**THE PLAN is `PIPELINE_RATIONALE.md`. This file is just the menu of options/knobs, kept so we don't forget branches.** Where this conflicts with the plan, the plan wins.

## Corrected thesis (see plan)
The ZS error is a **long-context focus/misread problem, not a reasoning problem**. The fix is **retrieval-augmented FOCUSED QA**: decompose → retrieve real quotes → extract+normalize true facts → re-answer. The post-retrieval step EXTRACTS+NORMALIZES facts; it does NOT diagnose the error (diagnosing makes the model rationalize the ZS — the expQ mistake). Detection/gate is a SEPARATE concern (whether to swap ZS for the QA answer), not the spine.

## Older framing (kept for the detection sub-problem only — it is the GATE, not the spine)
- The detect/correct/verdict cascade below is the QUALITY-CONTROL path (decide whether to swap), not the main correction path.
- Metric for the gate: recall of real errors (a miss is worst), then over-flag.

## DETECT — functional knobs (a detection pipeline = one level on each)
| knob | levels | notes |
|---|---|---|
| A. decompose question | off · planner (P0, list required facts) · CoT-decompose | what the question demands |
| B. analyze ZS | extract-facts (P1) · blind-paraphrase (R1, hides ownership) | restate ZS claims |
| C. grounding (evidence finder reads) | full-note · **union-evidence (dual extract / memo)** · retrieved-spans | recall lever |
| D. find strategy | plain · CoT · clean (WRONG/CORRECT/EVIDENCE) · few-shot · natural(×k) | how it reasons |
| E. confirm | merged-into-D · separate-verify · k-vote (union/majority/all) | precision lever |

## CONFIRM / CORRECT / VERDICT knobs
- CONFIRM variations (also usable standalone): k-vote, separate-verify, gate∧diagnoser conjunction.
- CORRECT: arm {source_led (tight WRONG/CORRECT/EVIDENCE), raicl (retrieved fix example), few-shot}; input form {error-only, full-note, summary}.
- VERDICT: {C3_cot, C3_strict, accept_all, adaptive/confidence-routed, k-vote}.

## Combine-vs-separate questions (open)
- decompose + analyze + ground + find as ONE monolithic CoT vs separate calls. ← **next diagnostic**
- find + confirm merged vs separate.
- confirm + error-formulation merged vs separate.
- gate (union) + diagnosis: separate strategies (union natural for recall, CoT for precision) vs same strategy.

## Results so far (evidence)
- **Cascade (309, projected to 962):** zeroshot 88.56%. Best deployable +0.6–0.9pp; oracle-verdict ceiling +1.6pp. Diagnoser-alone ≈ gated; blind_plain best blind diagnoser; raicl ≥ source_led; verdict OVER-rejects precise pipelines.
- **union:** recall 92–96%, over-flag 74% (OR of 3 natural compares @T0.7). It pulls plausible evidence but can't be trusted alone.
- **blind diagnosis:** recall 22–27% — it RE-DETECTS from the note, discarding union's recall (88% of its flags are already inside union; conjunction adds only corroboration, ~+0.4pp by trimming one break).
- **expL pilot (union-evidence + plain-verify, separate):** USEFUL 17 vs blind 8 (recall recovered 22→70%) BUT over-flag stayed 60% — the plain separate-verify did NOT prune. Only the weakest cell tested.

## Untested cells / open ideas (the anti-forgetting list)
- [ ] **union-evidence × CoT × merged** (CoT narrows on union's evidence) — likely best, never run.
- [ ] union-evidence × clean(WRONG/CORRECT/EVIDENCE) × merged.
- [ ] question-decompose (planner) on/off effect on recall+localization.
- [ ] few-shot as a find strategy (on union-evidence).
- [ ] retrieved-spans grounding vs union-evidence vs full-note.
- [ ] CORRECTION few-shot arm; input-form ablation re-confirm.
- [ ] VERDICT adaptive / less-conservative (recover the gap to +1.6pp oracle).
- [ ] tighter separate-verify ("confirm only if note CLEARLY contradicts").
- [ ] **monolithic CoT (all DETECT functions in one) + step-by-step failure attribution** ← running now.

## Decompose → blind-locate architecture (the new direction, proving stage by stage)
Replaces "model checks its own answer" (it rationalizes — expM: 26/27 misses are the model concluding "fine"). Instead:
1. **DECOMPOSE** (sees Q + ZS): ask where the evidence supporting the answer comes from → a plain list of things to look up. NO "reconstruct your reasoning" (LLM has no memory) — just "where would the evidence be". Two signals, proven COMPLEMENTARY and they map to the two error types (~19 value / ~21 omission):
   - ARM A (answer-anchored, granular per-claim) → catches VALUE errors + hallucinations.
   - ARM B (question-driven OPEN look-ups, e.g. "all treatments given") → catches OMISSION + WRONG-FRAME (cases A misses entirely).
   - **MERGE = ONE prompt works** IF it explicitly asks for both kinds WITH concrete open examples (observed: open item leads, specifics follow, on all 3 discriminating cases). The naive merge (expN) collapsed to answer-anchored for lack of the open-lookup framing. → use single MERGED-ONE call.
2. **BLIND LOCATE** (sees note + look-up list + Q, NOT the ZS answer): walk the note, report what it actually says per item or "not stated". Blind = the model can't rationalize its own answer.
3. **COMPARE** located-evidence vs ZS → value mismatch / omission = the error.
This one engine feeds detection, correction-input, AND the gate (below).

## GATE — where to stop false positives / bad fixes (user's 3 ideas; same located-evidence check, different placement)
- [ ] **(1) detection-as-gate (PRE-correction):** if the blind-locate CONFIRMS all ZS claims → no error → never enter correction. If detection is reliable/precise enough it IS the gate; clean cases get no correction (kills false positives). Highest value at base rate (over-flag is the enemy).
- [ ] **(2) post-correction fix-check (POST):** after correcting, re-check the CORRECTED answer against the located note evidence — if it still contradicts / didn't resolve the flagged item, it's a BAD fix → reject (catch breaks). Grounded in evidence, not an A/B preference.
- [ ] **(3) combine into verdict:** fold (1)/(2) into the verdict stage, using located evidence instead of just "which answer is better".
These are complementary, not exclusive — (1) pre + (2) post can both run. Prove separately.

## Method
Collect-once-project: run all variations on the SAME 309 cases so any composition is a comparable projection (as the cascade did). Score on USEFUL recall first. Prove each stage on a judge-free metric (recall/precision vs ground truth) before merging.

## Next investigation (this session)
Monolithic CoT detection (`expM_cot_introspect.py`): run one CoT doing decompose→claims→evidence→compare→conclude as a natural trace; then attribute each failure to the FIRST broken step.

### RESULT (expM, 40 wrong + 20 correct) — the bottleneck is FINDING, not concluding
- Monolithic CoT recall **32%** (13/40), over-flag **10%** — no better than blind.
- Of 27 misses, **26 (96%) are GENUINE analysis misses**: the model's own reasoning concludes "the answer is fine" when it is actually wrong. Only 1 found-but-didn't-flag. (LLM-classified the conclusions semantically — the regex/GPT-step-attributor mislabeled 23 as "conclusion failures" because they were all-supported-but-wrong; reading the conclusions corrected it. Method note: trust semantic reads, not labels — burned 3× by instruction-echo in the traces.)
- The misses are dominated by **OMISSIONS** (answer incomplete; the CoT validates the claims that ARE present, all SUPPORTED, but never checks the required facts are all present) and **LENIENCY** (accepts "note supports this but doesn't give the exact value" as SUPPORTED).
- So the model genuinely rationalizes the wrong answer as supported. This is WHY union has higher recall: union extracts "facts that actually answer the question" (question_facts) INDEPENDENT of the answer, surfacing omissions; the CoT anchors on the answer's own claims.
- **Stage to redesign = the FIND step (knob A + D), not the conclusion:** (1) decompose the question into REQUIRED facts and check each is present AND exact in the answer (catch omissions); (2) strict comparison (approximate ≠ supported). Test next.
