from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Variant:
    family: str
    variant_id: str
    label: str
    status: str
    source: str
    evidence: str
    notes: str


VARIANTS: list[Variant] = [
    Variant(
        "baseline",
        "zeroshot",
        "Zero-shot",
        "selected_baseline",
        "legacy/step8_multimodel_icl/generate_step8.py",
        "Full 5-fold Step 8 outputs exist for all main models.",
        "Reference condition for all paired comparisons.",
    ),
    Variant(
        "baseline",
        "gtr_note_pos_k1",
        "RA-ICL positive note retrieval",
        "candidate",
        "legacy/step8_multimodel_icl/generate_step8.py; legacy/pilot_12_ra_icl",
        "Step 8 and Pilot 12 compare retrieved positive examples.",
        "Uses fold-specific correct pools and GTR note embeddings.",
    ),
    Variant(
        "baseline",
        "gtr_note_neg_k1",
        "RA-ICL negative retrieval",
        "candidate",
        "legacy/step8_multimodel_icl/generate_step8.py",
        "Included in Step 8 and negative k-sweep.",
        "Often useful as a mistake-to-avoid prompt; can hurt if over-applied.",
    ),
    Variant(
        "baseline",
        "gtr_note_posneg_k1",
        "Contrastive RA-ICL",
        "candidate",
        "legacy/step8_multimodel_icl/generate_step8.py",
        "Step 8 compares positive+negative examples.",
        "Keeps positive style and negative failure mode side by side.",
    ),
    Variant(
        "baseline",
        "cot_evidence",
        "Evidence-first CoT",
        "candidate",
        "legacy/step8_multimodel_icl/generate_step8.py",
        "Full Step 8 condition.",
        "Asks model to extract evidence before answer.",
    ),
    Variant(
        "baseline",
        "cot_conclusion",
        "Conclusion-first CoT",
        "candidate",
        "legacy/step8_multimodel_icl/generate_step8.py",
        "Full Step 8 condition.",
        "Asks model to answer first, then explain.",
    ),
    Variant(
        "baseline",
        "multiturn",
        "Multi-turn few-shot",
        "pilot_candidate",
        "legacy/step8_multimodel_icl/generate_step8.py",
        "Full Step 8 condition.",
        "Uses chat-template multiturn demonstration where supported.",
    ),
    Variant(
        "judge",
        "gpt4o_stage1_binary_T0",
        "Trusted GPT-4o binary judge",
        "selected_judge",
        "legacy/step9_self_correction/v2/judge.py; legacy/ichl/judges/gpt4o_stage1_binary_judge.py",
        "Validated against human A∩B gold subset; target >=92% agreement and kappa >=0.74.",
        "Use for final paired labels and corrected-answer labels.",
    ),
    Variant(
        "judge",
        "legacy_step8_T01",
        "Legacy Step 8 GPT-4o judge",
        "legacy",
        "legacy/step8_multimodel_icl/evaluate_step8_binary.py",
        "Existing Step 8 evaluated CSVs use this in many places.",
        "Retained for provenance and label-drift comparison.",
    ),
    Variant(
        "correction",
        "step9_v2_D2_union_V2",
        "D2 detection + union evidence retrieval + V2 verdict",
        "selected_current",
        "legacy/step9_self_correction/v2/run_pipeline.py; multi_model_pilot.py",
        "Latest pre-atom implementation with audit logs and evidence quote verification.",
        "Canonical implementation for refactor execution.",
    ),
    Variant(
        "correction",
        "regen_count_compare",
        "Zero-shot regeneration + count-compare verdict",
        "candidate",
        "legacy/step9_self_correction/v2/regen_pilot.py",
        "Pilot/fullscale outputs exist under output/step9_v2/multi_model.",
        "No detection gate; useful shoulder-by-shoulder against guided correction.",
    ),
    Variant(
        "correction",
        "regen_v3_reconciled",
        "Regeneration V3 reconciled verdict",
        "pilot_candidate",
        "legacy/step9_self_correction/v2/regen_v3_pilot.py; reconcile_v3.py",
        "Reconciles old regex/Qwen3 disagreement rule after pilot runs.",
        "Included as pilot evidence, not default final.",
    ),
    Variant(
        "correction",
        "error_taxonomy_S1_P1Pool_V1",
        "Older S1/P1+Pool/V1 taxonomy pipeline",
        "legacy_reference",
        "legacy/step9_self_correction/error_taxonomy",
        "Written final-results doc exists; some tests differ from v2.",
        "Preserved for comparison and prompt provenance.",
    ),
    Variant(
        "prompt_engineering",
        "ichl_detection_correction_verdict",
        "ICHL prompt-engineering variants",
        "pilot_family",
        "legacy/ichl/prompt_engineering",
        "Contains prompt seeds, correction sub-variants, parser pilots, and fullscale scripts.",
        "Use decision matrix before promoting any sub-variant to selected.",
    ),
]


def write_matrix(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Variant Decision Matrix",
        "",
        "| Family | Variant | Status | Source | Evidence | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for v in VARIANTS:
        lines.append(
            f"| {v.family} | `{v.variant_id}`<br>{v.label} | {v.status} | "
            f"`{v.source}` | {v.evidence} | {v.notes} |"
        )
    out.write_text("\n".join(lines) + "\n")
    return out

