#!/usr/bin/env python3
"""Summarize Gate 1 multirun bakeoff arms side-by-side."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / 'runs' / 'selfdetect_raicl_verdict'


def arm_slug(detect_k: int, gen_k: int, verdict_k: int, note_ctx: str) -> str:
    if detect_k > 1 or gen_k > 1:
        k_tag = f"dk{detect_k}_gk{gen_k}_vk{verdict_k}"
    else:
        k_tag = f"vk{verdict_k}"
    return (
        f"qwen2.5-7b-instruct_input-qwen2.5-7b-instruct_nw25_nc25_seed42_"
        f"meta_plan_confirm_natural_gpt4o-mini-helper-v2_operation_guided_"
        f"multi_dimension_{k_tag}_{note_ctx}"
    )


def load_summary(slug: str) -> dict | None:
    p = RUNS_DIR / slug / 'summary.json'
    if not p.exists():
        return None
    return json.loads(p.read_text())


def fmt(v) -> str:
    if v is None:
        return '-'
    if isinstance(v, float):
        return f'{v:.2f}'
    return str(v)


def main(note_ctx: str = 'dynamic_spans') -> int:
    arms = [
        ('Arm 1 K=1', 1, 1, 1),
        ('Arm 2 K=3-det', 3, 1, 1),
        ('Arm 3 K=3-det+ver', 3, 1, 3),
        ('Arm 4 K=3-det+gen', 3, 3, 1),
    ]
    cols = [
        'n', 'detected', 'accepted', 'fixes', 'breaks', 'net',
        'transition_judged', 'transition_fixes', 'transition_breaks', 'transition_net',
        'correction_candidate_fixes', 'correction_candidate_breaks', 'correction_candidate_net',
        'errors',
    ]
    header = f"{'Arm':<22} " + ' '.join(f'{c[:5]:<6}' for c in cols)
    print(header)
    print('-' * len(header))
    for label, dk, gk, vk in arms:
        slug = arm_slug(dk, gk, vk, note_ctx)
        s = load_summary(slug)
        if s is None:
            print(f"{label:<22} MISSING: runs/selfdetect_raicl_verdict/{slug}")
            continue
        m = s.get('summary', {})
        vals = [fmt(m.get(c)) for c in cols]
        print(f"{label:<22} " + ' '.join(f'{v:<6}' for v in vals))
        # Also print K vote distributions when present.
        det_votes = m.get('detect_vote_incorrect_counts')
        if det_votes:
            print(f"  detection K-vote (count INCORRECT samples): {det_votes}")
        chosen_idx = m.get('chosen_candidate_idx')
        if chosen_idx:
            print(f"  chosen candidate idx (gen-k>1): {chosen_idx}")
    return 0


if __name__ == '__main__':
    ctx = sys.argv[1] if len(sys.argv) > 1 else 'dynamic_spans'
    raise SystemExit(main(ctx))
