"""Prompt-engineering optimization loop.

Round 0: evaluate every seed against pilot data → baseline
Round k: top-N → rule-based mutate (bone) → LLM polish (flesh) → evaluate →
         merge + prune back to top-N
Stop: Δ best-score < ε for 2 consecutive rounds, OR max_rounds reached.

Output (under out_dir):
    config.yaml
    pilot_data.jsonl
    rounds/round_NN.json           # per-round pool + scores + provenance
    raw_outputs/<candidate>/<item_id>.json   # every LLM call logged
    per_item/<candidate>.jsonl     # full evaluate_cell per-item rows
    final_ranking.json
    top_candidates/candidate_01.txt ... candidate_0N.txt

Per Notion `Claude: Module: Prompt Engineering Iteration` and
`Claude: Principle: Experiment Audit Guidelines`.

Design choice 2026-04-22: use `evaluate_cell` (not the legacy metric shim)
so per-item records and parser agreement are tracked across rounds. One
vLLM client and one MLX parser client are built ONCE per target and reused
across all variants — avoids TCP reconnects / model reloads.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ichl.clients.base import LLMClient
from ichl.clients.factory import make_client
from ichl.prompt_engineering.evaluator import evaluate_cell
from ichl.prompt_engineering.mutator import llm_polish, rule_based_mutate
from ichl.prompt_engineering.mutator.llm import PolishedVariant
from ichl.prompt_engineering.mutator.rules import Skeleton
from ichl.prompt_engineering.parsers import LLMParser, RegexParser
from ichl.prompt_engineering.pool import (
    SeedPrompt,
    VariationPrimitive,
    load_seeds,
    load_variations,
)


@dataclass
class Candidate:
    """A prompt candidate inside the optimization pool."""
    name: str
    template: str
    score: float = float("-inf")
    round_added: int = 0
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    top_candidates: list[Candidate]
    rankings: list[Candidate]
    history: list[dict[str, Any]]
    run_dir: Path


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def _write_round(path: Path, round_idx: int, pool: list[Candidate]) -> None:
    path.write_text(json.dumps({
        "round": round_idx,
        "pool": [asdict(c) for c in pool],
    }, indent=2, default=str))


def _candidate_to_dict(c: Candidate) -> dict[str, Any]:
    return asdict(c)


# ─────────────────────── the loop ───────────────────────

def run(
    *,
    step: str,
    target_client_name: str,
    base_prompt_pool: Path | str,
    variation_pool: Path | str,
    pilot_data: list[dict[str, Any]],
    max_tokens: int,
    out_dir: Path | str,
    tool_model: str = "gpt-4o",
    top_n_candidates: int = 3,
    max_rounds: int = 5,
    epsilon: float = 0.02,
    variants_per_candidate: int = 3,
    seed: int = 20260422,
    regex_pattern: str | None = None,
    llm_parser_user_tpl: str | None = None,
    enable_thinking: bool | None = None,
) -> RunResult:
    """Run the prompt-engineering loop end-to-end for ONE target.

    Args:
        step:               ICL step name (e.g. 'detection') — used as a tag
        target_client_name: the vLLM target to evaluate against (one model only)
        base_prompt_pool:   path to seeds.yaml
        variation_pool:     path to variations YAML (detection.yaml / general.yaml)
        pilot_data:         the 40-item pilot (loaded by caller)
        max_tokens:         per-target from Step 0
        out_dir:            run directory
        tool_model:         GPT-4o (or mlx-qwen35) for llm_polish at T=1
        top_n_candidates:   how many to keep at each prune
        max_rounds:         Rounds 1..K upper bound
        epsilon:            Δ threshold for stop condition
        variants_per_candidate: new variants generated per parent per round
        seed:               RNG seed for primitive sampling
        regex_pattern:      override default regex; pass `None` to use defaults
        llm_parser_user_tpl: override LLM-parser user prompt
        enable_thinking:    Qwen3-style thinking toggle
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rounds").mkdir(exist_ok=True)
    raw_log_dir = out_dir / "raw_outputs"
    raw_log_dir.mkdir(exist_ok=True)
    per_item_dir = out_dir / "per_item"
    per_item_dir.mkdir(exist_ok=True)

    # Persist config for audit.
    config_snapshot = {
        "step": step, "target_client_name": target_client_name,
        "base_prompt_pool": str(base_prompt_pool),
        "variation_pool": str(variation_pool),
        "tool_model": tool_model,
        "max_tokens": max_tokens,
        "top_n_candidates": top_n_candidates,
        "max_rounds": max_rounds, "epsilon": epsilon,
        "variants_per_candidate": variants_per_candidate,
        "seed": seed,
        "regex_pattern": regex_pattern,
    }
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config_snapshot, sort_keys=False))

    # Build shared clients + parsers (reused across every variant).
    target_client: LLMClient = make_client(target_client_name)
    parsers = [
        RegexParser(pattern=regex_pattern) if regex_pattern else RegexParser(),
        LLMParser(user_template=llm_parser_user_tpl) if llm_parser_user_tpl else LLMParser(),
    ]

    seeds: list[SeedPrompt] = load_seeds(base_prompt_pool)
    primitives: list[VariationPrimitive] = load_variations(variation_pool)
    rng = random.Random(seed)

    # Freeze pilot_data to disk for traceability.
    with open(out_dir / "pilot_data.jsonl", "w") as f:
        for item in pilot_data:
            f.write(json.dumps(item, default=str) + "\n")

    # Dedup tracker: template-hash → best known score.
    seen: dict[str, float] = {}

    def _evaluate_variant(name: str, template: str, round_idx: int) -> tuple[float, dict[str, Any]]:
        """Run evaluate_cell for a single variant; returns (score, summary)."""
        per_item_path = per_item_dir / f"{name}.jsonl"
        log_dir = raw_log_dir / name
        result = evaluate_cell(
            candidate_name=name,
            prompt_template=template,
            pilot_data=pilot_data,
            target_client=target_client,
            parsers=parsers,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            log_dir=log_dir,
            per_item_path=per_item_path,
        )
        return result.score, result.summary.as_dict()

    # ── Round 0: evaluate seeds ──
    print(f"\n=== Round 0 / {target_client_name} — evaluating {len(seeds)} seeds ===")
    pool: list[Candidate] = []
    for s in seeds:
        h = _hash(s.template)
        score, summary = _evaluate_variant(s.name, s.template, 0)
        seen[h] = score
        pool.append(Candidate(
            name=s.name, template=s.template, score=score, round_added=0,
            provenance={"seed": s.name, "origin": "round_0_baseline", "summary": summary},
        ))
        print(f"  {s.name}: acc={score:.3f}")
    pool.sort(key=lambda c: -c.score)
    _write_round(out_dir / "rounds" / "round_00.json", 0, pool)

    history: list[dict[str, Any]] = [{
        "round": 0,
        "candidates": [{"name": c.name, "score": c.score} for c in pool],
    }]

    # ── Rounds k ≥ 1 ──
    for k in range(1, max_rounds + 1):
        print(f"\n=== Round {k} / {target_client_name} — mutating top-{top_n_candidates} ===")
        parents = pool[:top_n_candidates]
        new_candidates: list[Candidate] = []

        for parent in parents:
            parent_seed = SeedPrompt(name=parent.name, template=parent.template)
            all_skeletons = rule_based_mutate(parent_seed, primitives)
            # Round-robin over primitives so different rounds try different rules.
            rng.shuffle(all_skeletons)

            n_rule = max(1, variants_per_candidate // 2)
            n_llm = max(0, variants_per_candidate - n_rule)

            # Rule-only skeletons (bone)
            picked_rule: list[Skeleton] = []
            for sk in all_skeletons:
                if len(picked_rule) >= n_rule:
                    break
                h = _hash(sk.template)
                if h in seen:
                    continue
                picked_rule.append(sk)
                seen[h] = float("-inf")  # reserve

            # LLM-polished variants (flesh)
            picked_llm: list[PolishedVariant] = []
            polish_pool = [sk for sk in all_skeletons if _hash(sk.template) in seen]
            polish_pool_unique = [sk for sk in all_skeletons[:n_llm * 2]]
            for sk in polish_pool_unique:
                if len(picked_llm) >= n_llm:
                    break
                try:
                    polished_list = llm_polish(
                        sk, tool_model=tool_model, n_variants=1,
                        temperature=1.0, log_dir=raw_log_dir / "polish",
                    )
                except Exception as e:
                    print(f"    [polish-error] {sk.name}: {e}")
                    continue
                for p in polished_list:
                    h = _hash(p.template)
                    if h in seen:
                        continue
                    picked_llm.append(p)
                    seen[h] = float("-inf")
                    if len(picked_llm) >= n_llm:
                        break

            for sk in picked_rule:
                new_candidates.append(Candidate(
                    name=f"r{k:02d}__{sk.name}",
                    template=sk.template,
                    round_added=k,
                    provenance={"parent": parent.name, "rule": sk.rule_name,
                                "kind": sk.rule_kind, "polish": None},
                ))
            for p in picked_llm:
                new_candidates.append(Candidate(
                    name=f"r{k:02d}__{p.name}",
                    template=p.template,
                    round_added=k,
                    provenance={"parent": parent.name, "polish": p.tool_model,
                                "parent_skeleton": p.parent_skeleton},
                ))

        # Evaluate new variants.
        print(f"  generated {len(new_candidates)} new variants; evaluating...")
        for c in new_candidates:
            score, summary = _evaluate_variant(c.name, c.template, k)
            c.score = score
            c.provenance["summary"] = summary
            seen[_hash(c.template)] = score
            print(f"    {c.name}: acc={score:.3f}")

        pool = sorted(pool + new_candidates, key=lambda c: -c.score)[:top_n_candidates]
        history.append({
            "round": k,
            "candidates": [{"name": c.name, "score": c.score} for c in pool],
        })
        _write_round(out_dir / "rounds" / f"round_{k:02d}.json", k, pool)

        # Stop: Δ best-score < ε for 2 consecutive rounds
        if k >= 2:
            d1 = history[-1]["candidates"][0]["score"] - history[-2]["candidates"][0]["score"]
            d2 = history[-2]["candidates"][0]["score"] - history[-3]["candidates"][0]["score"]
            if d1 < epsilon and d2 < epsilon:
                print(f"\n(stop) Δ-best-score < {epsilon} for 2 rounds; halting at round {k}")
                break

    # ── Final ranking ──
    final_ranking = sorted(pool, key=lambda c: -c.score)
    (out_dir / "final_ranking.json").write_text(
        json.dumps([_candidate_to_dict(c) for c in final_ranking], indent=2, default=str)
    )
    top_dir = out_dir / "top_candidates"
    top_dir.mkdir(exist_ok=True)
    top_n = final_ranking[:top_n_candidates]
    for i, c in enumerate(top_n, start=1):
        (top_dir / f"candidate_{i:02d}.txt").write_text(
            f"# name: {c.name}\n# score: {c.score:.3f}\n# round_added: {c.round_added}\n\n{c.template}"
        )

    return RunResult(
        top_candidates=top_n, rankings=final_ranking,
        history=history, run_dir=out_dir,
    )
