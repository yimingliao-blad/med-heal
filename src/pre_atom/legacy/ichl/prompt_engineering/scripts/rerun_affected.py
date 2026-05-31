"""Surgical re-run of iteration items that had parser issues.

For each iteration per_item JSONL, identify rows with:
  - chosen_verdict == UNKNOWN  OR
  - regex_verdict != llm_verdict  OR
  - finish_reason == "length"  OR
  - unclosed `<think>` in text_clean

Re-issue just those items with the NEW max_tokens from step0_token_budget.json
and the same parsers. Overwrite those rows in the JSONL.

The top_candidates/ text files and iteration_summary.json are untouched — call
`recompute_ranking.py` next to refresh those from the corrected per_item rows.

Usage:
    python -m ichl.prompt_engineering.scripts.rerun_affected \\
        --run-dir <run_dir> --targets qwen2.5-7b-instruct qwen3-8b
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ichl.clients.factory import make_client
from ichl.common import vllm_manager
from ichl.common.pilot_loader import load_detection_pilot
from ichl.prompt_engineering.evaluator import SYSTEM_MSG, _format_prompt
from ichl.prompt_engineering.parsers import LLMParser, RegexParser


def _load_parser_config(sub_pilot_dir: Path, target: str) -> tuple[str | None, str | None]:
    pc = sub_pilot_dir / "parser_configs"
    search = [pc / target, pc / "shared_qwen_verdict_only", sub_pilot_dir]
    regex = None; llm = None
    for d in search:
        rf = d / "regex_pattern.txt"
        if regex is None and rf.exists():
            regex = rf.read_text().strip()
        lf = d / "llm_parser_prompt.txt"
        if llm is None and lf.exists():
            llm = lf.read_text()
        if regex and llm: break
    return regex, llm


def _is_affected(row: dict[str, Any]) -> bool:
    if row.get("chosen_verdict") == "UNKNOWN":
        return True
    if row.get("regex_verdict") != row.get("llm_verdict"):
        return True
    if row.get("finish_reason") == "length":
        return True
    text = row.get("text_clean") or ""
    if "<think>" in text and "</think>" not in text:
        return True
    return False


def _load_cell_template(run_dir: Path, target: str, candidate_name: str) -> str | None:
    """Resolve a candidate's template from iteration rounds or seeds."""
    # Check rounds/round_NN.json files (they carry .template per candidate)
    rounds_dir = run_dir / "iteration" / target / "rounds"
    for rf in sorted(rounds_dir.glob("round_*.json")):
        data = json.loads(rf.read_text())
        for c in data.get("pool", []):
            if c.get("name") == candidate_name:
                return c.get("template")
    # Also check the top_candidates/ files (they have headers stripped).
    tc_dir = run_dir / "iteration" / target / "top_candidates"
    for tc in sorted(tc_dir.glob("*.txt")):
        text = tc.read_text()
        if f"# name: {candidate_name}" in text:
            lines = text.splitlines()
            header_end = 0
            for i, ln in enumerate(lines):
                if not ln.startswith("#") and ln.strip():
                    header_end = i; break
            return "\n".join(lines[header_end:]).strip()
    # Also check final_ranking.json
    fr_path = run_dir / "iteration" / target / "final_ranking.json"
    if fr_path.exists():
        fr = json.loads(fr_path.read_text())
        for c in fr:
            if c.get("name") == candidate_name:
                return c.get("template")
    return None


def _reparse(row: dict[str, Any], parsers: list) -> dict[str, Any]:
    """Re-run BOTH parsers on an existing row's text_clean; update verdicts."""
    text_clean = row.get("text_clean") or ""
    parser_results = {}
    for p in parsers:
        pr = p.parse(text_clean)
        parser_results[p.name] = {
            "verdict": pr.verdict, "match_text": pr.match_text,
            "match_pos": pr.match_pos, "latency_s": pr.latency_s,
            "raw_response": pr.raw_response,
        }
    regex_v = parser_results.get("regex", {}).get("verdict", "UNKNOWN")
    llm_v = parser_results.get("llm", {}).get("verdict", "UNKNOWN")
    # Chosen: regex first, LLM fallback
    chosen_parser = "regex"; chosen_verdict = regex_v
    if regex_v == "UNKNOWN" and llm_v != "UNKNOWN":
        chosen_parser = "llm"; chosen_verdict = llm_v
    expected = "CORRECT" if row["binary_correct"] == 1 else "INCORRECT"
    row.update({
        "regex_verdict": regex_v,
        "regex_match_pos": parser_results.get("regex", {}).get("match_pos", -1),
        "regex_match_text": parser_results.get("regex", {}).get("match_text", ""),
        "llm_verdict": llm_v,
        "llm_latency_ms": round(parser_results.get("llm", {}).get("latency_s", 0.0) * 1000, 1),
        "llm_raw": parser_results.get("llm", {}).get("raw_response", ""),
        "agree": regex_v == llm_v,
        "chosen_parser": chosen_parser,
        "chosen_verdict": chosen_verdict,
        "verdict_correct": 1 if chosen_verdict == expected else 0,
    })
    return row


def _rerun_item(row: dict[str, Any], template: str, client, parsers, max_tokens: int,
                item_lookup: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Re-call target model for one item with new max_tokens; reparse; return updated row."""
    pid = int(row["patient_id"])
    # Pull the original item for note/question
    pilot_item = item_lookup.get(pid)
    if not pilot_item:
        return row  # can't find pilot item; leave untouched
    prompt = _format_prompt(template, pilot_item)
    t0 = time.monotonic()
    resp = client.call(
        system=SYSTEM_MSG, user=prompt,
        temperature=0.0, max_tokens=max_tokens,
    )
    lat = time.monotonic() - t0
    # Update the fields from the new response
    row["raw_response"] = resp.raw_text
    row["text_clean"] = resp.text
    row["response_latency_ms"] = round(resp.latency * 1000, 1)
    row["call_latency_ms"] = round(lat * 1000, 1)
    row["finish_reason"] = resp.finish_reason
    row["usage"] = resp.usage
    # Re-parse with new output
    return _reparse(row, parsers)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    sub_pilot_dir = run_dir / "sub_pilot"
    budgets = json.loads((run_dir / "step0_token_budget.json").read_text())

    for target in args.targets:
        print(f"\n{'='*60}\n=== rerun affected / {target}\n{'='*60}")
        vllm_manager.ensure_model(target, log_dir=run_dir / "logs")
        client = make_client(target)
        regex_pat, llm_tpl = _load_parser_config(sub_pilot_dir, target)
        parsers = [
            RegexParser(pattern=regex_pat) if regex_pat else RegexParser(),
            LLMParser(user_template=llm_tpl) if llm_tpl else LLMParser(),
        ]
        max_tokens = int(budgets["per_target"][target]["chosen_max_tokens"])
        print(f"  max_tokens = {max_tokens}")

        # Pre-load pilot data for item lookup (note, question, model_answer).
        pilot = load_detection_pilot(target)
        item_lookup: dict[int, dict[str, Any]] = {int(i["patient_id"]): i for i in pilot}

        per_item_dir = run_dir / "iteration" / target / "per_item"
        n_cells = 0; n_total_affected = 0
        for jf in sorted(per_item_dir.glob("*.jsonl")):
            candidate_name = jf.stem
            rows = []
            for line in jf.open():
                line = line.strip()
                if not line: continue
                try: rows.append(json.loads(line))
                except: continue
            affected_idx = [i for i, r in enumerate(rows) if _is_affected(r)]
            if not affected_idx:
                continue

            template = _load_cell_template(run_dir, target, candidate_name)
            if template is None:
                print(f"  [skip] {candidate_name[:60]}: template not found")
                continue

            print(f"  {candidate_name[:80]}: {len(affected_idx)} affected")
            for idx in affected_idx:
                rows[idx] = _rerun_item(
                    rows[idx], template, client, parsers, max_tokens, item_lookup,
                )

            # Overwrite JSONL atomically.
            tmp = jf.with_suffix(".jsonl.tmp")
            with tmp.open("w") as f:
                for r in rows:
                    f.write(json.dumps(r, default=str) + "\n")
            tmp.replace(jf)
            n_cells += 1
            n_total_affected += len(affected_idx)

        print(f"\n  {target}: re-ran {n_total_affected} items across {n_cells} cells")


if __name__ == "__main__":
    main()
