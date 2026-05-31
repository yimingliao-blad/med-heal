"""T0 Anchor — Parser Sub-Pilot driver.

Per Notion principle "Claude: Principle: Regex Parser Unreliability" and the
T0 plan's Parser Sub-Pilot section: for every (target, sub-variant-run),
re-parse the 5 Step-0 raw outputs with BOTH a regex parser and an MLX LLM
parser, send both extracted answers through the Stage-1 binary GPT-4o judge,
and flag runs with <95 % agreement as "MLX-as-primary" for pilot/full-scale.

Inputs read from:
    output/ichl/correction/runs/<run_dir>/raw_outputs/<target>/<sv>/<item>.json
    output/step8/<target>/fold_<k>/zeroshot_evaluated_binary.csv  (ground_truth)

Outputs written to:
    output/ichl/correction/runs/<run_dir>/parser_sub_pilot/
        per_item/<target>_<sv>.jsonl      — 5 rows per run
        summary.json                      — aggregate agreement per (target, sv)
        summary.md                        — human-readable table + primary flags

Execution notes:
  - MLX calls: ThreadPoolExecutor(max_workers=2) per MEMORY rule.
  - GPT-4o calls: sequential (MEMORY user preference).
  - No vLLM dependency.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

# Make sure we can import ichl when run as a script
_SRC_ROOT = Path(__file__).resolve().parents[3]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from ichl.prompt_engineering.correction.parsers import (  # noqa: E402
    ExtractedAnswer,
    MLXCorrectionParser,
    RegexCorrectionParser,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "output"
    / "ichl"
    / "correction"
    / "runs"
    / "20260423_1909_t0_anchor"
)
STEP8_DIR = PROJECT_ROOT / "output" / "step8"

TARGETS = [
    "deepseek-r1-distill-llama-8b",
    "qwen2.5-7b-instruct",
    "qwen3-8b",
    "llama-3.1-8b-instruct",
]

MLX_PRIMARY_THRESHOLD = 0.95  # agreement below this flag → MLX-as-primary


# ─────────────────────── data loading ───────────────────────


def _load_ground_truth_map(target: str) -> dict[tuple[int, int], str]:
    """Return (fold, patient_id) -> ground_truth for a target.

    Reads output/step8/<target>/fold_<k>/zeroshot_evaluated_binary.csv.
    """
    gt: dict[tuple[int, int], str] = {}
    for fold in range(5):
        f = STEP8_DIR / target / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        if not f.exists():
            print(f"[WARN] ground-truth file missing: {f}")
            continue
        df = pd.read_csv(f)
        fold_col = "fold_id" if "fold_id" in df.columns else "fold"
        for _, r in df.iterrows():
            try:
                pid = int(r["patient_id"])
                fd = int(r[fold_col])
            except Exception:
                continue
            gt[(fd, pid)] = str(r.get("ground_truth", "") or "")
    return gt


def _list_raw_output_files(run_dir: Path, target: str) -> dict[str, list[Path]]:
    """Return {sub_variant: [json files]} under run_dir/raw_outputs/<target>."""
    target_dir = run_dir / "raw_outputs" / target
    out: dict[str, list[Path]] = {}
    if not target_dir.exists():
        return out
    for sv_dir in sorted(target_dir.iterdir()):
        if not sv_dir.is_dir():
            continue
        files = sorted(sv_dir.glob("*.json"))
        if files:
            out[sv_dir.name] = files
    return out


def _load_raw_output(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


# ─────────────────────── text-similarity diagnostic ───────────────────────


def _char_overlap(a: str, b: str) -> float:
    """Crude 90%-char-overlap proxy: Jaccard over char 3-grams."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    grams_a = {a[i:i + 3] for i in range(max(1, len(a) - 2))}
    grams_b = {b[i:i + 3] for i in range(max(1, len(b) - 2))}
    if not grams_a or not grams_b:
        return 0.0
    inter = len(grams_a & grams_b)
    union = len(grams_a | grams_b)
    return inter / union if union > 0 else 0.0


# ─────────────────────── Stage-1 binary judge ───────────────────────


_JUDGE_SYSTEM = (
    "You are a medical expert evaluating an AI model's answer to a clinical question."
)


def _judge_binary(client, note: str, question: str, ground_truth: str,
                  model_answer: str, *, gpt_model: str = "gpt-4o") -> int | None:
    """Stage-1 binary GPT-4o judge. Returns 1, 0, or None.

    Matches src/ichl/judges/gpt4o_stage1_binary_judge.py exactly (the canonical Stage-1 binary judge).
    """
    user_content = (
        f"DISCHARGE SUMMARY:\n{note}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CORRECT ANSWER (Ground Truth):\n{ground_truth}\n\n"
        f"MODEL'S ANSWER:\n{model_answer}\n\n"
        f"Task: Evaluate if the model's answer is correct compared to the ground truth.\n\n"
        f"Respond with ONLY a single digit:\n"
        f"1 = Correct\n"
        f"0 = Incorrect"
    )
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=gpt_model,
                messages=messages,
                max_tokens=10,
                temperature=0.1,
            )
            content = resp.choices[0].message.content.strip()
            if "1" in content and "0" not in content:
                return 1
            if "0" in content:
                return 0
            return None
        except Exception as e:  # noqa: BLE001
            if attempt < 4:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"    judge ERROR: {e}")
                return None
    return None


# ─────────────────────── sub-pilot runner ───────────────────────


def _gather_mlx_parse(items: list[dict[str, Any]], target: str, sv: str,
                      mlx: MLXCorrectionParser) -> list[ExtractedAnswer]:
    """Run MLX parser on all items for one (target, sv) with C=2."""
    results: list[ExtractedAnswer | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {
            ex.submit(mlx.parse, items[i]["raw_text"], target, sv): i
            for i in range(len(items))
        }
        for f in as_completed(futs):
            idx = futs[f]
            try:
                results[idx] = f.result()
            except Exception as e:  # noqa: BLE001
                results[idx] = ExtractedAnswer(
                    answer_text="UNKNOWN", notes=f"mlx future error: {e}",
                    parser_name="mlx",
                )
    return [r if r is not None else ExtractedAnswer("UNKNOWN", notes="missing",
                                                    parser_name="mlx") for r in results]


def _run_one_svrun(
    target: str,
    sv: str,
    raw_files: list[Path],
    gt_map: dict[tuple[int, int], str],
    regex_parser: RegexCorrectionParser,
    mlx_parser: MLXCorrectionParser,
    judge_client,
    *,
    gpt_model: str,
    per_item_out: Path,
) -> dict[str, Any]:
    """Process one (target, sub-variant-run). Writes per-item JSONL + returns summary."""
    print(f"  [{target} / {sv}] loading {len(raw_files)} raw outputs")

    # Load raw outputs (the `raw_text` in Step-0 JSON is the full model
    # response BEFORE any think-strip; `text` is already think-stripped for
    # Qwen3. For parser evaluation we want the RAW text so the regex can
    # exercise its boundary heuristics.).
    items: list[dict[str, Any]] = []
    for p in raw_files:
        d = _load_raw_output(p)
        items.append({
            "path": p,
            "pilot_item_id": d.get("pilot_item_id"),
            "patient_id": int(d.get("patient_id")),
            "fold": int(d.get("fold")),
            "question": d.get("user_prompt", "").split("\n\nQuestion:", 1)[-1]
            .split("\n\nAnswer:", 1)[0].strip() if "\n\nQuestion:" in d.get("user_prompt", "") else "",
            "note": d.get("user_prompt", "").split("Discharge notes:\n", 1)[-1]
            .split("\n\nQuestion:", 1)[0].strip()
            if "Discharge notes:\n" in d.get("user_prompt", "") else "",
            "A0": d.get("A0", ""),
            "A0_binary_correct": d.get("A0_binary_correct"),
            "raw_text": d.get("raw_text", "") or d.get("text", ""),
            "text_clean_runner": d.get("text", ""),
            "finish_reason": d.get("finish_reason"),
            "completion_tokens": d.get("completion_tokens"),
        })

    # Regex parse (fast, local).
    regex_out: list[ExtractedAnswer] = []
    for it in items:
        regex_out.append(regex_parser.parse(it["raw_text"], target, sv))

    # MLX parse (parallel, C=2).
    t_mlx_start = time.monotonic()
    mlx_out = _gather_mlx_parse(items, target, sv, mlx_parser)
    t_mlx_elapsed = time.monotonic() - t_mlx_start
    print(f"  [{target} / {sv}] MLX parse done in {t_mlx_elapsed:.1f}s")

    # Judge both extractions, sequentially (2 calls per item).
    rows: list[dict[str, Any]] = []
    for it, rx, mx in zip(items, regex_out, mlx_out):
        key = (it["fold"], it["patient_id"])
        ground_truth = gt_map.get(key, "")
        if not ground_truth:
            print(f"    [WARN] no GT for {target} fold={key[0]} pid={key[1]}")

        label_regex = _judge_binary(
            judge_client, it["note"], it["question"], ground_truth,
            rx.answer_text, gpt_model=gpt_model,
        )
        time.sleep(0.1)  # gentle rate-limit
        label_mlx = _judge_binary(
            judge_client, it["note"], it["question"], ground_truth,
            mx.answer_text, gpt_model=gpt_model,
        )
        time.sleep(0.1)

        text_exact = int(rx.answer_text.strip() == mx.answer_text.strip())
        char_overlap = _char_overlap(rx.answer_text, mx.answer_text)

        row = {
            "target": target,
            "sub_variant": sv,
            "pilot_item_id": it["pilot_item_id"],
            "patient_id": it["patient_id"],
            "fold": it["fold"],
            "ground_truth_present": bool(ground_truth),
            "regex": {
                "answer_text": rx.answer_text,
                "notes": rx.notes,
            },
            "mlx": {
                "answer_text": mx.answer_text,
                "notes": mx.notes,
                "latency_s": mx.latency_s,
                "finish_reason": (mx.extra or {}).get("finish_reason"),
            },
            "label_regex": label_regex,
            "label_mlx": label_mlx,
            "label_agree": int(
                label_regex is not None and label_mlx is not None
                and label_regex == label_mlx
            ),
            "text_exact": text_exact,
            "char_overlap": char_overlap,
            "regex_is_unknown": int(rx.answer_text.strip() == "UNKNOWN"),
            "mlx_is_unknown": int(mx.answer_text.strip() == "UNKNOWN"),
        }
        rows.append(row)

    # Write per-item JSONL.
    per_item_out.parent.mkdir(parents=True, exist_ok=True)
    with open(per_item_out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"  [{target} / {sv}] wrote {per_item_out}")

    # Aggregate.
    n = len(rows)
    n_agree = sum(r["label_agree"] for r in rows)
    n_either_none = sum(
        1 for r in rows if r["label_regex"] is None or r["label_mlx"] is None
    )
    denom = n - n_either_none if n > n_either_none else n
    agreement_rate = (n_agree / denom) if denom > 0 else 0.0
    summary = {
        "target": target,
        "sub_variant_id": sv,
        "n": n,
        "n_label_agree": n_agree,
        "n_judge_none": n_either_none,
        "agreement_rate": agreement_rate,
        "n_text_exact": sum(r["text_exact"] for r in rows),
        "mean_char_overlap": (sum(r["char_overlap"] for r in rows) / n) if n else 0.0,
        "n_regex_unknown": sum(r["regex_is_unknown"] for r in rows),
        "n_mlx_unknown": sum(r["mlx_is_unknown"] for r in rows),
        "mlx_latency_s_mean": (
            sum(r["mlx"]["latency_s"] for r in rows) / n
        ) if n else 0.0,
        "flag_mlx_as_primary": bool(agreement_rate < MLX_PRIMARY_THRESHOLD),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="T0 Anchor parser sub-pilot")
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--targets", nargs="+", default=TARGETS)
    parser.add_argument("--gpt_model", default="gpt-4o")
    parser.add_argument(
        "--mlx_url", default="http://192.168.68.107:8800/v1/chat/completions",
        help="MLX server chat-completions endpoint (default: Mac Studio 192.168.68.107:8800)",
    )
    parser.add_argument(
        "--mlx_model",
        default="/Users/madblade/Projects/local-llm/models/mlx/Qwen3.5-27B-6bit-NexVeridian",
        help="MLX model ID as reported by /v1/models",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    out_dir = run_dir / "parser_sub_pilot"
    per_item_dir = out_dir / "per_item"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_item_dir.mkdir(parents=True, exist_ok=True)

    # Build clients
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Source .env first.")
        sys.exit(1)
    from openai import OpenAI
    judge_client = OpenAI(api_key=api_key)

    regex_parser = RegexCorrectionParser()
    mlx_parser = MLXCorrectionParser(url=args.mlx_url, model_name=args.mlx_model)

    # Ground truth per target
    gt_maps: dict[str, dict[tuple[int, int], str]] = {}
    for t in args.targets:
        gt_maps[t] = _load_ground_truth_map(t)
        print(f"[GT] {t}: {len(gt_maps[t])} items")

    all_summaries: list[dict[str, Any]] = []
    t_total_start = time.monotonic()

    for target in args.targets:
        sv_map = _list_raw_output_files(run_dir, target)
        if not sv_map:
            print(f"[{target}] no raw outputs found; skipping")
            continue
        print(f"\n[{target}] sub-variant-runs found: {list(sv_map)}")
        for sv, files in sv_map.items():
            per_item_out = per_item_dir / f"{target}_{sv}.jsonl"
            summary = _run_one_svrun(
                target=target, sv=sv, raw_files=files,
                gt_map=gt_maps[target],
                regex_parser=regex_parser, mlx_parser=mlx_parser,
                judge_client=judge_client, gpt_model=args.gpt_model,
                per_item_out=per_item_out,
            )
            all_summaries.append(summary)

    # Write aggregate summary.
    elapsed = time.monotonic() - t_total_start
    payload = {
        "run_dir": str(run_dir),
        "targets": args.targets,
        "threshold_mlx_primary": MLX_PRIMARY_THRESHOLD,
        "wall_clock_s": elapsed,
        "summaries": all_summaries,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(payload, f, indent=2)

    # Human-readable md
    md = ["# T0 Anchor — Parser Sub-Pilot Summary", ""]
    md.append(f"Run dir: `{run_dir}`  ")
    md.append(f"MLX-primary threshold: `{MLX_PRIMARY_THRESHOLD:.2f}`  ")
    md.append(f"Wall clock: `{elapsed:.1f}s`  ")
    md.append("")
    md.append("| target | sub-variant | n | agreement | text-exact | regex-UNK | mlx-UNK | mlx-primary |")
    md.append("|---|---|---:|---:|---:|---:|---:|:---:|")
    for s in all_summaries:
        flag = "YES" if s["flag_mlx_as_primary"] else ""
        md.append(
            f"| {s['target']} | {s['sub_variant_id']} | {s['n']} "
            f"| {s['agreement_rate']:.2f} | {s['n_text_exact']}/{s['n']} "
            f"| {s['n_regex_unknown']} | {s['n_mlx_unknown']} | {flag} |"
        )
    md.append("")
    md.append("Flag definition: `flag_mlx_as_primary = agreement_rate < threshold`. "
              "When flagged, the MLX parser becomes authoritative for the "
              "corresponding (target, sub-variant-run); regex still runs for audit.")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")

    print(f"\nWROTE {out_dir / 'summary.json'}")
    print(f"WROTE {out_dir / 'summary.md'}")
    print(f"wall_clock_s = {elapsed:.1f}")


if __name__ == "__main__":
    main()
