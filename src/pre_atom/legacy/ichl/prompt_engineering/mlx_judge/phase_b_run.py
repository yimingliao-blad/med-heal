"""Phase B: Run V0 / V4 / V8 MLX judge baselines on the 300-item dev set.

Outputs:
    output/ichl/mlx_judge/phase_b/{v0,v4,v8}_dev.jsonl   (per-item preds)
    output/ichl/mlx_judge/phase_b/summary.json
    output/ichl/mlx_judge/phase_b/summary.md

Run:
    PYTHONPATH=src .venv/bin/python -m ichl.prompt_engineering.mlx_judge.phase_b_run
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

from ichl.prompt_engineering.mlx_judge.icl_selector import load_pool, select_balanced
from ichl.prompt_engineering.mlx_judge.mlx_judge import MLXJudge

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SPLIT_DIR = PROJECT_ROOT / "output" / "ichl" / "mlx_judge" / "splits"
PHASE_B_DIR = PROJECT_ROOT / "output" / "ichl" / "mlx_judge" / "phase_b"
PHASE_B_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_JSONL = PROJECT_ROOT / "output" / "EHRNoteQA_processed.jsonl"

CONCURRENCY = 2  # MLX C=2 per MEMORY
SEED = 42


# ───────────── Note lookup ─────────────
def build_note_lookup() -> dict[str, str]:
    """patient_id → note (prefer note_1; fallback concatenated)."""
    lookup: dict[str, str] = {}
    with open(PROCESSED_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = str(r["patient_id"])
            note = (r.get("note_1") or "").strip()
            if not note:
                # fall back to any non-empty note
                for k in ("note_2", "note_3"):
                    v = (r.get(k) or "").strip()
                    if v:
                        note = v
                        break
            lookup[pid] = note
    return lookup


# ───────────── Data loading ─────────────
def load_jsonl(p: Path) -> list[dict]:
    rows: list[dict] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def attach_notes(rows: list[dict], note_lookup: dict[str, str]) -> list[dict]:
    for r in rows:
        pid = str(r["patient_id"])
        r["note"] = note_lookup.get(pid, "")
    return rows


# ───────────── Runner ─────────────
def run_variant(
    variant: str,
    dev_rows: list[dict],
    icl_examples: list[dict] | None,
    judge: MLXJudge,
    out_path: Path,
) -> list[dict]:
    """Run one variant over all dev items with C=2 concurrency. Save per-item."""
    print(f"[{variant}] N={len(dev_rows)}, ICL examples={len(icl_examples) if icl_examples else 0}")

    results: list[dict | None] = [None] * len(dev_rows)
    t0 = time.monotonic()

    def _one(i: int, row: dict) -> tuple[int, dict]:
        out = judge.judge(
            note=row.get("note", ""),
            question=row["question"],
            ground_truth=row["ground_truth"],
            model_answer=row["model_answer"],
            prompt_version=variant,
            icl_examples=icl_examples,
        )
        # attach identifying info
        rec = {
            "target": row["target"],
            "patient_id": row["patient_id"],
            "fold_id": row["fold_id"],
            "gold_label": int(row["binary_correct"]),
            "mlx_label": out["label"],
            "raw": out["raw"],
            "latency_s": out["latency_s"],
            "prompt_tokens": out["prompt_tokens"],
            "completion_tokens": out["completion_tokens"],
            "success": out["success"],
            "error": out.get("error"),
            "prompt_version": variant,
        }
        return i, rec

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = [ex.submit(_one, i, row) for i, row in enumerate(dev_rows)]
        done = 0
        for fut in as_completed(futures):
            i, rec = fut.result()
            results[i] = rec
            done += 1
            if done % 25 == 0 or done == len(dev_rows):
                elapsed = time.monotonic() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(dev_rows) - done) / rate if rate > 0 else 0
                print(f"  [{variant}] {done}/{len(dev_rows)}  elapsed={elapsed:.1f}s  eta={eta:.0f}s")

    # Save per-item
    with open(out_path, "w") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")

    wall = time.monotonic() - t0
    print(f"[{variant}] DONE in {wall:.1f}s → {out_path}")
    return results  # type: ignore[return-value]


# ───────────── Metrics ─────────────
def compute_metrics(items: list[dict], scope_filter=None) -> dict:
    if scope_filter is not None:
        items = [r for r in items if scope_filter(r)]
    n_total = len(items)
    if n_total == 0:
        return {"n": 0}

    none_count = sum(1 for r in items if r["mlx_label"] is None)
    parsed = [r for r in items if r["mlx_label"] is not None]
    n_parsed = len(parsed)

    latencies = [r["latency_s"] for r in items if r.get("latency_s", -1) > 0]
    mean_lat = sum(latencies) / len(latencies) if latencies else 0.0

    if n_parsed == 0:
        return {
            "n": n_total,
            "n_parsed": 0,
            "none_rate": none_count / n_total,
            "mean_latency_s": mean_lat,
        }

    y_true = [r["gold_label"] for r in parsed]
    y_pred = [r["mlx_label"] for r in parsed]
    agree = sum(int(a == b) for a, b in zip(y_true, y_pred)) / n_parsed

    try:
        kappa = float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        kappa = float("nan")

    # Confusion matrix (labels=[0,1]) → [[tn, fp], [fn, tp]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])

    prf = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    prec_0, prec_1 = float(prf[0][0]), float(prf[0][1])
    rec_0, rec_1 = float(prf[1][0]), float(prf[1][1])
    f1_0, f1_1 = float(prf[2][0]), float(prf[2][1])

    # Agreement including None as disagreement (total denominator)
    agree_total = sum(int(r["gold_label"] == r["mlx_label"]) for r in items) / n_total

    return {
        "n": n_total,
        "n_parsed": n_parsed,
        "none_rate": none_count / n_total,
        "agreement_parsed": agree,
        "agreement_total": agree_total,
        "cohen_kappa": kappa,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "label1": {"precision": prec_1, "recall": rec_1, "f1": f1_1},
        "label0": {"precision": prec_0, "recall": rec_0, "f1": f1_0},
        "mean_latency_s": mean_lat,
    }


def per_target_metrics(items: list[dict]) -> dict:
    out: dict[str, dict] = {}
    targets = sorted({r["target"] for r in items})
    for t in targets:
        out[t] = compute_metrics(items, scope_filter=lambda r, _t=t: r["target"] == _t)
    return out


# ───────────── Summary writer ─────────────
def _fmt(x, n=3):
    if x is None:
        return "-"
    try:
        if isinstance(x, float):
            return f"{x:.{n}f}"
    except Exception:
        pass
    return str(x)


def write_summary(all_results: dict[str, list[dict]], walltimes: dict[str, float]) -> dict:
    summary: dict = {"variants": {}, "walltimes_s": walltimes}
    for v, items in all_results.items():
        summary["variants"][v] = {
            "overall": compute_metrics(items),
            "per_target": per_target_metrics(items),
            "n_calls": len(items),
            "n_success": sum(1 for r in items if r.get("success")),
        }

    (PHASE_B_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # Markdown summary
    lines: list[str] = []
    lines.append("# Phase B — MLX Judge Baseline Results")
    lines.append("")
    lines.append(f"**Dev set**: 300 items, 4 targets × 2 labels (~37–38 per cell).")
    lines.append(f"**Model**: Qwen3.5-27B-6bit via MLX ({'http://192.168.68.107:8800'}).")
    lines.append(f"**Concurrency**: C=2.")
    lines.append("")

    # Wall times
    lines.append("## Wall-clock")
    lines.append("")
    for v, w in walltimes.items():
        lines.append(f"- **{v}**: {w:.1f} s")
    lines.append("")

    # Overall metrics
    lines.append("## Overall metrics per variant")
    lines.append("")
    lines.append("| Variant | N | Parsed | None% | Agree (parsed) | Agree (total) | κ | Lat s |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for v, info in summary["variants"].items():
        o = info["overall"]
        lines.append(
            f"| {v} | {o['n']} | {o['n_parsed']} | {_fmt(o['none_rate'])} | "
            f"{_fmt(o.get('agreement_parsed'))} | {_fmt(o.get('agreement_total'))} | "
            f"{_fmt(o.get('cohen_kappa'))} | {_fmt(o.get('mean_latency_s'), 2)} |"
        )
    lines.append("")

    # Per-target agreement table
    lines.append("## Agreement by target × variant (agreement_parsed)")
    lines.append("")
    # targets from first variant
    first_v = next(iter(summary["variants"]))
    targets = sorted(summary["variants"][first_v]["per_target"].keys())
    lines.append("| Target | " + " | ".join(summary["variants"].keys()) + " |")
    lines.append("|" + "---|" * (len(summary["variants"]) + 1))
    for t in targets:
        row = [t]
        for v in summary["variants"]:
            pt = summary["variants"][v]["per_target"].get(t, {})
            row.append(_fmt(pt.get("agreement_parsed")))
        lines.append("| " + " | ".join(row) + " |")
    # Overall row
    row = ["**OVERALL**"]
    for v in summary["variants"]:
        row.append(_fmt(summary["variants"][v]["overall"].get("agreement_parsed")))
    lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Per-class metrics
    lines.append("## Per-class metrics (label 1 = correct)")
    lines.append("")
    lines.append("| Variant | Prec(1) | Rec(1) | F1(1) | Prec(0) | Rec(0) | F1(0) | TP | TN | FP | FN |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for v, info in summary["variants"].items():
        o = info["overall"]
        if "label1" not in o:
            continue
        cm = o["confusion_matrix"]
        lines.append(
            f"| {v} | {_fmt(o['label1']['precision'])} | {_fmt(o['label1']['recall'])} | {_fmt(o['label1']['f1'])} | "
            f"{_fmt(o['label0']['precision'])} | {_fmt(o['label0']['recall'])} | {_fmt(o['label0']['f1'])} | "
            f"{cm['tp']} | {cm['tn']} | {cm['fp']} | {cm['fn']} |"
        )
    lines.append("")

    # Lift
    lines.append("## Lift summary")
    lines.append("")
    if all(v in summary["variants"] for v in ("V0", "V4", "V8")):
        a0 = summary["variants"]["V0"]["overall"].get("agreement_parsed") or 0
        a4 = summary["variants"]["V4"]["overall"].get("agreement_parsed") or 0
        a8 = summary["variants"]["V8"]["overall"].get("agreement_parsed") or 0
        k0 = summary["variants"]["V0"]["overall"].get("cohen_kappa") or 0
        k4 = summary["variants"]["V4"]["overall"].get("cohen_kappa") or 0
        k8 = summary["variants"]["V8"]["overall"].get("cohen_kappa") or 0
        lines.append(f"- V0→V4 agreement lift: {a4 - a0:+.3f} ({a0:.3f} → {a4:.3f})")
        lines.append(f"- V4→V8 agreement lift: {a8 - a4:+.3f} ({a4:.3f} → {a8:.3f})")
        lines.append(f"- V0→V4 κ lift: {k4 - k0:+.3f} ({k0:.3f} → {k4:.3f})")
        lines.append(f"- V4→V8 κ lift: {k8 - k4:+.3f} ({k4:.3f} → {k8:.3f})")
        best = max(("V0", "V4", "V8"), key=lambda v: (
            summary["variants"][v]["overall"].get("cohen_kappa") or -1))
        lines.append("")
        lines.append(f"**Recommended Phase C start point**: **{best}** (highest κ).")

    (PHASE_B_DIR / "summary.md").write_text("\n".join(lines))
    return summary


# ───────────── Main ─────────────
def main():
    print("Loading splits…")
    dev_rows = load_jsonl(SPLIT_DIR / "dev.jsonl")
    pool_rows = load_pool(SPLIT_DIR / "train_pool.jsonl")
    print(f"  dev={len(dev_rows)}, pool={len(pool_rows)}")

    print("Building note lookup…")
    note_lookup = build_note_lookup()
    dev_rows = attach_notes(dev_rows, note_lookup)
    # also attach notes to the pool for ICL examples
    pool_rows = attach_notes(pool_rows, note_lookup)
    missing = sum(1 for r in dev_rows if not r.get("note"))
    print(f"  note lookup built. missing_notes_in_dev={missing}/{len(dev_rows)}")

    print("Selecting ICL examples…")
    icl_v4 = select_balanced(pool_rows, k_right=2, k_wrong=2, seed=SEED)
    icl_v8 = select_balanced(pool_rows, k_right=4, k_wrong=4, seed=SEED)
    print(f"  V4 ICL: {len(icl_v4)} examples ({[(e['target'], e['binary_correct']) for e in icl_v4]})")
    print(f"  V8 ICL: {len(icl_v8)} examples ({[(e['target'], e['binary_correct']) for e in icl_v8]})")

    import os as _os
    client_name = _os.environ.get("MLX_JUDGE_CLIENT", "mlx-qwen35")
    print(f"Client: {client_name}")
    judge = MLXJudge(client_name=client_name)
    all_results: dict[str, list[dict]] = {}
    walltimes: dict[str, float] = {}

    for variant, icl in [("V0", None), ("V4", icl_v4), ("V8", icl_v8)]:
        out_path = PHASE_B_DIR / f"{variant.lower()}_dev.jsonl"
        t0 = time.monotonic()
        results = run_variant(variant, dev_rows, icl, judge, out_path)
        walltimes[variant] = time.monotonic() - t0
        all_results[variant] = results

    print("Writing summary…")
    summary = write_summary(all_results, walltimes)
    print(f"Done. Summary → {PHASE_B_DIR / 'summary.md'}")

    # Print headline
    print("\n=== Headline ===")
    for v, info in summary["variants"].items():
        o = info["overall"]
        print(
            f"{v}: agree_parsed={_fmt(o.get('agreement_parsed'))}  "
            f"κ={_fmt(o.get('cohen_kappa'))}  none={_fmt(o.get('none_rate'))}  "
            f"wall={walltimes[v]:.0f}s"
        )


if __name__ == "__main__":
    main()
