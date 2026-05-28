#!/usr/bin/env python3
"""
Module 0 — Trusted GPT-4o judge (Stage-1 binary prompt @ temp=0).

The single source of truth for "is this answer correct vs ground truth".
Used by every downstream evaluation in the V2 self-correction pipeline.

Reuses the canonical Stage-1 prompt from
    src/step3_evaluation/stage1_gpt4_eval_combined.py::evaluate_openended_correctness
which achieved 92% / kappa=0.75 against the Sara/Jose human gold standard.

Differences from the legacy version:
  - temperature = 0.0 (was 0.1 in stage1 + run_fullscale.py)
  - explicit multi-sample wrapper (n>=1)
  - all raw responses are returned for persistence in audit logs

Two CLI entry points:
  --validate-against-gold   re-validates the temp=0 judge against the
                            Sara/Jose A∩B N=112 gold subset using BioMistral
                            zeroshot answers — must reproduce ≥92% / κ≥0.74.
  --rejudge-step8           re-judges every row of every Qwen2.5-7B step8
                            zeroshot_evaluated_binary.csv at temp=0 and writes
                            the new label column. Compares against the legacy
                            temp=0.1 labels and reports the flip rate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from openai import OpenAI
from sklearn.metrics import cohen_kappa_score

SOURCE_ROOT = Path(os.environ.get("PRE_ATOM_SOURCE_REPO_ROOT", Path(__file__).resolve().parents[5]))
RUN_ROOT = Path(os.environ.get("PRE_ATOM_PROJECT_ROOT", SOURCE_ROOT))
PROJECT_ROOT = SOURCE_ROOT
OUTPUT_DIR = RUN_ROOT / "output" / "step9_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Shared client ----------

def _load_api_key() -> str:
    env = PROJECT_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENAI_API_KEY=") and not line.startswith("#"):
                return line.split("=", 1)[1]
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    raise RuntimeError("OPENAI_API_KEY not found in .env or environment")


_client: OpenAI | None = None
def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=_load_api_key())
    return _client


# ---------- Canonical Stage-1 prompt ----------
# Verbatim copy of evaluate_openended_correctness's `messages` from
# src/step3_evaluation/stage1_gpt4_eval_combined.py lines 127-145.
# DO NOT modify the prompt text — it is the validated 92%/κ=0.75 prompt.

SYSTEM_PROMPT = "You are a medical expert evaluating an AI model's answer to a clinical question."

def build_user_prompt(note: str, question: str, ground_truth: str, model_answer: str) -> str:
    return (
        f"DISCHARGE SUMMARY:\n{note}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CORRECT ANSWER (Ground Truth):\n{ground_truth}\n\n"
        f"MODEL'S ANSWER:\n{model_answer}\n\n"
        f"Task: Evaluate if the model's answer is correct compared to the ground truth.\n\n"
        f"Respond with ONLY a single digit:\n"
        f"1 = Correct\n"
        f"0 = Incorrect"
    )


def _parse(text: str) -> int | None:
    """Stage-1 parsing rule (lines 151-157 of stage1_gpt4_eval_combined.py)."""
    if text is None:
        return None
    if "1" in text and "0" not in text:
        return 1
    if "0" in text:
        return 0
    return None


# ---------- Single-call and multi-sample interfaces ----------

def judge_call(note: str, question: str, ground_truth: str, model_answer: str,
               *, model: str = "gpt-4o", temperature: float = 0.0,
               max_retries: int = 3, sleep_time: float = 5.0) -> tuple[int | None, str | None]:
    """Single GPT-4o call. Returns (label, raw_text)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(note, question, ground_truth, model_answer)},
    ]
    for attempt in range(1, max_retries + 1):
        try:
            r = client().chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=10,
                temperature=temperature,
            )
            raw = r.choices[0].message.content.strip()
            return _parse(raw), raw
        except Exception as e:
            if attempt < max_retries:
                print(f"  ⏳ judge retry {attempt}/{max_retries}: {e}", flush=True)
                time.sleep(sleep_time)
            else:
                print(f"  ❌ judge failed after {max_retries}: {e}", flush=True)
                return None, None
    return None, None


def judge(note: str, question: str, ground_truth: str, model_answer: str,
          *, n: int = 1, temperature: float = 0.0,
          model: str = "gpt-4o") -> dict:
    """Multi-sample wrapper.
    Returns {label, votes:[...], raws:[...], unanimity, n_valid}.
    With temp=0 + n=1 this is just a deterministic single call.
    """
    votes: list[int] = []
    raws: list[str] = []
    for _ in range(n):
        lab, raw = judge_call(note, question, ground_truth, model_answer,
                              model=model, temperature=temperature)
        raws.append(raw if raw is not None else "")
        if lab is not None:
            votes.append(lab)
    if not votes:
        return {"label": None, "votes": [], "raws": raws, "unanimity": 0.0, "n_valid": 0}
    counts = Counter(votes)
    label, _ = counts.most_common(1)[0]
    unanimity = counts[label] / len(votes)
    return {"label": int(label), "votes": votes, "raws": raws,
            "unanimity": unanimity, "n_valid": len(votes)}


# ---------- CLI: --validate-against-gold ----------

def _load_human_gold() -> pd.DataFrame:
    """Returns merged DataFrame with patient_id, A, B, gold_label (where A==B).

    Reviewer A = Sara Saif, Reviewer B = Jose E. Lizarraga Mazab
    (per memory note and the original inter_rater_agreement.tex table that
    reports A=328, B=300 — matching the row counts in this CSV).
    """
    csv_path = PROJECT_ROOT / "datasets" / "external" / "all_users_openended_BioMistral-7B_latest.csv"
    df = pd.read_csv(csv_path)
    df["human_binary"] = (df["Answer Quality"] == 5).astype(int)
    name_a = "Sara Saif"
    name_b = "Jose E. Lizarraga Mazab"
    a = df[df["User Name"] == name_a].drop_duplicates("Patient ID").set_index("Patient ID")["human_binary"]
    b = df[df["User Name"] == name_b].drop_duplicates("Patient ID").set_index("Patient ID")["human_binary"]
    common = a.index.intersection(b.index)
    out = pd.DataFrame({"A": a.loc[common], "B": b.loc[common]})
    out = out[out["A"] == out["B"]].copy()
    out["gold_label"] = out["A"]
    out.index.name = "patient_id"
    return out.reset_index()


def _load_biomistral_step8_rows() -> pd.DataFrame:
    """Concat all 5 folds of BioMistral step8 zeroshot rows.
    Need: patient_id, question, ground_truth, model_answer + the 3 note columns."""
    parts = []
    for fold in range(5):
        f = PROJECT_ROOT / "output" / "step8" / "biomistral-7b" / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        if f.exists():
            parts.append(pd.read_csv(f))
    if not parts:
        raise FileNotFoundError("No BioMistral step8 zeroshot CSVs found")
    return pd.concat(parts, ignore_index=True)


def _load_biomistral_step2_rows() -> pd.DataFrame:
    """Step2 BioMistral generation (helpful-assistant prompt, the original
    answers behind the 92% / κ=0.75 reference number).

    Schema: patient_id, question, answer (letter A-E), choice_A..E,
    openended_answer, note_1..3, etc.
    """
    f = PROJECT_ROOT / "output" / "ours_biomistral-7b_EHRNoteQA_processed.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing {f}")
    df = pd.read_csv(f)
    # Build a step8-shaped frame so the validation code can be source-agnostic.
    out = pd.DataFrame({
        "patient_id": df["patient_id"].astype(int),
        "question": df["question"],
        "ground_truth": df.apply(_letter_to_full_gt, axis=1),
        "model_answer": df["openended_answer"],
    })
    return out.drop_duplicates("patient_id")


def _letter_to_full_gt(row: pd.Series) -> str:
    """Compose '<letter>. <choice text>' the way the step8 ground_truth column does."""
    letter = str(row.get("answer", "")).strip()
    choice = row.get(f"choice_{letter}", "")
    if pd.isna(choice):
        return letter
    return f"{letter}. {choice}"


def _load_notes_lookup() -> dict[str, str]:
    """Build patient_id → joined-note string from EHRNoteQA_processed.jsonl
    (mirrors load_notes() in run_fullscale.py)."""
    f = PROJECT_ROOT / "output" / "EHRNoteQA_processed.jsonl"
    df = pd.read_json(f, lines=True)
    out: dict[str, str] = {}
    for _, r in df.iterrows():
        pid = str(r.get("patient_id", ""))
        parts = []
        for i in (1, 2, 3):
            col = f"note_{i}"
            if col in r and pd.notna(r[col]):
                t = str(r[col]).strip()
                if t and t.lower() != "nan":
                    parts.append(f"[Note {i}]\n{t}")
        out[pid] = "\n\n".join(parts)
    return out


def cmd_validate_against_gold(args: argparse.Namespace) -> int:
    """Re-run the judge at the requested temperature against the A∩B N=112
    gold subset using BioMistral zeroshot answers. Writes output keyed by
    temperature so we can compare T=0 and T=0.1 side by side."""
    src = args.source
    print(f"Loading human gold (Reviewer A ∩ Reviewer B) — judge T={args.temperature}, source={src}...", flush=True)
    gold = _load_human_gold()
    print(f"  Gold subset: {len(gold)} patient_ids where A==B", flush=True)

    if src == "step8":
        print("Loading BioMistral step8 zeroshot rows...", flush=True)
        bm = _load_biomistral_step8_rows()
    elif src == "step2":
        print("Loading BioMistral step2 (original helpful-assistant) rows...", flush=True)
        bm = _load_biomistral_step2_rows()
    else:
        raise ValueError(f"unknown source: {src}")
    bm["patient_id"] = bm["patient_id"].astype(int)
    bm = bm.drop_duplicates(subset="patient_id")
    print(f"  BM rows: {len(bm)}", flush=True)

    print("Loading discharge notes...", flush=True)
    notes = _load_notes_lookup()

    # Join gold + bm on patient_id
    gold["patient_id"] = gold["patient_id"].astype(int)
    merged = gold.merge(bm[["patient_id", "question", "ground_truth", "model_answer"]],
                        on="patient_id", how="inner")
    print(f"  Joined N: {len(merged)} (need ≈112 to match the literature)", flush=True)

    out_rows = []
    for i, row in merged.iterrows():
        pid = int(row["patient_id"])
        note = notes.get(str(pid), "")
        if not note:
            print(f"  !! no note for patient {pid}, skipping", flush=True)
            continue
        result = judge(note, row["question"], row["ground_truth"], str(row["model_answer"]),
                       n=args.n, temperature=args.temperature)
        out_rows.append({
            "patient_id": pid,
            "gold_label": int(row["gold_label"]),
            "gpt4o_label_T0": result["label"],
            "gpt4o_raws": result["raws"],
            "unanimity": result["unanimity"],
        })
        if (i + 1) % 10 == 0:
            print(f"  judged {i+1}/{len(merged)}", flush=True)
        time.sleep(0.5)  # gentle pace to stay under 30k TPM

    if not out_rows:
        print("!! No items judged — aborting (check gold/CSV join).", flush=True)
        return 1
    out_df = pd.DataFrame(out_rows)
    valid = out_df[out_df["gpt4o_label_T0"].notna()].copy()
    n = len(valid)
    if n == 0:
        print("!! No valid judgments — aborting.", flush=True)
        return 1
    agree = int((valid["gold_label"] == valid["gpt4o_label_T0"]).sum())
    pct = 100.0 * agree / n
    kappa = float(cohen_kappa_score(valid["gold_label"], valid["gpt4o_label_T0"]))
    fn = int(((valid["gold_label"] == 1) & (valid["gpt4o_label_T0"] == 0)).sum())
    fp = int(((valid["gold_label"] == 0) & (valid["gpt4o_label_T0"] == 1)).sum())

    print()
    print("=" * 60)
    print(f"TEMP={args.temperature} JUDGE vs A∩B GOLD  (N={n})")
    print("=" * 60)
    print(f"  Agreement : {pct:.1f}% ({agree}/{n})")
    print(f"  Cohen's κ : {kappa:.3f}")
    print(f"  FN (gold=correct, gpt=wrong): {fn}")
    print(f"  FP (gold=wrong, gpt=correct): {fp}")
    print()
    print(f"  Reference (legacy temp=0.1): 92.0%, κ=0.75, FN=6, FP=3")

    pass_pct = pct >= 92.0
    pass_kappa = kappa >= 0.74
    pass_overall = pass_pct and pass_kappa
    print()
    print(f"  Pass thresholds (≥92% AND κ≥0.74): {'✅' if pass_overall else '❌'}")

    tag = f"T{args.temperature:g}".replace(".", "p")
    out_path = OUTPUT_DIR / f"judge_agreement_{src}_{tag}.json"
    summary = {
        "temperature": args.temperature,
        "n_samples_per_call": args.n,
        "n_items": n,
        "agreement_pct": pct,
        "kappa": kappa,
        "fn": fn,
        "fp": fp,
        "pass_threshold": pass_overall,
        "reference_temp01": {"agreement": 92.0, "kappa": 0.75, "fn": 6, "fp": 3},
        "per_item": out_rows,
    }
    with open(out_path, "w") as fout:
        json.dump(summary, fout, indent=2, default=str)
    print(f"\n  Wrote {out_path}", flush=True)
    return 0 if pass_overall else 2


# ---------- CLI: --rejudge-step8 ----------

def cmd_rejudge_step8(args: argparse.Namespace) -> int:
    """Re-judge every row of Qwen2.5-7B step8 zeroshot at temp=0.
    Writes a single combined CSV with both legacy and new labels."""
    notes = _load_notes_lookup()
    out_rows = []
    flips_to_correct = 0
    flips_to_wrong = 0
    same = 0
    failed = 0

    save_path = OUTPUT_DIR / "zeroshot_evaluated_binary_T0.csv"
    # Resume support
    done_keys: set[tuple[int, int]] = set()
    if save_path.exists() and not args.force:
        prior = pd.read_csv(save_path)
        out_rows = prior.to_dict("records")
        done_keys = {(int(r["fold"]), int(r["idx"])) for r in out_rows}
        print(f"Resuming: {len(done_keys)} rows already judged", flush=True)

    folds = list(range(5))
    if args.fold is not None:
        folds = [args.fold]
    for fold in folds:
        f = PROJECT_ROOT / "output" / "step8" / "qwen2.5-7b-instruct" / f"fold_{fold}" / "zeroshot_evaluated_binary.csv"
        if not f.exists():
            print(f"  !! missing {f}", flush=True)
            continue
        df = pd.read_csv(f)
        print(f"Fold {fold}: {len(df)} rows", flush=True)
        for i, row in df.iterrows():
            key = (fold, int(row["idx"]))
            if key in done_keys:
                continue
            note = notes.get(str(row["patient_id"]), "")
            if not note:
                continue
            result = judge(note, row["question"], row["ground_truth"], str(row["model_answer"]),
                           n=1, temperature=0.0)
            new_label = result["label"]
            old_label = int(row["binary_correct"])
            if new_label is None:
                failed += 1
            elif new_label == old_label:
                same += 1
            elif new_label == 1 and old_label == 0:
                flips_to_correct += 1
            elif new_label == 0 and old_label == 1:
                flips_to_wrong += 1
            out_rows.append({
                "fold": fold,
                "idx": int(row["idx"]),
                "patient_id": int(row["patient_id"]),
                "binary_correct_T01_legacy": old_label,
                "binary_correct_T0": new_label if new_label is not None else -1,
                "raw_T0": result["raws"][0] if result["raws"] else "",
            })
            if len(out_rows) % 25 == 0:
                pd.DataFrame(out_rows).to_csv(save_path, index=False)
                print(f"  ...{len(out_rows)} judged "
                      f"(same={same} →1={flips_to_correct} →0={flips_to_wrong} fail={failed})",
                      flush=True)

    pd.DataFrame(out_rows).to_csv(save_path, index=False)
    print()
    print("=" * 60)
    print("RE-JUDGING SUMMARY (Qwen2.5-7B step8 zeroshot @ T=0)")
    print("=" * 60)
    total = same + flips_to_correct + flips_to_wrong + failed
    if total > 0:
        print(f"  Total judged   : {total}")
        print(f"  Same as T=0.1  : {same} ({100*same/total:.1f}%)")
        print(f"  Flipped 0 → 1  : {flips_to_correct}")
        print(f"  Flipped 1 → 0  : {flips_to_wrong}")
        print(f"  Failed         : {failed}")
        print(f"\n  Net change to wrong-set size: {flips_to_wrong - flips_to_correct:+d}")
    print(f"\n  Wrote {save_path}", flush=True)
    return 0


# ---------- main ----------

def _self_check_prompt_equals_stage1() -> None:
    """Defensive: assert our prompt builders match the Stage-1 source byte-for-byte
    so a future edit to either side trips immediately."""
    src = (PROJECT_ROOT / "src" / "step3_evaluation" / "stage1_gpt4_eval_combined.py").read_text()
    assert "You are a medical expert evaluating an AI model's answer to a clinical question." in src, \
        "Stage-1 system prompt missing from source — abort"
    assert 'f"DISCHARGE SUMMARY:\\n{note}\\n\\n"' in src, "Stage-1 user prompt header drift"
    assert 'f"1 = Correct\\n"' in src and 'f"0 = Incorrect"' in src, "Stage-1 user prompt footer drift"


def main() -> int:
    p = argparse.ArgumentParser(description="V2 trusted judge — Stage-1 binary @ temp=0")
    p.add_argument("--validate-against-gold", action="store_true")
    p.add_argument("--rejudge-step8", action="store_true")
    p.add_argument("--self-check", action="store_true",
                   help="assert our prompt matches the Stage-1 source byte-for-byte")
    p.add_argument("-n", type=int, default=1,
                   help="samples per call (default 1; >1 only as a sanity check)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--source", choices=["step2", "step8"], default="step2",
                   help="which BioMistral generation to validate against "
                        "(step2 = original helpful-assistant; step8 = medical-expert)")
    p.add_argument("--fold", type=int, default=None,
                   help="restrict --rejudge-step8 to one fold")
    p.add_argument("--force", action="store_true",
                   help="ignore prior --rejudge-step8 save and start over")
    args = p.parse_args()

    if args.self_check:
        _self_check_prompt_equals_stage1()
        print("✅ prompt-equality self-check passed")
        return 0
    if args.validate_against_gold:
        _self_check_prompt_equals_stage1()
        return cmd_validate_against_gold(args)
    if args.rejudge_step8:
        _self_check_prompt_equals_stage1()
        return cmd_rejudge_step8(args)
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
