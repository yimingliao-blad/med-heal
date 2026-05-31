"""Phase 1: build the production pool index covering all 962 EHRNoteQA items.

For each item, store:
  - row_id (global, 0..961)
  - patient_id, fold_membership (which fold's test set; 0..4)
  - question, note (text), GT (text)
  - BM_zeroshot (text — donor model's answer)
  - per-target-model labels (correct/wrong, for downstream pool filtering if ever needed)
  - nomic embeddings: q, note, BM_zs (768-d each)  ← retrieval matching
  - NER entity sets (per-category)

Existing cache:
  - nomic_question.npy, nomic_note.npy, nomic_bm_zs.npy: 769 (fold_0/train) — extend to 962
  - ner_magistral.jsonl: 769 — extend by running NER on the missing 193 items via Magistral

Output:
  - output/ichl/retrieval_study/pool_index/embs.npz   (3 stacked arrays)
  - output/ichl/retrieval_study/pool_index/items.jsonl (962 rows: metadata + NER)
  - output/ichl/retrieval_study/pool_index/labels.csv  (per-target correctness)

Two-step:
  Step 1: NER extension (192 items via Magistral, ~5 min)
  Step 2: Nomic encoding extension (962 items × 3 fields, ~30 s on GPU)
  Step 3: Per-target label assembly (no compute, just file reads)

If Magistral vLLM is not running, this script boots it.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EHR = ROOT / "output" / "EHRNoteQA_processed.jsonl"
BM_GEN = ROOT / "output" / "ours_biomistral-7b_EHRNoteQA_processed.csv"
NER_FILE = ROOT / "output" / "ichl" / "retrieval_study" / "ner_magistral.jsonl"
OUT_DIR = ROOT / "output" / "ichl" / "retrieval_study"
EMB_CACHE = OUT_DIR / "emb_cache"
POOL_DIR = OUT_DIR / "pool_index"
POOL_DIR.mkdir(parents=True, exist_ok=True)

NOMIC_MODEL = "nomic-ai/nomic-embed-text-v1.5"
MAGISTRAL_URL = "http://localhost:8003/v1"
MAGISTRAL_MODEL = "Magistral-Small-2509-AWQ"

NER_SYSTEM = "You are a clinical information extractor."
NER_USER_TMPL = """Extract clinical entities from this clinical note. Return a JSON object with EXACTLY these 5 keys:
- "medications": list of distinct drug/medication names (just names, no doses, no duplicates)
- "doses": list of "drug NAME — DOSE — ROUTE — FREQ" strings (e.g., "metformin — 500 mg — PO — BID"); use "?" for missing fields
- "procedures": list of distinct surgical, diagnostic, or therapeutic procedures
- "lab_values": list of "TEST = VALUE" strings (e.g., "creatinine = 2.1", "Hgb = 8.5")
- "diagnoses": list of distinct clinical conditions / diagnoses

Normalize entities to lowercase. De-duplicate. Keep entries terse and canonical (use generic drug names, not brand). If a category has no entities, return [] for it.

NOTE:
{note}

Respond with ONLY a JSON object, no prose."""


def parse_json(text: str) -> dict:
    if not text: return {"_err": "empty"}
    s = text.strip()
    if s.startswith("```"):
        m = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
        if m: s = m.group(1).strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            try: return json.loads(m.group(0))
            except Exception: pass
    return {"_err": "no_json", "_raw": s[:200]}


def load_all_items() -> list[dict]:
    df = pd.read_json(EHR, lines=True)
    print(f"  EHRNoteQA: {len(df)} items")
    bm_df = pd.read_csv(BM_GEN)
    bm_col = "openended_answer"
    bm_by_pid = {int(r["patient_id"]): str(r.get(bm_col, "") or "") for _, r in bm_df.iterrows()}
    out = []
    for i, r in df.iterrows():
        pid = int(r["patient_id"])
        letter = str(r.get("answer", "")).strip().upper()
        gt_text = str(r.get(f"choice_{letter}", "")).strip() if letter else ""
        gt = f"{letter}: {gt_text}" if (letter and gt_text) else gt_text
        parts = []
        for j in [1, 2, 3]:
            v = r.get(f"note_{j}")
            if pd.notna(v) and str(v).strip() and str(v).lower() != "nan":
                parts.append(str(v))
        # Use [Note i]-prefixed step8 format (matches how step8 generation + judge see notes)
        # NEVER truncate per [Workflow] No Silent Truncation. Max EHRNoteQA note = 5709 nomic
        # tokens, fits 8K+ embedding/judge context. If a future tool can't handle full notes,
        # change the tool, do NOT truncate the note here.
        note_step8 = "\n\n".join(f"[Note {j+1}]\n{p}" for j, p in enumerate(parts))
        out.append({
            "row_id": int(i), "patient_id": pid,
            "category": str(r.get("category", "")),
            "question": str(r.get("question", "")),
            "ground_truth": gt,
            "note_text": note_step8,
            "bm_zeroshot": bm_by_pid.get(pid, ""),
        })
    return out


def get_fold_assignment() -> dict[int, int]:
    """Return {patient_id: fold_id} where fold_id ∈ {0..4} indicates which fold's test set."""
    out = {}
    for fid in range(5):
        test_file = ROOT / "output" / "folds" / f"fold_{fid}" / "test.jsonl"
        if not test_file.exists(): continue
        for l in test_file.open():
            try:
                r = json.loads(l)
                out[int(r["patient_id"])] = fid
            except Exception: pass
    return out


def magistral_alive() -> bool:
    try:
        from openai import OpenAI
        c = OpenAI(base_url=MAGISTRAL_URL, api_key="not-needed", timeout=2)
        ms = c.models.list().data
        return any("Magistral" in m.id for m in ms)
    except Exception:
        return False


def boot_magistral() -> int | None:
    """Boot Magistral vLLM and wait for ready. Returns PID or None on failure."""
    log = "/tmp/vllm_magistral_phase1.log"
    cmd = [
        ".venv/bin/python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "cyankiwi/Magistral-Small-2509-AWQ-4bit",
        "--served-model-name", "Magistral-Small-2509-AWQ",
        "--tokenizer-mode", "mistral",
        "--max-model-len", "14336",
        "--gpu-memory-utilization", "0.93",
        "--port", "8003",
        "--dtype", "auto",
    ]
    print(f"  booting Magistral... (log: {log})")
    p = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    pid = p.pid
    print(f"  vLLM PID: {pid}")
    # Wait up to 5 min
    for i in range(60):
        if magistral_alive():
            print(f"  Magistral READY after {i*5} s")
            return pid
        time.sleep(5)
    print(f"  Magistral failed to come up in 5 min; check {log}")
    return None


def ner_extract(items: list[dict], workers: int = 4) -> list[dict]:
    from openai import OpenAI
    client = OpenAI(base_url=MAGISTRAL_URL, api_key="not-needed", timeout=300)
    def one(it):
        user = NER_USER_TMPL.format(note=it["note_text"])
        try:
            r = client.chat.completions.create(
                model=MAGISTRAL_MODEL,
                messages=[{"role": "system", "content": NER_SYSTEM},
                          {"role": "user", "content": user}],
                temperature=0.0, max_tokens=800,
            )
            content = r.choices[0].message.content or ""
            return {"row_id": it["row_id"], "patient_id": it["patient_id"],
                    "entities": parse_json(content), "raw": content[:500],
                    "comp_tok": r.usage.completion_tokens if r.usage else None,
                    "prompt_tok": r.usage.prompt_tokens if r.usage else None}
        except Exception as e:
            return {"row_id": it["row_id"], "patient_id": it["patient_id"],
                    "_err": str(e)[:200]}
    out = []
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(one, items), 1):
            out.append(r)
            if i % 50 == 0:
                dt = time.monotonic() - t0
                eta = dt * (len(items) - i) / i
                print(f"    NER {i}/{len(items)}  elapsed={dt:.0f}s  eta={eta:.0f}s")
    return out


def encode_nomic(texts: list[str], device: str = "cuda") -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(NOMIC_MODEL, trust_remote_code=True, device=device)
    embs = model.encode(texts, batch_size=16, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=True)
    del model
    import gc, torch; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return embs


def main():
    print("=== Phase 1: build production pool index for full 962 ===\n")
    print("Loading items...")
    items = load_all_items()
    print(f"  loaded {len(items)} items")
    print("\nLoading fold assignment...")
    fold_map = get_fold_assignment()
    for it in items:
        it["fold_test_member"] = fold_map.get(it["patient_id"])  # 0..4

    # === Step 1: NER extension ===
    print("\n--- Step 1: NER extension (Magistral) ---")
    existing_ner = {}
    if NER_FILE.exists():
        for l in NER_FILE.open():
            r = json.loads(l)
            existing_ner[int(r["row_id"])] = r
    # Filter to items NOT already covered. Note: existing NER is keyed by pool's row_id,
    # which referred to fold_0/train. Now items are global EHRNoteQA. Match by patient_id.
    existing_pid = {existing_ner[rid]["patient_id"]: existing_ner[rid] for rid in existing_ner}
    print(f"  existing NER covers {len(existing_pid)} patients")
    missing = [it for it in items if it["patient_id"] not in existing_pid]
    print(f"  need to extract for {len(missing)} new patients")

    booted_magistral = False
    if missing:
        if not magistral_alive():
            pid = boot_magistral()
            if pid is None:
                print("  ERROR: cannot boot Magistral; aborting")
                return
            booted_magistral = True
        else:
            print("  Magistral already alive")
        new_ner = ner_extract(missing, workers=4)
        n_err = sum(1 for r in new_ner if "_err" in r or (isinstance(r.get("entities"), dict) and "_err" in r["entities"]))
        print(f"  NER extracted: {len(new_ner) - n_err}/{len(new_ner)} OK")
        # Build a lookup by patient_id from new_ner
        for r in new_ner:
            existing_pid[r["patient_id"]] = r

        # Save NER incrementally so we don't lose work if next step crashes
        merged_path = POOL_DIR / "ner_magistral_962.jsonl"
        with merged_path.open("w") as f:
            for it in items:
                ner = existing_pid.get(it["patient_id"], {"_err": "missing"})
                f.write(json.dumps({"row_id": it["row_id"], "patient_id": it["patient_id"],
                                    "entities": ner.get("entities", {}),
                                    "comp_tok": ner.get("comp_tok"), "prompt_tok": ner.get("prompt_tok")},
                                    default=str) + "\n")
        print(f"  saved partial NER index: {merged_path}")

        # Free GPU before nomic encoding (Magistral occupies ~23 GB; nomic needs CUDA)
        if booted_magistral:
            print("  killing Magistral to free GPU for nomic...")
            subprocess.run(["pkill", "-f", "vllm.entrypoints"], check=False)
            time.sleep(8)
            # Also kill any stray EngineCore
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]
                ).decode().strip().splitlines()
                for ppid in out:
                    if ppid.strip().isdigit():
                        subprocess.run(["kill", "-9", ppid.strip()], check=False)
            except Exception:
                pass
            time.sleep(4)

    # === Step 2: Nomic encoding (full 962) ===
    print("\n--- Step 2: Nomic encoding for 962 items ---")
    cache_q = POOL_DIR / "nomic_question.npy"
    cache_note = POOL_DIR / "nomic_note.npy"
    cache_zs = POOL_DIR / "nomic_bm_zs.npy"
    if cache_q.exists() and cache_note.exists() and cache_zs.exists():
        print("  cache hit (all 3 nomic arrays)")
        emb_q = np.load(cache_q)
        emb_note = np.load(cache_note)
        emb_zs = np.load(cache_zs)
    else:
        print("  encoding 962 items × 3 fields...")
        t0 = time.monotonic()
        emb_q = encode_nomic([it["question"] for it in items])
        np.save(cache_q, emb_q)
        emb_note = encode_nomic([it["note_text"] for it in items])
        np.save(cache_note, emb_note)
        emb_zs = encode_nomic([it["bm_zeroshot"] for it in items])
        np.save(cache_zs, emb_zs)
        print(f"  done in {time.monotonic()-t0:.1f}s; shapes: q={emb_q.shape}, note={emb_note.shape}, zs={emb_zs.shape}")

    # Save merged 962-item NER
    merged_ner_path = POOL_DIR / "ner_magistral_962.jsonl"
    print(f"\nSaving merged NER for 962 items \u2192 {merged_ner_path.name}")
    with merged_ner_path.open("w") as f:
        for it in items:
            ner = existing_pid.get(it["patient_id"], {"_err": "missing"})
            f.write(json.dumps({"row_id": it["row_id"], "patient_id": it["patient_id"],
                                "entities": ner.get("entities", {}),
                                "comp_tok": ner.get("comp_tok"), "prompt_tok": ner.get("prompt_tok")},
                                default=str) + "\n")

    # === Step 3: Build items.jsonl (the index file) ===
    print("\n--- Step 3: Build items.jsonl with all metadata ---")
    items_path = POOL_DIR / "items.jsonl"
    with items_path.open("w") as f:
        for it in items:
            ner = existing_pid.get(it["patient_id"], {"_err": "missing"})
            f.write(json.dumps({
                "row_id": it["row_id"],
                "patient_id": it["patient_id"],
                "category": it["category"],
                "fold_test_member": it["fold_test_member"],
                "question": it["question"],
                "ground_truth": it["ground_truth"],
                "note_text_truncated": it["note_text"],
                "bm_zeroshot": it["bm_zeroshot"],
                "ner_entities": ner.get("entities", {}),
            }, default=str) + "\n")
    print(f"  saved: {items_path}")

    # === Step 4: Summary ===
    n_with_ner = sum(1 for it in items
                      if existing_pid.get(it["patient_id"], {}).get("entities") and
                      not (isinstance(existing_pid[it["patient_id"]].get("entities"), dict)
                           and "_err" in existing_pid[it["patient_id"]]["entities"]))
    print(f"\n=== Index summary ===")
    print(f"  total items: {len(items)}")
    print(f"  items with valid NER: {n_with_ner}")
    print(f"  fold distribution:")
    from collections import Counter
    c = Counter(it["fold_test_member"] for it in items)
    for fid in sorted(c):
        print(f"    fold {fid}: {c[fid]}")
    print(f"\n  Pool index written: {POOL_DIR}")
    print(f"    items.jsonl, ner_magistral_962.jsonl, nomic_question.npy, nomic_note.npy, nomic_bm_zs.npy")


if __name__ == "__main__":
    main()
