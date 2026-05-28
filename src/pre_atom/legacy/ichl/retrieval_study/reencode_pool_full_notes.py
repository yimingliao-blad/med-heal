"""Re-encode the 962-item pool index with FULL EHRNoteQA notes (no truncation).

Replaces the existing pool_index/{nomic_question.npy, nomic_note.npy,
nomic_bm_zs.npy} which were computed against truncated 3000-char notes.

Per [Workflow] No Silent Truncation: every EHRNoteQA note fits in
nomic-embed-text-v1.5's 8192-token window (max observed: 5709 tokens), so
no truncation is needed.

Also rewrites items.jsonl to replace `note_text_truncated` with `note_text`
(full step8-format note with [Note i] headers).

Backups saved with `.truncated_3000.bak` suffix.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
POOL_DIR = ROOT / "output" / "ichl" / "retrieval_study" / "pool_index"
NOTES_FILE = ROOT / "output" / "EHRNoteQA_processed.jsonl"

NOMIC_MODEL = "nomic-ai/nomic-embed-text-v1.5"


def step8_format_note(row: dict) -> str:
    parts = []
    for i in [1, 2, 3]:
        v = row.get(f"note_{i}")
        if v and str(v).strip() and str(v).lower() != "nan":
            parts.append(f"[Note {i}]\n{str(v).strip()}")
    return "\n\n".join(parts)


def main():
    # Step 1: load existing items.jsonl + EHRNoteQA full notes
    print("Loading items.jsonl + EHRNoteQA full notes...")
    items = [json.loads(l) for l in (POOL_DIR / "items.jsonl").open()]
    notes_by_pid: dict[int, dict] = {}
    for line in NOTES_FILE.open():
        if not line.strip(): continue
        r = json.loads(line)
        notes_by_pid[int(r["patient_id"])] = r

    # Replace truncated note with full
    print(f"  pool items: {len(items)}")
    print(f"  EHRNoteQA full notes: {len(notes_by_pid)}")

    full_texts = []
    questions = []
    bm_zs_texts = []
    for it in items:
        pid = int(it["patient_id"])
        if pid not in notes_by_pid:
            print(f"  WARNING: pid {pid} not in EHRNoteQA full-notes file")
            full_texts.append(it.get("note_text_truncated", ""))
        else:
            full_texts.append(step8_format_note(notes_by_pid[pid]))
        questions.append(it["question"])
        bm_zs_texts.append(it.get("bm_zeroshot", ""))

    # Sanity stats
    char_max = max(len(t) for t in full_texts)
    char_min = min(len(t) for t in full_texts)
    print(f"  full-note char counts: min={char_min} max={char_max}")

    # Step 2: backup existing .npy and items.jsonl
    print("\nBacking up existing artifacts (suffix .truncated_3000.bak)...")
    for name in ["nomic_question.npy", "nomic_note.npy", "nomic_bm_zs.npy", "items.jsonl"]:
        src = POOL_DIR / name
        if src.exists():
            dst = POOL_DIR / (name + ".truncated_3000.bak")
            shutil.copy2(src, dst)
            print(f"  {src.name} -> {dst.name}")

    # Step 3: encode with nomic on GPU (vLLM must be stopped)
    print("\nEncoding nomic embeddings on GPU...")
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")
    model = SentenceTransformer(NOMIC_MODEL, trust_remote_code=True, device=device)

    t0 = time.monotonic()
    print(f"\n[1/3] question embeddings ({len(questions)} items)...")
    emb_q = model.encode(questions, batch_size=16, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)
    print(f"  shape={emb_q.shape}  elapsed={time.monotonic()-t0:.0f}s")

    t1 = time.monotonic()
    print(f"\n[2/3] note embeddings ({len(full_texts)} items, FULL notes)...")
    emb_n = model.encode(full_texts, batch_size=4, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)
    print(f"  shape={emb_n.shape}  elapsed={time.monotonic()-t1:.0f}s")

    t2 = time.monotonic()
    print(f"\n[3/3] BM zeroshot embeddings ({len(bm_zs_texts)} items)...")
    emb_bm = model.encode(bm_zs_texts, batch_size=16, show_progress_bar=False,
                          convert_to_numpy=True, normalize_embeddings=True)
    print(f"  shape={emb_bm.shape}  elapsed={time.monotonic()-t2:.0f}s")

    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Step 4: save new embeddings + updated items.jsonl
    print("\nSaving new embeddings + items.jsonl with full notes...")
    np.save(POOL_DIR / "nomic_question.npy", emb_q)
    np.save(POOL_DIR / "nomic_note.npy", emb_n)
    np.save(POOL_DIR / "nomic_bm_zs.npy", emb_bm)
    print(f"  saved 3 .npy files")

    with (POOL_DIR / "items.jsonl").open("w") as f:
        for it, full_text in zip(items, full_texts):
            it["note_text"] = full_text         # NEW canonical field
            it["note_text_truncated"] = full_text  # legacy alias for compat (now NOT truncated)
            f.write(json.dumps(it) + "\n")
    print(f"  saved items.jsonl with full notes ({len(items)} rows)")

    print(f"\nDONE. Total elapsed: {time.monotonic()-t0:.0f}s")
    print(f"  pool index now reflects full untruncated notes per [Workflow] No Silent Truncation.")


if __name__ == "__main__":
    main()
