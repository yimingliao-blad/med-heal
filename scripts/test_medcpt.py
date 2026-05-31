#!/usr/bin/env python3
"""MedCPT (medical embedding) vs GTR for ranking the gold meds on case 1, plus section-level check."""
import sys, re
sys.path.insert(0, "scripts"); sys.path.insert(0, "src/pre_atom/legacy/step9_self_correction/v2")
import numpy as np, torch
import phase2b_extract_compare_detection as P2
from note_span_index import get_embedder, split_sentences
from transformers import AutoTokenizer, AutoModel

rows = {(r["fold"], r["idx"]): r for r in P2.load_rows(40, 20, 42)}
import json
looked = {(r["fold"], r["idx"]): r.get("lookup", "") for r in (json.loads(l) for l in open("runs/expO_decompose_locate/qwen25_nw40_nc20_seed42/records.jsonl"))}
row = next(r for r in rows.values() if "discharge medications prescribed to the patient during his first and second" in r["question"])
note, q = row["note"], row["question"]
items = [ln.strip(" -*0123456789.").strip() for ln in looked[(row["fold"], row["idx"])].splitlines() if ln.strip()]
queries = [q] + items
sents = split_sentences(note)
GOLD = ["anastrozole", "acetaminophen", "alprazolam", "bisacodyl", "cyclobenzaprine", "docusate", "hydromorphone"]

# --- GTR ---
emb = get_embedder()
sv = emb.encode(sents, normalize_embeddings=True, show_progress_bar=False)
qv = emb.encode(queries, normalize_embeddings=True, show_progress_bar=False)
gtr_score = (sv @ qv.T).max(axis=1)
gtr_order = np.argsort(-gtr_score)

# --- MedCPT ---
print("loading MedCPT...", flush=True)
qt = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder"); qm = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").eval()
at = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder"); am = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").eval()
def enc(model, tok, texts, ml):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            e = tok(texts[i:i+32], truncation=True, padding=True, max_length=ml, return_tensors="pt")
            out.append(model(**e).last_hidden_state[:, 0, :])
    return torch.cat(out).numpy()
print("encoding...", flush=True)
a_emb = enc(am, at, sents, 256)
q_emb = enc(qm, qt, queries, 64)
med_score = (a_emb @ q_emb.T).max(axis=1)
med_order = np.argsort(-med_score)

def rank(order, term):
    for i, si in enumerate(order, 1):
        if term in sents[si].lower():
            return i
    return None
print(f"\n=== gold-med rank: GTR vs MedCPT (of {len(sents)} sentences) ===")
print(f"{'med':16} {'GTR rank':>9} {'MedCPT rank':>12}")
for m in GOLD:
    print(f"{m:16} {str(rank(gtr_order,m)):>9} {str(rank(med_order,m)):>12}")
print("\n(lower rank = retrieved sooner; top-10 is the cutoff)")
