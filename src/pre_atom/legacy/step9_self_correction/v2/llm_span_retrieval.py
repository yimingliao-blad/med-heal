#!/usr/bin/env python3
"""
LLM-based note-span retrieval (R3 in the retriever bake-off).

The premise: embeddings cannot bridge clinical inference. On idx=51 the
evidence sentence is "She was unable to void and was bladder scanned showing
700cc of retained urine", but the queries all contain "complication" and
"first rib resection" — zero token overlap with the evidence. Cosine
similarity ranks the evidence sentence outside the top-5 even with K=5
multi-query agreement scoring.

The Qwen2.5-7B target model, however, can recognise that "unable to void"
+ "bladder scanned" IS a postoperative complication. So we use Qwen2.5
itself as the retriever:

  - number every sentence in the note
  - ask Qwen2.5 to cite the 1-3 most relevant sentence numbers
  - K samples
  - majority-vote across samples on which numbers are cited
  - return the cited sentences in order of citation count

Settings: temp=0.7, K=5, single LLM call per sample. Each sample is
independent. Hallucinated citations (numbers outside the valid range) are
filtered. Empty cite lists are tolerated; if all 5 samples cite nothing,
returns [].
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

from detection_format_bakeoff import build_chatml, vllm_gen, vllm_chat
from note_span_index import split_sentences

LLM_RETRIEVE_SYS = "You are a strict medical expert reading discharge notes."

LLM_RETRIEVE_PROMPT = """The following is a discharge note. Each sentence is
numbered. Read the question carefully and identify which sentences in the note
contain the information needed to answer the question correctly.

Discharge note (numbered sentences):
{numbered_note}

Question: {question}

List up to {max_per_sample} sentence numbers, one per line, in order of importance
(most important first). Use only the sentence numbers shown above. Look across
the WHOLE note — chief complaint, hospital course, results, medications,
discharge diagnoses — anywhere the relevant facts may be. If no sentence in
the note bears on the question, reply NONE.

Format your answer as:
LINE 1: <number>
LINE 2: <number>   (or omit)
LINE 3: <number>   (or omit)
"""

# If the line looks like "LINE k: <number>" or "<k>) <number>" we want the
# number AFTER the colon / closing paren, not the index marker before it.
_RE_LINE_LABEL = re.compile(r"^\s*(?:LINE\s*\d+|^\d+[.)])\s*[:.\-]?\s*(.*)$",
                            re.IGNORECASE)
_RE_BARE_NUM = re.compile(r"\b(\d{1,3})\b")


def _parse_cited_numbers(raw: str, max_n: int) -> list[int]:
    """Extract numeric citations from a model response.

    Strategy:
      1. For each line, strip the leading "LINE k:" or "k)" label if present.
      2. Look for a number in the *remaining* text. If found and in range,
         that's the citation for this line.
      3. If no label, look for any number in the line.
      4. Out-of-range or duplicate citations are dropped.
      5. Stop at the first 'NONE' line (allows partial cite lists).
    """
    if not raw:
        return []
    out: list[int] = []
    seen: set[int] = set()
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.upper().startswith("NONE") or s.upper() == "NONE":
            break
        # Strip the leading "LINE k:" / "k)" label if present
        m = _RE_LINE_LABEL.match(s)
        rest = m.group(1) if m else s
        if rest.strip().upper().startswith("NONE"):
            continue
        # Take the FIRST number in the rest of the line
        nm = _RE_BARE_NUM.search(rest)
        if not nm:
            continue
        n = int(nm.group(1))
        if 1 <= n <= max_n and n not in seen:
            out.append(n)
            seen.add(n)
    return out


def llm_topk_spans(note: str, question: str, error_description: str = "",
                   *, port: int, k: int = 5,
                   max_per_sample: int = 5,
                   max_topk: int = 5) -> dict:
    """Returns:
        {
          'numbered_note': "1) ...\\n2) ...\\n...",
          'samples': [{'raw': str, 'cited': [int, ...]} × k],
          'citation_votes': {sentence_number: vote_count, ...},
          'top_sentences': [{'sentence_number': int, 'sentence': str,
                             'votes': int} × up to max_topk],
          'n_sentences': int,
        }
    """
    sents = split_sentences(note)
    if not sents:
        return {"numbered_note": "", "samples": [], "citation_votes": {},
                "top_sentences": [], "n_sentences": 0}
    numbered = "\n".join(f"{i+1}) {s}" for i, s in enumerate(sents))
    user = LLM_RETRIEVE_PROMPT.format(
        numbered_note=numbered,
        question=question,
        max_per_sample=max_per_sample,
    )

    samples: list[dict] = []
    all_cites: list[int] = []
    for _ in range(k):
        raw = vllm_chat(LLM_RETRIEVE_SYS, user, port,
                        max_tokens=120, temperature=0.7)
        cites = _parse_cited_numbers(raw, max_n=len(sents))[:max_per_sample]
        samples.append({"raw": raw, "cited": cites})
        all_cites.extend(cites)

    votes = Counter(all_cites)
    top: list[dict] = []
    for num, count in votes.most_common(max_topk):
        top.append({
            "sentence_number": num,
            "sentence": sents[num - 1],
            "votes": count,
            "vote_fraction": count / k,
        })
    return {
        "numbered_note": numbered,
        "samples": samples,
        "citation_votes": dict(votes),
        "top_sentences": top,
        "n_sentences": len(sents),
    }


# ---------- Self-test on idx=51 ----------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from judge import _load_notes_lookup

    notes = _load_notes_lookup()
    note = notes["12152580"]
    question = "What was the main postop complication from the infraclavicular first rib resection procedure?"
    err = ("The previous answer claimed there was no postoperative complication, "
           "but the question asks specifically what the main complication was.")
    print("Running LLM-based span retrieval on idx=51...")
    result = llm_topk_spans(note, question, err, port=8003, k=5)
    print(f"\nTotal sentences in note: {result['n_sentences']}")
    print(f"\nPer-sample citations:")
    for i, s in enumerate(result["samples"], 1):
        print(f"  sample {i}: cited={s['cited']}")
        print(f"    raw: {s['raw'][:200]}")
    print(f"\nVote tally: {result['citation_votes']}")
    print(f"\nTop sentences:")
    for t in result["top_sentences"]:
        print(f"  [{t['sentence_number']}] votes={t['votes']}/5: {t['sentence'][:160]}")
