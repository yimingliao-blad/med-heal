#!/usr/bin/env python3
"""
Module 2 — Multi-sample voting harness.

Wraps any single-call function so we can sample N times, take the majority,
and persist the full vote distribution + raw outputs. This is the boundary-
testing primitive: the user described "51% becomes 100% with enough samples;
49% never will" — every non-deterministic stage in the V2 pipeline goes
through vote_call so we can measure exactly that.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Callable


def vote_call(call_fn: Callable[..., tuple[Any, str]],
              n: int = 5,
              key_fn: Callable[[Any], Any] | None = None,
              **call_kwargs: Any) -> dict:
    """Run `call_fn` n times and return a structured majority-vote result.

    Args:
        call_fn: function returning (parsed_value, raw_text). Must accept any
                 kwargs in call_kwargs (typically: prompt, temperature, ...).
        n: number of samples.
        key_fn: optional projection from a parsed value to its hashable
                "vote key". If None, the parsed value itself is used.
                Useful when the parsed value is a dict and only one field
                drives the vote (e.g. dict["verdict"]).
        call_kwargs: forwarded to call_fn.

    Returns:
        {
          "n_requested": n,
          "n_valid":     int,
          "samples":     [{"parsed": ..., "raw": "..."} × n],
          "vote_keys":   [hashable × n_valid],
          "majority":    parsed value of the majority pick (one of the samples),
          "majority_key": the vote key,
          "vote_distribution": {key: count, ...},
          "unanimity":   count(majority) / n_valid,
          "entropy":     Shannon entropy in bits over the distribution,
        }
    """
    samples = []
    vote_keys = []
    valid_indices = []
    for i in range(n):
        parsed, raw = call_fn(**call_kwargs)
        samples.append({"parsed": parsed, "raw": raw})
        if parsed is None:
            continue
        k = key_fn(parsed) if key_fn else parsed
        try:
            hash(k)
        except TypeError:
            k = repr(k)
        vote_keys.append(k)
        valid_indices.append(i)

    if not vote_keys:
        return {
            "n_requested": n,
            "n_valid": 0,
            "samples": samples,
            "vote_keys": [],
            "majority": None,
            "majority_key": None,
            "vote_distribution": {},
            "unanimity": 0.0,
            "entropy": 0.0,
        }

    counts = Counter(vote_keys)
    majority_key, majority_count = counts.most_common(1)[0]
    # Pick the first sample whose key == majority_key
    for idx in valid_indices:
        k = key_fn(samples[idx]["parsed"]) if key_fn else samples[idx]["parsed"]
        try:
            hash(k)
        except TypeError:
            k = repr(k)
        if k == majority_key:
            majority_parsed = samples[idx]["parsed"]
            break
    n_valid = len(vote_keys)
    unanimity = majority_count / n_valid
    # Shannon entropy in bits
    entropy = 0.0
    for c in counts.values():
        p = c / n_valid
        entropy -= p * math.log2(p) if p > 0 else 0.0
    return {
        "n_requested": n,
        "n_valid": n_valid,
        "samples": samples,
        "vote_keys": [str(k) for k in vote_keys],
        "majority": majority_parsed,
        "majority_key": str(majority_key),
        "vote_distribution": {str(k): v for k, v in counts.items()},
        "unanimity": unanimity,
        "entropy": entropy,
    }


# ---------- Self-test ----------
if __name__ == "__main__":
    import random
    rng = random.Random(0)

    def fake_caller(p: float = 0.7) -> tuple[int, str]:
        v = 1 if rng.random() < p else 0
        return v, f"fake-raw-{v}"

    r = vote_call(fake_caller, n=11, p=0.7)
    print(f"n_valid={r['n_valid']} majority={r['majority']} dist={r['vote_distribution']} "
          f"unan={r['unanimity']:.2f} ent={r['entropy']:.3f}")

    # Test None handling
    def sometimes_none() -> tuple[int | None, str]:
        v = rng.choice([0, 1, None])
        return v, str(v)
    r = vote_call(sometimes_none, n=10)
    print(f"n_valid={r['n_valid']} majority={r['majority']} dist={r['vote_distribution']}")
