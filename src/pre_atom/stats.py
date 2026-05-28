from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def mcnemar_exact(breaks: int, fixes: int) -> dict[str, Any]:
    """Exact one-sided binomial McNemar test for fixes > breaks."""
    n = breaks + fixes
    if n == 0:
        return {"n_discordant": 0, "p_value_one_sided": None, "statistic": None}
    # P(X >= fixes), X ~ Binomial(n, 0.5)
    p = sum(math.comb(n, k) for k in range(fixes, n + 1)) / (2**n)
    stat = ((abs(fixes - breaks) - 1) ** 2 / n) if n else None
    return {"n_discordant": n, "p_value_one_sided": p, "statistic_chi2_cc": stat}


def bootstrap_delta(zs: np.ndarray, final: np.ndarray, n_boot: int = 10000, seed: int = 42) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(zs)
    if n == 0:
        return {"mean_delta": None, "ci95": [None, None], "p_delta_gt_0": None}
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = final[idx].mean() - zs[idx].mean()
    return {
        "mean_delta": float(deltas.mean()),
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "p_delta_gt_0": float((deltas > 0).mean()),
    }


def paired_summary(df: pd.DataFrame, *, fold_col: str = "fold") -> dict[str, Any]:
    required = {"zeroshot_correct", "final_correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    zs = df["zeroshot_correct"].astype(int).to_numpy()
    final = df["final_correct"].astype(int).to_numpy()
    fixes = int(((zs == 0) & (final == 1)).sum())
    breaks = int(((zs == 1) & (final == 0)).sum())
    same_correct = int(((zs == 1) & (final == 1)).sum())
    same_wrong = int(((zs == 0) & (final == 0)).sum())
    out: dict[str, Any] = {
        "n": int(len(df)),
        "zeroshot_accuracy": float(zs.mean()) if len(zs) else None,
        "final_accuracy": float(final.mean()) if len(final) else None,
        "delta": float(final.mean() - zs.mean()) if len(zs) else None,
        "fixes": fixes,
        "breaks": breaks,
        "same_correct": same_correct,
        "same_wrong": same_wrong,
        "mcnemar": mcnemar_exact(breaks, fixes),
        "bootstrap": bootstrap_delta(zs, final),
    }
    if fold_col in df.columns:
        folds = []
        for fold, g in df.groupby(fold_col):
            z = g["zeroshot_correct"].astype(int)
            f = g["final_correct"].astype(int)
            folds.append(
                {
                    "fold": int(fold),
                    "n": int(len(g)),
                    "zeroshot_accuracy": float(z.mean()),
                    "final_accuracy": float(f.mean()),
                    "delta": float(f.mean() - z.mean()),
                    "fixes": int(((z == 0) & (f == 1)).sum()),
                    "breaks": int(((z == 1) & (f == 0)).sum()),
                }
            )
        out["per_fold"] = folds
        loo = []
        for fold in sorted(df[fold_col].unique()):
            g = df[df[fold_col] != fold]
            if len(g):
                loo.append({"held_out_fold": int(fold), "delta": float(g["final_correct"].mean() - g["zeroshot_correct"].mean())})
        out["leave_one_fold_out"] = loo
    return out


def write_stats(input_csv: str | Path, output_dir: str | Path) -> dict[str, Any]:
    df = pd.read_csv(input_csv)
    summary = paired_summary(df)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paired_stats.json").write_text(json.dumps(summary, indent=2))
    rows = []
    rows.append(
        {
            "n": summary["n"],
            "zeroshot_accuracy": summary["zeroshot_accuracy"],
            "final_accuracy": summary["final_accuracy"],
            "delta": summary["delta"],
            "fixes": summary["fixes"],
            "breaks": summary["breaks"],
            "mcnemar_p_one_sided": summary["mcnemar"]["p_value_one_sided"],
            "bootstrap_ci_low": summary["bootstrap"]["ci95"][0],
            "bootstrap_ci_high": summary["bootstrap"]["ci95"][1],
        }
    )
    pd.DataFrame(rows).to_csv(out_dir / "paired_stats.csv", index=False)
    return summary

