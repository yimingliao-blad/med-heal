#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pre_atom.stats import write_stats  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: scripts/run_stats.py paired_outcomes.csv [output_dir]")
        return 2
    input_csv = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "reports" / "stats"
    summary = write_stats(input_csv, output_dir)
    print(f"wrote {output_dir}")
    print(f"n={summary['n']} delta={summary['delta']:.4f} fixes={summary['fixes']} breaks={summary['breaks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

