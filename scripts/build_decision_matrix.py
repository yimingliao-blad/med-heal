#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pre_atom.registry import write_matrix  # noqa: E402


if __name__ == "__main__":
    out = write_matrix(ROOT / "reports" / "VARIANT_DECISION_MATRIX.md")
    print(out)

