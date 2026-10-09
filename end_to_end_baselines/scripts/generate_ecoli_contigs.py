"""Fragment the E. coli genome (realbench/ecoli.fna) into large non-overlapping
contigs to test BWA-MEM2 against a divergent reference (Salmonella) with
long-range extension DP (ad-hoc experiment, not part of the standard dataset
suite).

Usage:
    uv run python end_to_end_baselines/scripts/generate_ecoli_contigs.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "end_to_end_baselines" / "scripts"))

from generate_test_reads import generate_reads  # noqa: E402

_SRC = _REPO / "end_to_end_baselines" / "data" / "realbench" / "ecoli.fna"
_OUT = _REPO / "end_to_end_baselines" / "data" / "realbench" / "ecoli_contigs_50k.fq"

if __name__ == "__main__":
    generate_reads(_SRC, _OUT, read_len=50_000, n_reads=90, step=50_000)
