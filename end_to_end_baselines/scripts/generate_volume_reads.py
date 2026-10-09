"""Generate large-volume MT-human read sets to dilute BWA-MEM2's fixed index-load
cost relative to the Smith-Waterman extension work (ad-hoc experiment, not part
of the standard dataset suite).

Usage:
    uv run python end_to_end_baselines/scripts/generate_volume_reads.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "end_to_end_baselines" / "scripts"))

from generate_test_reads import generate_reads  # noqa: E402

_SRC = _REPO / "third_party" / "ksw2" / "test" / "MT-human.fa"
_OUT = _REPO / "end_to_end_baselines" / "data"

if __name__ == "__main__":
    for n in (2_000, 20_000):
        generate_reads(_SRC, _OUT / f"mt_reads_vol{n}.fq", read_len=1000, n_reads=n, step=7)
