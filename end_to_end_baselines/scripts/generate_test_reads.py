"""Fragment MT-human.fa into synthetic ONT-style FASTQ reads.

Produces end_to_end_baselines/data/mt_reads.fq — 200 overlapping 1000 bp
reads that give ~200 extension calls per minimap2 run, enough for stable
timing across repeated runs.

Usage:
    uv run python end_to_end_baselines/scripts/generate_test_reads.py
"""
from __future__ import annotations

import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

pass  # stdlib only


def _load_fasta(path: pathlib.Path) -> str:
    seq_parts = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq_parts.append(line.upper())
    return "".join(seq_parts)


def generate_reads(
    source_fa: pathlib.Path,
    out_fq: pathlib.Path,
    read_len: int = 1000,
    n_reads: int = 200,
    step: int = 75,   # overlap step; total coverage = n_reads*step / seq_len
) -> None:
    seq = _load_fasta(source_fa)
    if len(seq) < read_len:
        raise ValueError(f"Source sequence too short ({len(seq)} < {read_len})")

    with out_fq.open("w") as fh:
        for i in range(n_reads):
            start = (i * step) % (len(seq) - read_len)
            read = seq[start : start + read_len]
            qual = "I" * len(read)   # fake Phred 40 quality
            fh.write(f"@read_{i:04d}\n{read}\n+\n{qual}\n")

    print(f"Wrote {n_reads} × {read_len} bp reads to {out_fq}")


if __name__ == "__main__":
    source = _REPO / "end_to_end_baselines" / "data" / "MT-human.fa"
    out    = _REPO / "end_to_end_baselines" / "data" / "mt_reads.fq"
    out.parent.mkdir(parents=True, exist_ok=True)
    generate_reads(source, out)
