"""Generate a synthetic tandem-repeat reference and long reads for extension benchmarking.

Creates:
  data/tandem_ref.fa    — reference: 50 tandem copies of a 5 kbp unit, each with
                          3% within-copy divergence (like real rDNA arrays)
  data/tandem_reads.fq  — 200 reads of 15 kbp, each spanning ~3 tandem copies

Design rationale:
  Perfectly identical copies overwhelm minimap2's chaining step (chaining is O(n^2)
  in the number of seed hits, so 200 identical copies is catastrophic).  Real tandem
  arrays (rDNA, centromeric satellites) have 2-5% within-copy divergence, which
  lets minimap2 distinguish copies and keeps chaining fast.

  With 50 copies and 3% divergence, each read matches ~10-20 candidate positions.
  minimap2 calls ksw_extz2 for left+right extensions of every candidate chain.
  At 15 kbp per extension × ~15 chains × 200 reads, the extension kernel dominates.

Usage:
    uv run python end_to_end_baselines/scripts/generate_tandem_reads.py [--seed N]
"""
from __future__ import annotations

import argparse
import pathlib
import random

_REPO = pathlib.Path(__file__).resolve().parents[2]
_DATA = _REPO / "end_to_end_baselines" / "data"

UNIT_LEN      = 5_000   # one tandem unit (bp)
N_COPIES      = 50      # number of tandem copies in reference
COPY_DIVERGE  = 0.03    # within-copy divergence from the consensus (realistic rDNA)
READ_LEN      = 15_000  # read length: spans ~3 units
N_READS       = 200     # reads per run
READ_ERROR    = 0.05    # per-read substitution error (ONT-style)
ALPHABET      = "ACGT"


def _random_seq(length: int, rng: random.Random) -> str:
    return "".join(rng.choices(ALPHABET, k=length))


def _diverge(seq: str, rate: float, rng: random.Random) -> str:
    """Return a copy of seq with `rate` fraction of positions substituted."""
    bases = list(seq)
    for i in range(len(bases)):
        if rng.random() < rate:
            bases[i] = rng.choice([b for b in ALPHABET if b != bases[i]])
    return "".join(bases)


def write_tandem_ref(out: pathlib.Path, seed: int) -> str:
    rng = random.Random(seed)
    consensus = _random_seq(UNIT_LEN, rng)
    copies = [_diverge(consensus, COPY_DIVERGE, rng) for _ in range(N_COPIES)]
    ref = "".join(copies)
    with out.open("w") as fh:
        fh.write(f">tandem_ref copies={N_COPIES} unit={UNIT_LEN} diverge={COPY_DIVERGE:.0%}\n")
        for i in range(0, len(ref), 80):
            fh.write(ref[i : i + 80] + "\n")
    total_kb = len(ref) // 1000
    print(
        f"Wrote tandem reference: {total_kb} kbp  "
        f"({N_COPIES} copies × {UNIT_LEN} bp, {COPY_DIVERGE:.0%} within-copy divergence)  → {out}"
    )
    return ref


def write_reads(ref: str, out: pathlib.Path, seed: int) -> None:
    rng = random.Random(seed + 1)
    ref_len = len(ref)
    # Space reads across the first copy so they sample different phases
    step = max(1, UNIT_LEN // N_READS)
    with out.open("w") as fh:
        for i in range(N_READS):
            start = (i * step) % (ref_len - READ_LEN)
            seq   = _diverge(ref[start : start + READ_LEN], READ_ERROR, rng)
            qual  = "I" * len(seq)
            fh.write(f"@tandem_read_{i:04d}\n{seq}\n+\n{qual}\n")
    print(
        f"Wrote {N_READS} × {READ_LEN // 1000} kbp reads "
        f"(read error={READ_ERROR:.0%})  → {out}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _DATA.mkdir(parents=True, exist_ok=True)
    ref   = write_tandem_ref(_DATA / "tandem_ref.fa", args.seed)
    write_reads(ref, _DATA / "tandem_reads.fq", args.seed)
