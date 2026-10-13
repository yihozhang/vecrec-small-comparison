"""Real-world dataset: subsample of the full Loman/Quick ultra-long E. coli
ONT run vs K-12 MG1655.

This fetch pulls the full CLIMB release and keeps the N longest reads
(default 500).

Usage:
    uv run python end_to_end_baselines/scripts/fetch_ecoli_ultralong_full.py
    uv run python end_to_end_baselines/scripts/fetch_ecoli_ultralong_full.py --n 500
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess

_REPO = pathlib.Path(__file__).resolve().parents[2]
_DATA = _REPO / "end_to_end_baselines" / "data" / "realbench"

READS_URL = "https://s3.climb.ac.uk/nanopore/ecoli_allreads.fasta"
REF_URL = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
           "?db=nuccore&id=NC_000913.3&rettype=fasta&retmode=text")

RAW = _DATA / "ecoli_allreads_full.fasta"
READS = _DATA / "ecoli_ultralong_full_reads.fa"
REF = _DATA / "ecoli_k12_mg1655.fna"  # shared with ecoli-strains-real / ecoli-hifi-real


def _iter_fasta(path: pathlib.Path):
    name, parts = None, []
    with path.open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(parts)
                name, parts = line[1:].split()[0], []
            else:
                parts.append(line)
        if name is not None:
            yield name, "".join(parts)


def subsample_longest(src: pathlib.Path, dst: pathlib.Path, n: int) -> None:
    print(f"Selecting {n} longest reads from {src.name} ...")
    records = list(_iter_fasta(src))
    records.sort(key=lambda kv: len(kv[1]), reverse=True)
    keep = records[:n]
    with dst.open("w") as fh:
        for name, seq in keep:
            fh.write(f">{name}\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i : i + 80] + "\n")
    lengths = [len(s) for _, s in keep]
    print(f"Wrote {len(keep)} reads to {dst.name} "
          f"(len min={min(lengths)} med={sorted(lengths)[len(lengths)//2]} "
          f"max={max(lengths)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=500,
                    help="number of longest reads to keep (default 500)")
    args = ap.parse_args()

    _DATA.mkdir(parents=True, exist_ok=True)

    if not RAW.exists():
        print(f"Downloading full Loman/Quick ultra-long E. coli ONT reads -> {RAW.name} ...")
        subprocess.run(["curl", "-sS", "-L", "--max-time", "1800", "-o", str(RAW), READS_URL], check=True)
    else:
        print(f"{RAW.name} already present, skipping.")

    if not REF.exists():
        print(f"Downloading E. coli K-12 MG1655 reference -> {REF.name} ...")
        subprocess.run(["curl", "-sS", "-L", "--max-time", "120", "-o", str(REF), REF_URL], check=True)
    else:
        print(f"{REF.name} already present, skipping.")

    # Always (re)write the subsample so --n changes take effect.
    subsample_longest(RAW, READS, args.n)
    print("Done.")


if __name__ == "__main__":
    main()
