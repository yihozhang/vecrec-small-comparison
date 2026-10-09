"""Real-world dataset: real ONT ultra-long E. coli reads vs. the real E. coli
K-12 MG1655 reference genome.

Unlike sv-fair/tandem (synthetic reads) and crossspecies-real (real genomes but
a cross-species pair prone to real structural-rearrangement chimeras), this is
real sequencing data mapped to the reference of the *same* organism it was
sequenced from — real basecalling error, real read-length distribution
(212bp-393kbp, "ultra-long" nanopore prep), but without the cross-species
chimeric-split noise. That combination tends to produce plenty of large,
genuinely-in-band gap-fill DP (good for VecRec's per-cell throughput) while
keeping the alignment structure "boring" (mostly co-linear), which is where
VecRec's own banded static-pruning path is expected to have the best shot at
beating stock's approx-max + default-band shortcut.

Reads: 371 real ultra-long ONT reads of E. coli, originally from Josh Quick's
"ultra-long read" nanopore protocol post (http://lab.loman.net/2017/03/09/
ultrareads-for-nanopore/), redistributed by wdecoster/nanotest (GitHub).
Reference: E. coli K-12 MG1655 (NCBI RefSeq NC_000913.3), fetched via NCBI
eutils.

Usage:
    uv run python end_to_end_baselines/scripts/fetch_ecoli_ultralong.py
"""
from __future__ import annotations

import pathlib
import subprocess

_REPO = pathlib.Path(__file__).resolve().parents[2]
_DATA = _REPO / "end_to_end_baselines" / "data" / "realbench"

READS_URL = "https://raw.githubusercontent.com/wdecoster/nanotest/master/reads.fa.gz"
REF_URL = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
           "?db=nuccore&id=NC_000913.3&rettype=fasta&retmode=text")

READS = _DATA / "ecoli_ultralong_reads.fa"
REF = _DATA / "ecoli_k12_mg1655.fna"


def main() -> None:
    _DATA.mkdir(parents=True, exist_ok=True)

    if not READS.exists():
        gz = READS.with_suffix(READS.suffix + ".gz")
        print(f"Downloading real ultra-long E. coli ONT reads -> {READS.name} ...")
        subprocess.run(["curl", "-sS", "-L", "--max-time", "120", "-o", str(gz), READS_URL], check=True)
        subprocess.run(["gunzip", "-f", str(gz)], check=True)
    else:
        print(f"{READS.name} already present, skipping.")

    if not REF.exists():
        print(f"Downloading E. coli K-12 MG1655 reference -> {REF.name} ...")
        subprocess.run(["curl", "-sS", "-L", "--max-time", "120", "-o", str(REF), REF_URL], check=True)
    else:
        print(f"{REF.name} already present, skipping.")

    print("Done.")


if __name__ == "__main__":
    main()
