"""Real-world dataset: P. aeruginosa PA14 contigs vs PAO1 reference.

Same-species strain pair with large pathogenicity-island / accessory-genome
differences (classic PA14 vs PAO1 comparison). Mirrors ecoli-strains-real:
chop the query into 50 kbp contigs and map under map-ont, where VecRec is
expected to win exact DP while banding/z-drop cost real fidelity.

Usage:
    uv run python end_to_end_baselines/scripts/fetch_pa_strains.py
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
_DATA = _REPO / "end_to_end_baselines" / "data" / "realbench"
sys.path.insert(0, str(_REPO / "end_to_end_baselines" / "scripts"))

REF_URL = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
           "?db=nuccore&id=NC_002516.2&rettype=fasta&retmode=text")  # PAO1
QRY_URL = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
           "?db=nuccore&id=CP000438.1&rettype=fasta&retmode=text")  # PA14

REF = _DATA / "pa_pao1.fna"
PA14 = _DATA / "pa_pa14.fna"
CONTIGS = _DATA / "pa_pa14_contigs_50k.fq"


def main() -> None:
    _DATA.mkdir(parents=True, exist_ok=True)

    if not REF.exists():
        print(f"Downloading P. aeruginosa PAO1 reference -> {REF.name} ...")
        subprocess.run(["curl", "-sS", "-L", "--max-time", "300", "-o", str(REF), REF_URL], check=True)
    else:
        print(f"{REF.name} already present, skipping.")

    if not PA14.exists():
        print(f"Downloading P. aeruginosa PA14 genome -> {PA14.name} ...")
        subprocess.run(["curl", "-sS", "-L", "--max-time", "300", "-o", str(PA14), QRY_URL], check=True)
    else:
        print(f"{PA14.name} already present, skipping.")

    from generate_test_reads import generate_reads
    generate_reads(PA14, CONTIGS, read_len=50_000, n_reads=100, step=50_000)
    print("Done.")


if __name__ == "__main__":
    main()
