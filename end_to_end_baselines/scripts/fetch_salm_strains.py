"""Real-world dataset: S. Typhimurium SL1344 contigs vs LT2 reference.

Same-serovar strain pair (not cross-species). Real prophage / island
differences between SL1344 and LT2, chopped into 50 kbp contigs for map-ont
ablation — a same-species control against the losing crossspecies-real/asm20
regime.

Usage:
    uv run python end_to_end_baselines/scripts/fetch_salm_strains.py
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
_DATA = _REPO / "end_to_end_baselines" / "data" / "realbench"
sys.path.insert(0, str(_REPO / "end_to_end_baselines" / "scripts"))

REF_URL = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
           "?db=nuccore&id=NC_003197.2&rettype=fasta&retmode=text")  # LT2
QRY_URL = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
           "?db=nuccore&id=FQ312003.1&rettype=fasta&retmode=text")  # SL1344

REF = _DATA / "salm_lt2.fna"
SL1344 = _DATA / "salm_sl1344.fna"
CONTIGS = _DATA / "salm_sl1344_contigs_50k.fq"


def main() -> None:
    _DATA.mkdir(parents=True, exist_ok=True)

    if not REF.exists():
        print(f"Downloading S. Typhimurium LT2 reference -> {REF.name} ...")
        subprocess.run(["curl", "-sS", "-L", "--max-time", "300", "-o", str(REF), REF_URL], check=True)
    else:
        print(f"{REF.name} already present, skipping.")

    if not SL1344.exists():
        print(f"Downloading S. Typhimurium SL1344 genome -> {SL1344.name} ...")
        subprocess.run(["curl", "-sS", "-L", "--max-time", "300", "-o", str(SL1344), QRY_URL], check=True)
    else:
        print(f"{SL1344.name} already present, skipping.")

    from generate_test_reads import generate_reads
    generate_reads(SL1344, CONTIGS, read_len=50_000, n_reads=100, step=50_000)
    print("Done.")


if __name__ == "__main__":
    main()
