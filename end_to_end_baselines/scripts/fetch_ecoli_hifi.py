"""Real-world dataset: real PacBio HiFi E. coli reads.

530 real PacBio HiFi reads of E. coli K-12 (SRA accession SRR10971019, first
15MiB subset redistributed via Zenodo for testing), mapped against the real
E. coli K-12 MG1655 reference (shared with ecoli-ultralong-real). HiFi's near
-zero error rate (Q30+, >99.9% per-base accuracy) means indels are rare, so
minimap2's real default band rarely clips and z-drop rarely fires -- the
profile expected to most favor VecRec's approx-on path, in contrast to the
noisy ONT ultra-long reads.

Usage:
    uv run python end_to_end_baselines/scripts/fetch_ecoli_hifi.py
"""
from __future__ import annotations

import pathlib
import subprocess

_REPO = pathlib.Path(__file__).resolve().parents[2]
_DATA = _REPO / "end_to_end_baselines" / "data" / "realbench"

READS_URL = "https://zenodo.org/api/records/19050385/files/SRR10971019.first15MB.fastq/content"
REF_URL = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
           "?db=nuccore&id=NC_000913.3&rettype=fasta&retmode=text")  # K-12 MG1655

READS = _DATA / "ecoli_hifi_reads.fastq"
REF = _DATA / "ecoli_k12_mg1655.fna"           # shared with ecoli-ultralong-real


def main() -> None:
    _DATA.mkdir(parents=True, exist_ok=True)

    if not READS.exists():
        print(f"Downloading real PacBio HiFi E. coli reads -> {READS.name} ...")
        subprocess.run(["curl", "-sS", "-L", "--max-time", "120", "-o", str(READS), READS_URL], check=True)
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
