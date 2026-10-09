"""Real-world dataset: same-species, structurally-divergent genome comparison.

E. coli K-12 MG1655 (already fetched for ecoli-ultralong-real/ecoli-hifi-real)
vs. E. coli O157:H7 Sakai -- the same species as K-12, but a different
pathotype with ~1.4 Mbp of extra genomic islands (prophages, O-islands) and
core-genome SNP divergence of roughly 1-2%. The intent: same-species (so no
cross-species chimeric-alignment confound), collinear across most of the core
backbone, but with real, substantial per-window divergence -- large enough to
force big gap-fill DP (good for VecRec), unlike the ~99.9%-identical HiFi
reads which barely need gap-fills at all.

Caveat found by a quick probe: O157:H7's genomic islands are frequent enough
(~140 alignment segments for a single 5.5 Mbp query at -x asm10) that z-drop
still fires often here -- this pair is not as "clean" (collinear) as hoped,
though still less structurally chaotic than the E. coli/Salmonella
cross-species pair (crossspecies-real).

Query is chopped into 50kbp contigs (matching the crossspecies-real /
generate_ecoli_contigs.py convention) rather than aligned as one giant
5.5Mbp sequence, so per-read timing stays comparable across datasets.

Usage:
    uv run python end_to_end_baselines/scripts/fetch_ecoli_strains.py
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
_DATA = _REPO / "end_to_end_baselines" / "data" / "realbench"
sys.path.insert(0, str(_REPO / "end_to_end_baselines" / "scripts"))

REF_URL = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
           "?db=nuccore&id=NC_000913.3&rettype=fasta&retmode=text")  # K-12 MG1655
QRY_URL = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
           "?db=nuccore&id=NC_002695.2&rettype=fasta&retmode=text")  # O157:H7 Sakai

REF = _DATA / "ecoli_k12_mg1655.fna"           # shared with ecoli-ultralong-real
O157 = _DATA / "ecoli_o157h7.fna"
CONTIGS = _DATA / "ecoli_o157h7_contigs_50k.fq"


def main() -> None:
    _DATA.mkdir(parents=True, exist_ok=True)

    if not REF.exists():
        print(f"Downloading E. coli K-12 MG1655 reference -> {REF.name} ...")
        subprocess.run(["curl", "-sS", "-L", "--max-time", "120", "-o", str(REF), REF_URL], check=True)
    else:
        print(f"{REF.name} already present, skipping.")

    if not O157.exists():
        print(f"Downloading E. coli O157:H7 Sakai genome -> {O157.name} ...")
        subprocess.run(["curl", "-sS", "-L", "--max-time", "120", "-o", str(O157), QRY_URL], check=True)
    else:
        print(f"{O157.name} already present, skipping.")

    from generate_test_reads import generate_reads
    generate_reads(O157, CONTIGS, read_len=50_000, n_reads=100, step=50_000)
    print("Done.")


if __name__ == "__main__":
    main()
