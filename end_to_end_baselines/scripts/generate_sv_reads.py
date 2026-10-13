"""Generate a realistic "large-extension" benchmark for the minimap2 VecRec drop-in.

Motivation
----------
minimap2's ordinary map-ont extension matrices are small because chaining
anchors densely. To exercise large extension DP matrices we need LARGE
inter-anchor gaps.

This generator synthesizes reads with two EXACT-match flanks (strong, unambiguous
anchors) surrounding a highly DIVERGENT core (substitutions + indels dense enough
that no k-mer seeds survive). minimap2 chains the two flanks and then fills the
divergent core with a single ~(core x core) gap-fill DP. Sweeping the core length
scales the extension matrix from ~285^2 (core=500) to ~1000^2 (core=3000).

This models real divergent/variable regions flanked by conserved sequence
(hypervariable segments, divergent alleles, SV-flanking drift) and needs no
download -- the flanks/cores are sampled from the real yeast chrXII sequence
already vendored in the repo.

Usage
-----
    uv run python end_to_end_baselines/scripts/generate_sv_reads.py \
        [--core 2000] [--n-reads 60] [--flank 600] [--seed 7] [--out PATH]

Fair comparison (IMPORTANT)
---------------------------
The VecRec drop-in always computes the EXACT full matrix (no banding, no z-drop,
no approx-max). To compare apples-to-apples, stock minimap2 must be forced into
the same regime:

    MM_NO_APPROX=1 uv run python end_to_end_baselines/scripts/run_e2e.py \
        --ref end_to_end_baselines/data/yeast/chrXII.fa \
        --reads end_to_end_baselines/data/sv_reads.fq --zdrop 1000000 --bw 100000

  * --zdrop 1000000  : disables z-drop early termination (both builds)
  * --bw 100000      : forces the full (unbanded) DP matrix (both builds); without
                       it minimap2 may band some gap-fills and the full-matrix
                       drop-in aborts on its no-banding assertion.
  * MM_NO_APPROX=1   : forces stock ksw2 to the EXACT max path (drops
                       KSW_EZ_APPROX_MAX), matching VecRec's exact full matrix.

Without MM_NO_APPROX=1, stock uses an approximate-max path for gap fills (no
full-H store); that is not the same computation.
"""
from __future__ import annotations

import argparse
import pathlib
import random

_REPO = pathlib.Path(__file__).resolve().parents[2]
_REF = _REPO / "end_to_end_baselines" / "data" / "yeast" / "chrXII.fa"
_OUT_DEFAULT = _REPO / "end_to_end_baselines" / "data" / "sv_reads.fq"

ALPHABET = "ACGT"


def load_fasta(path: pathlib.Path) -> str:
    return "".join(
        ln.strip() for ln in path.open() if not ln.startswith(">")
    ).upper()


def diverge_core(seq: str, rng: random.Random,
                 p_sub: float, p_ins: float, p_del: float) -> str:
    """Apply dense substitutions + indels so no k-mer seed survives the core."""
    out: list[str] = []
    for c in seq:
        r = rng.random()
        if r < p_del:
            continue
        if r < p_del + p_ins:
            out.append(rng.choice(ALPHABET))
            out.append(c)
        elif r < p_del + p_ins + p_sub:
            out.append(rng.choice([x for x in ALPHABET if x != c]))
        else:
            out.append(c)
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core", type=int, default=2000,
                    help="divergent core length in bp (extension matrix ~= core^2). Default 2000.")
    ap.add_argument("--n-reads", type=int, default=60)
    ap.add_argument("--flank", type=int, default=600,
                    help="exact-match flank length each side (the anchors). Default 600.")
    ap.add_argument("--p-sub", type=float, default=0.20)
    ap.add_argument("--p-ins", type=float, default=0.08)
    ap.add_argument("--p-del", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ref", type=pathlib.Path, default=_REF)
    ap.add_argument("--out", type=pathlib.Path, default=_OUT_DEFAULT)
    args = ap.parse_args()

    ref = load_fasta(args.ref)
    span = 2 * args.flank + args.core
    if len(ref) <= span + 1:
        ap.error(f"reference too short ({len(ref)} bp) for span {span} bp")

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for i in range(args.n_reads):
            st = rng.randint(0, len(ref) - span - 1)
            left = ref[st : st + args.flank]
            core_src = ref[st + args.flank : st + args.flank + args.core]
            right = ref[st + args.flank + args.core : st + span]
            core = diverge_core(core_src, rng, args.p_sub, args.p_ins, args.p_del)
            read = left + core + right
            fh.write(f"@sv_core{args.core}_{i:04d}\n{read}\n+\n{'I' * len(read)}\n")

    edit = args.p_sub + args.p_ins + args.p_del
    print(
        f"Wrote {args.n_reads} reads (~{span} bp: {args.flank} exact flank + "
        f"{args.core} bp core @ {edit:.0%} edit + {args.flank} exact flank)\n"
        f"  reference: {args.ref}\n"
        f"  reads:     {args.out}\n"
        f"  extension DP per read ~= {args.core}x{args.core} (~{args.core**2} cells)\n"
        f"Run the FAIR comparison with:\n"
        f"  MM_NO_APPROX=1 uv run python end_to_end_baselines/scripts/run_e2e.py \\\n"
        f"      --ref {args.ref} --reads {args.out} --zdrop 1000000 --bw 100000"
    )


if __name__ == "__main__":
    main()
