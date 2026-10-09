"""Per-read SAM parity check: minimap2_original vs minimap2_vecrec.

Runs both binaries on the same reads/reference and compares, per primary
alignment, the mapping coordinate (RNAME:POS), the CIGAR, the alignment score
(AS:i:), and MAPQ. Reports how many reads agree on each field. The VecRec
extension uses a pragmatic (optimal-score) traceback, so CIGAR may differ from
ksw2 only on equal-score tiebreaks while POS/AS stay identical.

Usage:
    uv run python end_to_end_baselines/scripts/compare_e2e_sam.py [--zdrop N]
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
_REF_DEFAULT = _REPO / "end_to_end_baselines" / "data" / "MT-human.fa"
_READS_DEFAULT = _REPO / "end_to_end_baselines" / "data" / "mt_reads.fq"

BINARIES = {
    "original": _REPO / "bin" / "minimap2_original",
    "vecrec": _REPO / "bin" / "minimap2_vecrec",
}


def run_sam(binary: pathlib.Path, reads, ref, flags):
    res = subprocess.run([str(binary)] + flags + [str(ref), str(reads)],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{binary.name} exited {res.returncode}\n{res.stderr[:500]}")
    return res.stdout


def parse_primary(sam: str):
    """Return {qname: (rname, pos, mapq, cigar, AS)} for primary alignments."""
    out = {}
    for line in sam.splitlines():
        if line.startswith("@") or not line.strip():
            continue
        f = line.split("\t")
        if len(f) < 6:
            continue
        flag = int(f[1])
        if flag & 0x900:  # skip secondary (0x100) / supplementary (0x800)
            continue
        qname, rname, pos, mapq, cigar = f[0], f[2], f[3], f[4], f[5]
        score = None
        for tag in f[11:]:
            if tag.startswith("AS:i:"):
                score = int(tag[5:])
                break
        out[qname] = (rname, pos, mapq, cigar, score)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reads", type=pathlib.Path, default=_READS_DEFAULT)
    ap.add_argument("--ref", type=pathlib.Path, default=_REF_DEFAULT)
    ap.add_argument("--zdrop", type=int, default=1000000,
                    help="z-drop value; large = effectively off (the VecRec "
                         "integration assumes z-drop disabled)")
    ap.add_argument("--bw", type=int, default=None,
                    help="-r bandwidth override; default None = minimap2's own "
                         "preset default band (i.e. real banding applies). Pass "
                         "a large value (e.g. 100000) to match the unbanded "
                         "'exact' scenario regime -- omitting this previously "
                         "made every parity check run banded regardless of the "
                         "scenario's own bw setting.")
    ap.add_argument("--preset", type=str, default="map-ont", help="-x preset")
    ap.add_argument("--show", type=int, default=8, help="max divergent reads to print")
    args = ap.parse_args()

    flags = ["-ax", args.preset, "-t", "1", "--secondary=no", "-z", str(args.zdrop)]
    if args.bw is not None:
        flags += ["-r", f"{args.bw},{args.bw}"]
    for name, b in BINARIES.items():
        if not b.exists():
            print(f"ERROR: {name} binary missing: {b}", file=sys.stderr)
            sys.exit(1)

    orig = parse_primary(run_sam(BINARIES["original"], args.reads, args.ref, flags))
    vrec = parse_primary(run_sam(BINARIES["vecrec"], args.reads, args.ref, flags))

    common = sorted(set(orig) & set(vrec))
    only_o = set(orig) - set(vrec)
    only_v = set(vrec) - set(orig)

    n = len(common)
    same_loc = same_cigar = same_as = same_mapq = 0
    cigar_diffs = []
    loc_diffs = []
    as_diffs = []
    for q in common:
        ro, po, mo, co, so = orig[q]
        rv, pv, mv, cv, sv = vrec[q]
        loc_ok = (ro == rv and po == pv)
        same_loc += loc_ok
        same_cigar += (co == cv)
        same_mapq += (mo == mv)
        same_as += (so == sv)
        if not loc_ok:
            loc_diffs.append((q, f"{ro}:{po}", f"{rv}:{pv}"))
        if so != sv:
            as_diffs.append((q, so, sv))
        if co != cv:
            cigar_diffs.append((q, so, sv, co, cv))

    print(f"reads aligned by both: {n}")
    print(f"  only original: {len(only_o)}   only vecrec: {len(only_v)}")
    if n:
        print(f"  same RNAME:POS : {same_loc}/{n} ({100*same_loc/n:.1f}%)")
        print(f"  same MAPQ      : {same_mapq}/{n} ({100*same_mapq/n:.1f}%)")
        print(f"  same AS score  : {same_as}/{n} ({100*same_as/n:.1f}%)")
        print(f"  same CIGAR     : {same_cigar}/{n} ({100*same_cigar/n:.1f}%)")

    if loc_diffs:
        print(f"\nlocation differences ({len(loc_diffs)}):")
        for q, a, b in loc_diffs[:args.show]:
            print(f"  {q}: orig={a} vecrec={b}")
    if as_diffs:
        print(f"\nscore (AS) differences ({len(as_diffs)}):")
        for q, a, b in as_diffs[:args.show]:
            print(f"  {q}: orig AS={a} vecrec AS={b} (delta={('NA' if a is None or b is None else b-a)})")
    if cigar_diffs:
        equal_score = sum(1 for (_, so, sv, _, _) in cigar_diffs if so == sv)
        print(f"\nCIGAR differences ({len(cigar_diffs)}; {equal_score} at equal AS score):")
        for q, so, sv, co, cv in cigar_diffs[:args.show]:
            print(f"  {q} (AS o={so} v={sv}):")
            print(f"    orig:   {co[:80]}{'...' if len(co) > 80 else ''}")
            print(f"    vecrec: {cv[:80]}{'...' if len(cv) > 80 else ''}")

    # Exit non-zero only if mapping coordinates or scores diverge (CIGAR
    # tiebreak differences at equal score are acceptable for the pragmatic TB).
    bad = bool(loc_diffs or as_diffs or only_o or only_v)
    print("\nRESULT:", "DIVERGENCE in POS/AS/coverage" if bad else
          "POS + AS + coverage identical (CIGAR may differ at equal score)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
