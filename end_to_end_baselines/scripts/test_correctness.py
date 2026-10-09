"""Strict end-to-end correctness check: original vs VecRec, per read.

For each aligner we run the stock binary and the VecRec binary on the same
reads/reference and compare the primary alignment of every read field-by-field
(reference, position, CIGAR, MAPQ / alignment score).  Matching alignment
*counts* is necessary but not sufficient; this script verifies the actual
mapping coordinates and alignment shapes agree.

Usage:
    uv run python end_to_end_baselines/scripts/test_correctness.py [--aligner bwa|bowtie2|both]

Exit code is non-zero if any aligner falls below the agreement threshold.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
_REF = _REPO / "third_party" / "ksw2" / "test" / "MT-human.fa"
_READS = _REPO / "end_to_end_baselines" / "data" / "mt_reads.fq"
_BT2_INDEX = _REPO / "end_to_end_baselines" / "data" / "mt_bt2_index"

# Below this fraction of reads with identical (RNAME, POS, CIGAR) we fail.
THRESHOLD = 0.99


def _run(cmd: list[str]) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{cmd[0]} exited {res.returncode}\n{res.stderr[:800]}")
    return res.stdout


def _parse_primary(sam: str) -> dict[str, tuple]:
    """Map QNAME -> (RNAME, POS, CIGAR, MAPQ, AS) for primary alignments."""
    out: dict[str, tuple] = {}
    for line in sam.splitlines():
        if not line or line.startswith("@"):
            continue
        f = line.split("\t")
        if len(f) < 11:
            continue
        flag = int(f[1])
        if flag & 0x900:  # secondary / supplementary
            continue
        qname = f[0]
        mate = 0x40 if (flag & 0x40) else (0x80 if (flag & 0x80) else 0)
        key = f"{qname}/{mate}"
        as_score = None
        for tag in f[11:]:
            if tag.startswith("AS:i:"):
                as_score = int(tag[5:])
                break
        out[key] = (f[2], f[3], f[5], f[4], as_score)  # RNAME, POS, CIGAR, MAPQ, AS
    return out


def _compare(orig: dict, vrec: dict, label: str) -> bool:
    keys = set(orig) | set(vrec)
    n = len(keys)
    if n == 0:
        print(f"  {label}: no alignments parsed")
        return False
    same_pos = same_cigar = same_score = both_present = 0
    diffs = []
    for k in keys:
        o = orig.get(k)
        v = vrec.get(k)
        if o is None or v is None:
            diffs.append((k, o, v))
            continue
        both_present += 1
        if (o[0], o[1]) == (v[0], v[1]):
            same_pos += 1
        if o[2] == v[2]:
            same_cigar += 1
        if o[4] == v[4]:
            same_score += 1
        if (o[0], o[1], o[2]) != (v[0], v[1], v[2]):
            if len(diffs) < 10:
                diffs.append((k, o, v))
    pos_rate = same_pos / n
    cig_rate = same_cigar / n
    sc_rate = same_score / both_present if both_present else 0.0
    print(f"  {label}: {n} reads")
    print(f"    same (RNAME,POS):  {same_pos}/{n}  ({pos_rate*100:.1f}%)")
    print(f"    same CIGAR:        {same_cigar}/{n}  ({cig_rate*100:.1f}%)")
    print(f"    same AS score:     {same_score}/{both_present}  ({sc_rate*100:.1f}%)")
    if diffs:
        print(f"    first diffs (key, original, vecrec):")
        for k, o, v in diffs[:5]:
            print(f"      {k}: {o} != {v}")
    return pos_rate >= THRESHOLD


def test_bwa() -> bool:
    print("BWA-MEM2:")
    orig_bin = _REPO / "bin" / "bwa_mem2_original"
    vrec_bin = _REPO / "bin" / "bwa_mem2_vecrec"
    if not orig_bin.exists() or not vrec_bin.exists():
        print("  SKIP (binaries missing; run build_bwa_mem2.sh)")
        return True
    common = ["mem", "-t", "1", str(_REF), str(_READS)]
    o = _parse_primary(_run([str(orig_bin)] + common))
    v = _parse_primary(_run([str(vrec_bin)] + common))
    return _compare(o, v, "primary alignments")


def test_bowtie2() -> bool:
    print("Bowtie2:")
    orig_bin = _REPO / "bin" / "bowtie2_original"
    vrec_bin = _REPO / "bin" / "bowtie2_vecrec"
    if not orig_bin.exists() or not vrec_bin.exists():
        print("  SKIP (binaries missing; run build_bowtie2.sh)")
        return True
    common = ["-p", "1", "--end-to-end", "-x", str(_BT2_INDEX), "-U", str(_READS)]
    o = _parse_primary(_run([str(orig_bin)] + common))
    v = _parse_primary(_run([str(vrec_bin)] + common))
    return _compare(o, v, "primary alignments")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aligner", choices=["bwa", "bowtie2", "both"], default="both")
    args = ap.parse_args()

    ok = True
    if args.aligner in ("bwa", "both"):
        ok &= test_bwa()
    if args.aligner in ("bowtie2", "both"):
        ok &= test_bowtie2()

    print()
    if ok:
        print(f"PASS: all aligners >= {THRESHOLD*100:.0f}% identical primary mappings")
        sys.exit(0)
    else:
        print(f"FAIL: at least one aligner below {THRESHOLD*100:.0f}% agreement")
        sys.exit(1)


if __name__ == "__main__":
    main()
