"""End-to-end minimap2 pipeline comparison: original ksw2 vs vecrec extension.

Times the FULL minimap2 pipeline (index + seed + chain + extend + output) for
both binaries and prints a comparison table.

Usage:
    uv run python end_to_end_baselines/scripts/run_e2e.py [--reps N]

Requires:
    bin/minimap2_original and bin/minimap2_vecrec  (built by build_minimap2.sh)
    end_to_end_baselines/data/mt_reads.fq          (built by generate_test_reads.py)
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import subprocess
import sys
import time

_REPO = pathlib.Path(__file__).resolve().parents[2]

_REF_DEFAULT   = _REPO / "end_to_end_baselines" / "data" / "MT-human.fa"
_READS_DEFAULT = _REPO / "end_to_end_baselines" / "data" / "mt_reads.fq"

BINARIES = [
    ("minimap2-original", _REPO / "bin" / "minimap2_original"),
    ("minimap2-vecrec",   _REPO / "bin" / "minimap2_vecrec"),
]

# minimap2 flags: -x map-ont preset for long-read reference alignment;
# -t 1: single thread for deterministic timing; -a: SAM output (counts with grep -v @)
MM2_FLAGS = ["-ax", "map-ont", "-t", "1", "--secondary=no"]


def run_once(binary: pathlib.Path, reads: pathlib.Path, ref: pathlib.Path, flags: list[str]) -> tuple[float, int]:
    """Return (wall_seconds, n_primary_alignments)."""
    t0 = time.perf_counter()
    result = subprocess.run(
        [str(binary)] + flags + [str(ref), str(reads)],
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        raise RuntimeError(
            f"{binary.name} exited {result.returncode}\n{result.stderr[:500]}"
        )
    # SAM output: count non-header lines (one per aligned read)
    n_alns = sum(1 for line in result.stdout.splitlines() if not line.startswith("@"))
    return elapsed, n_alns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=10,
                        help="Number of repeated runs per binary (default: 10)")
    parser.add_argument("--reads", type=pathlib.Path, default=_READS_DEFAULT,
                        help="FASTQ reads file (default: mt_reads.fq)")
    parser.add_argument("--ref", type=pathlib.Path, default=_REF_DEFAULT,
                        help="FASTA reference file (default: MT-human.fa)")
    parser.add_argument("--zdrop", type=int, default=None,
                        help="Override minimap2 zdrop value (-z); a very high value "
                             "effectively disables z-drop early termination")
    parser.add_argument("--bw", type=int, default=None,
                        help="Override minimap2 alignment bandwidth (-r NUM,NUM); a "
                             "value >= read length forces the full (unbanded) DP matrix")
    parser.add_argument("--extra-binary", action="append", default=[], metavar="NAME:PATH",
                        help="Additional binary to include in the comparison "
                             "(e.g. --extra-binary minimap2-vecrec-i8:bin/minimap2_vecrec_i8). "
                             "Repeatable.")
    args = parser.parse_args()
    READS = args.reads
    REF   = args.ref
    mm2_flags = list(MM2_FLAGS)
    if args.zdrop is not None:
        mm2_flags += ["-z", f"{args.zdrop},{args.zdrop}"]
    if args.bw is not None:
        mm2_flags += ["-r", f"{args.bw},{args.bw}"]

    if not REF.exists():
        print(f"ERROR: reference not found: {REF}", file=sys.stderr)
        sys.exit(1)
    if not READS.exists():
        print(f"ERROR: reads not found: {READS}", file=sys.stderr)
        print("       Run:  uv run python end_to_end_baselines/scripts/generate_test_reads.py",
              file=sys.stderr)
        sys.exit(1)

    binaries = list(BINARIES)
    for spec in args.extra_binary:
        name, _, path = spec.partition(":")
        if not path:
            print(f"ERROR: --extra-binary must be NAME:PATH, got {spec!r}", file=sys.stderr)
            sys.exit(1)
        binaries.append((name, pathlib.Path(path)))

    print(f"Reference: {REF.name}  ({REF.stat().st_size // 1024} KB)")
    print(f"Reads:     {READS.name}  ({READS.stat().st_size // 1024} KB)")
    print(f"Flags:     {' '.join(mm2_flags)}")
    print(f"Reps:      {args.reps}")
    print()

    results = {}
    for name, binary in binaries:
        if not binary.exists():
            print(f"  {name:25s}  SKIP  (binary not found: {binary})")
            continue

        times = []
        n_alns_list = []
        print(f"  {name}: ", end="", flush=True)
        for _ in range(args.reps):
            t, n = run_once(binary, READS, REF, mm2_flags)
            times.append(t)
            n_alns_list.append(n)
            print(".", end="", flush=True)
        print()

        med  = statistics.median(times)
        stdev = statistics.stdev(times) if len(times) > 1 else 0.0
        p10  = sorted(times)[max(0, len(times)//10 - 1)]
        p90  = sorted(times)[min(len(times)-1, -len(times)//10 - 1)]
        results[name] = (med, p10, p90, n_alns_list[0])

    print()
    print(f"{'binary':25s}  {'median':>8s}  {'p10–p90':>18s}  {'alignments':>10s}")
    print("-" * 70)
    for name, (med, p10, p90, n) in results.items():
        print(f"{name:25s}  {med*1000:7.1f}ms  [{p10*1000:.0f}–{p90*1000:.0f}ms]  {n:>10d}")

    if len(results) == 2:
        names = list(results)
        orig_t = results[names[0]][0]
        vrec_t = results[names[1]][0]
        speedup = orig_t / vrec_t if vrec_t > 0 else float("inf")
        print()
        print(f"Speedup (original / vecrec):  {speedup:.2f}×")

        orig_n = results[names[0]][3]
        vrec_n = results[names[1]][3]
        if orig_n == vrec_n:
            print(f"Alignment count:              {orig_n} = {vrec_n}  ✓")
        else:
            print(f"Alignment count:              original={orig_n}  vecrec={vrec_n}  ← differ (score approx)")


if __name__ == "__main__":
    main()
