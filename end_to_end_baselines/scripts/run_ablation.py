"""VecRec minimap2 ablation sweep.

Rebuilds bin/minimap2_vecrec once per compile-time ablation variant (the VR_*
switches are compile-time, so each variant is a separate build), runs it on a
reads/reference set with the phase timers on, and tabulates per-call cost so you
can attribute where the extend primitive spends time.

Each variant reports (from the vecrec build's stderr):
  - us/call mean, p50, p95   (from [extend-timing])
  - init / dp / traceback us/call   (from [vecrec-phases])

Timing-only variants (NO_TRACEBACK / DUMMY_PROFILE / NO_HSTORE / NO_ENDPOINT_SCAN)
produce WRONG alignments; build_minimap2.sh auto-enables VR_UNSAFE so the run
still completes. They exist to attribute cost, not to check correctness.

Usage:
    uv run python end_to_end_baselines/scripts/run_ablation.py \
        [--ref REF.fa] [--reads READS.fq] [--reps 3] [--extra-opts "-O4,4 -E2,2"]

Requires: a working build toolchain (clang++, cc) and the same env as
build_minimap2.sh (e.g. LD_PRELOAD / ZLIB_DIR on some hosts).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
_BUILD = _REPO / "end_to_end_baselines" / "scripts" / "build_minimap2.sh"
_VECREC = _REPO / "bin" / "minimap2_vecrec"
_REF_DEFAULT = _REPO / "end_to_end_baselines" / "data" / "MT-human.fa"
_READS_DEFAULT = _REPO / "end_to_end_baselines" / "data" / "mt_reads.fq"

# name -> VECREC_EXTRA_DEFS (compile-time). "" is the correct full kernel.
VARIANTS = [
    ("full",            ""),
    ("force_2p",        "-DVR_FORCE_2P=1"),
    ("no_traceback",    "-DVR_NO_TRACEBACK=1"),
    ("dummy_profile",   "-DVR_DUMMY_PROFILE=1"),
    ("no_endpoint",     "-DVR_NO_ENDPOINT_SCAN=1"),
    ("no_hstore",       "-DVR_NO_HSTORE=1"),
    ("dp_only",         "-DVR_NO_TRACEBACK=1 -DVR_DUMMY_PROFILE=1 -DVR_NO_HSTORE=1 -DVR_NO_ENDPOINT_SCAN=1"),
]

_RE_TIMING = re.compile(r"mean=([\d.]+)\s+p50=([\d.]+)\s+p95=([\d.]+)")
_RE_PHASES = re.compile(r"init=([\d.]+) us\s+dp=([\d.]+) us\s+tb=([\d.]+) us")


def build(defs: str) -> None:
    env = dict(os.environ, VECREC_PHASE_TIMING="1", VECREC_EXTRA_DEFS=defs)
    r = subprocess.run(["bash", str(_BUILD)], env=env,
                       capture_output=True, text=True)
    if r.returncode != 0 or not _VECREC.exists():
        sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:])
        raise RuntimeError(f"build failed for defs={defs!r}")


def run(ref, reads, opts, reps):
    flags = ["-ax", "map-ont", "-t1", "--secondary=no", "-z", "1000000", *opts.split()]
    best = None
    for _ in range(reps):
        r = subprocess.run([str(_VECREC), *flags, str(ref), str(reads)],
                           capture_output=True, text=True)
        err = r.stderr
        mt = _RE_TIMING.search(err)
        mp = _RE_PHASES.search(err)
        if not mt:
            continue
        mean = float(mt.group(1))
        rec = {
            "mean": mean, "p50": float(mt.group(2)), "p95": float(mt.group(3)),
            "init": float(mp.group(1)) if mp else float("nan"),
            "dp":   float(mp.group(2)) if mp else float("nan"),
            "tb":   float(mp.group(3)) if mp else float("nan"),
        }
        if best is None or rec["mean"] < best["mean"]:
            best = rec
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default=str(_REF_DEFAULT))
    ap.add_argument("--reads", default=str(_READS_DEFAULT))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--extra-opts", default="",
                    help="extra minimap2 opts, e.g. '-O4,4 -E2,2' to force the 1-piece path")
    args = ap.parse_args()

    print(f"ref={args.ref}\nreads={args.reads}\nopts='map-ont -z1000000 {args.extra_opts}'  reps={args.reps}\n")
    hdr = f"{'variant':<16}{'us/call':>9}{'p50':>8}{'p95':>8}{'init':>8}{'dp':>8}{'tb':>8}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for name, defs in VARIANTS:
        build(defs)
        rec = run(args.ref, args.reads, args.extra_opts, args.reps)
        if rec is None:
            print(f"{name:<16}{'(no timing output)':>41}")
            continue
        rows.append((name, rec))
        print(f"{name:<16}{rec['mean']:>9.2f}{rec['p50']:>8.2f}{rec['p95']:>8.2f}"
              f"{rec['init']:>8.2f}{rec['dp']:>8.2f}{rec['tb']:>8.2f}")

    if rows:
        full = next((r for n, r in rows if n == "full"), rows[0][1])
        print("\ndelta vs full (us/call saved by removing the phase):")
        for name, rec in rows:
            if name == "full":
                continue
            print(f"  {name:<16}{full['mean'] - rec['mean']:+.2f}")
    print("\nNOTE: timing-only variants emit wrong alignments (VR_UNSAFE auto-set); "
          "use them only to attribute per-phase cost. Rebuild with no VECREC_EXTRA_DEFS "
          "to restore a correct binary.")


if __name__ == "__main__":
    main()
