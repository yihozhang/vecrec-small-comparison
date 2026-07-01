#!/usr/bin/env python3
"""Generate (if needed), build, and run benchmarks; aggregate results to a table + CSV.

Usage:
    uv run python scripts/run_all.py            # all registered benchmarks
    uv run python scripts/run_all.py <name> ... # a subset
    uv run python scripts/run_all.py --regen    # force-regenerate kernel headers
    uv run python scripts/run_all.py --iters 50 --warmup 5
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_ROOT, "scripts")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _SCRIPTS)

from specs.registry import SPECS  # noqa: E402
from gen_kernels import gen_one  # noqa: E402

GEN = os.path.join(_ROOT, "generated")
BUILD = os.path.join(_ROOT, "build")
RESULTS_CSV = os.path.join(_ROOT, "results.csv")

COLUMNS = ["name", "display_name", "elements", "serial_us", "baseline_us", "opt_us",
           "opt_vs_serial", "opt_vs_baseline", "baseline_vs_serial",
           "max_abs_baseline", "max_abs_opt", "status"]


def run_one(spec, regen: bool, extra_args: list[str]) -> dict | None:
    # Specs with no expression (e.g. seam-carving) have no VecRec kernel; build and run
    # their serial-only benchmark, which reports status UNSUPPORTED.
    if spec.build_expr is not None:
        header = os.path.join(GEN, spec.kernel_name + ".h")
        if regen or not os.path.exists(header):
            gen_one(spec.name)
    # Build (relative target so it matches the Makefile's pattern rules).
    subprocess.run(["make", f"build/{spec.kernel_name}"], cwd=_ROOT, check=True)
    # Run and capture the machine-readable RESULT line.
    proc = subprocess.run(
        [os.path.join(BUILD, spec.kernel_name), *extra_args],
        cwd=_ROOT, capture_output=True, text=True,
    )
    sys.stdout.write(proc.stdout)
    if proc.returncode not in (0, 1):  # 0=PASS, 1=FAIL are both valid runs
        sys.stderr.write(proc.stderr)
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT,"):
            f = line.split(",")
            vals = [spec.name, spec.display_name, int(f[2])] + [float(x) for x in f[3:11]] + [f[11]]
            return dict(zip(COLUMNS, vals))
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("names", nargs="*", help="benchmark name(s); default = all registered")
    ap.add_argument("--regen", action="store_true", help="force regeneration of kernel headers")
    ap.add_argument("--iters", type=int)
    ap.add_argument("--warmup", type=int)
    args = ap.parse_args()

    extra: list[str] = []
    if args.iters is not None:
        extra += ["--iters", str(args.iters)]
    if args.warmup is not None:
        extra += ["--warmup", str(args.warmup)]

    names = args.names or list(SPECS)
    rows = []
    for name in names:
        if name not in SPECS:
            raise SystemExit(f"unknown benchmark '{name}'. Known: {', '.join(sorted(SPECS))}")
        r = run_one(SPECS[name], args.regen, extra)
        if r is not None:
            rows.append(r)

    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    W = 96
    print("\n" + "=" * W)
    print(f"{'benchmark':22s} {'serial us':>10s} {'base us':>10s} {'opt us':>10s} "
          f"{'opt/ser':>8s} {'opt/base':>9s} {'status':>11s}")
    print("-" * W)
    for r in rows:
        print(f"{r['name']:22s} {r['serial_us']:10.1f} {r['baseline_us']:10.1f} {r['opt_us']:10.1f} "
              f"{r['opt_vs_serial']:7.2f}x {r['opt_vs_baseline']:8.2f}x {r['status']:>11s}")
    print("=" * W)
    print("opt = VecRec best schedule;  base = VecRec un-optimized schedule (codegen only);"
          "  ser = C++ serial loop")
    print(f"wrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
