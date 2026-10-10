"""Tables 1 and 2: VecRec kernels vs serial C++ loops (groups ``wild`` / ``leetcode``).

Pipeline per benchmark (same as the original ``scripts/run_all.py``):
  1. VecRec schedule search -> self-contained header ``generated/<kernel>.h``
     (best schedule + un-optimized baseline schedule), via
     ``vecrec.codegen.find_best_schedule_and_emit_header``.
  2. ``make build/<kernel>`` compiles ``benchmarks/<kernel>.cpp`` (serial loop +
     harness) against that header.
  3. Run the binary; it prints a machine-readable ``RESULT,...`` line with the
     best-of-N times of serial / baseline / optimized and the error vs serial.

Results go to ``results/<group>/<host>.csv`` (one row per benchmark).
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from ae.layout import MICRO_GROUPS
from ae.paths import BUILD, GENERATED, ROOT, merge_csv, result_file

COLUMNS = ["name", "display_name", "elements", "serial_us", "baseline_us", "opt_us",
           "opt_vs_serial", "opt_vs_baseline", "baseline_vs_serial",
           "max_abs_baseline", "max_abs_opt", "status"]


def gen_kernel(spec, regen: bool = False) -> Path:
    """Search the best schedule for ``spec`` and emit ``generated/<kernel>.h``."""
    from vecrec.codegen import find_best_schedule_and_emit_header
    from vecrec.expr import pp

    GENERATED.mkdir(exist_ok=True)
    header = GENERATED / f"{spec.kernel_name}.h"
    if header.exists() and not regen:
        print(f"[gen] {spec.name}: reusing {header.relative_to(ROOT)}", flush=True)
        return header
    print(f"[gen] {spec.name} ({spec.dim}, {spec.semiring}) -> {header.relative_to(ROOT)}", flush=True)
    result = find_best_schedule_and_emit_header(spec.build_expr(), str(header), spec.kernel_name)
    print(
        f"      best:     {result['best_throughput_mbs']:8.1f} MB/s | "
        f"epi_mult={result['best_ele_per_iter_multiple']} rpi={result['best_rows_per_iter']} | "
        f"{result['num_schedules']} schedules\n"
        f"      baseline: {result['baseline_throughput_mbs']:8.1f} MB/s | "
        f"epi_mult={result['baseline_ele_per_iter_multiple']} rpi={result['baseline_rows_per_iter']}",
        flush=True,
    )
    meta = dict(result)
    meta["best_schedule"] = pp(meta.pop("best_expr"))
    meta["baseline_schedule"] = pp(meta.pop("baseline_expr"))
    meta.update(name=spec.name, dim=spec.dim, semiring=spec.semiring)
    (GENERATED / f"{spec.kernel_name}.meta.json").write_text(json.dumps(meta, indent=2))
    return header


def build_kernel(spec) -> Path:
    subprocess.run(["make", f"build/{spec.kernel_name}"], cwd=ROOT, check=True)
    return BUILD / spec.kernel_name


def run_kernel(spec, extra_args: list[str]) -> dict | None:
    """Run one benchmark binary and parse its ``RESULT,...`` line."""
    proc = subprocess.run([str(BUILD / spec.kernel_name), *extra_args],
                          cwd=ROOT, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode not in (0, 1):  # 0 = PASS, 1 = correctness FAIL; both report
        sys.stderr.write(proc.stderr)
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT,"):
            f = line.split(",")
            vals = [spec.name, spec.display_name, int(f[2])] + [float(x) for x in f[3:11]] + [f[11]]
            return dict(zip(COLUMNS, vals))
    return None


def exec_group(group: str, host: str, *, regen: bool = False, iters: int | None = None,
               warmup: int | None = None, only: list[str] | None = None) -> Path:
    from specs.registry import SPECS

    extra: list[str] = []
    if iters is not None:
        extra += ["--iters", str(iters)]
    if warmup is not None:
        extra += ["--warmup", str(warmup)]

    out = result_file(group, host)
    rows: list[dict] = []
    for _label, name in MICRO_GROUPS[group]:
        if only and name not in only:
            continue
        spec = SPECS[name]
        gen_kernel(spec, regen=regen)
        build_kernel(spec)
        r = run_kernel(spec, extra)
        if r is None:
            print(f"[warn] {name}: no RESULT line (run failed)", file=sys.stderr)
            continue
        rows.append(r)
        # Flush after every benchmark so partial results survive an interrupt;
        # rows of benchmarks not selected by --only are kept.
        merge_csv(out, rows, COLUMNS, key="name")

    print(f"\n{'benchmark':40s} {'serial us':>10s} {'base us':>10s} {'opt us':>10s} "
          f"{'opt/ser':>8s} {'opt/base':>9s} {'status':>7s}")
    for r in rows:
        print(f"{r['name']:40s} {r['serial_us']:10.1f} {r['baseline_us']:10.1f} {r['opt_us']:10.1f} "
              f"{r['opt_vs_serial']:7.2f}x {r['opt_vs_baseline']:8.2f}x {r['status']:>7s}")
    print(f"wrote {out}")
    return out
