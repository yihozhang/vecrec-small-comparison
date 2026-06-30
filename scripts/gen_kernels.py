#!/usr/bin/env python3
"""Drive VecRec to find the best schedule for each benchmark and emit a kernel header.

Usage:
    uv run python scripts/gen_kernels.py <name> [<name> ...]
    uv run python scripts/gen_kernels.py --all
    uv run python scripts/gen_kernels.py --list

For each benchmark, calls vecrec.codegen.find_best_schedule_and_emit_header(), which
enumerates candidate schedules, times them, sweeps codegen parameters on the best
ones, and writes a self-contained header to generated/<kernel_name>.h. A sidecar
generated/<kernel_name>.meta.json records the chosen schedule's throughput and config.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from specs.registry import SPECS  # noqa: E402
from vecrec.codegen import find_best_schedule_and_emit_header  # noqa: E402

GENERATED_DIR = os.path.join(_ROOT, "generated")


def gen_one(name: str) -> dict | None:
    if name not in SPECS:
        raise SystemExit(f"unknown benchmark '{name}'. Known: {', '.join(sorted(SPECS))}")
    spec = SPECS[name]
    if spec.build_expr is None:
        print(f"[skip] {spec.name}: {spec.note or 'unsupported'}", flush=True)
        return None
    os.makedirs(GENERATED_DIR, exist_ok=True)
    header = os.path.join(GENERATED_DIR, spec.kernel_name + ".h")
    print(f"[gen] {spec.name}  ({spec.dim}, {spec.semiring})  ->  {header}", flush=True)
    result = find_best_schedule_and_emit_header(spec.build_expr(), header, spec.kernel_name)
    print(
        f"      best:     {result['best_throughput_mbs']:8.1f} MB/s | "
        f"epi_mult={result['best_ele_per_iter_multiple']} rpi={result['best_rows_per_iter']} | "
        f"ele_per_iter={result['ele_per_iter']} | {result['num_schedules']} schedules",
        flush=True,
    )
    print(
        f"      baseline: {result['baseline_throughput_mbs']:8.1f} MB/s | "
        f"epi_mult={result['baseline_ele_per_iter_multiple']} rpi={result['baseline_rows_per_iter']} | "
        f"ele_per_iter={result['baseline_ele_per_iter']}"
        + ("  (== best schedule)" if result['baseline_is_best'] else ""),
        flush=True,
    )
    meta = dict(result)
    meta["name"] = spec.name
    meta["dim"] = spec.dim
    meta["semiring"] = spec.semiring
    with open(os.path.join(GENERATED_DIR, spec.kernel_name + ".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("names", nargs="*", help="benchmark name(s) to generate")
    ap.add_argument("--all", action="store_true", help="generate all registered benchmarks")
    ap.add_argument("--list", action="store_true", help="list registered benchmarks and exit")
    args = ap.parse_args()

    if args.list:
        for name, spec in SPECS.items():
            print(f"{name:24s} -> {spec.kernel_name:24s} ({spec.dim}, {spec.semiring})")
        return

    names = list(SPECS) if args.all else args.names
    if not names:
        ap.error("provide benchmark name(s), --all, or --list")
    for name in names:
        gen_one(name)


if __name__ == "__main__":
    main()
