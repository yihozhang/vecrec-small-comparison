"""Measure memory pressure (cache/LLC traffic) of tiled vs untiled VecRec drivers.

This tool uses `perf stat` hardware counters to measure whether column/2D tiling
reduces memory traffic, most importantly LLC-load-misses (≈ main-memory accesses).

It builds the fastest VecRec schedule for a recurrence (same selection as bench.py),
then runs the SAME binary under `perf stat` with different input.json tiling configs
so any counter delta is attributable to the traversal order, not the kernel.

Hybrid-CPU note: on Intel P/E-core parts, counters split into cpu_core/cpu_atom.
We pin to one P-core with taskset and read the cpu_core PMU.

Usage (from repo root, with clang18 on PATH):
    uv run python Reptile/perf_pressure.py --kinds nw,affine,sw --size 8192 --tile 512
    uv run python Reptile/perf_pressure.py --kinds nw --size 16384 --tile 1024 --cpu 0
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from Reptile import bench, reptile

# Counters that describe memory pressure. LLC-load-misses is the headline metric
# (last-level-cache misses ≈ DRAM accesses). We also report L1 misses and the
# generic cache-references/misses, plus instructions/cycles for context.
DEFAULT_EVENTS = [
    "instructions",
    "cycles",
    "cache-misses",
    "L1-dcache-load-misses",
    "LLC-loads",
    "LLC-load-misses",
    "LLC-stores",
    "LLC-store-misses",
]


def _run_perf(binary: Path, json_path: str, events: list[str], cpu: int) -> dict[str, float]:
    """Run one driver under perf stat pinned to a P-core; return {event: count}."""
    # paranoid=1 forbids per-CPU system-wide counting, so count the child process
    # (allowed) and pin it to a P-core with taskset (perf's child inherits affinity).
    cmd = [
        "taskset", "-c", str(cpu),
        "perf", "stat", "-x", ",",
        "-e", ",".join(events),
        str(binary), json_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    counts: dict[str, float] = {}
    for line in proc.stderr.splitlines():
        parts = line.split(",")
        if len(parts) < 3:
            continue
        raw, _unit, event = parts[0], parts[1], parts[2]
        # On hybrid parts perf emits cpu_core/<event>/ and cpu_atom/<event>/.
        base = event.split("/")[1] if "/" in event else event
        if base not in events:
            continue
        # Prefer the cpu_core reading; skip atom (≈0 when pinned to a P-core).
        if "cpu_atom" in event:
            continue
        try:
            counts[base] = float(raw)
        except ValueError:
            counts.setdefault(base, float("nan"))
    return counts


def _measure(kind: str, size: int, tile: int, cpu: int, warmup: int, iters: int,
             events: list[str]) -> None:
    print(f"\n=== memory pressure: {kind} {size}x{size} (tile {tile}, cpu {cpu}, "
          f"{iters} timed iters) ===")
    vr = bench.build_fastest_vecrec(kind, bench.SELECT_SIZE, warmup=2, iterations=6)
    if vr is None:
        print(f"  [{kind}] no VecRec schedule built; skipping")
        return
    vr_bin, label = vr
    inputs, expected = bench._reference_score(kind, size)

    approaches = [("untiled", 0, 0), ("row", tile, 0), ("col", 0, tile), ("block", tile, tile)]
    rows: list[tuple[str, dict[str, float]]] = []
    for name, tr, tc in approaches:
        jpath = bench._write_input(inputs, name, tr, tc, warmup, iters, verify=0)
        try:
            counts = _run_perf(vr_bin, jpath, events, cpu)
        finally:
            Path(jpath).unlink(missing_ok=True)
        rows.append((name, counts))

    base = dict(rows[0][1])
    # Headline table: absolute LLC-load-misses + ratio vs untiled, plus a few more.
    show = ["LLC-load-misses", "LLC-store-misses", "cache-misses", "L1-dcache-load-misses", "instructions"]
    hdr = f"  [{kind}] schedule {label}"
    print(hdr)
    print(f"    {'approach':<9} " + " ".join(f"{e.replace('-',' '):>22}" for e in show))
    for name, counts in rows:
        cells = []
        for e in show:
            v = counts.get(e, float("nan"))
            b = base.get(e, float("nan"))
            ratio = (v / b) if (b and b == b and v == v and b != 0) else float("nan")
            cells.append(f"{v:>12.3e} ({ratio:4.2f}x)" if v == v else f"{'n/a':>22}")
        print(f"    {name:<9} " + " ".join(cells))
    print("    (ratios are vs untiled; <1.00x = tiling reduced that counter)")


def main() -> int:
    ap = argparse.ArgumentParser(description="perf-stat memory pressure of tiled vs untiled VecRec")
    ap.add_argument("--kinds", default="nw,affine,sw")
    ap.add_argument("--size", type=int, default=8192)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--cpu", type=int, default=0, help="P-core to pin/count on")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--events", default=",".join(DEFAULT_EVENTS))
    args = ap.parse_args()

    events = [e.strip() for e in args.events.split(",") if e.strip()]
    for kind in [k.strip() for k in args.kinds.split(",") if k.strip()]:
        _measure(kind, args.size, args.tile, args.cpu, args.warmup, args.iters, events)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
