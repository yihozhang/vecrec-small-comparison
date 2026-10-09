#!/usr/bin/env python3
"""Generate a LaTeX table comparing throughput across architectures.

Consumes the tidy result CSVs written by ``run.py --csv`` (one per architecture).
Each CSV has one row per implementation. For every benchmark this script keeps
*all* library baselines plus one representative vecrec kernel:

    baselines -- every non-vecrec library implementation (one row each)
    opt       -- the fastest vecrec schedule (one row, labelled "VecRec")

The table lists each baseline's throughput (Melem/s) and, on the VecRec row,
its throughput plus the speedup of opt over the fastest *correct* library
baseline. The scalar vecrec schedule (``...:k0``) is intentionally not shown.

Usage:
    python -m bio_kernel_evals.make_table FILE:ARCH [FILE:ARCH ...]

Example:
    uv run python -m bio_kernel_evals.make_table \
        m2-results.csv:"Apple M2" \
        skylake-result.csv:"Skylake" \
        zen5-result.csv:"Zen 5"
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from typing import Optional


def parse_source(spec: str) -> tuple[str, str]:
    """Split a "path:arch name" spec into (path, arch)."""
    if ":" not in spec:
        sys.exit(f"error: expected FILE:ARCH, got {spec!r}")
    path, arch = spec.split(":", 1)
    if not path or not arch:
        sys.exit(f"error: both file and arch name required in {spec!r}")
    return path, arch


@dataclass
class Variant:
    """One implementation's timing for a single benchmark."""

    impl: str
    us: float          # median wall time (microseconds)
    throughput: float  # elements / us == Melem/s


@dataclass
class Benchmark:
    """All library baselines plus the fastest vecrec kernel for one benchmark."""

    baselines: dict[str, Variant] = field(default_factory=dict)  # impl -> Variant
    opt: Optional[Variant] = None            # fastest vecrec schedule
    best_baseline: Optional[Variant] = None  # fastest *correct* library baseline

    @property
    def speedup(self) -> Optional[float]:
        """Opt over the fastest correct library baseline (smaller wall time)."""
        if self.opt is None or self.opt.us <= 0 or self.best_baseline is None:
            return None
        return self.best_baseline.us / self.opt.us


def _float(value: object) -> Optional[float]:
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f


def _reduce(rows: list[dict]) -> Optional[Benchmark]:
    """Collect every baseline and the fastest vecrec kernel for one benchmark."""
    usable = []
    for row in rows:
        us = _float(row.get("median_us"))
        elements = _float(row.get("elements"))
        if us is None or us <= 0 or elements is None or elements <= 0:
            continue  # unbuilt / errored / untimed row
        usable.append((row, us, elements))
    if not usable:
        return None

    def variant(entry) -> Variant:
        row, us, elements = entry
        return Variant(impl=row["impl"], us=us, throughput=elements / us)

    vecrec = [e for e in usable if e[0].get("kind") == "vecrec"]
    baselines = [e for e in usable if e[0].get("kind") != "vecrec"]

    bench = Benchmark()
    if vecrec:
        bench.opt = variant(min(vecrec, key=lambda e: e[1]))  # fastest schedule
    for e in baselines:
        v = variant(e)
        bench.baselines[v.impl] = v
    if baselines:
        # Speedup denominator: fastest *correct* baseline (fall back to all).
        correct = [e for e in baselines if e[0].get("correct") != "False"]
        pool = correct or baselines
        bench.best_baseline = variant(min(pool, key=lambda e: e[1]))
    return bench


def load(path: str) -> tuple[list[str], dict[str, Benchmark]]:
    """Return (benchmark order, {display_name: Benchmark}) for one CSV."""
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("display_name")
            if not name:
                continue
            if name not in groups:
                groups[name] = []
                order.append(name)
            groups[name].append(row)

    reduced: dict[str, Benchmark] = {}
    kept: list[str] = []
    for name in order:
        bench = _reduce(groups[name])
        if bench is not None:
            reduced[name] = bench
            kept.append(name)
    return kept, reduced


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def fmt(x: Optional[float]) -> str:
    return "--" if x is None else f"{x:.1f}"


def fmt_speedup(x: Optional[float]) -> str:
    return "--" if x is None else f"{x:.2f}"


def opt_label(bench: Optional[Benchmark]) -> str:
    """"VecRec (k5)" from the opt kernel's schedule suffix, else "VecRec"."""
    if bench is None or bench.opt is None:
        return "VecRec"
    _, _, sched = bench.opt.impl.partition(":")
    return f"VecRec ({sched})" if sched else "VecRec"


def baseline_order(benchmarks_by_arch: list[Benchmark], name: str) -> list[str]:
    """Union of baseline impls for a benchmark, fastest (highest Melem/s) first.

    Throughput for ranking is the best value seen for that impl across archs so
    the row order is stable even when an arch is missing a baseline.
    """
    best: dict[str, float] = {}
    for bench in benchmarks_by_arch:
        if bench is None:
            continue
        for impl, v in bench.baselines.items():
            best[impl] = max(best.get(impl, 0.0), v.throughput)
    return sorted(best, key=lambda impl: (-best[impl], impl))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a LaTeX throughput-comparison table from run.py CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "sources",
        nargs="+",
        metavar="FILE:ARCH",
        help='results CSV and its architecture name, e.g. m2-results.csv:"Apple M2"',
    )
    args = ap.parse_args()

    sources = [parse_source(s) for s in args.sources]
    data = [(arch, *load(path)) for path, arch in sources]  # (arch, order, benchmarks)

    # Benchmark order taken from the first file; append any extras seen later.
    order = list(data[0][1])
    seen = set(order)
    for _, arch_order, _ in data[1:]:
        for name in arch_order:
            if name not in seen:
                seen.add(name)
                order.append(name)

    n_arch = len(data)
    per_arch_cols = 2  # Melem/s, speedup
    col_spec = "ll" + "|" + "|".join(["r" * per_arch_cols] * n_arch)

    out: list[str] = []
    out.append(r"\begin{table}[t]")
    out.append(r"\centering")
    out.append(r"\resizebox{\textwidth}{!}{%")
    out.append(r"\small")
    out.append(r"\begin{tabular}{" + col_spec + "}")
    out.append(r"\toprule")

    # Architecture header spanning each 2-column group.
    arch_header = ["", ""]
    for arch, _, _ in data:
        arch_header.append(
            r"\multicolumn{%d}{c}{%s}" % (per_arch_cols, latex_escape(arch))
        )
    out.append(" & ".join(arch_header) + r" \\")

    # cmidrule under each architecture group.
    rules = []
    for i in range(n_arch):
        start = 3 + i * per_arch_cols
        end = start + per_arch_cols - 1
        rules.append(r"\cmidrule(lr){%d-%d}" % (start, end))
    out.append(" ".join(rules))

    # Sub-header: throughput + speedup per arch.
    sub = ["Benchmark", "Implementation"]
    for _ in data:
        sub += [r"Melem/s", r"\(\times\)"]
    out.append(" & ".join(sub) + r" \\")
    out.append(r"\midrule")

    for gi, name in enumerate(order):
        benches = [benchmarks.get(name) for _, _, benchmarks in data]
        impls = baseline_order(benches, name)

        if gi > 0:
            out.append(r"\midrule")

        first = True

        def emit(label: str, per_arch_cells: list[str], bold: bool = False) -> None:
            nonlocal first
            head = latex_escape(name) if first else ""
            first = False
            impl_cell = (r"\textbf{%s}" % label) if bold else label
            out.append(" & ".join([head, impl_cell, *per_arch_cells]) + r" \\")

        for impl in impls:
            cells: list[str] = []
            for bench in benches:
                v = bench.baselines.get(impl) if bench is not None else None
                cells += [fmt(v.throughput if v else None), "--"]
            emit(latex_escape(impl), cells)

        # VecRec (opt) row: throughput + speedup vs the fastest correct baseline.
        label = next(
            (opt_label(b) for b in benches if b is not None and b.opt is not None),
            "VecRec",
        )
        cells = []
        for bench in benches:
            opt = bench.opt if bench is not None else None
            cells += [
                fmt(opt.throughput if opt else None),
                fmt_speedup(bench.speedup if bench is not None else None),
            ]
        emit(label, cells, bold=True)

    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"}")  # end resizebox
    out.append(
        r"\caption{Throughput (millions of elements per second) for every library "
        r"baseline and the fastest vecrec schedule (\textbf{VecRec}) across "
        r"architectures. Speedup (\(\times\)) is VecRec over the fastest correct "
        r"library baseline.}"
    )
    out.append(r"\label{tab:throughput}")
    out.append(r"\end{table}")

    print("\n".join(out))


if __name__ == "__main__":
    main()
