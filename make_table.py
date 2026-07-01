#!/usr/bin/env python3
"""Generate a LaTeX table comparing throughput across architectures.

For each benchmark and architecture, report the throughput (in Melem/s) for
the serial, baseline, and opt variants, plus the speedup of opt over the best
(fastest) of serial and baseline.

Usage:
    python make_table.py FILE:ARCH [FILE:ARCH ...]

Example:
    python make_table.py \
        m2-results.csv:"Apple M2" \
        skylake-result.csv:"Skylake" \
        zen5-result.csv:"Zen 5"
"""

import argparse
import csv
import sys


def parse_source(spec):
    """Split a "path:arch name" spec into (path, arch)."""
    if ":" not in spec:
        sys.exit(f"error: expected FILE:ARCH, got {spec!r}")
    path, arch = spec.split(":", 1)
    if not path or not arch:
        sys.exit(f"error: both file and arch name required in {spec!r}")
    return path, arch


def load(path):
    """Return {benchmark_name: row_dict} for one results.csv file."""
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("display_name")
            if not name:
                continue
            rows[name] = row
    return rows


def throughput(elements, time_us):
    """Elements per microsecond == Melem/s (millions of elements per second)."""
    return elements / time_us


def latex_escape(text):
    return text.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def fmt(x):
    return f"{x:.1f}"


def main():
    ap = argparse.ArgumentParser(
        description="Generate a LaTeX throughput-comparison table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "sources",
        nargs="+",
        metavar="FILE:ARCH",
        help='results.csv file and its architecture name, e.g. m2-results.csv:"Apple M2"',
    )
    args = ap.parse_args()

    sources = [parse_source(s) for s in args.sources]
    data = [(arch, load(path)) for path, arch in sources]

    # Benchmark order taken from the first file; append any extras seen later.
    order = list(data[0][1].keys())
    seen = set(order)
    for _, rows in data[1:]:
        for name in rows:
            if name not in seen:
                seen.add(name)
                order.append(name)

    n_arch = len(data)
    per_arch_cols = 4  # serial, baseline, opt, speedup
    col_spec = "l" + "|" + "|".join(["r" * per_arch_cols] * n_arch)

    out = []
    out.append(r"\begin{table}[t]")
    out.append(r"\resizebox{\textwidth}{!}{%")
    out.append(r"\centering")
    out.append(r"\small")
    out.append(r"\begin{tabular}{" + col_spec + "}")
    out.append(r"\toprule")

    # Architecture header spanning its 4 columns.
    arch_header = ["Benchmark"]
    for arch, _ in data:
        arch_header.append(
            r"\multicolumn{%d}{c}{%s}" % (per_arch_cols, latex_escape(arch))
        )
    out.append(" & ".join(arch_header) + r" \\")

    # cmidrule under each architecture group.
    rules = []
    for i in range(n_arch):
        start = 2 + i * per_arch_cols
        end = start + per_arch_cols - 1
        rules.append(r"\cmidrule(lr){%d-%d}" % (start, end))
    out.append(" ".join(rules))

    # Sub-header: throughput columns + speedup.
    sub = [""]
    for _ in data:
        sub += ["Serial", "Base", "Opt", "Speedup"]
    out.append(" & ".join(sub) + r" \\")
    out.append(
        r" & "
        + " & ".join(
            [r"\multicolumn{3}{c}{(Melem/s)} & (\(\times\))"] * n_arch
        )
        + r" \\"
    )
    out.append(r"\midrule")

    for name in order:
        cells = [latex_escape(name)]
        for _, rows in data:
            row = rows.get(name)
            if row is None:
                cells += ["--"] * per_arch_cols
                continue
            elements = float(row["elements"])
            serial = float(row["serial_us"])
            baseline = float(row["baseline_us"])
            opt = float(row["opt_us"])
            best = min(serial, baseline)  # fastest -> smallest time
            speedup = best / opt
            cells += [
                fmt(throughput(elements, serial)),
                fmt(throughput(elements, baseline)),
                fmt(throughput(elements, opt)),
                f"{speedup:.2f}",
            ]
        out.append(" & ".join(cells) + r" \\")

    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(r"}")  # end resizebox
    out.append(
        r"\caption{Throughput (millions of elements per second) for the serial, "
        r"baseline, and optimized (opt) implementations across architectures. "
        r"Speedup is opt over the faster of serial and baseline.}"
    )
    out.append(r"\label{tab:throughput}")
    out.append(r"\end{table}")

    print("\n".join(out))


if __name__ == "__main__":
    main()
