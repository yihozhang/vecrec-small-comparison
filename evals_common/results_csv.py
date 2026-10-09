"""Serialize evaluation results to a tidy CSV (one row per implementation).

The harness produces a list of :class:`~evals_common.metrics.ImplResult` rows
per case (vecrec schedules + library baselines). This module flattens those into
a long/tidy CSV -- one line per ``(case, implementation)`` -- that captures the
timing and correctness fields needed to reconstruct a comparison table later.

``bio_kernel_evals/make_table.py`` consumes these CSVs (one per architecture) to
emit a cross-architecture LaTeX table. Keeping the writer here means the column
schema lives next to the :class:`ImplResult` it is derived from.
"""

from __future__ import annotations

import csv
from typing import Optional, Sequence, TextIO

from evals_common.metrics import ImplResult, effective_seconds

#: CSV column order. ``make_table.py`` reads by column *name* (DictReader), so
#: the order here is only for human readability of the file.
CSV_FIELDS = [
    "display_name",   # case label incl. size, e.g. "needleman-wunsch (1024x1024, seed=0)"
    "case",           # base case name, e.g. "needleman-wunsch"
    "rows",
    "cols",
    "seed",
    "elements",       # DP cells for this row (== rows*cols for full-matrix kernels)
    "impl",           # implementation label, e.g. "vecrec_nw_tb:k5" or "seqan3_nw"
    "kind",           # "vecrec" or "baseline"
    "work_mode",
    "median_us",      # median wall time in microseconds (blank if unavailable)
    "gcups",          # billions of cell updates/s (blank if the case reports wall time only)
    "speedup_vs_best",
    "correct",        # "True"/"False"/"" (unknown / not checked)
    "score",
    "note",
]


def result_kind(impl: str) -> str:
    """Classify an implementation label as a vecrec kernel or a baseline."""
    return "vecrec" if impl.startswith("vecrec") else "baseline"


def result_row(case, r: ImplResult) -> dict:
    """Build one tidy CSV row (a dict keyed by :data:`CSV_FIELDS`)."""
    params = case.sweep_params()
    seconds = effective_seconds(r)
    return {
        "display_name": case.display_name(),
        "case": case.name,
        "rows": params.get("rows", ""),
        "cols": params.get("cols", ""),
        "seed": params.get("seed", ""),
        "elements": r.cells,
        "impl": r.impl,
        "kind": result_kind(r.impl),
        "work_mode": r.work_mode,
        "median_us": f"{seconds * 1e6:.4f}" if seconds > 0 else "",
        "gcups": f"{r.gcups:.6f}" if r.gcups is not None else "",
        "speedup_vs_best": (
            f"{r.speedup_vs_best:.4f}" if r.speedup_vs_best is not None else ""
        ),
        "correct": "" if r.correct is None else str(bool(r.correct)),
        "score": "" if r.score is None else repr(float(r.score)),
        "note": r.note,
    }


class ResultsCsvWriter:
    """Incrementally append per-case result rows to a CSV file.

    Use as a context manager and pass :meth:`write_case` as the harness
    ``on_results`` sink::

        with ResultsCsvWriter("results.csv") as w:
            run_suite(cases, on_results=w.write_case)

    Rows are flushed after every case so partial results survive an interrupt.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._fh: Optional[TextIO] = None
        self._writer: Optional[csv.DictWriter] = None

    def __enter__(self) -> "ResultsCsvWriter":
        self._fh = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._fh.flush()
        return self

    def write_case(self, case, results: Sequence[ImplResult]) -> None:
        if self._writer is None or self._fh is None:
            raise RuntimeError("ResultsCsvWriter used outside its context manager")
        for r in results:
            self._writer.writerow(result_row(case, r))
        self._fh.flush()

    def __exit__(self, *exc: object) -> bool:
        if self._fh is not None:
            self._fh.close()
        self._fh = None
        self._writer = None
        return False
