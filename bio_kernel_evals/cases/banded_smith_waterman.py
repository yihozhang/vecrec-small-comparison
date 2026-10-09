"""Banded Smith-Waterman local sequence alignment.

vecrec kernel: the same TropMax 2D recurrence as SmithWaterman, compiled
with ``band_width > 0`` so that each row's column sweep is restricted to
the symmetric diagonal band
    ``[max(0, row - band_width), min(cols, row + band_width + 1))``
aligned to ``epi`` (the element-per-iteration count of the schedule).

A warm-up pass runs ``kernel.run()`` from col=0 to ``col_start`` without
writing to the output array, priming ProfileSignal2D row pointers, the
RepeaterVec ring buffer, and SRecurse carry so the first in-band column
receives correct predecessor values.

For SW (local alignment) the TropMax floor is 0, so cells outside the band
are initialised to 0 rather than -inf.  The warm-up correctly propagates
the true out-of-band values, which are typically small or zero for sequences
whose optimal path stays near the diagonal.  Scores match SeqAn3 banded SW
for near-diagonal alignments; may differ slightly near the band boundary.
"""

from __future__ import annotations

import numpy as np

from vecrec.expr import (
    Num,
    Recurse2D,
    TVKernel2D,
    Type,
    Var2D,
    with_types,
)
from vecrec.expr.base import KernelExpr2D, SignalExpr
from vecrec.util import ElementType

from evals_common.cpp import CppBaseline
from evals_common.harness import Baseline, EvalCase

from ._alignment import banded_sw_reference, sequence_inputs

GAP = -1
BAND_WIDTH = 128  # default diagonal half-width


def _build_expr(element_type: ElementType = ElementType.I16) -> SignalExpr:
    with with_types(Type.TropMax, element_type):
        s = Var2D("s")
        neginf = Num(Type.TropMax.zero())
        gap = Num(GAP)
        kernel: KernelExpr2D = TVKernel2D(
            [
                [neginf, gap],   # M(i, j-1) + g
                [gap, s.at(0)],  # M(i-1, j) + g  and  M(i-1, j-1) + s(i,j)
            ]
        )
        return Recurse2D(kernel, Num(0)).at(0)


class BandedSmithWaterman(EvalCase):
    name = "banded-smith-waterman"
    vecrec_bytes_per_cell = 4
    report_gcups = False
    boundary_convention = (
        "vecrec: TropMax banded local; reference: full-matrix Smith-Waterman"
    )

    def __init__(
        self,
        rows: int = 128,
        cols: int = 128,
        band_width: int = BAND_WIDTH,
        seed: int = 0,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.band_width = band_width
        self.seed = seed
        self.n_elements = rows * cols

    def display_name(self) -> str:
        return f"{self.name} {self.rows}x{self.cols} bw={self.band_width}"

    @classmethod
    def sweep_variants(cls) -> list[BandedSmithWaterman]:
        return [cls(rows=n, cols=n, band_width=BAND_WIDTH) for n in [128, 256, 512, 1024]]

    def make_inputs(self) -> dict[str, object]:
        return sequence_inputs(
            self.rows, self.cols, gap_open=0, gap_extend=-GAP, seed=self.seed
        )

    def reference(self, **inputs: object) -> np.ndarray:
        return banded_sw_reference(inputs["s"], GAP, self.band_width).flatten()

    def optimal_score(self, matrix: np.ndarray) -> float:
        return float(matrix.max())

    def kernel_expr(self, element_type: ElementType = ElementType.I16) -> SignalExpr:
        return _build_expr(element_type)

    def baselines(self) -> list[Baseline]:
        return [
            Baseline(
                "numpy_loop",
                lambda **kw: banded_sw_reference(
                    kw["s"], GAP, self.band_width
                ).flatten(),
            )
        ]

    def cpp_baselines(self) -> list:
        from evals_common.vecrec_cpp import vecrec_traceback_baselines_multi_width

        return [
            *vecrec_traceback_baselines_multi_width(
                self.kernel_expr, kind="sw", band_width=self.band_width
            ),
            # SeqAn3 banded SW: directly comparable symmetric-band baseline.
            CppBaseline(
                "seqan3_sw_banded",
                "seqan3_sw",
                "sw_banded",
                extra_payload={"band_width": self.band_width},
            ),
            # SeqAn3 full-matrix SW: correctness reference (not a banded comparison).
            CppBaseline("seqan3_sw_fullmat", "seqan3_sw", "sw"),
        ]


if __name__ == "__main__":
    from evals_common.harness import run_cases

    raise SystemExit(run_cases([BandedSmithWaterman()]))
