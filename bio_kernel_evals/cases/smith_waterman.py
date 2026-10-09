"""Smith-Waterman local sequence alignment.

vecrec kernel: the 2D max-plus recurrence (mirrors
``vec-rec/benchmarks/trop/smith_waterman.py``) where every cell may "restart"
from 0 (local alignment)::

    M(i,j) = max( 0, M(i-1,j-1)+s(x_i,y_j), M(i-1,j)+g, M(i,j-1)+g )

Baselines: native C++ libraries (SeqAn3, ...) configured with a match/mismatch
scoring scheme and linear gap, validated by optimal-score equality against a
textbook Smith-Waterman DP. See ``_alignment.py`` for the sequence/score bridge.
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

from evals_common.harness import Baseline, EvalCase
from evals_common.parasail import parasail_baselines

from ._alignment import alignment_sweep_variants, sequence_inputs, sw_reference

# Linear gap penalty (per gap character). The C++ drivers use the equivalent
# affine encoding cost(L) = gap_open + L*gap_extend, so gap_open=0, gap_extend=1.
GAP = -1
_BOUND = np.int64(np.iinfo(np.int64).min // 8)


def _build_expr(element_type: ElementType = ElementType.I16) -> SignalExpr:
    with with_types(Type.TropMax, element_type):
        s = Var2D("s")
        neginf = Num(Type.TropMax.zero())
        gap = Num(GAP)
        kernel: KernelExpr2D = TVKernel2D(
            [
                [neginf, gap],     # row 0: M(i, j-1) + g
                [gap, s.at(0)],    # row 1: M(i-1, j) + g  and  M(i-1, j-1) + s(i,j)
            ]
        )
        # Num(0) supplies the TropMax "one" (= 0) at every column, so each cell
        # can restart from 0 -- local alignment.
        return Recurse2D(kernel, Num(0)).at(0)


class SmithWaterman(EvalCase):
    name = "smith-waterman"
    vecrec_bytes_per_cell = 4  # I16 s input (2B) + I16 output (2B)
    report_gcups = False
    # Local alignment: the kernel emits the M value matrix (restart-at-0) and the
    # C++ driver tracks the optimum driver-side -- sw_traceback_full scans the full
    # matrix for the global max (the local-alignment score + start cell), then
    # walks back to a restart cell. The global max is a full reduction that does
    # not fit vecrec's streaming model, so it lives in the driver, not the kernel.
    boundary_convention = (
        "vecrec: TropMax local (restart-at-0); baselines: textbook Smith-Waterman"
    )

    def __init__(self, rows: int = 128, cols: int = 128, seed: int = 0) -> None:
        self.rows = rows
        self.cols = cols
        self.seed = seed
        self.n_elements = rows * cols

    @classmethod
    def sweep_variants(cls) -> list[SmithWaterman]:
        return alignment_sweep_variants(cls)

    def make_inputs(self) -> dict[str, object]:
        # Linear gap: SeqAn3 cost(L) = gap_open + L*gap_extend, so a per-char
        # penalty of |GAP| is gap_open=0, gap_extend=-GAP.
        return sequence_inputs(
            self.rows, self.cols, gap_open=0, gap_extend=-GAP, seed=self.seed
        )

    def reference(self, **inputs: object) -> np.ndarray:
        return sw_reference(inputs["s"], GAP).flatten()

    def optimal_score(self, matrix: np.ndarray) -> float:
        return float(matrix.max())  # local optimum

    def kernel_expr(self, element_type: ElementType = ElementType.I16) -> SignalExpr:
        # Present so the harness skips the random-input run_vecrec path; SW does
        # not (yet) emit a vecrec row (see class docstring above).
        return _build_expr(element_type)

    def baselines(self) -> list[Baseline]:
        return [Baseline("numpy_loop", lambda **kw: sw_reference(kw["s"], GAP).flatten())]

    def cpp_baselines(self) -> list:
        # NOTE: SSW (bin/ssw_sw) is built but intentionally not wired here. It is
        # an affine-only local aligner (gap_open=0 is degenerate), so it cannot
        # reproduce this case's pure linear-gap Smith-Waterman.
        from evals_common.cpp import CppBaseline
        from evals_common.vecrec_cpp import vecrec_traceback_baselines_multi_width

        return [
            *vecrec_traceback_baselines_multi_width(self.kernel_expr, kind="sw"),
            CppBaseline("seqan3_sw", "seqan3_sw", "sw"),
            *parasail_baselines("sw"),
        ]


if __name__ == "__main__":
    from evals_common.harness import run_cases

    raise SystemExit(run_cases([SmithWaterman()]))
