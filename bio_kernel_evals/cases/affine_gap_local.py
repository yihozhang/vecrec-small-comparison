"""Affine-gap LOCAL (Smith-Waterman + Gotoh) sequence alignment -- SCORE-ONLY.

VecRec kernel: collapsed 2-state affine (H + E; F resolved in-row via
Recurse), same shape as ``affine_gap.py``'s global recurrence but with a
local restart floor at 0 (``Ht = max(diag, E, 0)``), matching Smith-Waterman
+ affine gap costs. This recurrence doesn't exist anywhere else in the suite:
``smith_waterman.py`` is local with a *linear* gap, and ``affine_gap.py`` is
*global* with an affine gap -- this is the missing local+affine combination.

Baseline: ``ksw_ll_i16`` -- ksw2's vectorized Farrar-striped local aligner
(SSE int16). Both compute the local (Smith-Waterman + affine) optimum, so
scores are correctness-checked against a textbook affine-local DP.
``ksw_ll_i16`` is score-only (no traceback) and not bandable (full striped
matrix), so the VecRec side uses the score-only driver mode (fused in-sweep
max reduce, O(row) memory, no traceback) and this case is unbanded only.
"""

from __future__ import annotations

import numpy as np

from vecrec.expr import (
    Convolve, Impulse, Ith, Num, PointwiseMul, Recurse, RepeaterVec, RVec2D,
    SAdd, SignalVector, SignalVectorExpr, TIKernel, Type, Var2D, with_types,
)
from vecrec.expr.base import SignalExpr
from vecrec.util import ElementType

from evals_common.cpp import CppBaseline
from evals_common.harness import Baseline, EvalCase

from ._alignment import SWEEP_SIZES, affine_sw_reference, sequence_inputs

# Affine gap constants (negative scores). gap(L) = GO + L*GE.
GO = -4
GE = -2
GO_GE = GO + GE


def _build_expr() -> SignalExpr:
    """Collapsed 2-state affine LOCAL recurrence (restart-at-0), I16.

        E(i,j)  = max( H(i-1,j) + GO+GE,  E(i-1,j) + GE )
        Ht(i,j) = max( H(i-1,j-1) + s(i,j),  E(i,j),  0 )   <- local restart
        F(i,j)  = max( Ht(i,j-1) + GO+GE,  F(i,j-1) + GE )
        H(i,j)  = max( Ht(i,j),  F(i,j) )

    The local optimum is the global max over H (fused in-sweep reduce).
    """
    with with_types(Type.TropMax, ElementType.I16):
        s = Var2D("s")
        neginf = Type.TropMax.zero()
        one = Type.TropMax.one()
        z = TIKernel.z()

        def body(prev: RVec2D) -> SignalVectorExpr:
            prev_h = prev.at(0).at(0)  # H(i-1, j)
            prev_e = prev.at(1).at(0)  # E(i-1, j)
            s_row = s.at(0)
            E = SAdd(PointwiseMul(prev_h, Num(GO_GE)), PointwiseMul(prev_e, Num(GE)))
            diag = SAdd(
                PointwiseMul(Convolve(z, prev_h), s_row),
                PointwiseMul(Impulse(one), s_row),
            )
            # max(diag, E, 0): the Num(0) is the local restart floor.
            Ht = SAdd(SAdd(diag, E), Num(0))
            F = Recurse(TIKernel([neginf, GE]), Convolve(TIKernel([neginf, GO_GE]), Ht))
            H = SAdd(Ht, F)
            return SignalVector([H, E])

        return Ith(RepeaterVec(2, body).at(0), 0)


class AffineGapLocal(EvalCase):
    name = "affine-gap-local"
    vecrec_bytes_per_cell = 4
    report_gcups = False
    boundary_convention = (
        "vecrec: TropMax affine-local (restart-at-0), score-only (O(row), no traceback); "
        "ksw2: ksw_ll_i16 SIMD Farrar striped local affine"
    )

    def __init__(self, rows: int = 256, cols: int = 256, seed: int = 0) -> None:
        self.rows = rows
        self.cols = cols
        self.band_width = 0  # ksw_ll_i16 is not bandable (full striped matrix)
        self.seed = seed
        self.n_elements = rows * cols

    def display_name(self) -> str:
        return f"{self.name} {self.rows}x{self.cols}"

    @classmethod
    def sweep_variants(cls) -> list[AffineGapLocal]:
        return [cls(rows=n, cols=n) for n in SWEEP_SIZES]

    def make_inputs(self) -> dict[str, object]:
        return sequence_inputs(
            self.rows, self.cols, gap_open=-GO, gap_extend=-GE, seed=self.seed
        )

    def reference(self, **inputs: object) -> np.ndarray:
        return affine_sw_reference(inputs["s"], GO, GE).flatten()

    def optimal_score(self, matrix: np.ndarray) -> float:
        return float(matrix.max())  # local optimum

    def kernel_expr(self) -> SignalExpr:
        return _build_expr()

    def baselines(self) -> list[Baseline]:
        return []

    def cpp_baselines(self) -> list:
        from evals_common.vecrec_cpp import vecrec_traceback_baselines

        return [
            CppBaseline("ksw2_ll_i16", "ksw2_align", "ll_i16"),
            *vecrec_traceback_baselines(self.kernel_expr(), kind="affine_sw", score_only=True),
        ]


if __name__ == "__main__":
    from evals_common.harness import run_cases

    raise SystemExit(run_cases([AffineGapLocal()]))
