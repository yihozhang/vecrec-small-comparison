"""VecRec's 2-piece dual-affine drop-in kernel vs ksw2 SIMD extension (``ksw_extd2_sse``).

This is the SAME recurrence as the real minimap2 drop-in kernel
(``end_to_end_baselines/scripts/generate_kernel.py``'s
``_build_minimap2_dual_affine_expr``) -- a 3-state dual-affine (H, E1, E2
carried across rows; F1, F2 resolved in-row), built here with compile-time
map-ont gap constants instead of the drop-in's runtime-patched sentinels.
This file is the controlled, synthetic-data correctness/perf anchor for that
shared recurrence.

``baselines/ksw2/align.cpp`` calls ``ksw_extd2_sse`` with ``flag=0`` (i.e.
*without* ``KSW_EZ_EXTZ_ONLY``) and ``zdrop=-1`` (disabled) -- "gap-fill"
mode per ``ksw2_extd2_sse.c``'s own backtrack logic: it backtracks from the
corner ``(qlen, tlen)`` and ``ez.score`` IS the global corner H(qlen,tlen),
the same quantity a textbook 2-piece Gotoh DP computes. So unbanded is
directly correctness-checked against ``dual_affine_reference`` here, unlike
a genuine ``KSW_EZ_EXTZ_ONLY`` (free-end) call.

Banded has no dual-affine reference in ``_alignment.py`` (only single-piece
``banded_affine_reference`` exists), so ``reference()`` returns ``None`` and
the ksw2 banded score is not checked against anything -- this is a coverage
gap, not evidence the scores actually disagree; a banded 2-piece reference
would need to be added to make banded correctness-checkable here.

Unbanded (full matrix) and banded (``w``) variants are swept.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from vecrec.expr import (
    Convolve, Impulse, Ith, Num, PointwiseMul,
    Recurse, RepeaterVec, RVec2D, SAdd, SignalVector, SignalVectorExpr,
    TIKernel, Type, Var2D, with_types,
)
from vecrec.expr.base import SignalExpr
from vecrec.util import ElementType

from evals_common.cpp import CppBaseline
from evals_common.harness import Baseline, EvalCase

from ._alignment import SWEEP_SIZES, dual_affine_reference, sequence_inputs

# minimap2 map-ont gap parameters (positive penalty magnitudes)
Q  = 4;   E  = 2    # piece 1: short gaps
Q2 = 24;  E2 = 1    # piece 2: long gaps  (crossover at L=20)

BAND_WIDTH = 128


def _build_dual_affine_expr(element_type: ElementType = ElementType.I16) -> SignalExpr:
    """3-state dual-affine recurrence matching ksw_extd2_sse (map-ont params).

    States carried across rows: H (best score), E1 (gap-in-query piece 1),
                                                 E2 (gap-in-query piece 2).
    Within-row recurrences:     F1 (gap-in-target piece 1),
                                F2 (gap-in-target piece 2).
    """
    GO  = -Q;   GE  = -E    # piece 1 scores (negative = penalty)
    GO2 = -Q2;  GE2 = -E2   # piece 2 scores
    GO_GE   = GO  + GE      # = -6
    GO2_GE2 = GO2 + GE2     # = -25

    with with_types(Type.TropMax, element_type):
        s      = Var2D("s")
        neginf = Type.TropMax.zero()
        one    = Type.TropMax.one()
        z      = TIKernel.z()

        def body(prev: RVec2D) -> SignalVectorExpr:
            prev_h  = prev.at(0).at(0)
            prev_e1 = prev.at(1).at(0)
            prev_e2 = prev.at(2).at(0)
            s_row   = Ith(s, 0)

            E1 = SAdd(PointwiseMul(prev_h, Num(GO_GE)),   PointwiseMul(prev_e1, Num(GE)))
            E2 = SAdd(PointwiseMul(prev_h, Num(GO2_GE2)), PointwiseMul(prev_e2, Num(GE2)))

            diag = SAdd(
                PointwiseMul(Convolve(z, prev_h), s_row),
                PointwiseMul(Impulse(one), s_row),
            )
            Ht = SAdd(SAdd(diag, E1), E2)

            F1 = Recurse(TIKernel([neginf, GE]),  Convolve(TIKernel([neginf, GO_GE]),   Ht))
            F2 = Recurse(TIKernel([neginf, GE2]), Convolve(TIKernel([neginf, GO2_GE2]), Ht))

            H = SAdd(SAdd(Ht, F1), F2)
            return SignalVector([H, E1, E2])

        return Ith(RepeaterVec(3, body).at(0), 0)


class ExtensionDualAffine(EvalCase):
    name = "extension-dual-affine"
    vecrec_bytes_per_cell = 6  # I16 x 3 states
    report_gcups = False
    boundary_convention = (
        "vecrec: TropMax 3-state dual-affine (corner) -- the real minimap2 "
        "2-piece drop-in kernel; ksw2: ksw_extd2_sse SIMD 2-piece affine, called "
        "in gap-fill mode (flag=0) so its score is the same global corner"
    )

    def __init__(self, rows: int = 256, cols: int = 256, band_width: int = 0, seed: int = 0) -> None:
        self.rows = rows
        self.cols = cols
        self.band_width = band_width
        self.seed = seed
        self.n_elements = rows * cols

    def display_name(self) -> str:
        band = f" bw={self.band_width}" if self.band_width else " full"
        return f"{self.name} {self.rows}x{self.cols}{band}"

    @classmethod
    def sweep_variants(cls) -> list[ExtensionDualAffine]:
        variants: list[ExtensionDualAffine] = []
        for n in SWEEP_SIZES:
            variants.append(cls(rows=n, cols=n, band_width=0))
            variants.append(cls(rows=n, cols=n, band_width=BAND_WIDTH))
        return variants

    def make_inputs(self) -> dict[str, object]:
        inputs = sequence_inputs(
            self.rows, self.cols, gap_open=Q, gap_extend=E, seed=self.seed
        )
        inputs["gap_open2"] = Q2
        inputs["gap_extend2"] = E2
        return inputs

    def reference(self, **inputs: object) -> Optional[np.ndarray]:
        if self.band_width:
            return None  # no banded 2-piece reference; cross-schedule check only
        return dual_affine_reference(inputs["s"], -Q, -E, -Q2, -E2).flatten()

    def optimal_score(self, matrix: np.ndarray) -> float:
        return float(matrix[-1])  # global corner

    def kernel_expr(self, element_type: ElementType = ElementType.I16) -> SignalExpr:
        return _build_dual_affine_expr(element_type)

    def baselines(self) -> list[Baseline]:
        return []

    def cpp_baselines(self) -> list:
        from evals_common.vecrec_cpp import vecrec_traceback_baselines_multi_width

        bw = self.band_width
        ksw_bw = bw if bw else -1
        gap2 = {"band_width": ksw_bw, "gap_open2": Q2, "gap_extend2": E2}
        return [
            CppBaseline(
                "ksw2_extd2_sse", "ksw2_align", "extd2_sse",
                extra_payload=gap2,
            ),
            # Isolated approx-max A/B -- see extension_affine.py's identical
            # comment. This is the kernel minimap2's map-ont preset actually
            # dispatches to (q!=q2 or e!=e2), so this row is the closer
            # analog of the real e2e `approx-max` scenario than the 1-piece
            # case above, minus all pipeline overhead.
            CppBaseline(
                "ksw2_extd2_sse_approx", "ksw2_align", "extd2_sse",
                extra_payload={**gap2, "approx_max": 1},
            ),
            # I8/I16/I32 width sweep -- see extension_affine.py's identical comment.
            *vecrec_traceback_baselines_multi_width(
                self.kernel_expr, kind="affine", band_width=bw, variant="dual"
            ),
        ]


if __name__ == "__main__":
    from evals_common.harness import run_cases

    raise SystemExit(run_cases([ExtensionDualAffine()]))
