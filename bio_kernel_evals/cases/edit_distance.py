"""Edit-distance (Levenshtein) global alignment.

A different metric from the scoring-matrix alignments: every substitution,
insertion, and deletion costs 1, and we minimize total cost (min-plus / TropMin
semiring).

vecrec kernel: an NW-shaped 2D recurrence in TropMin where ``s[i,j]`` is the
substitution *cost* (0 if equal else 1) and the gap cost is 1::

    D(i,j) = min( D(i-1,j-1)+s(i,j), D(i-1,j)+1, D(i,j-1)+1 )

Baseline: edlib (Myers bit-vector edit distance), validated by optimal-distance
equality against a textbook Levenshtein DP.
"""

from __future__ import annotations

import numpy as np

from vecrec.expr import (
    Impulse,
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

from ._alignment import (
    alignment_sweep_variants,
    edit_cost_inputs,
    levenshtein_reference,
    similar_edit_cost_inputs,
)

GAP = 1  # indel cost (positive; TropMin minimizes)


def _build_expr(element_type: ElementType = ElementType.I16) -> SignalExpr:
    with with_types(Type.TropMin, element_type):
        s = Var2D("s")
        posinf = Num(Type.TropMin.zero())  # +inf: min identity, no self-reference
        gap = Num(GAP)
        kernel: KernelExpr2D = TVKernel2D(
            [
                [posinf, gap],     # row 0: D(i, j-1) + 1
                [gap, s.at(0)],    # row 1: D(i-1, j) + 1  and  D(i-1, j-1) + s(i,j)
            ]
        )
        return Recurse2D(kernel, Impulse(0)).at(0)


class EditDistance(EvalCase):
    name = "edit-distance"
    vecrec_bytes_per_cell = 4  # I16 s input (2B) + I16 output (2B)
    report_gcups = False
    boundary_convention = (
        "vecrec: TropMin unit-cost; baselines: textbook Levenshtein / edlib"
    )

    def __init__(self, rows: int = 128, cols: int = 128, seed: int = 0) -> None:
        self.rows = rows
        self.cols = cols
        self.seed = seed
        self.n_elements = rows * cols

    @classmethod
    def sweep_variants(cls) -> list[EditDistance]:
        return alignment_sweep_variants(cls)

    def make_inputs(self) -> dict[str, object]:
        return edit_cost_inputs(self.rows, self.cols, seed=self.seed)

    def reference(self, **inputs: object) -> np.ndarray:
        return levenshtein_reference(str(inputs["x"]), str(inputs["y"])).flatten()

    def optimal_score(self, matrix: np.ndarray) -> float:
        return float(matrix[-1])  # D[n, m]: total edit distance

    def kernel_expr(self, element_type: ElementType = ElementType.I16) -> SignalExpr:
        return _build_expr(element_type)

    def baselines(self) -> list[Baseline]:
        return [
            Baseline(
                "numpy_loop",
                lambda **kw: levenshtein_reference(str(kw["x"]), str(kw["y"])).flatten(),
            )
        ]

    def cpp_baselines(self) -> list[CppBaseline]:
        from evals_common.vecrec_cpp import vecrec_traceback_baselines_multi_width

        return [
            *vecrec_traceback_baselines_multi_width(self.kernel_expr, kind="levenshtein"),
            CppBaseline("edlib", "edlib_editdist", "edit"),
            # Exact end-to-end Levenshtein via WFA2-lib (wavefront edit distance).
            CppBaseline("wfa2_edit", "wfa2_align", "edit"),
        ]


class EditDistanceSim65(EditDistance):
    """WFA2-only timing variant on ~65% identity pairs (substitution-mutated).

    Same Levenshtein reference as the parent; only ``wfa2_edit`` is wired so
    ``--sweep`` can show WFA2 when edit distance s is moderate (O(ns+s^2)).
    """

    name = "edit-distance-sim65"
    boundary_convention = (
        "WFA2-only; ~65% identity pairs (y = substitution-mutated copy of x)"
    )

    def make_inputs(self) -> dict[str, object]:
        return similar_edit_cost_inputs(self.rows, self.cols, seed=self.seed)

    def baselines(self) -> list[Baseline]:
        return []

    def cpp_baselines(self) -> list[CppBaseline]:
        return [CppBaseline("wfa2_edit", "wfa2_align", "edit")]

    @classmethod
    def sweep_variants(cls) -> list[EditDistanceSim65]:
        return alignment_sweep_variants(cls)


if __name__ == "__main__":
    from evals_common.harness import run_cases

    raise SystemExit(run_cases([EditDistance()]))
