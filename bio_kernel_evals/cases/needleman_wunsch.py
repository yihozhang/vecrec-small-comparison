"""Needleman-Wunsch global sequence alignment.

vecrec kernel: the 2D max-plus recurrence (mirrors
``vec-rec/benchmarks/trop/needleman_wunsch.py``)::

    M(i,j) = max( M(i-1,j-1)+s(x_i,y_j), M(i-1,j)+g, M(i,j-1)+g )

Baselines: native C++ libraries configured with a match/mismatch scoring scheme
and linear gap, validated by optimal-score equality (bottom-right cell) against a
textbook global DP. See ``_alignment.py`` for the sequence/score bridge.
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

from evals_common.harness import Baseline, EvalCase
from evals_common.parasail import parasail_baselines

from ._alignment import MATCH, MISMATCH, alignment_sweep_variants, nw_reference, sequence_inputs, similar_sequence_inputs

GAP = -1
# SConvolveOp / SRecurseOp for TropMax use INT64_MIN/8 as the out-of-bounds sentinel.
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
        return Recurse2D(kernel, Impulse(0)).at(0)


class NeedlemanWunsch(EvalCase):
    name = "needleman-wunsch"
    vecrec_bytes_per_cell = 4  # I16 s input (2B) + I16 output (2B)
    report_gcups = False
    boundary_convention = (
        "vecrec: TropMax global; baselines: textbook Needleman-Wunsch"
    )

    def alignment_score(self, query: str, ref: str) -> float:
        """Linear-gap score: match/mismatch per column, GAP per gap column."""
        score = 0
        for a, b in zip(query, ref):
            if a == "-" or b == "-":
                score += GAP
            else:
                score += MATCH if a == b else MISMATCH
        return float(score)

    def __init__(self, rows: int = 128, cols: int = 128, seed: int = 0) -> None:
        self.rows = rows
        self.cols = cols
        self.seed = seed
        self.n_elements = rows * cols

    @classmethod
    def sweep_variants(cls) -> list[NeedlemanWunsch]:
        return alignment_sweep_variants(cls)

    def make_inputs(self) -> dict[str, object]:
        # Linear gap encoded as affine: gap_open=0, gap_extend=-GAP.
        return sequence_inputs(
            self.rows, self.cols, gap_open=0, gap_extend=-GAP, seed=self.seed
        )

    def reference(self, **inputs: object) -> np.ndarray:
        return nw_reference(inputs["s"], GAP).flatten()

    def optimal_score(self, matrix: np.ndarray) -> float:
        return float(matrix[-1])  # global optimum: bottom-right cell

    def kernel_expr(self, element_type: ElementType = ElementType.I16) -> SignalExpr:
        return _build_expr(element_type)

    def baselines(self) -> list[Baseline]:
        return [Baseline("numpy_loop", lambda **kw: nw_reference(kw["s"], GAP).flatten())]

    def cpp_baselines(self) -> list:
        from evals_common.cpp import CppBaseline
        from evals_common.vecrec_cpp import (
            vecrec_traceback_baselines,
            vecrec_traceback_baselines_multi_width,
        )

        return [
            # Sweeps I8/I16/I32 element widths x every SIMD schedule for each,
            # so the fastest (width, schedule) pair wins -- not just the fastest
            # schedule at a fixed I16. I8 is expected to fail correctness once
            # scores exceed its saturating range at larger N; see the helper's
            # docstring.
            *vecrec_traceback_baselines_multi_width(self.kernel_expr, kind="nw"),
            CppBaseline("seqan3_nw", "seqan3_nw", "nw"),
            *parasail_baselines("nw"),
            # Exact end-to-end gap-linear NW via WFA2-lib (wavefront). Penalties
            # are mapped to WFA2's distance form in the driver; cigar->score is
            # already the similarity corner score (match < 0 path).
            CppBaseline("wfa2_nw", "wfa2_align", "nw"),
            # SIMD 1-piece affine GLOBAL (native SSE2 on x86, sse2neon on ARM).
            # gap_open=0 degenerates its affine gap(L) = gapo + L*gape into pure
            # linear gap, matching nw_reference exactly.
            CppBaseline("ksw2_gg2_sse", "ksw2_align", "gg2_sse"),
            # SIMD linear-gap X-drop extension kernel (score-only, no CIGAR).
            # Called unbanded with xdrop disabled, it reports the global-corner
            # linear score -- matches nw_reference exactly, so it's a real
            # correctness anchor, not just a timing comparison. Paired with a
            # score-only VecRec driver (O(row), no traceback) so memory
            # footprint and timed span match this baseline's no-CIGAR kernel.
            # NOTE(artifact): both are disabled -- the paper compares full-matrix
            # + traceback implementations only, and a score-only kernel is not a
            # like-for-like baseline for VecRec's traceback-timed kernels.
            # CppBaseline("ksw2_extf2_sse", "ksw2_align", "extf2_sse", extra_payload={"band_width": -1}),
            # *vecrec_traceback_baselines(self.kernel_expr(), kind="nw", score_only=True),
        ]


class NeedlemanWunschSim65(NeedlemanWunsch):
    """WFA2-only timing variant on ~65% identity pairs (substitution-mutated).

    The parent case uses independent random DNA (~25% identity). This subclass
    keeps the same NW recurrence/reference but feeds a similar pair and wires
    only ``wfa2_nw``, so ``--sweep`` can show WFA2's O(ns+s^2) regime without
    re-running every other library.
    """

    name = "needleman-wunsch-sim65"
    identity = 0.65
    boundary_convention = (
        "WFA2-only; ~65% identity pairs (y = substitution-mutated copy of x)"
    )

    def make_inputs(self) -> dict[str, object]:
        return similar_sequence_inputs(
            self.rows, self.cols, gap_open=0, gap_extend=-GAP, seed=self.seed,
            identity=self.identity,
        )

    def baselines(self) -> list[Baseline]:
        return []

    def cpp_baselines(self) -> list:
        from evals_common.cpp import CppBaseline

        return [CppBaseline("wfa2_nw", "wfa2_align", "nw")]

    @classmethod
    def sweep_variants(cls) -> list[NeedlemanWunschSim65]:
        return alignment_sweep_variants(cls)


class NeedlemanWunschSim85(NeedlemanWunschSim65):
    """WFA2-only timing variant on ~85% identity pairs (the paper's dagger row)."""

    name = "needleman-wunsch-sim85"
    identity = 0.85
    boundary_convention = (
        "WFA2-only; ~85% identity pairs (y = substitution-mutated copy of x)"
    )


if __name__ == "__main__":
    from evals_common.harness import run_cases

    raise SystemExit(run_cases([NeedlemanWunsch()]))
