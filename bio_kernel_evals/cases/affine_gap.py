"""Affine-gap (Gotoh) global sequence alignment.

VecRec kernel: collapsed 2-state affine (H + E; F resolved in-row via Recurse),
with on-the-fly substitution via ``SubstSignal2D`` (same scheduling path as
``vec-rec/benchmarks/trop/affine_gap.py``)::

    E(i,j) = max( H(i-1,j) + GO+GE,  E(i-1,j) + GE )
    Ht(i,j) = max( H(i-1,j-1) + s(i,j), E(i,j) )
    F(i,j) = max( Ht(i,j-1) + GO+GE, F(i,j-1) + GE )
    H(i,j) = max( Ht(i,j), F(i,j) )

Baselines: native C++ libraries with an affine gap scheme, validated by
optimal-score equality against a textbook Gotoh DP (gap of length L costs
``gap_open + L*gap_extend``). See ``_alignment.py`` for the bridge.
"""

from __future__ import annotations

import numpy as np

from vecrec.expr import (
    Convolve,
    Impulse,
    Ith,
    Num,
    PointwiseMul,
    Recurse,
    RepeaterVec,
    RVec2D,
    SAdd,
    SignalVector,
    SignalVectorExpr,
    TIKernel,
    Type,
    Var2D,
    with_types,
)

# ComputedSignal2D (on-the-fly substitution) was dropped from newer vec-rec builds.
# It is only needed by the unused `_subst_signal` helper below; the active kernel
# path uses Var2D("s") + a match/mismatch profile, so import it optionally.
try:
    from vecrec.expr import ComputedSignal2D  # type: ignore
except ImportError:  # pragma: no cover - depends on vec-rec version
    ComputedSignal2D = None  # type: ignore
from vecrec.expr.base import SignalExpr
from vecrec.util import ElementType

from evals_common.harness import Baseline, EvalCase
from evals_common.parasail import parasail_baselines

from ._alignment import MATCH, MISMATCH, affine_reference, alignment_sweep_variants, sequence_inputs, similar_sequence_inputs

GO = -11           # gap open penalty
GE = -1            # gap extend penalty
GO_GE = GO + GE    # combined open+extend: -12

# On-the-fly substitution (vec-rec/benchmarks/trop/affine_gap.py). Scores are
# computed in-kernel from int8 sequence buffers — no O(N^2) profile/materialized s.
SUBST_SIGNAL_PRELUDE = r"""
template <typename vec_type, int n_rows, int block_size = 1, int block_row = 0>
struct SubstSignal2D {
    using elt_type = typename ElementType<vec_type>::type;
    static constexpr int lanes = sizeof(vec_type) / sizeof(elt_type);

    const int8_t *__restrict__ s1;
    const int8_t *__restrict__ s2;
    int stride;
    elt_type match_score;
    elt_type mismatch_score;
    int current_row;
    int my_col;
    vec_type s1v[n_rows];

    SubstSignal2D() = delete;
    SubstSignal2D(const int8_t *__restrict__ s1, const int8_t *__restrict__ s2, int stride,
                  elt_type match_score, elt_type mismatch_score)
        : s1(s1), s2(s2), stride(stride),
          match_score(match_score), mismatch_score(mismatch_score),
          current_row(block_row), my_col(0) { load_s1_row(); }

    inline __attribute__((always_inline)) void load_s1_row() {
        _Pragma("unroll") for (int i = 0; i < n_rows; i++) {
            int r = current_row - i;
            elt_type a = (r >= 0) ? (elt_type)s1[r] : (elt_type)(-1);
            s1v[i] = a;
        }
    }

    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base) {
        my_col += lanes;
        vec_type s2v;
        if constexpr (lanes == 1) {
            s2v = (elt_type)s2[my_col - lanes];
        } else {
            _Pragma("unroll") for (int k = 0; k < lanes; k++)
                s2v[k] = (elt_type)s2[my_col - lanes + k];
        }
        const vec_type mv = match_score;
        const vec_type mmv = mismatch_score;
        _Pragma("unroll") for (int i = 0; i < n_rows; i++) {
            if (current_row == 0) {
                // ProfileSignal2D boundary row uses sym[0]=0 (alphabet row A).
                reg[base + i] = (elt_type(0) == s2v) ? mv : mmv;
                // Column 0 is profile[0][0]=0 (uninitialized); Impulse seeds M[0,0]=0.
                if (my_col == lanes) reg[base + i][0] = elt_type(0);
            } else {
                reg[base + i] = (s1v[i] == s2v) ? mv : mmv;
            }
        }
    }

    void seek_col(int col_start) { my_col = col_start; }

    void reset_and_next_row() {
        current_row++;
        my_col = 0;
        load_s1_row();
    }
};
"""


def _subst_signal() -> ComputedSignal2D:
    # setup_code is emitted by evals_common/vecrec_cpp (reads in.x / in.y from input.json).
    return ComputedSignal2D(
        "s",
        cpp_struct="SubstSignal2D",
        prelude=SUBST_SIGNAL_PRELUDE,
        setup_code="",
        make_args=f"s1.data(), s2.data(), phys_w, {MATCH}, {MISMATCH}",
    )


def _build_expr(element_type: ElementType = ElementType.I16) -> SignalExpr:
    # Collapsed 2-state affine. Default I16 -- int16 lanes give 2x SIMD width
    # vs I32 (32 lanes/zmm instead of 16), matching the parasail scan_16
    # baseline -- but cpp_baselines() also sweeps I8/I32 so autoscheduling can
    # pick whichever width is actually fastest for a given N.
    with with_types(Type.TropMax, element_type):
        s = Var2D("s")   # precomputed profile (driver builds profile[4][phys]),
                         # matching parasail's profile lookup; avoids on-the-fly subst.
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
            Ht = SAdd(diag, E)
            F = Recurse(TIKernel([neginf, GE]), Convolve(TIKernel([neginf, GO_GE]), Ht))
            H = SAdd(Ht, F)
            return SignalVector([H, E])

        return Ith(RepeaterVec(2, body).at(0), 0)


class AffineGap(EvalCase):
    name = "affine-gap"
    vecrec_bytes_per_cell = 8  # I32 output (4B); subst computed on-the-fly
    report_gcups = False
    boundary_convention = (
        "vecrec: TropMax collapsed 2-state affine (I32) + SubstSignal2D; "
        "baselines: textbook affine global"
    )

    def alignment_score(self, query: str, ref: str) -> float:
        """Gotoh score: match/mismatch per column, a gap of length L costs GO + L*GE."""
        score, in_gap = 0, None
        for a, b in zip(query, ref):
            if a == "-" or b == "-":
                kind = "q" if a == "-" else "r"
                if in_gap != kind:
                    score += GO
                    in_gap = kind
                score += GE
            else:
                in_gap = None
                score += MATCH if a == b else MISMATCH
        return float(score)

    def __init__(self, rows: int = 128, cols: int = 128, seed: int = 0) -> None:
        self.rows = rows
        self.cols = cols
        self.seed = seed
        self.n_elements = rows * cols

    @classmethod
    def sweep_variants(cls) -> list[AffineGap]:
        return alignment_sweep_variants(cls)

    def make_inputs(self) -> dict[str, object]:
        return sequence_inputs(
            self.rows, self.cols, gap_open=-GO, gap_extend=-GE, seed=self.seed
        )

    def reference(self, **inputs: object) -> np.ndarray:
        return affine_reference(inputs["s"], GO, GE).flatten()

    def optimal_score(self, matrix: np.ndarray) -> float:
        return float(matrix[-1])

    def kernel_expr(self, element_type: ElementType = ElementType.I16) -> SignalExpr:
        return _build_expr(element_type)

    def baselines(self) -> list[Baseline]:
        return [Baseline("numpy_loop", lambda **kw: affine_reference(kw["s"], GO, GE).flatten())]

    def cpp_baselines(self) -> list:
        from evals_common.cpp import CppBaseline
        from evals_common.vecrec_cpp import vecrec_traceback_baselines_multi_width

        return [
            *vecrec_traceback_baselines_multi_width(self.kernel_expr, kind="affine"),
            *parasail_baselines("nw", prefix="affine"),
            # Exact end-to-end gap-affine Gotoh via WFA2-lib (wavefront).
            # Driver maps harness penalties into WFA2 distance form; cigar->score
            # is the similarity corner (match < 0 path), matching affine_reference.
            CppBaseline("wfa2_affine", "wfa2_align", "affine"),
            # SIMD 1-piece affine GLOBAL (native SSE2 on x86, sse2neon on ARM).
            # ksw2's gap(L) = gapo + L*gape is the same convention as
            # affine_reference's gap_open + L*gap_extend, so scores match exactly.
            CppBaseline("ksw2_gg2_sse", "ksw2_align", "gg2_sse"),
            # SIMD 1-piece affine EXTENSION (the kernel minimap2's ksw_extz2_sse
            # actually runs). Called in gap-fill mode (flag=0, zdrop=-1) so
            # ez.score is the global corner H(qlen,tlen) -- correctness-checkable
            # against affine_reference.
            CppBaseline(
                "ksw2_extz2_sse", "ksw2_align", "extz2_sse",
                extra_payload={"band_width": -1},
            ),
            # SIMD 2-piece dual-affine EXTENSION (ksw_extd2_sse) degenerated to
            # 1-piece by setting the second gap pair equal to the first
            # (q2=q, e2=e), so it computes the same Gotoh corner and is
            # correctness-checked against affine_reference.
            CppBaseline(
                "ksw2_extd2_sse", "ksw2_align", "extd2_sse",
                extra_payload={"band_width": -1, "gap_open2": -GO, "gap_extend2": -GE},
            ),
        ]


class AffineGapSim65(AffineGap):
    """WFA2-only timing variant on ~65% identity pairs (substitution-mutated).

    Same Gotoh recurrence/reference as the parent; only ``wfa2_affine`` is
    wired so ``--sweep`` can show WFA2 under moderate similarity without
    re-running parasail/ksw2/VecRec.
    """

    name = "affine-gap-sim65"
    identity = 0.65
    boundary_convention = (
        "WFA2-only; ~65% identity pairs (y = substitution-mutated copy of x)"
    )

    def make_inputs(self) -> dict[str, object]:
        return similar_sequence_inputs(
            self.rows, self.cols, gap_open=-GO, gap_extend=-GE, seed=self.seed,
            identity=self.identity,
        )

    def baselines(self) -> list[Baseline]:
        return []

    def cpp_baselines(self) -> list:
        from evals_common.cpp import CppBaseline

        return [CppBaseline("wfa2_affine", "wfa2_align", "affine")]

    @classmethod
    def sweep_variants(cls) -> list[AffineGapSim65]:
        return alignment_sweep_variants(cls)


class AffineGapSim85(AffineGapSim65):
    """WFA2-only timing variant on ~85% identity pairs (the paper's dagger row)."""

    name = "affine-gap-sim85"
    identity = 0.85
    boundary_convention = (
        "WFA2-only; ~85% identity pairs (y = substitution-mutated copy of x)"
    )


if __name__ == "__main__":
    from evals_common.harness import run_cases

    raise SystemExit(run_cases([AffineGap()]))
