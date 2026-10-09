"""Isolated check that the VecRec semi-global ("glocal") affine kernel computes
Bowtie2's end-to-end inner-loop recurrence correctly.

The end-to-end recurrence is read-global / reference-local: the alignment may
start at any reference column with no penalty, so the boundary row above the
first read row is 0 everywhere (H(-1, j) = 0).  This is exactly what the new
`RepeaterVec(..., boundary=(one, neginf))` codegen path produces.  We run the
generated kernel over deterministic sequences and compare H against a plain
Python semi-global affine DP.

Usage:
    uv run python end_to_end_baselines/scripts/verify_semiglobal_kernel.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from vecrec.codegen import CodeGen, generate_kernel_executable
from vecrec.expr import (
    ComputedSignal2D, Convolve, Impulse, Ith, Num, PointwiseMul, Recurse,
    RepeaterVec, RVec2D, SAdd, SignalVector, SignalVectorExpr, TIKernel, Type,
    with_types,
)
from vecrec.optimize import get_candidate_schedules
from vecrec.util import ElementType

GO, GE = -5, -3
GO_GE = GO + GE
MATCH, MISMATCH = 2, -6
ALPHA = 4
_NEGINF = int(np.iinfo(np.int32).min // 8)

# On-the-fly match/mismatch substitution (same struct the affine benchmark uses).
SUBST_PRELUDE = r"""
template <typename vec_type, int n_rows, int block_size = 1, int block_row = 0>
struct SubstSignal2D {
    using elt_type = typename ElementType<vec_type>::type;
    static constexpr int lanes = sizeof(vec_type) / sizeof(elt_type);
    const int8_t *__restrict__ s1; const int8_t *__restrict__ s2;
    int stride; elt_type match_score; elt_type mismatch_score;
    int current_row; int my_col; vec_type s1v[n_rows];
    SubstSignal2D() = delete;
    SubstSignal2D(const int8_t *s1, const int8_t *s2, int stride, elt_type m, elt_type mm)
        : s1(s1), s2(s2), stride(stride), match_score(m), mismatch_score(mm),
          current_row(block_row), my_col(0) { load_s1_row(); }
    inline __attribute__((always_inline)) void load_s1_row() {
        _Pragma("unroll") for (int i = 0; i < n_rows; i++) {
            int r = current_row - i; s1v[i] = (r >= 0) ? (elt_type)s1[r] : (elt_type)(-1);
        }
    }
    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base) {
        vec_type s2v;
        if constexpr (lanes == 1) { s2v = (elt_type)s2[my_col]; }
        else { _Pragma("unroll") for (int k = 0; k < lanes; k++) s2v[k] = (elt_type)s2[my_col + k]; }
        const vec_type mv = match_score; const vec_type mmv = mismatch_score;
        _Pragma("unroll") for (int i = 0; i < n_rows; i++) reg[base + i] = (s1v[i] == s2v) ? mv : mmv;
        my_col += lanes;
    }
    void reset_and_next_row() { current_row++; my_col = 0; load_s1_row(); }
    void seek_col(int col_start) { my_col = col_start; }
};
"""
SUBST_SETUP = (
    f"std::vector<int8_t> s1(N / row_size + 64), s2(row_size);\n"
    f"for (int i = 0; i < N / row_size; i++) s1[i] = (int8_t)(i % {ALPHA});\n"
    f"for (int j = 0; j < row_size; j++) s2[j] = (int8_t)(j % {ALPHA});\n"
)


def build_expr() -> SignalExpr:  # type: ignore[name-defined]
    with with_types(Type.TropMax, ElementType.I32):
        s = ComputedSignal2D("s", cpp_struct="SubstSignal2D", prelude=SUBST_PRELUDE,
                             setup_code=SUBST_SETUP,
                             make_args=f"s1.data(), s2.data(), row_size, {MATCH}, {MISMATCH}")
        neginf = Type.TropMax.zero()
        one = Type.TropMax.one()
        z = TIKernel.z()

        def body(prev: RVec2D) -> SignalVectorExpr:
            prev_h = prev.at(0).at(0)
            prev_e = prev.at(1).at(0)
            s_row = Ith(s, 0)
            E_new = SAdd(PointwiseMul(prev_h, Num(GO_GE)), PointwiseMul(prev_e, Num(GE)))
            diag = SAdd(PointwiseMul(Convolve(z, prev_h), s_row),
                        PointwiseMul(Impulse(one), s_row))
            Ht = SAdd(diag, E_new)
            F = Recurse(TIKernel([neginf, GE]), Convolve(TIKernel([neginf, GO_GE]), Ht))
            H = SAdd(Ht, F)
            return SignalVector([H, E_new])

        # Semi-global boundary: H(-1, j) = 0 for all j.
        return Ith(RepeaterVec(2, body, boundary=(one, neginf)).at(0), 0)


def reference(nrow: int, ncol: int) -> np.ndarray:
    s1 = np.array([i % ALPHA for i in range(nrow)])
    s2 = np.array([j % ALPHA for j in range(ncol)])

    def sub(i, j):
        return MATCH if s1[i] == s2[j] else MISMATCH

    H = np.full((nrow, ncol), _NEGINF, dtype=np.int64)
    E = np.full((nrow, ncol), _NEGINF, dtype=np.int64)
    F = np.full((nrow, ncol), _NEGINF, dtype=np.int64)
    for i in range(nrow):
        for j in range(ncol):
            # H(-1, j) = 0 for all j (free reference start); H(i, -1) = -inf.
            h_diag = 0 if i == 0 else (int(H[i - 1, j - 1]) if j > 0 else _NEGINF)
            diag = h_diag + sub(i, j)
            h_up = 0 if i == 0 else int(H[i - 1, j])      # H(-1, j) = 0
            e_up = int(E[i - 1, j]) if i > 0 else _NEGINF
            E[i, j] = max(h_up + GO_GE, e_up + GE)
            h_left = int(H[i, j - 1]) if j > 0 else _NEGINF
            f_left = int(F[i, j - 1]) if j > 0 else _NEGINF
            F[i, j] = max(h_left + GO_GE, f_left + GE)
            H[i, j] = max(diag, int(E[i, j]), int(F[i, j]))
    return H.flatten()


def main() -> None:
    expr = build_expr()
    scheds = get_candidate_schedules(expr, 256)
    codegen = CodeGen()
    row_size = 256
    nrow = 8
    n = nrow * row_size

    ok_all = True
    for idx in (0, len(scheds) // 2, len(scheds) - 1):
        runner = generate_kernel_executable(codegen, scheds[idx], row_size=row_size)
        out, _, _ = runner.run({}, n=n)
        exp = reference(nrow, row_size)
        m = min(len(out), len(exp))
        a = np.maximum(np.asarray(out[:m], dtype=np.int64), _NEGINF // 2)
        b = np.maximum(np.asarray(exp[:m], dtype=np.int64), _NEGINF // 2)
        same = np.array_equal(a, b)
        ok_all &= same
        print(f"  schedule k{idx}: {'PASS' if same else 'FAIL'}")
        if not same:
            bad = int(np.where(a != b)[0][0])
            print(f"    first diff at flat idx {bad}: got {a[bad]}, expected {b[bad]}")

    print("PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
