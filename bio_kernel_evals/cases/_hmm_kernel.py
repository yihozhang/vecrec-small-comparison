"""vecrec kernel expression for the profile-HMM Viterbi recurrence (float TropMax).

Mirrors HMMER's core glocal M/I/D Viterbi (see ``_hmm.py``). Structurally this is
"affine-gap on a profile" (cf. ``vec-rec/benchmarks/trop/affine_gap.py``): a
``RepeaterVec`` carrying M/I/D across rows, M/I from the previous row (vectorized
across model position k) and D a within-row recurrence. The difference from affine
is that every transition coefficient is **position-varying** (depends on k), so the
constant ``TIKernel`` gap penalties become ``Var2D`` signals / ``TVKernel`` entries.

All score inputs are full ``L x (M+1)`` ``Var2D`` arrays materialized by the driver,
pre-aligned so the kernel reads each coefficient directly at output position k (the
``k-1`` transitions are pre-shifted; emissions are indexed by the observed residue;
``begin`` carries TBM(k-1) in row 0 and the zero element elsewhere). This avoids any
in-kernel coefficient shifting and the begin (first-row source) needs no special
primitive. The kernel emits ``H = max(M, I, D)`` as a 4th (output-only) component;
the corner ``H[L-1][M]`` is the Viterbi score ``max(M[L][M], D[L][M])``.

Input variable names (sorted alphabetically by vecrec codegen -> constructor order):
  a_dd, a_dm, a_ii, a_im, a_md, a_mi, a_mm, begin, emitI, emitM
"""

from __future__ import annotations

from vecrec.expr import (
    Convolve,
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
    TVKernel,
    Type,
    Var2D,
    with_types,
)
from vecrec.expr.base import SignalExpr
from vecrec.util import ElementType

# Ordered list of the 10 materialized 2D inputs (alphabetical == codegen ctor order).
INPUT_NAMES = ("a_dd", "a_dm", "a_ii", "a_im", "a_md", "a_mi", "a_mm",
               "begin", "emitI", "emitM")


def build_expr() -> SignalExpr:
    with with_types(Type.TropMax, ElementType.Float):
        emitM = Var2D("emitM")
        emitI = Var2D("emitI")
        begin = Var2D("begin")
        a_mm = Var2D("a_mm")
        a_im = Var2D("a_im")
        a_dm = Var2D("a_dm")
        a_mi = Var2D("a_mi")
        a_ii = Var2D("a_ii")
        a_md = Var2D("a_md")
        a_dd = Var2D("a_dd")
        neginf = Num(Type.TropMax.zero())
        z = TIKernel.z()

        def body(prev: RVec2D) -> SignalVectorExpr:
            pM = prev.at(0).at(0)  # M(i-1, k)
            pI = prev.at(1).at(0)  # I(i-1, k)
            pD = prev.at(2).at(0)  # D(i-1, k)   (prev.at(3) = prev H, unused output channel)

            # M(i,k) = emitM(k) + max( M/I/D(i-1,k-1)+T*M(k-1), begin(k) )
            # Convolve(z, ·) shifts by one column: value at k is the (k-1) predecessor.
            dM = PointwiseMul(Convolve(z, pM), Ith(a_mm, 0))
            dI = PointwiseMul(Convolve(z, pI), Ith(a_im, 0))
            dD = PointwiseMul(Convolve(z, pD), Ith(a_dm, 0))
            best_in = SAdd(SAdd(SAdd(dM, dI), dD), Ith(begin, 0))
            M = PointwiseMul(best_in, Ith(emitM, 0))

            # I(i,k) = emitI(k) + max( M(i-1,k)+TMI(k), I(i-1,k)+TII(k) )
            iM = PointwiseMul(pM, Ith(a_mi, 0))
            iI = PointwiseMul(pI, Ith(a_ii, 0))
            I = PointwiseMul(SAdd(iM, iI), Ith(emitI, 0))

            # D(i,k) = max( M(i,k-1)+TMD(k-1), D(i,k-1)+TDD(k-1) ) -- within-row recurrence
            # with position-varying feedback (TVKernel coefficients from Var2D rows).
            ext_D = Convolve(TVKernel([neginf, Ith(a_md, 0)]), M)
            D = Recurse(TVKernel([neginf, Ith(a_dd, 0)]), ext_D)

            return SignalVector([M, I, D])

        # Emit H = max(M, I, D) by combining the three component outputs (mirrors the
        # affine eval's H = max(M, Dx, Dy)); the corner H[L-1][M] is the Viterbi score.
        rv = RepeaterVec(3, body)
        return SAdd(SAdd(Ith(rv.at(0), 0), Ith(rv.at(1), 0)), Ith(rv.at(2), 0))
