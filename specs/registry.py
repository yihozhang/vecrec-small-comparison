"""Registry of recurrence specs to benchmark.

Each entry maps a benchmark name to the C++ kernel/struct name and a builder that
returns the *unscheduled* VecRec expression. ``scripts/gen_kernels.py`` feeds the
expression to ``vecrec.codegen.find_best_schedule_and_emit_header`` to find the best
schedule and emit ``generated/<kernel_name>.h``.

Expressions are reconstructed inline (with a pointer to the upstream spec under
``vec-rec/benchmarks/*`` where one exists) so this project depends only on the
``vecrec`` library, not on vec-rec's Python benchmark harness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

from vecrec.expr import (
    Convolve,
    Convolve2D,
    KernelMatrix,
    Num,
    PointwiseDiv,
    Recurse,
    Recurse2D,
    RecurseV,
    SignalVector,
    TIKernel,
    TIKernel2D,
    TVKernel,
    Type,
    Var,
    Var2D,
    VectorAt,
    with_types,
)
from vecrec.expr.base import SignalExpr, SignalExpr2D
from vecrec.expr.signal import ComputedSignal2D
from vecrec.expr.signal_ops import Ith, PointwiseMul, SSub
from vecrec.util import ElementType

INF = float("inf")


@dataclass(frozen=True)
class Spec:
    name: str          # benchmark name / slug (hyphenated) — dict key, CLI arg, CSV key
    kernel_name: str   # C++ struct / header name (underscores)
    dim: str           # "1d" or "2d" — informational
    semiring: str      # "arith" | "tropmin" — informational
    build_expr: Optional[Callable[[], "SignalExpr | SignalExpr2D"]]
    display_name: str = ""  # human-readable name for display (e.g. "Exponential Smoothing")
    description: str = ""
    note: str = ""     # e.g. why a benchmark is unsupported


# ── 1D arithmetic IIR filters ─────────────────────────────────────────────────

def _exponential_smoothing(a: float = 0.8) -> SignalExpr:
    # f(x) = a*f(x-1) + (1-a)*g(x)
    with with_types(Type.Arith, ElementType.Float):
        return Recurse(TIKernel([0, a]), Convolve(TIKernel([1 - a]), Var("g")))


def _double_exp_smoothing(alpha: float = 0.5, beta: float = 0.3) -> SignalExpr:
    # Holt's linear (double) exponential smoothing, rewritten so the trend b_t no
    # longer depends on the *current* level l_t (VecRec recurrence kernels only allow
    # state at t to depend on state at t-1):
    #   l_t = (1-a) l_{t-1} + (1-a) b_{t-1} + a    g_t
    #   b_t = -a*B l_{t-1} + (1-a*B) b_{t-1} + a*B g_t
    # Encoded as a 2x2 mutual recurrence (precedent: arith/mutual_recurse.py).
    # Output = the level l_t. Coupled kernel is stable for these (alpha, beta).
    with with_types(Type.Arith, ElementType.Float):
        oma = 1.0 - alpha          # (1 - alpha)
        ab = alpha * beta          # alpha*beta
        A = KernelMatrix(
            [
                [TIKernel([0.0, oma]), TIKernel([0.0, oma])],
                [TIKernel([0.0, -ab]), TIKernel([0.0, 1.0 - ab])],
            ]
        )
        G = SignalVector(
            [
                Convolve(TIKernel([alpha]), Var("g")),  # a*g_t  -> level
                Convolve(TIKernel([ab]), Var("g")),     # a*B*g_t -> trend
            ]
        )
        F = RecurseV(A, G)
        return VectorAt(F, 0)  # output the smoothed level l_t


def _first_order_high_pass(a: float = 0.8) -> SignalExpr:
    # vec-rec/benchmarks/arith/first_order_high_pass.py
    # f(x) = a*(f(x-1) + g(x) - g(x-1))
    with with_types(Type.Arith, ElementType.Float):
        return Recurse(TIKernel([0, a]), Convolve(TIKernel([a, -a]), Var("g")))


def _second_order_high_pass() -> SignalExpr:
    # vec-rec/benchmarks/arith/second_order_high_pass.py
    # f(x) = 1.6*f(x-1) - 0.64*f(x-2) + g(x) - 2*g(x-1) + g(x-2)
    with with_types(Type.Arith, ElementType.Float):
        return Recurse(TIKernel([0, 1.6, -0.64]), Convolve(TIKernel([1, -2, 1]), Var("g")))


def _second_order_low_pass() -> SignalExpr:
    # vec-rec/benchmarks/arith/second_order_ti.py
    # f(x) = 1.8*f(x-1) - 0.9*f(x-2) + g(x)
    with with_types(Type.Arith, ElementType.Float):
        return Recurse(TIKernel([0, 1.8, -0.9]), Var("g"))


# Recursive Gaussian (Young & van Vliet 1995, causal), sigma=2.0.
# vec-rec/benchmarks/arith/recursive_gaussian.py
_SIGMA = 2.0
_q = 1.31564 * (math.sqrt(1.0 + 0.490811 * _SIGMA * _SIGMA) - 1.0)  # sigma < 2.5
_b0 = 1.57825 + 2.44413 * _q + 1.4281 * _q**2 + 0.422205 * _q**3
_b1 = 2.44413 * _q + 2.85619 * _q**2 + 1.26661 * _q**3
_b2 = -1.4281 * _q**2 - 1.26661 * _q**3
_b3 = 0.422205 * _q**3
GAUSS_A1 = _b1 / _b0
GAUSS_A2 = _b2 / _b0
GAUSS_A3 = _b3 / _b0
GAUSS_B = 1.0 - (GAUSS_A1 + GAUSS_A2 + GAUSS_A3)


def _recursive_gaussian() -> SignalExpr:
    # f(x) = a1*f(x-1) + a2*f(x-2) + a3*f(x-3) + B*g(x)
    with with_types(Type.Arith, ElementType.Float):
        return Recurse(
            TIKernel([0, GAUSS_A1, GAUSS_A2, GAUSS_A3]),
            Convolve(TIKernel([GAUSS_B]), Var("g")),
        )


# ── 1D time-varying (non-uniform) recurrences ─────────────────────────────────

def _nonuniform_recursive_filter() -> SignalExpr:
    # vec-rec/benchmarks/arith/second_order_tv.py
    # f(x) = a(x)*f(x-1) + b(x)*f(x-2) + g(x)  — coefficients vary per sample.
    # Driven by an input stream g so it is a genuine (BIBO-stable) filter; the
    # benchmark keeps |a|+|b| < 1 so the time-varying recurrence stays bounded.
    with with_types(Type.Arith, ElementType.Float):
        return Recurse(TVKernel([Num(0.0), Var("a"), Var("b")]), Var("g"))


# w(x) = 1/x is a pure function of stream position, so instead of feeding it as a
# materialized input array we compute it on the fly with a ComputedSignal2D whose
# C++ struct tracks the column position and emits a fully-vectorized reciprocal
# (one vdivps/vdivpd per run() — no memory traffic for w). Instantiated by codegen
# as WelfordWeight2D<vec_type, n_rows, block_size, block_row>; used via Ith(_,0) as
# a 1D signal. x is 1-based (element i => x = i+1).
WELFORD_W_PRELUDE = r"""
template <typename vec_type, int n_rows, int block_size = 1, int block_row = 0>
struct WelfordWeight2D {
    using elt_type = typename ElementType<vec_type>::type;
    static constexpr int lanes = sizeof(vec_type) / sizeof(elt_type);
    int my_col;
    WelfordWeight2D() : my_col(0) {}
    template <typename Reg_t>
    inline __attribute__((always_inline)) void run(Reg_t &reg, int base) {
        vec_type wv;
        if constexpr (lanes == 1) {
            wv = (elt_type)1 / (elt_type)(my_col + 1);
        } else {
            vec_type posv;
            _Pragma("unroll") for (int k = 0; k < lanes; k++) posv[k] = (elt_type)(my_col + 1 + k);
            wv = (elt_type)1 / posv;   // fully-vectorized reciprocal
        }
        _Pragma("unroll") for (int i = 0; i < n_rows; i++) reg[base + i] = wv;
        my_col += lanes;
    }
    void reset_and_next_row() { my_col = 0; }
};
"""


def _welford_mean() -> SignalExpr:
    # Welford running mean: M(x) = M(x-1) + (a(x) - M(x-1)) / x
    #                            = (1 - w(x))*M(x-1) + w(x)*a(x),  w(x) = 1/x.
    with with_types(Type.Arith, ElementType.Float):
        w = Ith(ComputedSignal2D("w", cpp_struct="WelfordWeight2D",
                                 prelude=WELFORD_W_PRELUDE, setup_code="", make_args=""), 0)
        a = Var("a")
        return Recurse(TVKernel([Num(0.0), SSub(Num(1.0), w)]), PointwiseMul(w, a))


# Continued fraction via the Wallis–Euler recurrence:
#   h(x) = a(x)*h(x-1) + b(x)*h(x-2),   f(x) = h(x)/h(x-1)
# (skipping the final division would leave a rational approximant; see
# vec-rec/benchmarks/arith/continued_fraction.py).
#
# h grows/decays geometrically, so a single long stream overflows within ~100
# steps. We instead run many *independent* continued fractions of CF_SEGMENT
# steps each: one CF per harness "row" (row_size = CF_SEGMENT), relying on the
# recurrence state resetting at each row boundary. A seed input s injects h=1 at
# each segment's first step (after reset the history is zero, so h(0)=s(0)).
# Emitted in **double** (float64) so h stays finite across a segment; VecRec
# dilates this genuinely lag-1 recurrence to vectorize within each segment.
CF_SEGMENT = 512      # steps per continued fraction (== harness row_size)


def _continued_fraction() -> SignalExpr:
    with with_types(Type.Arith, ElementType.Double):
        # h(x) = a(x)*h(x-1) + b(x)*h(x-2) + s(x)
        h = Recurse(TVKernel([Num(0.0), Var("a"), Var("b")]), Var("s"))
        return PointwiseDiv(h, Convolve(TIKernel.z(), h))  # f = h(x)/h(x-1)


# ── 2D arithmetic ─────────────────────────────────────────────────────────────

def _box_filter_sat(D: int = 11) -> SignalExpr:
    # vec-rec/benchmarks/arith/box_filter_summed_area_table.py
    with with_types(Type.Arith, ElementType.Float):
        kernel = [[0.0] * D for _ in range(D)]
        kernel[0][0] = kernel[D - 1][D - 1] = 1.0
        kernel[D - 1][0] = kernel[0][D - 1] = -1.0
        fir = Convolve2D(TIKernel2D(kernel), Var2D("g"))
        sat_kernel = TIKernel2D([[0.0, 1.0], [1.0, -1.0]])
        return Recurse2D(sat_kernel, fir).at(0)


def _repeated_box_filter(D: int = 7) -> SignalExpr:
    # vec-rec/benchmarks/arith/repeated_box_filter.py — 3 cascaded SAT box passes.
    # Each pass divides by D*D, so it is a moving *average* (values stay bounded).
    with with_types(Type.Arith, ElementType.Float):
        def one_pass(sig: SignalExpr2D) -> SignalExpr2D:
            k = [[0.0] * D for _ in range(D)]
            k[0][0] = k[D - 1][D - 1] = 1.0
            k[D - 1][0] = k[0][D - 1] = -1.0
            df = Convolve2D(TIKernel2D(k), sig)
            df2 = PointwiseDiv(df, Num(float(D * D)))
            return Recurse2D(TIKernel2D([[0.0, 1.0], [1.0, -1.0]]), df2)

        b1 = one_pass(Var2D("g"))
        b2 = one_pass(b1)
        b3 = one_pass(b2)
        return b3.at(0)


# ── 2D tropical (min, +) ──────────────────────────────────────────────────────

_DT_KERNEL = [
    [INF, 1.0],
    [1.0, math.sqrt(2), math.sqrt(5)],
    [INF, math.sqrt(5)],
]


def _distance_transform() -> SignalExpr:
    # vec-rec/benchmarks/trop/distance_transform.py
    with with_types(Type.TropMin, ElementType.Float):
        g = Var2D("g")
        return Recurse2D(TIKernel2D(_DT_KERNEL), g.at(0)).at(0)


# ── Registry ──────────────────────────────────────────────────────────────────
SPECS: dict[str, Spec] = {
    s.name: s
    for s in [
        Spec("exponential-smoothing", "exponential_smoothing", "1d", "arith",
             _exponential_smoothing, display_name="Exponential Smoothing",
             description="f(x)=0.8*f(x-1)+0.2*g(x)"),
        Spec("double-exp-smoothing", "double_exp_smoothing", "1d", "arith",
             _double_exp_smoothing, display_name="Double Exponential Smoothing",
             description="Holt's double exponential smoothing (level+trend) as a 2x2 mutual recurrence"),
        Spec("box-filter-sat", "box_filter_sat", "2d", "arith",
             _box_filter_sat, display_name="Box Filter (Summed Area Table)",
             description="Box filter (D=11) via 2D difference kernel + summed area table"),
        Spec("repeated-box-filter", "repeated_box_filter", "2d", "arith",
             _repeated_box_filter, display_name="Repeated Box Filter",
             description="3-pass box filter (D=7) approximating a Gaussian"),
        Spec("first-order-high-pass", "first_order_high_pass", "1d", "arith",
             _first_order_high_pass, display_name="First-Order High Pass Filter",
             description="f(x)=0.8*(f(x-1)+g(x)-g(x-1))"),
        Spec("recursive-gaussian", "recursive_gaussian", "1d", "arith",
             _recursive_gaussian, display_name="Recursive Gaussian Filter",
             description="Causal recursive Gaussian (Young & van Vliet, sigma=2): 3rd-order IIR"),
        Spec("second-order-high-pass", "second_order_high_pass", "1d", "arith",
             _second_order_high_pass, display_name="Second-Order High Pass Filter",
             description="f(x)=1.6*f(x-1)-0.64*f(x-2)+g(x)-2*g(x-1)+g(x-2)"),
        Spec("second-order-low-pass", "second_order_low_pass", "1d", "arith",
             _second_order_low_pass, display_name="Second-Order Low Pass Filter",
             description="f(x)=1.8*f(x-1)-0.9*f(x-2)+g(x)"),
        Spec("distance-transform", "distance_transform", "2d", "tropmin",
             _distance_transform, display_name="Distance Transform",
             description="2D distance transform (TropMin) with knight-move kernel"),
        Spec("nonuniform-recursive-filter", "nonuniform_recursive_filter", "1d", "arith",
             _nonuniform_recursive_filter, display_name="Non-uniform Recursive Filter",
             description="Time-varying 2nd-order IIR: f(x)=a(x)*f(x-1)+b(x)*f(x-2)+g(x)"),
        Spec("welford-mean", "welford_mean", "1d", "arith",
             _welford_mean, display_name="Welford Mean",
             description="Welford running mean M(x)=M(x-1)+(a(x)-M(x-1))/x as a 1st-order TV recurrence"),
        Spec("continued-fraction", "continued_fraction", "1d", "arith",
             _continued_fraction, display_name="Continued Fraction",
             description="Wallis-Euler continued fraction h(x)=a(x)h(x-1)+b(x)h(x-2), f=h(x)/h(x-1); "
                         "batched 512-step segments in float64 (see registry note)"),
    ]
}
